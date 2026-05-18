"""Zhihu crawler integration via MediaCrawler_zhihu subprocess.

This module is the boundary between inkprint and the MediaCrawler_zhihu
third-party crawler living under crawlers/zhihu/. It rewrites the crawler's
config, spawns the crawler under its own venv, and harvests the output JSON.
"""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import samples

ROOT = Path(__file__).resolve().parent.parent
CRAWLER_DIR = ROOT / "crawlers" / "zhihu"
CONFIG_FILE = CRAWLER_DIR / "config" / "zhihu_config.py"
OUTPUT_DIR = CRAWLER_DIR / "data" / "zhihu" / "json"
SAMPLES_DIR = ROOT / "samples" / "zhihu"

# Proxy env vars must be unset before invoking the crawler — zhihu is a
# mainland-CN site that doesn't need a tunnel, and httpx picks up SOCKS proxies
# from env even when proxy=None is passed explicitly.
PROXY_ENV_KEYS = (
    "all_proxy",
    "ALL_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
)

CRAWL_TIMEOUT_SECONDS = 600

URL_LIST_PATTERN = re.compile(r"ZHIHU_CREATOR_URL_LIST\s*=\s*\[[^\]]*\]", re.DOTALL)


class CrawlerError(RuntimeError):
    """Anything that goes wrong inside the crawler boundary."""


@dataclass(frozen=True)
class CrawlResult:
    sample_path: Path
    item_count: int  # items in this raw dump
    total_count: int  # union size after merging with history
    added_count: int
    updated_count: int
    unchanged_count: int


def _rewrite_config(zhihu_url: str) -> None:
    if not CONFIG_FILE.exists():
        raise CrawlerError(
            f"crawler config not found: {CONFIG_FILE}. "
            "Did you clone MediaCrawler_zhihu under crawlers/zhihu/?"
        )
    source = CONFIG_FILE.read_text(encoding="utf-8")
    replacement = f'ZHIHU_CREATOR_URL_LIST = [\n    "{zhihu_url}",\n]'
    if not URL_LIST_PATTERN.search(source):
        raise CrawlerError("could not locate ZHIHU_CREATOR_URL_LIST in config")
    CONFIG_FILE.write_text(URL_LIST_PATTERN.sub(replacement, source), encoding="utf-8")


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


def _latest_output() -> Path:
    if not OUTPUT_DIR.exists():
        raise CrawlerError(f"no output dir produced: {OUTPUT_DIR}")
    candidates = sorted(OUTPUT_DIR.glob("creator_contents_*.json"))
    if not candidates:
        raise CrawlerError("crawler produced no creator_contents_*.json")
    return candidates[-1]


def _harvest(persona_id: int, output_file: Path) -> CrawlResult:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    persona_dir = SAMPLES_DIR / str(persona_id)
    persona_dir.mkdir(exist_ok=True)

    # Snapshot the prior union BEFORE copying the new dump in.
    prior = {
        item["content_id"]: item
        for item in samples.load_merged(persona_id)
        if isinstance(item.get("content_id"), str)
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = persona_dir / f"{timestamp}.json"
    shutil.copy2(output_file, target)

    with target.open(encoding="utf-8") as f:
        dump = json.load(f)
    dump = dump if isinstance(dump, list) else []

    added = updated = unchanged = 0
    for item in dump:
        cid = item.get("content_id")
        if not isinstance(cid, str):
            continue
        old = prior.get(cid)
        if old is None:
            added += 1
        elif samples.updated_ts(item) > samples.updated_ts(old):
            updated += 1
        else:
            unchanged += 1

    return CrawlResult(
        sample_path=target,
        item_count=len(dump),
        total_count=len(samples.load_merged(persona_id)),
        added_count=added,
        updated_count=updated,
        unchanged_count=unchanged,
    )


def sync(persona_id: int, zhihu_url: str) -> CrawlResult:
    """Run MediaCrawler against zhihu_url and harvest output into samples/.

    Blocks until the crawler exits (up to CRAWL_TIMEOUT_SECONDS). The first
    invocation triggers an interactive QR-code login in a Chrome window;
    subsequent invocations reuse the persisted CDP profile.
    """
    _rewrite_config(zhihu_url)
    cmd = ["uv", "run", "main.py", "--platform", "zhihu", "--lt", "qrcode", "--type", "creator"]
    proc = subprocess.run(
        cmd,
        cwd=CRAWLER_DIR,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=CRAWL_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise CrawlerError(f"crawler exited with code {proc.returncode}:\n{tail}")
    return _harvest(persona_id, _latest_output())
