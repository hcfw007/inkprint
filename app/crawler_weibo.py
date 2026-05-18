"""Weibo crawler integration via dataabc/weibo-crawler subprocess.

The third-party crawler is HTTP-only (no browser) but requires a logged-in
cookie. We hand it the user_id_list + cookie through its config.json, run it
under its own venv with proxy env stripped, then transform its native dump
into the common sample schema before writing to samples/weibo/{persona_id}/.

Cookie acquisition lives in browser_auth.py — this module just expects a
non-empty cookie string from the caller.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import json5

from . import samples

ROOT = Path(__file__).resolve().parent.parent
CRAWLER_DIR = ROOT / "crawlers" / "weibo"
CONFIG_FILE = CRAWLER_DIR / "config.json"
CONFIG_TEMPLATE = CRAWLER_DIR / "config_default.json.bak"
OUTPUT_BASE = CRAWLER_DIR / "weibo_data"
SAMPLES_DIR = ROOT / "samples" / "weibo"

# We override only the fields we actively control. Everything else (anti-ban
# tunables, user agents, retry budgets, ...) stays at the crawler's defaults
# so we do not have to track its schema across upgrades.
CONFIG_OVERRIDES: tuple[str, ...] = (
    "user_id_list",
    "cookie",
    "write_mode",
    "output_directory",
    "user_id_as_folder_name",
    "since_date",
    "end_date",
    "original_pic_download",
    "retweet_pic_download",
    "original_video_download",
    "retweet_video_download",
    "original_live_photo_download",
    "retweet_live_photo_download",
    "download_comment",
    "download_repost",
    "remove_html_tag",
)

PROXY_ENV_KEYS = (
    "all_proxy",
    "ALL_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
)

CRAWL_TIMEOUT_SECONDS = 900


class CrawlerError(RuntimeError):
    """Anything that goes wrong inside the weibo crawler boundary."""


@dataclass(frozen=True)
class CrawlResult:
    sample_path: Path
    item_count: int
    total_count: int
    added_count: int
    updated_count: int
    unchanged_count: int


def _read_template_config() -> dict:
    """Load the crawler's pristine config as our base.

    Prefer config_default.json.bak (a snapshot taken on first install,
    preserves the json5 comments); fall back to config.json if that's
    missing (works for both json5 originals and plain JSON rewrites).
    """
    sources = [p for p in (CONFIG_TEMPLATE, CONFIG_FILE) if p.exists()]
    if not sources:
        raise CrawlerError(
            f"no weibo config template found under {CRAWLER_DIR}. "
            "Did you clone dataabc/weibo-crawler under crawlers/weibo/?"
        )
    return json5.loads(sources[0].read_text(encoding="utf-8"))


def _write_config(uid: str, cookie: str) -> None:
    cfg = _read_template_config()
    overrides = {
        "user_id_list": [uid],
        "cookie": cookie,
        "write_mode": ["json"],
        "output_directory": "weibo_data",
        "user_id_as_folder_name": 1,
        "since_date": "2005-01-01",
        "end_date": "",
        "original_pic_download": 0,
        "retweet_pic_download": 0,
        "original_video_download": 0,
        "retweet_video_download": 0,
        "original_live_photo_download": 0,
        "retweet_live_photo_download": 0,
        "download_comment": 0,
        "download_repost": 0,
        "remove_html_tag": 1,
    }
    for key in CONFIG_OVERRIDES:
        if key in overrides:
            cfg[key] = overrides[key]
    # Seed the bak file on first run so future syncs always have a clean base.
    if not CONFIG_TEMPLATE.exists() and CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, CONFIG_TEMPLATE)
    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


def _latest_output(uid: str) -> Path:
    user_dir = OUTPUT_BASE / uid / "json"
    if not user_dir.exists():
        raise CrawlerError(f"crawler produced no output dir: {user_dir}")
    candidates = sorted(user_dir.glob("*.json"))
    if not candidates:
        raise CrawlerError(f"no json files under {user_dir}")
    return candidates[-1]


def _to_unix_ts(created_at: str) -> int:
    """weibo-crawler emits e.g. 'Mon May 18 13:00:00 +0800 2026' or ISO-ish strings."""
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(created_at, fmt).timestamp())
        except (ValueError, TypeError):
            continue
    return 0


def _normalize_weibo(item: dict) -> dict | None:
    text = item.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    wid = item.get("id")
    if wid is None:
        return None
    cid = str(wid)
    created = _to_unix_ts(item.get("created_at") or "")
    return {
        "platform": "weibo",
        "content_id": cid,
        "content_type": "weibo",
        "content_text": text,
        "content_url": f"https://m.weibo.cn/detail/{cid}",
        "title": "",
        "question_title": "",
        "created_time": created,
        "updated_time": created,
        "voteup_count": int(item.get("attitudes_count") or 0),
        "comment_count": int(item.get("comments_count") or 0),
        "reposts_count": int(item.get("reposts_count") or 0),
    }


def _harvest(persona_id: int, uid: str) -> CrawlResult:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    persona_dir = SAMPLES_DIR / str(persona_id)
    persona_dir.mkdir(exist_ok=True)

    prior = {
        item["content_id"]: item
        for item in samples.load_merged(persona_id, platform="weibo")
        if isinstance(item.get("content_id"), str)
    }

    raw_path = _latest_output(uid)
    with raw_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    raw_weibos = payload.get("weibo") if isinstance(payload, dict) else []
    normalized = [n for n in (_normalize_weibo(w) for w in raw_weibos) if n is not None]

    added = updated = unchanged = 0
    for item in normalized:
        old = prior.get(item["content_id"])
        if old is None:
            added += 1
        elif samples.updated_ts(item) > samples.updated_ts(old):
            updated += 1
        else:
            unchanged += 1

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = persona_dir / f"{timestamp}.json"
    target.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")

    return CrawlResult(
        sample_path=target,
        item_count=len(normalized),
        total_count=len(samples.load_merged(persona_id, platform="weibo")),
        added_count=added,
        updated_count=updated,
        unchanged_count=unchanged,
    )


def sync(persona_id: int, uid: str, cookie: str) -> CrawlResult:
    """Run weibo-crawler against uid+cookie and harvest into samples/weibo/."""
    if not cookie.strip():
        raise CrawlerError("empty cookie; trigger browser_auth.get_weibo_cookie() first")
    _write_config(uid, cookie)
    venv_python = CRAWLER_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise CrawlerError(
            f"weibo-crawler venv missing at {venv_python}. Run "
            "`cd crawlers/weibo && uv venv --python 3.11 .venv && "
            "uv pip install -r requirements.txt`."
        )
    proc = subprocess.run(
        [str(venv_python), "weibo.py"],
        cwd=CRAWLER_DIR,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=CRAWL_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise CrawlerError(f"weibo crawler exited {proc.returncode}:\n{tail}")
    return _harvest(persona_id, uid)
