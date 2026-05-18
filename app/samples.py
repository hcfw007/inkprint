"""Sample browser — read-only, multi-platform access to crawled posts.

Each platform's crawler writes timestamped dumps under
samples/{platform}/{persona_id}/*.json. Items are normalized to a common
shape (see required_fields below) so downstream code does not have to
branch on platform.

The "merged view" of a persona is the union across all platforms, deduped
by (platform, content_id) and kept fresh by updated_time.
"""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"

SNIPPET_LEN = 80

# Each normalized sample item should carry at minimum these fields. Crawlers
# inject platform="zhihu" / "weibo" and a stable string content_id.
COMMON_FIELDS = ("platform", "content_id", "content_text")


@dataclass(frozen=True)
class SampleSummary:
    idx: int
    platform: str
    content_type: str
    question_title: str
    snippet: str
    char_count: int
    voteup_count: int


def _platform_dirs() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return [p for p in SAMPLES_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]


def _persona_files(persona_id: int, platform: str | None = None) -> list[tuple[str, Path]]:
    """Return (platform, path) for every dump belonging to this persona."""
    pairs = []
    for plat_dir in _platform_dirs():
        if platform is not None and plat_dir.name != platform:
            continue
        persona_dir = plat_dir / str(persona_id)
        if persona_dir.exists():
            for path in sorted(persona_dir.glob("*.json")):
                pairs.append((plat_dir.name, path))
    return pairs


def _read_dump(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def updated_ts(item: dict) -> int:
    """Best-effort 'last edited' timestamp; falls back to created_time / 0."""
    return int(item.get("updated_time") or item.get("created_time") or 0)


def _normalize(item: dict, platform: str) -> dict | None:
    """Backfill platform; reject items without a content_id / content_text."""
    cid = item.get("content_id")
    if not isinstance(cid, str):
        return None
    if not isinstance(item.get("content_text"), str):
        return None
    if not item.get("platform"):
        item = {**item, "platform": platform}
    return item


def load_merged(persona_id: int, platform: str | None = None) -> list[dict]:
    """Union of every sample dump for this persona, deduped by (platform, content_id)."""
    merged: dict[tuple[str, str], dict] = {}
    for plat, path in _persona_files(persona_id, platform):
        for raw in _read_dump(path):
            item = _normalize(raw, plat)
            if item is None:
                continue
            key = (item["platform"], item["content_id"])
            existing = merged.get(key)
            if existing is None or updated_ts(item) >= updated_ts(existing):
                merged[key] = item
    return list(merged.values())


def _summarize(idx: int, item: dict) -> SampleSummary:
    text = item.get("content_text") or ""
    snippet = text.replace("\n", " ")[:SNIPPET_LEN]
    if len(text) > SNIPPET_LEN:
        snippet += "…"
    return SampleSummary(
        idx=idx,
        platform=item.get("platform") or "",
        content_type=item.get("content_type") or "",
        question_title=item.get("question_title") or item.get("title") or "",
        snippet=snippet,
        char_count=len(text),
        voteup_count=int(item.get("voteup_count") or 0),
    )


def list_for(persona_id: int) -> list[SampleSummary]:
    items = load_merged(persona_id)
    summaries = [_summarize(i, item) for i, item in enumerate(items)]
    summaries.sort(key=lambda s: (-s.voteup_count, -s.char_count))
    return summaries


def get_for(persona_id: int, idx: int) -> dict | None:
    items = load_merged(persona_id)
    if 0 <= idx < len(items):
        return items[idx]
    return None
