"""Sample browser — read-only access to the latest zhihu sample dump.

The sample JSON is treated as the source of truth for "what got crawled".
For each persona we always read the newest *.json under samples/zhihu/{id}/.
"""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples" / "zhihu"

SNIPPET_LEN = 80


@dataclass(frozen=True)
class SampleSummary:
    idx: int
    content_type: str
    question_title: str
    snippet: str
    char_count: int
    voteup_count: int


def _all_files(persona_id: int) -> list[Path]:
    persona_dir = SAMPLES_DIR / str(persona_id)
    if not persona_dir.exists():
        return []
    return sorted(persona_dir.glob("*.json"))


def _read_dump(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def updated_ts(item: dict) -> int:
    """Best-effort 'last edited' timestamp; falls back to created_time / 0."""
    return int(item.get("updated_time") or item.get("created_time") or 0)


def load_merged(persona_id: int) -> list[dict]:
    """Union of every sample dump for this persona, deduped by content_id.

    When the same content_id appears across multiple dumps we keep the
    copy with the newest updated_time. This way 'sync' is naturally
    incremental at the data layer even though the crawler itself does
    a full pass each time.
    """
    merged: dict[str, dict] = {}
    for path in _all_files(persona_id):
        for item in _read_dump(path):
            cid = item.get("content_id")
            if not isinstance(cid, str):
                continue
            existing = merged.get(cid)
            if existing is None or updated_ts(item) >= updated_ts(existing):
                merged[cid] = item
    return list(merged.values())


def _load(persona_id: int) -> list[dict]:
    return load_merged(persona_id)


def _summarize(idx: int, item: dict) -> SampleSummary:
    text = item.get("content_text") or ""
    snippet = text.replace("\n", " ")[:SNIPPET_LEN]
    if len(text) > SNIPPET_LEN:
        snippet += "…"
    return SampleSummary(
        idx=idx,
        content_type=item.get("content_type") or "",
        question_title=item.get("question_title") or item.get("title") or "",
        snippet=snippet,
        char_count=len(text),
        voteup_count=int(item.get("voteup_count") or 0),
    )


def list_for(persona_id: int) -> list[SampleSummary]:
    items = _load(persona_id)
    summaries = [_summarize(i, item) for i, item in enumerate(items)]
    summaries.sort(key=lambda s: (-s.voteup_count, -s.char_count))
    return summaries


def get_for(persona_id: int, idx: int) -> dict | None:
    items = _load(persona_id)
    if 0 <= idx < len(items):
        return items[idx]
    return None
