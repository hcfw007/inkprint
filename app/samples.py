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


def _latest_file(persona_id: int) -> Path | None:
    persona_dir = SAMPLES_DIR / str(persona_id)
    if not persona_dir.exists():
        return None
    candidates = sorted(persona_dir.glob("*.json"))
    return candidates[-1] if candidates else None


def _load(persona_id: int) -> list[dict]:
    path = _latest_file(persona_id)
    if path is None:
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


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
