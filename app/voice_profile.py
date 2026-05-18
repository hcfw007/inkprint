"""Voice profile generation — turns zhihu samples into a reusable VOICE PROFILE.

Pipeline:
  1. Load the latest sample JSON for a persona (samples/zhihu/{id}/*.json).
  2. Compute lightweight stats (count, length distribution, top question titles).
  3. Pick a representative subset that fits the LLM context window.
  4. Ask the LLM to produce a markdown profile following the brand-voice schema.
  5. Write the profile to profiles/{persona_id}.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from . import llm

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples" / "zhihu"
PROFILES_DIR = ROOT / "profiles"

SAMPLE_PICK_LIMIT = 30
SAMPLE_MIN_CHARS = 30


class ProfileError(RuntimeError):
    """Anything that goes wrong producing a profile."""


@dataclass(frozen=True)
class ProfileResult:
    profile_path: Path
    sample_count_total: int
    sample_count_used: int


def _latest_sample_file(persona_id: int) -> Path:
    persona_dir = SAMPLES_DIR / str(persona_id)
    if not persona_dir.exists():
        raise ProfileError(f"no samples for persona {persona_id}; sync zhihu first")
    candidates = sorted(persona_dir.glob("*.json"))
    if not candidates:
        raise ProfileError(f"no sample json under {persona_dir}")
    return candidates[-1]


def _load_samples(persona_id: int) -> list[dict]:
    path = _latest_sample_file(persona_id)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ProfileError(f"unexpected sample shape in {path}: not a list")
    return data


def _pick_representative(samples: list[dict]) -> list[dict]:
    """Pick up to SAMPLE_PICK_LIMIT items biased toward longer, more substantive posts.

    Strategy: filter out very short ones, then take the longest by content_text.
    Longer posts tend to carry more voice signal than one-liners.
    """
    eligible = [
        s
        for s in samples
        if isinstance(s.get("content_text"), str) and len(s["content_text"]) >= SAMPLE_MIN_CHARS
    ]
    eligible.sort(key=lambda s: len(s["content_text"]), reverse=True)
    return eligible[:SAMPLE_PICK_LIMIT]


def _build_prompt(samples_used: list[dict], total_count: int) -> list[dict[str, str]]:
    formatted = []
    for i, s in enumerate(samples_used, 1):
        question = s.get("question_title") or s.get("title") or ""
        text = s["content_text"]
        formatted.append(f"[样本 {i}] 问题：{question}\n回答：{text}")
    samples_block = "\n\n---\n\n".join(formatted)

    system = (
        "You are a voice profile analyst following the brand-voice methodology. "
        "Given real social-media samples from one author, produce a structured, "
        "operational VOICE PROFILE that downstream LLMs can read directly to "
        "imitate the author's voice. Be concrete and source-backed — every "
        "claim should be observable in the samples. If samples conflict, call "
        "out the split instead of averaging it into mush. The samples are in "
        "Chinese; the profile MUST also be in Chinese."
    )

    user = f"""请基于以下来自同一作者的知乎回答样本，输出 VOICE PROFILE。

样本总数：{total_count}（已选取最具代表性的 {len(samples_used)} 条）。

样本：

{samples_block}

请严格按照以下 schema 输出（markdown，标题保持英文 key，正文用中文）：

```
VOICE PROFILE
=============
Author:
Goal:
Confidence:  <low | medium | high，结合样本量和一致性判断>

Source Set
- 知乎回答 {len(samples_used)} 条（共 {total_count} 条）

Rhythm
- <句长、节奏、断句习惯>

Compression
- <密度 vs 解释；信息密集还是娓娓道来>

Capitalization
- <中文场景这一项通常写"中文，不适用"或备注英文夹杂时的习惯>

Parentheticals
- <括号使用：用来限定 / 补充 / 调侃 / 罕用>

Question Use
- <提问频率与目的：真问、反问、引导、罕用>

Claim Style
- <断言锋利度、是否摆事实、是否给数据>

Preferred Moves
- <具体能在样本里观察到的招式，比如"先抛结论再举例"、"用括号补一句吐槽"等>

Banned Moves
- <作者明显不做的事，比如"不卖鸡汤"、"不用感叹号"等>

CTA Rules
- <收尾习惯：留问题、给结论、轻调侃、无 CTA>

Channel Notes
- 知乎：<在这个平台的具体腔调>
- 其他平台：<待补>
```

不要写文学评论式的描述，每条都要简短可操作，能让另一个 AI 拿来直接复用。"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate(persona_id: int) -> ProfileResult:
    samples = _load_samples(persona_id)
    if not samples:
        raise ProfileError("sample file is empty")
    picked = _pick_representative(samples)
    if not picked:
        raise ProfileError(f"no samples passed the minimum length ({SAMPLE_MIN_CHARS} chars)")
    messages = _build_prompt(picked, total_count=len(samples))
    try:
        profile_md = llm.chat(messages)
    except llm.LLMError as e:
        raise ProfileError(str(e)) from e

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{persona_id}.md"
    profile_path.write_text(profile_md, encoding="utf-8")
    return ProfileResult(
        profile_path=profile_path,
        sample_count_total=len(samples),
        sample_count_used=len(picked),
    )


def read_existing(persona_id: int) -> str | None:
    path = PROFILES_DIR / f"{persona_id}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None
