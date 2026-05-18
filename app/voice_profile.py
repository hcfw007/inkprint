"""Voice profile generation — turns zhihu samples into a reusable VOICE PROFILE.

Pipeline:
  1. Load the latest sample JSON for a persona (samples/zhihu/{id}/*.json).
  2. Compute lightweight stats (count, length distribution, top question titles).
  3. Pick a representative subset that fits the LLM context window.
  4. Ask the LLM to produce a markdown profile following the brand-voice schema.
  5. Write the profile to profiles/{persona_id}.md.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import llm, samples

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "profiles"

SAMPLE_PICK_LIMIT = 30
SAMPLE_MIN_CHARS = 30


def _persona_dir(persona_id: int) -> Path:
    return PROFILES_DIR / str(persona_id)


def _migrate_legacy_flat(persona_id: int) -> None:
    """Move profiles/{id}.md (old layout) into profiles/{id}/v{ts}.md once.

    Idempotent and silent if there is nothing to migrate.
    """
    legacy = PROFILES_DIR / f"{persona_id}.md"
    if not legacy.exists():
        return
    persona_dir = _persona_dir(persona_id)
    persona_dir.mkdir(parents=True, exist_ok=True)
    legacy_ts = datetime.fromtimestamp(legacy.stat().st_mtime, tz=UTC)
    target = persona_dir / f"{legacy_ts.strftime('%Y%m%dT%H%M%SZ')}.md"
    if not target.exists():
        legacy.rename(target)
    else:
        legacy.unlink()


class ProfileError(RuntimeError):
    """Anything that goes wrong producing a profile."""


@dataclass(frozen=True)
class ProfileResult:
    profile_path: Path
    sample_count_total: int
    sample_count_used: int


@dataclass(frozen=True)
class ProfileVersion:
    name: str  # filename without extension, e.g. "20260518T123456Z"
    created_at: str  # human-readable
    path: Path


def _load_samples(persona_id: int) -> list[dict]:
    """Pull the merged sample union (latest version per content_id)."""
    items = samples.load_merged(persona_id)
    if not items:
        raise ProfileError(f"no samples for persona {persona_id}; sync first")
    return items


def _pick_representative(samples: list[dict]) -> list[dict]:
    """Pick up to SAMPLE_PICK_LIMIT items, biased toward substantive posts but
    balanced across platforms so a chatty long-form source does not starve
    out a terser short-form one.

    Strategy: bucket by platform, sort each bucket by length desc, then
    interleave round-robin until the global limit is hit.
    """
    eligible = [
        s
        for s in samples
        if isinstance(s.get("content_text"), str) and len(s["content_text"]) >= SAMPLE_MIN_CHARS
    ]
    buckets: dict[str, list[dict]] = {}
    for item in eligible:
        buckets.setdefault(item.get("platform") or "unknown", []).append(item)
    for plat in buckets:
        buckets[plat].sort(key=lambda s: len(s["content_text"]), reverse=True)

    picked: list[dict] = []
    while len(picked) < SAMPLE_PICK_LIMIT and any(buckets.values()):
        for plat in list(buckets.keys()):
            if not buckets[plat]:
                continue
            picked.append(buckets[plat].pop(0))
            if len(picked) >= SAMPLE_PICK_LIMIT:
                break
    return picked


PLATFORM_LABEL = {"zhihu": "知乎", "weibo": "微博"}


def _format_sample(idx: int, item: dict) -> str:
    platform = item.get("platform") or "unknown"
    label = PLATFORM_LABEL.get(platform, platform)
    text = item["content_text"]
    question = item.get("question_title") or item.get("title") or ""
    if platform == "zhihu" and question:
        return f"[样本 {idx} · {label}] 问题：{question}\n回答：{text}"
    return f"[样本 {idx} · {label}] {text}"


def _platform_breakdown(samples_used: list[dict], total_count: int) -> str:
    by_platform: dict[str, int] = {}
    for s in samples_used:
        by_platform[s.get("platform") or "unknown"] = (
            by_platform.get(s.get("platform") or "unknown", 0) + 1
        )
    parts = [f"{PLATFORM_LABEL.get(p, p)} {c} 条" for p, c in sorted(by_platform.items())]
    return (
        f"已选取最具代表性的 {len(samples_used)} 条（{' + '.join(parts)}），"
        f"来自共 {total_count} 条样本"
    )


def _build_prompt(samples_used: list[dict], total_count: int) -> list[dict[str, str]]:
    formatted = [_format_sample(i, s) for i, s in enumerate(samples_used, 1)]
    samples_block = "\n\n---\n\n".join(formatted)
    breakdown = _platform_breakdown(samples_used, total_count)

    system = (
        "You are a voice profile analyst following the brand-voice methodology. "
        "Given real social-media samples from one author across multiple platforms, "
        "produce a structured, operational VOICE PROFILE that downstream LLMs can "
        "read directly to imitate the author's voice. Be concrete and source-backed "
        "— every claim should be observable in the samples. If the author writes "
        "differently across platforms, call that split out in Channel Notes rather "
        "than averaging it into mush. The samples are in Chinese; the profile MUST "
        "also be in Chinese."
    )

    user = f"""请基于以下来自同一作者的多平台社媒样本，输出 VOICE PROFILE。

{breakdown}。

样本：

{samples_block}

请严格按照以下 schema 输出（markdown，标题保持英文 key，正文用中文）：

```
VOICE PROFILE
=============
Author:  <对作者的一段简短画像：身份、年龄段、领域、性格、立场倾向等>
Goal:    <作者发文的核心目的：分享、辩论、记录、社交等>
Confidence:  <low | medium | high，结合样本量和一致性判断>

Source Set
- <按平台分别列出实际使用的样本数>

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
- 知乎：<在这个平台的具体腔调，跟下面对比要有差异>
- 微博：<在这个平台的具体腔调，跟上面对比要有差异>
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

    persona_dir = _persona_dir(persona_id)
    persona_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    profile_path = persona_dir / f"{timestamp}.md"
    profile_path.write_text(profile_md, encoding="utf-8")
    return ProfileResult(
        profile_path=profile_path,
        sample_count_total=len(samples),
        sample_count_used=len(picked),
    )


def format_ts(stem: str) -> str:
    """Render a 20260518T123456Z stem as 2026-05-18 12:34:56 UTC, best effort."""
    try:
        dt = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return stem


def list_versions(persona_id: int) -> list[ProfileVersion]:
    _migrate_legacy_flat(persona_id)
    persona_dir = _persona_dir(persona_id)
    if not persona_dir.exists():
        return []
    versions = []
    for path in sorted(persona_dir.glob("*.md"), reverse=True):
        versions.append(ProfileVersion(name=path.stem, created_at=format_ts(path.stem), path=path))
    return versions


def read_existing(persona_id: int) -> str | None:
    versions = list_versions(persona_id)
    if not versions:
        return None
    return versions[0].path.read_text(encoding="utf-8")


def read_version(persona_id: int, name: str) -> str | None:
    _migrate_legacy_flat(persona_id)
    path = _persona_dir(persona_id) / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None
