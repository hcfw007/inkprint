"""Voice profile generation — turns zhihu samples into a reusable VOICE PROFILE.

Pipeline:
  1. Load the latest sample JSON for a persona (samples/zhihu/{id}/*.json).
  2. Compute lightweight stats (count, length distribution, top question titles).
  3. Pick a representative subset that fits the LLM context window.
  4. Ask the LLM to produce a markdown profile following the brand-voice schema.
  5. Write the profile to profiles/{persona_id}.md.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import llm, samples, search

AUTHOR_LINE_RE = re.compile(r"^Author\s*[:：]\s*(.*?)\s*$", re.MULTILINE)

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


def extract_author(profile_md: str) -> str:
    """Return the Author line content from a VOICE PROFILE, empty if missing."""
    match = AUTHOR_LINE_RE.search(profile_md)
    return match.group(1).strip() if match else ""


def apply_author_overrides(profile_md: str, author_text: str) -> str:
    """Overlay the user-confirmed Author line on top of an LLM-generated profile.

    Leaves the file on disk untouched — overlay happens at display time so
    profile history stays faithful to what the model produced.
    """
    if not author_text:
        return profile_md
    return AUTHOR_LINE_RE.sub(f"Author: {author_text}", profile_md, count=1)


def _build_prompt(
    samples_used: list[dict],
    total_count: int,
    author_text: str = "",
) -> list[dict[str, str]]:
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

    confirmed_facts = ""
    if author_text:
        confirmed_facts = (
            "\n以下是用户确认过的作者履历，请在 Author 字段直接使用，不要改写：\n"
            f"- Author: {author_text}\n"
        )

    user = f"""请基于以下来自同一作者的多平台社媒样本，输出 VOICE PROFILE。

{breakdown}。
{confirmed_facts}
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


def generate(persona_id: int, author_text: str = "") -> ProfileResult:
    samples = _load_samples(persona_id)
    if not samples:
        raise ProfileError("sample file is empty")
    picked = _pick_representative(samples)
    if not picked:
        raise ProfileError(f"no samples passed the minimum length ({SAMPLE_MIN_CHARS} chars)")
    messages = _build_prompt(picked, total_count=len(samples), author_text=author_text)
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


FORM_LONG = "long"
FORM_SHORT = "short"

COMPOSE_FORMS: dict[str, dict[str, str]] = {
    FORM_LONG: {
        "label": "长文",
        "length": "2000 到 4000 字之间，可以分段、用小标题",
        "images": ("可以在合适的位置用 [图：简短说明] 占位提示配图，建议 1-3 处。占位独占一行。"),
    },
    FORM_SHORT: {
        "label": "短文",
        "length": "140 字以内，单段更自然",
        "images": "不要配图，不要写图片占位。",
    },
}


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "搜索互联网获取实时事实信息。"
            "当你需要引用具体数据、日期、人物、事件细节但不确定时，必须先用本工具验证。"
            "不要凭印象编造数据。query 用简短具体的中文或英文关键词。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询，简短具体"},
            },
            "required": ["query"],
        },
    },
}


def _format_search_results(results: list[dict]) -> str:
    if not results:
        return "未找到相关结果"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n来源: {r['url']}\n摘要: {r['content']}")
    return "\n\n".join(lines)


def _search_handler(query: str) -> str:
    try:
        return _format_search_results(search.web_search(query))
    except search.SearchError as e:
        return f"搜索失败: {e}"


def compose(
    persona_id: int,
    form_type: str,
    topic: str,
    author_text: str = "",
) -> str:
    """Generate a post in the persona's voice for the given length type."""
    if form_type not in COMPOSE_FORMS:
        raise ProfileError(f"未知文本类型: {form_type}")
    profile_md = read_existing(persona_id)
    if not profile_md:
        raise ProfileError("尚未生成 voice profile，先生成 profile 再来写")
    profile_md = apply_author_overrides(profile_md, author_text)
    if not topic.strip():
        raise ProfileError("输入文本不能为空")
    form = COMPOSE_FORMS[form_type]

    search_enabled = search.is_available()
    fact_clause = (
        " 当文章涉及具体数据、日期、人物、事件细节时，先用 web_search 工具核实再下笔。"
        if search_enabled
        else ""
    )
    system = (
        "你是这位作者本人，按下方 VOICE PROFILE 描述的风格写作。严格遵守 Preferred "
        "Moves，避开 Banned Moves，参考 Channel Notes 中长度相近的平台腔调融合使用。"
        f"输出语言：中文。只输出正文，不要任何解释、标题、前后缀。{fact_clause}"
    )
    user = f"""下面是你的 VOICE PROFILE：

{profile_md}

---

现在请以你自己的口吻写一段「{form["label"]}」。
要求长度：{form["length"]}。
配图：{form["images"]}
话题或素材：

{topic.strip()}
"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        if search_enabled:
            return llm.chat_with_tools(
                messages=messages,
                tools=[WEB_SEARCH_TOOL],
                handlers={"web_search": _search_handler},
                temperature=0.7,
            )
        return llm.chat(messages, temperature=0.7)
    except llm.LLMError as e:
        raise ProfileError(str(e)) from e


def read_existing(persona_id: int) -> str | None:
    versions = list_versions(persona_id)
    if not versions:
        return None
    return versions[0].path.read_text(encoding="utf-8")


def read_version(persona_id: int, name: str) -> str | None:
    _migrate_legacy_flat(persona_id)
    path = _persona_dir(persona_id) / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None
