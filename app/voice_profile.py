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
AUTHOR_LINE_FULL_RE = re.compile(r"^Author\s*[:：].*\n?", re.MULTILINE)

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


@dataclass(frozen=True)
class ComposeResult:
    content: str
    trace: list[llm.TraceStep]


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


def strip_author(profile_md: str) -> str:
    """Drop the Author line entirely (line + trailing newline).

    Used when the author identity is being injected elsewhere (e.g. the
    `你是 {author}` opener in compose's system prompt) and we don't want it
    repeated inside the embedded VOICE PROFILE block.
    """
    return AUTHOR_LINE_FULL_RE.sub("", profile_md, count=1)


def _build_prompt(
    samples_used: list[dict],
    total_count: int,
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
        "also be in Chinese. Do NOT include any author bio / identity / 履历 fields — "
        "downstream code injects the author identity separately."
    )

    user = f"""请基于以下来自同一作者的多平台社媒样本，输出 VOICE PROFILE。

{breakdown}。

样本：

{samples_block}

请严格按照以下 schema 输出（markdown，标题保持英文 key，正文用中文）。
**不要输出 Author / 作者 / 履历 / 身份相关字段** —— 那块由调用方单独维护。

```
VOICE PROFILE
=============
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


FORM_LONG = "long"
FORM_SHORT = "short"

COMPOSE_FORMS: dict[str, dict[str, str]] = {
    FORM_LONG: {
        "label": "长文",
        "length": "600 到 2000 字之间，可以分段、用小标题，不要硬撑",
        "scenario": (
            "你正在为自己的个人公众号 / 博客 / 长文专栏写一篇**原创随笔或观察评论**。"
            "不是在回答任何人提的问题——是你主动想聊这件事，所以才坐下来写。"
        ),
        "images": ("可以在合适的位置用 [图：简短说明] 占位提示配图，建议 1-3 处。占位独占一行。"),
    },
    FORM_SHORT: {
        "label": "短文",
        "length": "140 字以内，单段更自然",
        "scenario": (
            "你正在自己的微博 / X / 朋友圈发一条**独立短帖**。"
            "不是回复谁，也不是回答提问，就是看到 / 想到一件事顺手发出来。"
        ),
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
    q_and_a: bool = False,
) -> ComposeResult:
    """Generate a post in the persona's voice for the given length type.

    q_and_a flips the default 'you are originating content' framing into
    'you are answering a question someone asked' — useful when the topic
    really is a Q the user wants the persona to answer in their voice.
    """
    if form_type not in COMPOSE_FORMS:
        raise ProfileError(f"未知文本类型: {form_type}")
    profile_md = read_existing(persona_id)
    if not profile_md:
        raise ProfileError("尚未生成 voice profile，先生成 profile 再来写")
    if not topic.strip():
        raise ProfileError("输入文本不能为空")
    form = COMPOSE_FORMS[form_type]

    search_enabled = search.is_available()
    fact_clause = ""
    if search_enabled:
        fact_clause = (
            "\n\n【事实核查规则·必读】\n"
            "你的训练数据有截止日期，2024 年之后的所有事件、获奖、数据、人事变动都可能过时。\n"
            "凡是话题里出现以下任一情况，**必须先调用 web_search 工具**，再下笔：\n"
            "- 出现明确年份/时间词（'今年'、'最新'、'近期'、'当前'、'2024'、'2025' 等）\n"
            "- 涉及具体数字、获奖归属、排名、比分、价格、版本号\n"
            "- 涉及活着的人物的当前状态（在哪个公司、参加什么比赛、最近言论）\n"
            "- 涉及最近的新闻事件、产品发布、政策变动\n"
            "**禁止凭记忆答题**。哪怕你 90% 确定，也搜一下确认再写。"
            "如果搜索结果跟你记忆冲突，以搜索为准。"
        )

    if q_and_a:
        stance = (
            "本次是**回答一个具体提问**。可以正常使用 Preferred Moves 中的答题动作（"
            "『先抛结论再展开』、『利益相关』、『—— 为什么…』等），按知乎答题模式来。"
        )
        scenario = "你正在知乎或论坛回答一个具体问题，下面给你看问题本身。"
    else:
        stance = (
            "本次是**原创发起内容**，不是回答任何人的提问。Preferred Moves 里的『答题动作』"
            "（『先抛结论再展开』、『利益相关』、『—— 为什么…』）**不要套用**——它们的语气、"
            "节奏、词汇偏好可以借鉴，但结构上不要装成在答题。\n"
            "禁止以下开场：『我认为...』、『先说结论...』、『说实话...』、『其实...』、"
            "『利益相关...』、『不请自来...』、任何反问『为什么...』作为首句。\n"
            "开场应该是观察、事件、画面或具体场景，不是表态。"
        )
        scenario = form["scenario"]

    # 身份来源优先级：用户人工 author_text → 旧版 profile 残留的 Author 行 → 兜底
    author_desc = author_text.strip() or extract_author(profile_md).strip()
    identity_line = f"你是 {author_desc}。" if author_desc else "你是这位作者本人。"
    # 旧版 profile 可能仍带 Author 行，剥掉避免和 identity_line 重复
    profile_for_prompt = strip_author(profile_md).lstrip("\n")

    system = (
        f"{identity_line}\n"
        "你的写作风格如下（VOICE PROFILE，基于该作者的多源样本提炼，严格按它来写）：\n\n"
        f"{profile_for_prompt}\n\n"
        f"{stance}\n"
        "严格避开 Banned Moves。输出语言：中文。只输出正文，不要任何解释、标题、前后缀。"
        f"{fact_clause}"
    )
    user = f"""【写作场景】
{scenario}

【字数】{form["length"]}
【配图】{form["images"]}

【{"问题" if q_and_a else "素材 / 想法 / 草稿"}】
{topic.strip()}
"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        if search_enabled:
            content, trace = llm.chat_with_tools(
                messages=messages,
                tools=[WEB_SEARCH_TOOL],
                handlers={"web_search": _search_handler},
                temperature=0.7,
                trace_tag=f"compose:p{persona_id}:{form_type}",
            )
        else:
            content = llm.chat(messages, temperature=0.7)
            prompt_chars = sum(len(m["content"]) for m in messages)
            trace = [
                llm.TraceStep("request", f"system + user 共 {prompt_chars} 字（无搜索）", ""),
                llm.TraceStep("final", f"最终正文 {len(content)} 字", ""),
            ]
    except llm.LLMError as e:
        raise ProfileError(str(e)) from e
    return ComposeResult(content=content, trace=trace)


def read_existing(persona_id: int) -> str | None:
    versions = list_versions(persona_id)
    if not versions:
        return None
    return versions[0].path.read_text(encoding="utf-8")


def read_version(persona_id: int, name: str) -> str | None:
    _migrate_legacy_flat(persona_id)
    path = _persona_dir(persona_id) / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None
