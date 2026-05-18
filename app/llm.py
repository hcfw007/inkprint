"""LLM client — OpenAI-compatible, provider-agnostic.

Configuration via env vars (loaded from .env at startup if present):

    LLM_BASE_URL   e.g. https://api.anthropic.com/v1
                       https://api.openai.com/v1
                       http://localhost:4000  (LiteLLM proxy)
    LLM_API_KEY    provider API key
    LLM_MODEL      model id understood by the configured base URL

Any OpenAI-compatible endpoint works because we use the openai SDK with a
custom base_url.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

# DeepSeek and other CN-hosted LLM providers must be reached directly — clash
# style auto-proxies will route foreign-looking domains through an overseas
# tunnel and then time out on the SSL handshake. trust_env=False ignores the
# system ALL_PROXY/HTTP_PROXY; users who actually need a tunnel (e.g. calling
# Anthropic from inside China) set LLM_PROXY explicitly.
LLM_HTTP_TIMEOUT_SECONDS = 120.0


class LLMError(RuntimeError):
    """LLM configuration or invocation failure."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class TraceStep:
    kind: str  # "request" | "tool_call" | "tool_result" | "final"
    label: str  # short summary, safe to print
    detail: str = ""  # longer detail, render in <details> on the UI


def load_config() -> LLMConfig:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    missing = [
        k
        for k, v in (("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model))
        if not v
    ]
    if missing:
        raise LLMError(f"missing LLM env vars: {', '.join(missing)}. See .env.example.")
    return LLMConfig(base_url=base_url, api_key=api_key, model=model)


def _build_http_client() -> httpx.Client:
    proxy = os.getenv("LLM_PROXY", "").strip() or None
    return httpx.Client(proxy=proxy, trust_env=False, timeout=LLM_HTTP_TIMEOUT_SECONDS)


def chat(messages: list[dict[str, str]], temperature: float = 0.3) -> str:
    cfg = load_config()
    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        http_client=_build_http_client(),
    )
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMError("LLM returned empty content")
    return content


def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handlers: dict[str, Callable[..., str]],
    temperature: float = 0.3,
    max_iterations: int = 6,
    trace_tag: str = "llm",
) -> tuple[str, list[TraceStep]]:
    """Run a chat-completion loop that lets the model invoke tools.

    The model alternates between requesting tool calls and producing final
    content. We dispatch each tool_call to ``handlers[name](**args)`` and
    append the result back as a ``role: "tool"`` message. The loop exits as
    soon as the model returns plain content (no tool_calls), or after
    ``max_iterations`` round-trips to keep runaway loops bounded.

    Returns (content, trace). Trace also gets a one-line summary printed to
    stdout so the dev sees what the model did without unfolding any UI.
    """
    cfg = load_config()
    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        http_client=_build_http_client(),
    )
    history = list(messages)
    trace: list[TraceStep] = []
    prompt_chars = sum(len(str(m.get("content") or "")) for m in history)
    trace.append(TraceStep("request", f"system + user 共 {prompt_chars} 字 → {cfg.model}", ""))
    print(f"[{trace_tag}] → request to {cfg.model}, prompt {prompt_chars} chars", flush=True)
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=cfg.model,
            messages=history,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            temperature=temperature,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            history.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                arg_repr = ", ".join(f"{k}={v!r}" for k, v in args.items())
                trace.append(TraceStep("tool_call", f"{name}({arg_repr})", ""))
                print(f"[{trace_tag}] → tool_call: {name}({arg_repr})", flush=True)
                if name in handlers:
                    try:
                        result = handlers[name](**args)
                    except Exception as e:
                        result = f"tool error: {e}"
                else:
                    result = f"unknown tool: {name}"
                result_str = str(result)
                preview = result_str[:120].replace("\n", " ")
                trace.append(
                    TraceStep("tool_result", f"返回 {len(result_str)} 字 · {preview}", result_str)
                )
                print(
                    f"[{trace_tag}] ← tool_result: {len(result_str)} chars · {preview}",
                    flush=True,
                )
                history.append({"role": "tool", "tool_call_id": call.id, "content": result_str})
            continue
        if msg.content:
            trace.append(TraceStep("final", f"最终正文 {len(msg.content)} 字", ""))
            print(f"[{trace_tag}] ← final content: {len(msg.content)} chars", flush=True)
            return msg.content, trace
        raise LLMError("model returned neither content nor tool_calls")
    raise LLMError(f"tool-call loop exceeded {max_iterations} iterations without final content")
