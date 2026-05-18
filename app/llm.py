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

import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
