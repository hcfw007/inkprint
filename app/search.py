"""Web search backend for fact-grounding during compose.

We use Tavily — it returns LLM-ready snippets (title + URL + short summary)
without us having to fetch and parse pages. Free tier covers 1000 calls/month.

If TAVILY_API_KEY is missing the search layer is treated as disabled: the
caller should skip passing the tool to the LLM rather than failing the
compose entirely.
"""

import os

import httpx

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
TAVILY_TIMEOUT_SECONDS = 30.0


class SearchError(RuntimeError):
    """Search call failed or is not configured."""


def is_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise SearchError("TAVILY_API_KEY not configured")
    if not query.strip():
        raise SearchError("empty query")
    proxy = os.getenv("LLM_PROXY", "").strip() or None
    payload = {
        "api_key": api_key,
        "query": query.strip(),
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }
    with httpx.Client(proxy=proxy, trust_env=False, timeout=TAVILY_TIMEOUT_SECONDS) as client:
        try:
            response = client.post(TAVILY_ENDPOINT, json=payload)
        except httpx.HTTPError as e:
            raise SearchError(f"network error: {e}") from e
    if response.status_code >= 400:
        raise SearchError(f"tavily returned {response.status_code}: {response.text[:200]}")
    data = response.json()
    results = data.get("results") or []
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": r.get("content") or "",
        }
        for r in results
    ]
