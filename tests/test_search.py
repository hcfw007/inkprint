"""search.py — Tavily 客户端全部 mock（httpx.Client 替换为 stub）。"""

import httpx
import pytest

from app import search


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


class FakeClient:
    """记录 post() 载荷并按脚本返回的 httpx.Client stub。"""

    def __init__(self, script, captured, **kwargs):
        self._script = script
        self._captured = captured
        self._captured["client_kwargs"] = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        self._captured["post"] = {"url": url, "json": json}
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def install_fake_client(monkeypatch, script, captured):
    monkeypatch.setattr(search.httpx, "Client", lambda **kw: FakeClient(script, captured, **kw))


@pytest.fixture()
def api_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


class TestAvailability:
    def test_available_with_key(self, api_key):
        assert search.is_available() is True

    def test_unavailable_without_key(self):
        assert search.is_available() is False

    def test_whitespace_key_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "   ")
        assert search.is_available() is False


class TestWebSearchValidation:
    def test_missing_key_raises(self):
        with pytest.raises(search.SearchError, match="TAVILY_API_KEY not configured"):
            search.web_search("q")

    def test_empty_query_raises(self, api_key):
        with pytest.raises(search.SearchError, match="empty query"):
            search.web_search("   ")


class TestWebSearchCalls:
    def test_success_maps_results(self, monkeypatch, api_key):
        captured = {}
        response = FakeResponse(
            data={
                "results": [
                    {"title": "标题", "url": "https://x", "content": "摘要", "extra": 1},
                    {"title": None, "url": None, "content": None},
                ]
            }
        )
        install_fake_client(monkeypatch, [response], captured)
        results = search.web_search("  关键词  ", max_results=3)
        assert results == [
            {"title": "标题", "url": "https://x", "content": "摘要"},
            {"title": "", "url": "", "content": ""},
        ]
        assert captured["post"]["url"] == search.TAVILY_ENDPOINT
        payload = captured["post"]["json"]
        assert payload["query"] == "关键词"
        assert payload["api_key"] == "tvly-test"
        assert payload["max_results"] == 3
        assert captured["client_kwargs"]["trust_env"] is False

    def test_empty_results(self, monkeypatch, api_key):
        captured = {}
        install_fake_client(monkeypatch, [FakeResponse(data={})], captured)
        assert search.web_search("q") == []

    def test_network_error_wrapped(self, monkeypatch, api_key):
        captured = {}
        install_fake_client(monkeypatch, [httpx.ConnectError("boom")], captured)
        with pytest.raises(search.SearchError, match="network error: boom"):
            search.web_search("q")

    def test_http_error_status(self, monkeypatch, api_key):
        captured = {}
        response = FakeResponse(status_code=429, text="rate limited")
        install_fake_client(monkeypatch, [response], captured)
        with pytest.raises(search.SearchError, match="tavily returned 429"):
            search.web_search("q")

    def test_proxy_env_used_for_client(self, monkeypatch, api_key):
        captured = {}
        monkeypatch.setenv("LLM_PROXY", "http://127.0.0.1:7890")
        install_fake_client(monkeypatch, [FakeResponse(data={"results": []})], captured)
        search.web_search("q")
        assert captured["client_kwargs"]["proxy"] == "http://127.0.0.1:7890"
