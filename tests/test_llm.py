"""llm.py — 全部 mock OpenAI client，验证配置校验与 chat / tool-loop 编排。"""

from types import SimpleNamespace

import pytest

from app import llm


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


def fake_response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage(content, tool_calls))])


def fake_tool_call(name: str, arguments: str, call_id: str = "call-1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def install_fake_openai(monkeypatch, script, captured):
    """script: list of fake responses (or exceptions) consumed per create() call."""

    class FakeCompletions:
        def create(self, **kwargs):
            captured.setdefault("create_calls", []).append(kwargs)
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    return captured


class TestLoadConfig:
    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "model-x")
        cfg = llm.load_config()
        assert cfg == llm.LLMConfig("https://api.example/v1", "sk-test", "model-x")

    def test_missing_vars_listed(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "model-x")
        with pytest.raises(llm.LLMError) as exc:
            llm.load_config()
        assert "LLM_BASE_URL" in str(exc.value) and "LLM_API_KEY" in str(exc.value)
        assert "LLM_MODEL" not in str(exc.value)

    def test_whitespace_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "   ")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "m")
        with pytest.raises(llm.LLMError, match="LLM_BASE_URL"):
            llm.load_config()


class TestHttpClient:
    def test_no_proxy_by_default(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            llm.httpx, "Client", lambda **kw: captured.update(kw) or SimpleNamespace()
        )
        llm._build_http_client()
        assert captured["proxy"] is None
        assert captured["trust_env"] is False
        assert captured["timeout"] == llm.LLM_HTTP_TIMEOUT_SECONDS

    def test_explicit_proxy_respected(self, monkeypatch):
        captured = {}
        monkeypatch.setenv("LLM_PROXY", "socks5://127.0.0.1:1080")
        monkeypatch.setattr(
            llm.httpx, "Client", lambda **kw: captured.update(kw) or SimpleNamespace()
        )
        llm._build_http_client()
        assert captured["proxy"] == "socks5://127.0.0.1:1080"


@pytest.fixture()
def llm_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "model-x")


class TestChat:
    def test_returns_content_and_configures_client(self, monkeypatch, llm_env):
        captured = {}
        install_fake_openai(monkeypatch, [fake_response(content="你好")], captured)
        out = llm.chat([{"role": "user", "content": "hi"}], temperature=0.7)
        assert out == "你好"
        assert captured["client_kwargs"]["base_url"] == "https://api.example/v1"
        assert captured["client_kwargs"]["api_key"] == "sk-test"
        call = captured["create_calls"][0]
        assert call["model"] == "model-x"
        assert call["temperature"] == 0.7

    def test_empty_content_raises(self, monkeypatch, llm_env):
        captured = {}
        install_fake_openai(monkeypatch, [fake_response(content=None)], captured)
        with pytest.raises(llm.LLMError, match="empty content"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_config_error_before_any_call(self, monkeypatch):
        captured = {}
        install_fake_openai(monkeypatch, [], captured)
        with pytest.raises(llm.LLMError, match="missing LLM env vars"):
            llm.chat([{"role": "user", "content": "hi"}])
        assert "client_kwargs" not in captured


class TestChatWithTools:
    def _messages(self):
        return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    def test_tool_loop_dispatch_and_trace(self, monkeypatch, llm_env):
        captured = {}
        calls = []
        script = [
            fake_response(tool_calls=[fake_tool_call("web_search", '{"query": "关键词"}')]),
            fake_response(content="最终正文"),
        ]
        install_fake_openai(monkeypatch, script, captured)
        content, trace = llm.chat_with_tools(
            messages=self._messages(),
            tools=[{"type": "function"}],
            handlers={"web_search": lambda query: calls.append(query) or f"结果:{query}"},
            temperature=0.5,
            trace_tag="t",
        )
        assert content == "最终正文"
        assert calls == ["关键词"]
        assert [t.kind for t in trace] == ["request", "tool_call", "tool_result", "final"]
        assert "web_search(query='关键词')" in trace[1].label
        assert "结果:关键词" in trace[2].detail
        # 第二轮请求带上 tool 结果
        second = captured["create_calls"][1]["messages"]
        assert {"role": "tool", "tool_call_id": "call-1", "content": "结果:关键词"} in second
        assert captured["create_calls"][0]["tools"] == [{"type": "function"}]

    def test_unknown_tool_and_handler_error(self, monkeypatch, llm_env):
        captured = {}
        script = [
            fake_response(
                tool_calls=[
                    fake_tool_call("ghost", "{}", call_id="a"),
                    fake_tool_call("boom", "{}", call_id="b"),
                ]
            ),
            fake_response(content="ok"),
        ]
        install_fake_openai(monkeypatch, script, captured)

        def broken(**_kwargs):
            raise ValueError("kaboom")

        _, trace = llm.chat_with_tools(
            messages=self._messages(), tools=[], handlers={"boom": broken}, trace_tag="t"
        )
        results = [t for t in trace if t.kind == "tool_result"]
        assert "unknown tool: ghost" in results[0].detail
        assert "tool error: kaboom" in results[1].detail

    def test_bad_tool_arguments_default_to_empty(self, monkeypatch, llm_env):
        captured = {}
        received = {}
        script = [
            fake_response(tool_calls=[fake_tool_call("echo", "not-json")]),
            fake_response(content="ok"),
        ]
        install_fake_openai(monkeypatch, script, captured)
        llm.chat_with_tools(
            messages=self._messages(),
            tools=[],
            handlers={"echo": lambda **kw: received.update(kw) or "done"},
            trace_tag="t",
        )
        assert received == {}

    def test_max_iterations_exceeded(self, monkeypatch, llm_env):
        captured = {}
        script = [fake_response(tool_calls=[fake_tool_call("echo", "{}")])] * 3
        install_fake_openai(monkeypatch, script, captured)
        with pytest.raises(llm.LLMError, match="exceeded 3 iterations"):
            llm.chat_with_tools(
                messages=self._messages(),
                tools=[],
                handlers={"echo": lambda **kw: "done"},
                max_iterations=3,
                trace_tag="t",
            )

    def test_neither_content_nor_tool_calls(self, monkeypatch, llm_env):
        captured = {}
        install_fake_openai(monkeypatch, [fake_response(content=None)], captured)
        with pytest.raises(llm.LLMError, match="neither content nor tool_calls"):
            llm.chat_with_tools(messages=self._messages(), tools=[], handlers={}, trace_tag="t")
