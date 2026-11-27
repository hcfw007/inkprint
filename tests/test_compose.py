"""compose() tests — LLM 与搜索全部 mock，只验证编排逻辑与 prompt 拼装。"""

import pytest

from app import search, voice_profile

PROFILE_MD = (
    "VOICE PROFILE\n=============\nAuthor: 张三\nRhythm\n- 短句\nBanned Moves\n- 不卖鸡汤\n"
)


@pytest.fixture()
def profile(profiles_dir):
    pdir = profiles_dir / "1"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "20260101T000000Z.md").write_text(PROFILE_MD, encoding="utf-8")
    return pdir


class TestComposeValidation:
    def test_unknown_form_type(self, profile):
        with pytest.raises(voice_profile.ProfileError, match="未知文本类型"):
            voice_profile.compose(1, "poem", "话题")

    def test_missing_profile(self, profiles_dir):
        with pytest.raises(voice_profile.ProfileError, match="尚未生成"):
            voice_profile.compose(1, "short", "话题")

    def test_empty_topic(self, profile):
        with pytest.raises(voice_profile.ProfileError, match="不能为空"):
            voice_profile.compose(1, "short", "   ")


class TestComposeWithoutSearch:
    def _run(self, monkeypatch, profile, **kwargs):
        captured = {}
        monkeypatch.setattr(search, "is_available", lambda: False)

        def fake_chat(messages, temperature=0.3):
            captured["messages"] = messages
            captured["temperature"] = temperature
            return "生成的正文"

        monkeypatch.setattr(voice_profile.llm, "chat", fake_chat)
        result = voice_profile.compose(1, "short", "聊聊测试", **kwargs)
        return result, captured

    def test_single_shot_and_trace(self, monkeypatch, profile):
        result, captured = self._run(monkeypatch, profile)
        assert result.content == "生成的正文"
        assert [t.kind for t in result.trace] == ["request", "final"]
        assert captured["temperature"] == 0.7
        assert "140 字以内" in captured["messages"][1]["content"]

    def test_identity_from_author_text(self, monkeypatch, profile):
        _, captured = self._run(monkeypatch, profile, author_text=" 某某 ")
        assert captured["messages"][0]["content"].startswith("你是 某某。")

    def test_identity_falls_back_to_author_line(self, monkeypatch, profile):
        _, captured = self._run(monkeypatch, profile)
        system = captured["messages"][0]["content"]
        assert system.startswith("你是 张三。")
        # profile 内嵌块不再重复 Author 行
        assert "Author" not in system

    def test_identity_default_when_no_author(self, monkeypatch, profiles_dir):
        pdir = profiles_dir / "1"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "20260101T000000Z.md").write_text("VOICE PROFILE\n- 无作者", encoding="utf-8")
        monkeypatch.setattr(search, "is_available", lambda: False)
        captured = {}

        def fake_chat(messages, temperature=0.3):
            captured["messages"] = messages
            return "正文"

        monkeypatch.setattr(voice_profile.llm, "chat", fake_chat)
        voice_profile.compose(1, "short", "话题")
        assert captured["messages"][0]["content"].startswith("你是这位作者本人。")

    def test_original_content_stance(self, monkeypatch, profile):
        _, captured = self._run(monkeypatch, profile)
        system = captured["messages"][0]["content"]
        assert "原创发起内容" in system
        assert "禁止以下开场" in system

    def test_q_and_a_stance(self, monkeypatch, profile):
        _, captured = self._run(monkeypatch, profile, q_and_a=True)
        system, user = captured["messages"][0]["content"], captured["messages"][1]["content"]
        assert "回答一个具体提问" in system
        assert "【问题】" in user

    def test_llm_error_wrapped(self, monkeypatch, profile):
        monkeypatch.setattr(search, "is_available", lambda: False)

        def boom(messages, temperature=0.3):
            raise voice_profile.llm.LLMError("quota exceeded")

        monkeypatch.setattr(voice_profile.llm, "chat", boom)
        with pytest.raises(voice_profile.ProfileError, match="quota exceeded"):
            voice_profile.compose(1, "short", "话题")


class TestComposeWithSearch:
    def test_tool_loop_path(self, monkeypatch, profile):
        monkeypatch.setattr(search, "is_available", lambda: True)
        captured = {}
        fake_trace = [voice_profile.llm.TraceStep("final", "done", "")]

        def fake_chat_with_tools(*, messages, tools, handlers, temperature, trace_tag):
            captured.update(
                messages=messages,
                tools=tools,
                handlers=handlers,
                temperature=temperature,
                tag=trace_tag,
            )
            return "带检索的正文", fake_trace

        monkeypatch.setattr(voice_profile.llm, "chat_with_tools", fake_chat_with_tools)
        result = voice_profile.compose(1, "long", "2025 年发生了什么")
        assert result.content == "带检索的正文"
        assert result.trace is fake_trace
        assert captured["tools"] == [voice_profile.WEB_SEARCH_TOOL]
        assert "web_search" in captured["handlers"]
        assert captured["temperature"] == 0.7
        assert captured["tag"] == "compose:p1:long"
        system = captured["messages"][0]["content"]
        assert "事实核查规则" in system and "必须先调用 web_search" in system

    def test_tool_loop_llm_error_wrapped(self, monkeypatch, profile):
        monkeypatch.setattr(search, "is_available", lambda: True)

        def boom(**_kwargs):
            raise voice_profile.llm.LLMError("tool loop died")

        monkeypatch.setattr(voice_profile.llm, "chat_with_tools", boom)
        with pytest.raises(voice_profile.ProfileError, match="tool loop died"):
            voice_profile.compose(1, "short", "话题")


class TestSearchHandler:
    def test_format_search_results_empty(self):
        assert voice_profile._format_search_results([]) == "未找到相关结果"

    def test_format_search_results_items(self):
        out = voice_profile._format_search_results(
            [{"title": "标题", "url": "https://x", "content": "摘要"}]
        )
        assert out == "[1] 标题\n来源: https://x\n摘要: 摘要"

    def test_search_handler_success(self, monkeypatch):
        monkeypatch.setattr(
            search, "web_search", lambda q: [{"title": "t", "url": "u", "content": "c"}]
        )
        assert voice_profile._search_handler("q") == "[1] t\n来源: u\n摘要: c"

    def test_search_handler_error(self, monkeypatch):
        def boom(q):
            raise search.SearchError("no key")

        monkeypatch.setattr(search, "web_search", boom)
        assert voice_profile._search_handler("q") == "搜索失败: no key"
