"""Route-layer tests for app/main.py (FastAPI via TestClient) — fully offline.

main.py is the unit under test: every route is exercised on its success
path, its parameter-error paths and its exception paths. Domain logic is
covered by the per-module suites; here LLM / search / crawlers / browser
auth are mocked or inert (env cleared by conftest), and data/ samples/
profiles/ are redirected into tmp_path.
"""

from pathlib import Path

import pytest
from conftest import make_sample
from fastapi.testclient import TestClient

from app import browser_auth, crawler_weibo, crawler_zhihu, main, personas, voice_profile
from app.crawler_weibo import CrawlResult as WeiboCrawlResult
from app.crawler_zhihu import CrawlResult as ZhihuCrawlResult

PROFILE_MD = "VOICE PROFILE\n=============\nRhythm\n- 快节奏"


@pytest.fixture()
def client(tmp_path, monkeypatch, db_path, profiles_dir, samples_dir):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    with TestClient(main.app) as c:
        yield c


def _create(client, name: str = "测试人格", description: str = "一段描述") -> int:
    resp = client.post(
        "/personas",
        data={"name": name, "description": description},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    return int(resp.headers["location"].rstrip("/").split("/")[-1])


def _write_profile(profiles_dir: Path, persona_id: int, name: str, text: str) -> Path:
    pdir = profiles_dir / str(persona_id)
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def _bind_source(pid: int, platform: str = "zhihu", identifier: str = "uid-1") -> int:
    result = personas.add_source(pid, platform, identifier)
    assert result["ok"]
    return int(personas.list_sources(pid)[-1]["id"])


class TestIndexAndNewForm:
    def test_index_empty(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "还没有人格" in resp.text

    def test_index_lists_personas(self, client):
        pid = _create(client, name="李四")
        resp = client.get("/")
        assert resp.status_code == 200
        assert "李四" in resp.text and f"/personas/{pid}" in resp.text

    def test_new_persona_form(self, client):
        resp = client.get("/personas/new")
        assert resp.status_code == 200
        assert 'action="/personas"' in resp.text


class TestCreatePersona:
    def test_create_success_redirects(self, client):
        pid = _create(client, name="张三")
        assert personas.get(pid)["name"] == "张三"

    def test_create_blank_name_400(self, client):
        resp = client.post("/personas", data={"name": "   "})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "name is required"

    def test_create_missing_name_422(self, client):
        resp = client.post("/personas", data={"description": "没有名字"})
        assert resp.status_code == 422


class TestShowPersona:
    def test_show_ok_without_profile(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}")
        assert resp.status_code == 200
        assert "测试人格" in resp.text and "尚未生成" in resp.text

    def test_show_404(self, client):
        resp = client.get("/personas/4242")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "persona not found"

    def test_show_sync_delta_flash(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}?added=3&updated=1&unchanged=2&total=6")
        assert "同步完成 · 新增 3 · 更新 1" in resp.text
        assert "累计 6 条" in resp.text

    def test_show_profile_and_author_flashes(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}?profile=ok&author=saved")
        assert "Voice profile 已生成" in resp.text
        assert "Author 履历已保存" in resp.text

    def test_show_detected_author_fills_placeholder(self, client, profiles_dir):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000Z", "Author: 王五\nRhythm\n")
        resp = client.get(f"/personas/{pid}")
        assert "王五" in resp.text


class TestDeletePersona:
    def test_delete_ok_purges_files(self, client, profiles_dir, samples_dir):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000Z", PROFILE_MD)
        (profiles_dir / f"{pid}.md").write_text("旧版", encoding="utf-8")
        sdir = samples_dir / "zhihu" / str(pid)
        sdir.mkdir(parents=True)
        (sdir / "dump.json").write_text("[]", encoding="utf-8")

        resp = client.post(f"/personas/{pid}/delete", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"
        assert not (profiles_dir / str(pid)).exists()
        assert not (profiles_dir / f"{pid}.md").exists()
        assert not sdir.exists()
        assert client.get(f"/personas/{pid}").status_code == 404

    def test_delete_404(self, client):
        resp = client.post("/personas/4242/delete")
        assert resp.status_code == 404


class TestSourceBindingRoutes:
    def test_add_source_ok(self, client):
        pid = _create(client)
        resp = client.post(
            f"/personas/{pid}/sources",
            data={"platform": "zhihu", "identifier": "https://www.zhihu.com/people/x"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == f"/personas/{pid}"
        source = personas.list_sources(pid)[0]
        assert source["platform"] == "zhihu"

    def test_add_source_persona_404(self, client):
        resp = client.post("/personas/4242/sources", data={"platform": "zhihu", "identifier": "x"})
        assert resp.status_code == 404

    def test_add_source_duplicate_400(self, client):
        pid = _create(client)
        _bind_source(pid, platform="zhihu")
        resp = client.post(
            f"/personas/{pid}/sources", data={"platform": "zhihu", "identifier": "other"}
        )
        assert resp.status_code == 400
        assert "already has a zhihu source" in resp.json()["detail"]

    def test_unbind_source_ok_purges_samples(self, client, samples_dir):
        pid = _create(client)
        sid = _bind_source(pid, platform="weibo")
        sdir = samples_dir / "weibo" / str(pid)
        sdir.mkdir(parents=True)
        (sdir / "dump.json").write_text("[]", encoding="utf-8")

        resp = client.post(f"/personas/{pid}/sources/{sid}/delete", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == f"/personas/{pid}"
        assert personas.list_sources(pid) == []
        assert not sdir.exists()

    def test_unbind_source_missing_404(self, client):
        pid = _create(client)
        resp = client.post(f"/personas/{pid}/sources/999/delete")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "source not found"

    def test_unbind_source_wrong_persona_404(self, client):
        pid, other = _create(client), _create(client, name="别人")
        sid = _bind_source(pid)
        resp = client.post(f"/personas/{other}/sources/{sid}/delete")
        assert resp.status_code == 404


class TestSamplesRoutes:
    def test_list_samples_ok(self, client, dump_factory):
        pid = _create(client)
        dump_factory(
            "zhihu",
            pid,
            [make_sample(1, question_title="问题甲", voteup_count=9), make_sample(2)],
        )
        resp = client.get(f"/personas/{pid}/samples")
        assert resp.status_code == 200
        assert "问题甲" in resp.text and "共 2 条" in resp.text

    def test_list_samples_empty(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}/samples")
        assert "还没有样本" in resp.text

    def test_list_samples_persona_404(self, client):
        assert client.get("/personas/4242/samples").status_code == 404

    def test_show_sample_ok(self, client, dump_factory):
        pid = _create(client)
        dump_factory(
            "zhihu",
            pid,
            [
                make_sample(
                    1,
                    text="样本正文内容",
                    question_title="问题标题",
                    question_detail="问题描述",
                    content_type="answer",
                    voteup_count=7,
                    comment_count=3,
                    content_url="https://www.zhihu.com/x",
                )
            ],
        )
        resp = client.get(f"/personas/{pid}/samples/0")
        assert resp.status_code == 200
        for fragment in ("问题标题", "问题描述", "样本正文内容", "https://www.zhihu.com/x"):
            assert fragment in resp.text

    def test_show_sample_idx_404(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}/samples/5")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "sample not found"

    def test_show_sample_persona_404(self, client):
        assert client.get("/personas/4242/samples/0").status_code == 404


class TestGenerateProfileRoute:
    def _dump_eligible(self, dump_factory, pid):
        dump_factory("zhihu", pid, [make_sample(1, text=f"知乎正文{'。' * 40}1", platform="zhihu")])

    def test_generate_ok_full_pipeline(self, client, dump_factory, profiles_dir, monkeypatch):
        pid = _create(client)
        self._dump_eligible(dump_factory, pid)
        monkeypatch.setattr(voice_profile.llm, "chat", lambda messages, temperature=0.3: PROFILE_MD)
        resp = client.post(f"/personas/{pid}/profile", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == f"/personas/{pid}?profile=ok"
        written = list((profiles_dir / str(pid)).glob("*.md"))
        assert len(written) == 1 and written[0].read_text(encoding="utf-8") == PROFILE_MD
        shown = client.get(resp.headers["location"])  # 跳转目标带 ?profile=ok
        assert "Voice profile 已生成" in shown.text and "Rhythm" in shown.text

    def test_generate_persona_404(self, client):
        resp = client.post("/personas/4242/profile")
        assert resp.status_code == 404

    def test_generate_profile_error_502(self, client):
        pid = _create(client)
        resp = client.post(f"/personas/{pid}/profile")
        assert resp.status_code == 502
        assert "no samples" in resp.json()["detail"]


class TestUpdateAuthorRoute:
    def test_update_author_ok(self, client):
        pid = _create(client)
        resp = client.post(
            f"/personas/{pid}/author",
            data={"author_text": "张三，28 岁，工程师"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == f"/personas/{pid}?author=saved"
        assert personas.get(pid)["author_text"] == "张三，28 岁，工程师"

    def test_update_author_blank_clears(self, client):
        pid = _create(client)
        personas.update_author(pid, "原值")
        client.post(f"/personas/{pid}/author", data={"author_text": "  "})
        assert personas.get(pid)["author_text"] is None

    def test_update_author_persona_404(self, client):
        resp = client.post("/personas/4242/author", data={"author_text": "x"})
        assert resp.status_code == 404


class TestComposeRoutes:
    def test_compose_form_ok(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}/compose")
        assert resp.status_code == 200
        assert "未配置 TAVILY_API_KEY" in resp.text

    def test_compose_form_404(self, client):
        assert client.get("/personas/4242/compose").status_code == 404

    def test_compose_submit_ok_no_search(self, client, profiles_dir, monkeypatch):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", PROFILE_MD)
        captured = {}

        def fake_chat(messages, temperature=0.3):
            captured["messages"] = messages
            return "成稿正文"

        monkeypatch.setattr(voice_profile.llm, "chat", fake_chat)
        resp = client.post(
            f"/personas/{pid}/compose",
            data={"form_type": "long", "topic": "聊聊独立开发"},
        )
        assert resp.status_code == 200
        assert "成稿正文" in resp.text and "执行轨迹" in resp.text
        system = captured["messages"][0]["content"]
        assert "VOICE PROFILE" in system and "原创发起内容" in system
        assert "事实核查规则" not in system

    def test_compose_submit_ok_with_search(self, client, profiles_dir, monkeypatch):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", PROFILE_MD)
        monkeypatch.setenv("TAVILY_API_KEY", "tk")
        captured = {}

        def fake_chat_tools(messages, tools=None, handlers=None, temperature=0.3, trace_tag=""):
            captured["messages"] = messages
            captured["tools"] = tools
            return "联网成稿", [voice_profile.llm.TraceStep("request", "已搜索", "")]

        monkeypatch.setattr(voice_profile.llm, "chat_with_tools", fake_chat_tools)
        resp = client.post(
            f"/personas/{pid}/compose", data={"form_type": "short", "topic": "最新数据"}
        )
        assert resp.status_code == 200
        assert "联网成稿" in resp.text
        assert captured["tools"] == [voice_profile.WEB_SEARCH_TOOL]
        assert "事实核查规则" in captured["messages"][0]["content"]

    def test_compose_submit_q_and_a_uses_answering_stance(self, client, profiles_dir, monkeypatch):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", PROFILE_MD)
        captured = {}

        def fake_chat(messages, temperature=0.3):
            captured["messages"] = messages
            return "回答成稿"

        monkeypatch.setattr(voice_profile.llm, "chat", fake_chat)
        resp = client.post(
            f"/personas/{pid}/compose",
            data={"form_type": "long", "topic": "为什么远程办公？", "q_and_a": "1"},
        )
        assert resp.status_code == 200
        assert "回答一个具体提问" in captured["messages"][0]["content"]

    def test_compose_submit_llm_error_renders_error(self, client, profiles_dir, monkeypatch):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", PROFILE_MD)

        def boom(messages, temperature=0.3):
            raise voice_profile.llm.LLMError("api down")

        monkeypatch.setattr(voice_profile.llm, "chat", boom)
        resp = client.post(f"/personas/{pid}/compose", data={"form_type": "long", "topic": "话题"})
        assert resp.status_code == 200
        assert "api down" in resp.text

    def test_compose_submit_no_profile_renders_error(self, client):
        pid = _create(client)
        resp = client.post(f"/personas/{pid}/compose", data={"form_type": "long", "topic": "话题"})
        assert resp.status_code == 200
        assert "尚未生成 voice profile" in resp.text

    def test_compose_submit_unknown_form_type_renders_error(self, client, profiles_dir):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", PROFILE_MD)
        resp = client.post(f"/personas/{pid}/compose", data={"form_type": "poem", "topic": "话题"})
        assert "未知文本类型: poem" in resp.text

    def test_compose_submit_blank_topic_renders_error(self, client, profiles_dir):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", PROFILE_MD)
        resp = client.post(f"/personas/{pid}/compose", data={"form_type": "long", "topic": "  "})
        assert "输入文本不能为空" in resp.text

    def test_compose_submit_missing_form_type_422(self, client):
        pid = _create(client)
        resp = client.post(f"/personas/{pid}/compose", data={"topic": "话题"})
        assert resp.status_code == 422

    def test_compose_submit_persona_404(self, client):
        resp = client.post("/personas/4242/compose", data={"form_type": "long", "topic": "x"})
        assert resp.status_code == 404


class TestProfileVersionsRoutes:
    def test_versions_ok(self, client, profiles_dir):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", "旧版内容")
        _write_profile(profiles_dir, pid, "20260201T000000.000Z", "新版内容")
        resp = client.get(f"/personas/{pid}/profile/versions")
        assert resp.status_code == 200
        assert "共 2 个版本" in resp.text
        body = resp.text
        assert body.index("2026-02-01 00:00:00 UTC") < body.index("2026-01-01 00:00:00 UTC")

    def test_versions_empty(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}/profile/versions")
        assert "还没有生成过 voice profile" in resp.text

    def test_versions_persona_404(self, client):
        assert client.get("/personas/4242/profile/versions").status_code == 404

    def test_show_version_ok(self, client, profiles_dir):
        pid = _create(client)
        _write_profile(profiles_dir, pid, "20260101T000000.000Z", "版本内容")
        resp = client.get(f"/personas/{pid}/profile/20260101T000000.000Z")
        assert resp.status_code == 200
        assert "版本内容" in resp.text and "2026-01-01 00:00:00 UTC" in resp.text

    def test_show_version_missing_404(self, client):
        pid = _create(client)
        resp = client.get(f"/personas/{pid}/profile/nope")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "profile version not found"

    def test_show_version_persona_404(self, client):
        assert client.get("/personas/4242/profile/nope").status_code == 404


class TestSyncSourceRoute:
    def test_sync_zhihu_ok(self, client, monkeypatch):
        pid = _create(client)
        sid = _bind_source(pid, platform="zhihu", identifier="https://www.zhihu.com/people/x")
        calls = {}

        def fake_sync(persona_id, zhihu_url):
            calls["args"] = (persona_id, zhihu_url)
            return ZhihuCrawlResult(Path("/tmp/samples/x.json"), 5, 5, 3, 1, 1)

        monkeypatch.setattr(crawler_zhihu, "sync", fake_sync)
        resp = client.post(f"/personas/{pid}/sources/{sid}/sync", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == f"/personas/{pid}?added=3&updated=1&unchanged=1&total=5"
        assert calls["args"] == (pid, "https://www.zhihu.com/people/x")
        source = personas.list_sources(pid)[0]
        assert source["last_synced_count"] == 5
        assert source["last_sample_path"] == "/tmp/samples/x.json"

    def test_sync_weibo_ok(self, client, monkeypatch):
        pid = _create(client)
        sid = _bind_source(pid, platform="weibo", identifier="12345")
        calls = {}

        async def fake_cookie():
            return "cookie-value"

        def fake_sync(persona_id, uid, cookie):
            calls["args"] = (persona_id, uid, cookie)
            return WeiboCrawlResult(Path("/tmp/samples/w.json"), 2, 2, 1, 1, 0)

        monkeypatch.setattr(browser_auth, "ensure_weibo_cookie", fake_cookie)
        monkeypatch.setattr(crawler_weibo, "sync", fake_sync)
        resp = client.post(f"/personas/{pid}/sources/{sid}/sync", follow_redirects=False)
        assert resp.status_code == 303
        assert calls["args"] == (pid, "12345", "cookie-value")

    def test_sync_unsupported_platform_400(self, client):
        pid = _create(client)
        sid = _bind_source(pid, platform="douban", identifier="u")
        resp = client.post(f"/personas/{pid}/sources/{sid}/sync")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "unsupported platform: douban"

    def test_sync_crawler_error_502(self, client, monkeypatch):
        pid = _create(client)
        sid = _bind_source(pid)

        def boom(persona_id, zhihu_url):
            raise crawler_zhihu.CrawlerError("crawler 挂了")

        monkeypatch.setattr(crawler_zhihu, "sync", boom)
        resp = client.post(f"/personas/{pid}/sources/{sid}/sync")
        assert resp.status_code == 502
        assert resp.json()["detail"] == "crawler 挂了"

    def test_sync_browser_auth_error_502(self, client, monkeypatch):
        pid = _create(client)
        sid = _bind_source(pid, platform="weibo", identifier="12345")

        async def boom():
            raise browser_auth.BrowserAuthError("没登录")

        monkeypatch.setattr(browser_auth, "ensure_weibo_cookie", boom)
        resp = client.post(f"/personas/{pid}/sources/{sid}/sync")
        assert resp.status_code == 502
        assert resp.json()["detail"] == "browser auth failed: 没登录"

    def test_sync_source_missing_404(self, client):
        pid = _create(client)
        resp = client.post(f"/personas/{pid}/sources/999/sync")
        assert resp.status_code == 404

    def test_sync_wrong_persona_404(self, client):
        pid, other = _create(client), _create(client, name="别人")
        sid = _bind_source(pid)
        resp = client.post(f"/personas/{other}/sources/{sid}/sync")
        assert resp.status_code == 404


class TestClearPlatformCookieRoute:
    def test_clear_weibo_ok(self, client, monkeypatch):
        cleared = []
        monkeypatch.setattr(browser_auth, "clear_cached", cleared.append)
        resp = client.post("/auth/weibo/clear", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"
        assert cleared == ["weibo"]

    def test_clear_unsupported_platform_400(self, client):
        resp = client.post("/auth/zhihu/clear")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "unsupported platform: zhihu"
