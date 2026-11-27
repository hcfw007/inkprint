"""crawler_weibo 编排层测试。

subprocess / 文件系统边界全部 mock：不真起爬虫、不碰 crawlers/ 目录，
只验证编排逻辑——config 读写、返回值解析、噪声过滤、CrawlerError 映射、超时路径。
"""

import json
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app import crawler_weibo as cw


@pytest.fixture()
def weibo_env(tmp_path: Path, samples_dir: Path, monkeypatch) -> Path:
    """把爬虫目录 / 输出 / 样本目录全部重定向进 tmp_path，返回爬虫根目录。"""
    crawler_dir = tmp_path / "crawlers" / "weibo"
    crawler_dir.mkdir(parents=True)
    monkeypatch.setattr(cw, "CRAWLER_DIR", crawler_dir)
    monkeypatch.setattr(cw, "CONFIG_FILE", crawler_dir / "config.json")
    monkeypatch.setattr(cw, "CONFIG_TEMPLATE", crawler_dir / "config_default.json.bak")
    monkeypatch.setattr(cw, "OUTPUT_BASE", crawler_dir / "weibo_data")
    monkeypatch.setattr(cw, "SAMPLES_DIR", samples_dir / "weibo")
    return crawler_dir


class FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def fake_run(monkeypatch, result: FakeProc | None = None, error: BaseException | None = None):
    """替换 subprocess.run：记录 (cmd, kwargs)，可注入返回值或异常。"""
    calls: list[tuple[list[str], dict]] = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if error is not None:
            raise error
        return result or FakeProc()

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


def write_raw_output(uid: str, weibos: list[dict]) -> Path:
    path = cw.OUTPUT_BASE / uid / f"{uid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weibo": weibos}, ensure_ascii=False), encoding="utf-8")
    return path


def write_prior(persona_id: int, items: list[dict]) -> Path:
    path = cw.SAMPLES_DIR / str(persona_id) / "old.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def write_template(content: str = '{"cookie": "old", "user_id_list": [999]}') -> None:
    cw.CONFIG_TEMPLATE.write_text(content, encoding="utf-8")


class TestReadTemplateConfig:
    def test_missing_both_sources_raises(self, weibo_env):
        with pytest.raises(cw.CrawlerError, match="no weibo config template"):
            cw._read_template_config()

    def test_prefers_bak_over_config(self, weibo_env):
        cw.CONFIG_TEMPLATE.write_text(
            '{"cookie": "from-bak", // 注释\n "keep": 1}', encoding="utf-8"
        )
        cw.CONFIG_FILE.write_text('{"cookie": "from-config"}', encoding="utf-8")
        assert cw._read_template_config() == {"cookie": "from-bak", "keep": 1}

    def test_falls_back_to_config_json(self, weibo_env):
        cw.CONFIG_FILE.write_text('{"cookie": "plain"}', encoding="utf-8")
        assert cw._read_template_config() == {"cookie": "plain"}


class TestWriteConfig:
    def test_overrides_controlled_fields_and_keeps_rest(self, weibo_env):
        template = {"keep_me": "x", "anti_ban": "default", "cookie": "old", "user_id_list": [999]}
        cw.CONFIG_TEMPLATE.write_text(json.dumps(template), encoding="utf-8")
        cw._write_config("12345", "COOKIE=1")
        cfg = json.loads(cw.CONFIG_FILE.read_text(encoding="utf-8"))
        assert cfg["user_id_list"] == ["12345"]
        assert cfg["cookie"] == "COOKIE=1"
        assert cfg["write_mode"] == ["json"]
        assert cfg["output_directory"] == "weibo_data"
        assert cfg["user_id_as_folder_name"] == 1
        assert cfg["since_date"] == "2005-01-01"
        assert cfg["remove_html_tag"] == 1
        assert cfg["keep_me"] == "x"
        assert cfg["anti_ban"] == "default"

    def test_seeds_bak_from_config_on_first_run(self, weibo_env):
        cw.CONFIG_FILE.write_text('{"cookie": "old"}', encoding="utf-8")
        cw._write_config("12345", "COOKIE=1")
        assert cw.CONFIG_TEMPLATE.read_text(encoding="utf-8") == '{"cookie": "old"}'


class TestCleanEnv:
    def test_strips_proxy_keys_keeps_others(self, monkeypatch):
        for key in cw.PROXY_ENV_KEYS:
            monkeypatch.setenv(key, "http://proxy:1")
        monkeypatch.setenv("MY_KEEP", "1")
        env = cw._clean_env()
        for key in cw.PROXY_ENV_KEYS:
            assert key not in env
        assert env["MY_KEEP"] == "1"


class TestLatestOutput:
    def test_missing_user_dir_raises(self, weibo_env):
        with pytest.raises(cw.CrawlerError, match="produced no output dir"):
            cw._latest_output("12345")

    def test_dir_without_json_raises(self, weibo_env):
        (cw.OUTPUT_BASE / "12345").mkdir(parents=True)
        with pytest.raises(cw.CrawlerError, match="no json files"):
            cw._latest_output("12345")

    def test_picks_last_sorted(self, weibo_env):
        for stem in ("a", "b"):
            p = cw.OUTPUT_BASE / "12345" / f"{stem}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("[]", encoding="utf-8")
        assert cw._latest_output("12345").name == "b.json"


class TestToUnixTs:
    def test_weibo_native_format(self):
        ts = cw._to_unix_ts("Mon May 18 13:00:00 +0800 2026")
        assert ts == int(datetime(2026, 5, 18, 5, 0, tzinfo=UTC).timestamp())

    @pytest.mark.parametrize("created", ["2026-01-02T03:04:05", "2026-01-02 03:04:05"])
    def test_iso_ish_formats(self, created):
        assert cw._to_unix_ts(created) == int(datetime(2026, 1, 2, 3, 4, 5).timestamp())

    @pytest.mark.parametrize("created", ["garbage", "", None, 123])
    def test_unparsable_returns_zero(self, created):
        assert cw._to_unix_ts(created) == 0


class TestNoiseFilter:
    @pytest.mark.parametrize(
        "text",
        ["转发微博", "Repost", "嗨", "   ", "@微博抽奖平台 新年抽奖", "我已参与转发", "瓜分奖金池"],
    )
    def test_noise_bodies(self, text):
        assert cw._looks_like_noise(text) is True

    def test_real_content_kept(self):
        assert cw._looks_like_noise("这是一条真实原创微博内容，聊聊最近的观察") is False


class TestNormalizeWeibo:
    def test_missing_or_blank_text_dropped(self):
        assert cw._normalize_weibo({"id": 1}) is None
        assert cw._normalize_weibo({"id": 1, "text": "  "}) is None
        assert cw._normalize_weibo({"id": 1, "text": 123}) is None

    def test_noise_text_dropped(self):
        assert cw._normalize_weibo({"id": 1, "text": "转发微博"}) is None

    def test_missing_id_dropped(self):
        assert cw._normalize_weibo({"text": "这是一条真实原创微博内容"}) is None

    def test_normalizes_fields(self):
        item = cw._normalize_weibo(
            {
                "id": 42,
                "text": "这是一条真实原创微博内容",
                "created_at": "2026-01-02T03:04:05",
                "attitudes_count": 5,
                "comments_count": 2,
                "reposts_count": 1,
            }
        )
        assert item["platform"] == "weibo"
        assert item["content_id"] == "42"
        assert item["content_url"] == "https://m.weibo.cn/detail/42"
        assert item["created_time"] == int(datetime(2026, 1, 2, 3, 4, 5).timestamp())
        assert (item["voteup_count"], item["comment_count"], item["reposts_count"]) == (5, 2, 1)

    def test_counts_default_to_zero(self):
        item = cw._normalize_weibo({"id": 1, "text": "这是一条真实原创微博内容"})
        assert (item["voteup_count"], item["comment_count"], item["reposts_count"]) == (0, 0, 0)


class TestHarvest:
    def test_counts_and_writes_normalized_dump(self, weibo_env):
        write_prior(
            7,
            [
                {
                    "platform": "weibo",
                    "content_id": "1",
                    "content_text": "旧内容甲",
                    "updated_time": 0,
                },
                {
                    "platform": "weibo",
                    "content_id": "2",
                    "content_text": "旧内容乙",
                    "updated_time": 10**12,
                },
            ],
        )
        write_raw_output(
            "12345",
            [
                {
                    "id": 1,
                    "text": "第一条足够长的原创微博内容",
                    "created_at": "2026-01-02T03:04:05",
                },
                {
                    "id": 2,
                    "text": "第二条足够长的原创微博内容",
                    "created_at": "2026-01-02T03:04:05",
                },
                {
                    "id": 3,
                    "text": "第三条足够长的原创微博内容",
                    "created_at": "2026-01-02T03:04:05",
                },
                {"id": 4, "text": "转发微博"},  # 噪声，应被过滤
            ],
        )
        result = cw._harvest(7, "12345")
        assert result.item_count == 3
        assert (result.added_count, result.updated_count, result.unchanged_count) == (1, 1, 1)
        assert result.total_count == 3
        assert result.sample_path.parent == cw.SAMPLES_DIR / "7"
        dumped = json.loads(result.sample_path.read_text(encoding="utf-8"))
        assert [d["content_id"] for d in dumped] == ["1", "2", "3"]

    def test_payload_with_empty_weibo_list(self, weibo_env):
        path = cw.OUTPUT_BASE / "12345" / "12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"weibo": []}', encoding="utf-8")
        result = cw._harvest(7, "12345")
        assert result.item_count == 0
        assert result.total_count == 0

    def test_payload_not_dict_counts_zero_and_warns(self, weibo_env):
        # 顶层就不是 dict（如裸 list）：按空结果处理，同时发出结构化告警。
        path = cw.OUTPUT_BASE / "12345" / "12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.warns(cw.CrawlerDumpWarning, match="reason=payload_not_dict") as record:
            result = cw._harvest(7, "12345")
        assert result.item_count == 0
        assert "skipped_non_dict=0" in str(record[0].message)

    def test_payload_missing_weibo_key_counts_zero_and_warns(self, weibo_env):
        # dict 缺 "weibo" 键：与 zhihu 对畸形 dump 的处理一致，视为空结果而非 TypeError；
        # 但不再静默归档空文件，必须发出告警。
        path = cw.OUTPUT_BASE / "12345" / "12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"ok": 1}', encoding="utf-8")
        with pytest.warns(cw.CrawlerDumpWarning, match="reason=missing_weibo_key"):
            result = cw._harvest(7, "12345")
        assert result.item_count == 0
        assert result.total_count == 0
        assert (result.added_count, result.updated_count, result.unchanged_count) == (0, 0, 0)

    def test_payload_weibo_null_counts_zero_and_warns(self, weibo_env):
        path = cw.OUTPUT_BASE / "12345" / "12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"weibo": null}', encoding="utf-8")
        with pytest.warns(cw.CrawlerDumpWarning, match="reason=weibo_not_a_list"):
            result = cw._harvest(7, "12345")
        assert result.item_count == 0
        assert result.total_count == 0

    def test_payload_weibo_non_list_counts_zero_and_warns(self, weibo_env):
        # 与 zhihu 的 test_non_list_dump_counts_zero 对齐：非 list 一律按空处理。
        path = cw.OUTPUT_BASE / "12345" / "12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"weibo": "oops"}', encoding="utf-8")
        with pytest.warns(cw.CrawlerDumpWarning, match="reason=weibo_not_a_list"):
            result = cw._harvest(7, "12345")
        assert result.item_count == 0
        assert result.total_count == 0

    def test_non_dict_elements_skipped_counted_and_warned(self, weibo_env):
        # 元素级防御：list 里混入字符串/数字/None 时跳过而非 AttributeError，
        # 跳过数写进同一条告警（reason=non_dict_items）。
        path = cw.OUTPUT_BASE / "12345" / "12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weibo": ["oops", {"id": 1, "text": "第一条足够长的原创微博内容"}, 42, None]}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.warns(cw.CrawlerDumpWarning) as record:
            result = cw._harvest(7, "12345")
        assert result.item_count == 1
        assert len(record) == 1
        message = str(record[0].message)
        assert "reason=non_dict_items" in message
        assert "skipped_non_dict=3" in message

    def test_clean_dump_emits_no_warning(self, weibo_env):
        # 正常 dump（含空 list）不产生告警，避免告警噪声稀释信号。
        write_raw_output("12345", [{"id": 1, "text": "第一条足够长的原创微博内容"}])
        raw = cw.OUTPUT_BASE / "12345" / "12345.json"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = cw._harvest(7, "12345")
            raw.write_text('{"weibo": []}', encoding="utf-8")
            cw._harvest(7, "12345")
        assert result.item_count == 1
        assert not [w for w in caught if issubclass(w.category, cw.CrawlerDumpWarning)]


class TestSync:
    def test_empty_cookie_raises_before_anything(self, weibo_env, monkeypatch):
        calls = fake_run(monkeypatch)
        with pytest.raises(cw.CrawlerError, match="empty cookie"):
            cw.sync(7, "12345", "   ")
        assert calls == []
        assert not cw.CONFIG_FILE.exists()

    def test_missing_venv_raises_with_hint(self, weibo_env, monkeypatch):
        write_template()
        calls = fake_run(monkeypatch)
        with pytest.raises(cw.CrawlerError, match="uv venv"):
            cw.sync(7, "12345", "COOKIE=1")
        assert calls == []
        assert cw.CONFIG_FILE.exists()  # config 先写好，装好 venv 重跑即可

    def test_success_runs_crawler_and_harvests(self, weibo_env, monkeypatch):
        write_template()
        for key in cw.PROXY_ENV_KEYS:
            monkeypatch.setenv(key, "http://proxy:1")
        venv_python = weibo_env / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        write_raw_output("12345", [{"id": 1, "text": "第一条足够长的原创微博内容"}])
        calls = fake_run(monkeypatch)
        result = cw.sync(7, "12345", "COOKIE=1")
        cmd, kwargs = calls[0]
        assert cmd == [str(venv_python), "weibo.py"]
        assert kwargs["cwd"] == cw.CRAWLER_DIR
        assert all(key not in kwargs["env"] for key in cw.PROXY_ENV_KEYS)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == cw.CRAWL_TIMEOUT_SECONDS == 900
        cfg = json.loads(cw.CONFIG_FILE.read_text(encoding="utf-8"))
        assert cfg["user_id_list"] == ["12345"]
        assert cfg["cookie"] == "COOKIE=1"
        assert result.added_count == 1

    def test_nonzero_exit_maps_to_crawler_error(self, weibo_env, monkeypatch):
        write_template()
        venv_python = weibo_env / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        fake_run(monkeypatch, FakeProc(returncode=3, stderr="boom"))
        with pytest.raises(cw.CrawlerError, match="exited 3") as excinfo:
            cw.sync(7, "12345", "COOKIE=1")
        assert "boom" in str(excinfo.value)

    def test_error_tail_falls_back_to_stdout(self, weibo_env, monkeypatch):
        write_template()
        venv_python = weibo_env / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        fake_run(monkeypatch, FakeProc(returncode=1, stderr="", stdout="out tail"))
        with pytest.raises(cw.CrawlerError, match="out tail"):
            cw.sync(7, "12345", "COOKIE=1")

    def test_error_tail_truncated_to_2000_chars(self, weibo_env, monkeypatch):
        write_template()
        venv_python = weibo_env / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        stderr = "A" * 2500 + "B" * 2500
        fake_run(monkeypatch, FakeProc(returncode=1, stderr=stderr))
        with pytest.raises(cw.CrawlerError) as excinfo:
            cw.sync(7, "12345", "COOKIE=1")
        assert "B" * 2000 in str(excinfo.value)
        assert "A" * 2500 not in str(excinfo.value)

    def test_timeout_expires_wrapped_as_crawler_error(self, weibo_env, monkeypatch):
        # 行为变更：TimeoutExpired 不再透传，包装成 CrawlerError（含超时秒数），
        # 路由层统一走既有 502 映射。原「透传」用例已按新行为更新。
        write_template()
        venv_python = weibo_env / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        error = subprocess.TimeoutExpired(cmd="weibo.py", timeout=cw.CRAWL_TIMEOUT_SECONDS)
        fake_run(monkeypatch, error=error)
        with pytest.raises(cw.CrawlerError, match=r"timed out after 900s") as excinfo:
            cw.sync(7, "12345", "COOKIE=1")
        assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
