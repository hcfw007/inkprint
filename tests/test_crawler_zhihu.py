"""crawler_zhihu 编排层测试。

subprocess / 文件系统边界全部 mock：不真起爬虫、不碰 crawlers/ 目录，
只验证编排逻辑——config 重写、参数组装、返回值解析、CrawlerError 映射、超时路径。
"""

import json
import subprocess
from pathlib import Path

import pytest

from app import crawler_zhihu as cz

CONFIG_SOURCE = (
    "CUSTOM = 1\n"
    'ZHIHU_CREATOR_URL_LIST = [\n    "https://www.zhihu.com/people/old",\n]\n'
    "AFTER = 2\n"
)


@pytest.fixture()
def zhihu_env(tmp_path: Path, samples_dir: Path, monkeypatch) -> Path:
    """把爬虫配置 / 输出 / 样本目录全部重定向进 tmp_path，返回 config 路径。"""
    crawler_dir = tmp_path / "crawlers" / "zhihu"
    config_file = crawler_dir / "config" / "zhihu_config.py"
    config_file.parent.mkdir(parents=True)
    monkeypatch.setattr(cz, "CRAWLER_DIR", crawler_dir)
    monkeypatch.setattr(cz, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cz, "OUTPUT_DIR", crawler_dir / "data" / "zhihu" / "json")
    monkeypatch.setattr(cz, "SAMPLES_DIR", samples_dir / "zhihu")
    return config_file


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


def write_dump(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestRewriteConfig:
    def test_missing_config_raises(self, zhihu_env):
        with pytest.raises(cz.CrawlerError, match="Did you clone"):
            cz._rewrite_config("https://www.zhihu.com/people/new")

    def test_missing_url_list_raises(self, zhihu_env):
        zhihu_env.write_text("OTHER = 1\n", encoding="utf-8")
        with pytest.raises(cz.CrawlerError, match="could not locate"):
            cz._rewrite_config("https://www.zhihu.com/people/new")

    def test_replaces_url_and_preserves_rest(self, zhihu_env):
        zhihu_env.write_text(CONFIG_SOURCE, encoding="utf-8")
        cz._rewrite_config("https://www.zhihu.com/people/new")
        content = zhihu_env.read_text(encoding="utf-8")
        assert '"https://www.zhihu.com/people/new"' in content
        assert "old" not in content
        assert "CUSTOM = 1" in content
        assert "AFTER = 2" in content


class TestCleanEnv:
    def test_strips_proxy_keys_keeps_others(self, monkeypatch):
        for key in cz.PROXY_ENV_KEYS:
            monkeypatch.setenv(key, "http://proxy:1")
        monkeypatch.setenv("MY_KEEP", "1")
        env = cz._clean_env()
        for key in cz.PROXY_ENV_KEYS:
            assert key not in env
        assert env["MY_KEEP"] == "1"
        assert "PATH" in env


class TestLatestOutput:
    def test_missing_output_dir_raises(self, zhihu_env):
        with pytest.raises(cz.CrawlerError, match="no output dir"):
            cz._latest_output()

    def test_no_matching_files_raises(self, zhihu_env):
        cz.OUTPUT_DIR.mkdir(parents=True)
        (cz.OUTPUT_DIR / "other.json").write_text("[]", encoding="utf-8")
        with pytest.raises(cz.CrawlerError, match="no creator_contents"):
            cz._latest_output()

    def test_picks_last_sorted(self, zhihu_env):
        for stem in ("creator_contents_20260201", "creator_contents_20260101"):
            (cz.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            (cz.OUTPUT_DIR / f"{stem}.json").write_text("[]", encoding="utf-8")
        assert cz._latest_output().name == "creator_contents_20260201.json"


class TestHarvest:
    def test_counts_added_updated_unchanged(self, zhihu_env, samples_dir):
        # 历史 dump：id=a 会被更新，id=b 保持不变
        write_dump(
            samples_dir / "zhihu" / "7" / "old.json",
            [
                {"content_id": "a", "content_text": "旧内容甲", "updated_time": 100},
                {"content_id": "b", "content_text": "旧内容乙", "updated_time": 10**12},
            ],
        )
        dump = [
            {"content_id": "a", "content_text": "新内容甲", "updated_time": 200},
            {"content_id": "b", "content_text": "旧内容乙", "updated_time": 10**12},
            {"content_id": "c", "content_text": "全新内容", "updated_time": 300},
            {"content_id": 99, "content_text": "非法 id 不计数"},
            {"content_text": "缺 id 不计数"},
        ]
        output = write_dump(cz.OUTPUT_DIR / "creator_contents_1.json", dump)
        result = cz._harvest(7, output)
        assert result.item_count == 5
        assert (result.added_count, result.updated_count, result.unchanged_count) == (1, 1, 1)
        assert result.total_count == 3
        assert result.sample_path.parent == samples_dir / "zhihu" / "7"
        assert result.sample_path.exists()

    def test_non_list_dump_counts_zero(self, zhihu_env):
        output = write_dump(cz.OUTPUT_DIR / "creator_contents_1.json", {"oops": True})
        result = cz._harvest(7, output)
        assert result.item_count == 0
        assert result.total_count == 0
        assert (result.added_count, result.updated_count, result.unchanged_count) == (0, 0, 0)


class TestSync:
    def test_success_runs_crawler_and_harvests(self, zhihu_env, monkeypatch):
        for key in cz.PROXY_ENV_KEYS:
            monkeypatch.setenv(key, "http://proxy:1")
        zhihu_env.write_text(CONFIG_SOURCE, encoding="utf-8")
        write_dump(
            cz.OUTPUT_DIR / "creator_contents_1.json",
            [{"content_id": "a", "content_text": "全新内容", "updated_time": 300}],
        )
        calls = fake_run(monkeypatch)
        result = cz.sync(7, "https://www.zhihu.com/people/new")
        cmd, kwargs = calls[0]
        assert cmd == [
            "uv",
            "run",
            "main.py",
            "--platform",
            "zhihu",
            "--lt",
            "qrcode",
            "--type",
            "creator",
        ]
        assert kwargs["cwd"] == cz.CRAWLER_DIR
        assert all(key not in kwargs["env"] for key in cz.PROXY_ENV_KEYS)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == cz.CRAWL_TIMEOUT_SECONDS == 600
        assert '"https://www.zhihu.com/people/new"' in zhihu_env.read_text(encoding="utf-8")
        assert result.added_count == 1

    def test_missing_config_short_circuits_before_run(self, zhihu_env, monkeypatch):
        calls = fake_run(monkeypatch)
        with pytest.raises(cz.CrawlerError):
            cz.sync(7, "https://www.zhihu.com/people/new")
        assert calls == []

    def test_nonzero_exit_maps_to_crawler_error(self, zhihu_env, monkeypatch):
        zhihu_env.write_text(CONFIG_SOURCE, encoding="utf-8")
        fake_run(monkeypatch, FakeProc(returncode=2, stderr="bad thing"))
        with pytest.raises(cz.CrawlerError, match="exited with code 2") as excinfo:
            cz.sync(7, "https://www.zhihu.com/people/new")
        assert "bad thing" in str(excinfo.value)

    def test_error_tail_falls_back_to_stdout(self, zhihu_env, monkeypatch):
        zhihu_env.write_text(CONFIG_SOURCE, encoding="utf-8")
        fake_run(monkeypatch, FakeProc(returncode=1, stderr="", stdout="out tail"))
        with pytest.raises(cz.CrawlerError, match="out tail"):
            cz.sync(7, "https://www.zhihu.com/people/new")

    def test_error_tail_truncated_to_2000_chars(self, zhihu_env, monkeypatch):
        zhihu_env.write_text(CONFIG_SOURCE, encoding="utf-8")
        stderr = "A" * 2500 + "B" * 2500
        fake_run(monkeypatch, FakeProc(returncode=1, stderr=stderr))
        with pytest.raises(cz.CrawlerError) as excinfo:
            cz.sync(7, "https://www.zhihu.com/people/new")
        assert "B" * 2000 in str(excinfo.value)
        assert "A" * 2500 not in str(excinfo.value)

    def test_timeout_expires_propagates_unwrapped(self, zhihu_env, monkeypatch):
        zhihu_env.write_text(CONFIG_SOURCE, encoding="utf-8")
        error = subprocess.TimeoutExpired(cmd="uv", timeout=cz.CRAWL_TIMEOUT_SECONDS)
        fake_run(monkeypatch, error=error)
        with pytest.raises(subprocess.TimeoutExpired):
            cz.sync(7, "https://www.zhihu.com/people/new")
