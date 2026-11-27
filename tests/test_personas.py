"""personas.py — persona / source_bindings CRUD 跑在临时 SQLite 上。"""

import pytest

from app import personas


@pytest.fixture()
def pid(db_path):
    return personas.create("  测试人格  ", "  描述  ")


class TestPersonaCrud:
    def test_create_strips_and_normalizes(self, db_path):
        pid = personas.create("  甲  ", "   ")
        row = personas.get(pid)
        assert row["name"] == "甲"
        assert row["description"] is None
        assert row["created_at"]

    def test_list_all_returns_every_persona(self, db_path):
        personas.create("甲", None)
        personas.create("乙", None)
        names = [r["name"] for r in personas.list_all()]
        # created_at 粒度为秒，同秒行序不保证；只验证集合与数量
        assert set(names) == {"甲", "乙"}
        assert len(names) == 2

    def test_get_missing_returns_none(self, db_path):
        assert personas.get(999) is None

    def test_update_author(self, db_path, pid):
        personas.update_author(pid, "  某某  ")
        assert personas.get(pid)["author_text"] == "某某"
        personas.update_author(pid, "   ")
        assert personas.get(pid)["author_text"] is None

    def test_delete_cascades_sources(self, db_path, pid):
        personas.add_source(pid, "zhihu", "uid-1")
        personas.delete(pid)
        assert personas.get(pid) is None
        assert personas.list_sources(pid) == []


class TestSources:
    def test_add_source_and_list(self, db_path, pid):
        assert personas.add_source(pid, " zhihu ", " uid-1 ") == {"ok": True}
        sources = personas.list_sources(pid)
        assert len(sources) == 1
        assert sources[0]["platform"] == "zhihu"
        assert sources[0]["identifier"] == "uid-1"
        assert sources[0]["last_synced_at"] is None

    def test_add_source_rejects_duplicate_platform(self, db_path, pid):
        personas.add_source(pid, "zhihu", "uid-1")
        result = personas.add_source(pid, "zhihu", "uid-2")
        assert result["ok"] is False
        assert "zhihu" in result["error"]

    def test_add_source_integrity_error(self, db_path, pid):
        personas.add_source(pid, "zhihu", "uid-1")
        # UNIQUE (persona_id, platform, identifier) 比平台去重更宽，直接构造冲突
        result = personas.add_source(pid, "weibo", "uid-1")
        assert result == {"ok": True}
        result = personas.add_source(pid + 100, "zhihu", "uid-9")
        # 不存在的 persona 会触发外键约束 → IntegrityError 被包装返回
        assert result["ok"] is False
        assert result["error"]

    def test_get_source_and_delete(self, db_path, pid):
        personas.add_source(pid, "zhihu", "uid-1")
        sid = personas.list_sources(pid)[0]["id"]
        assert personas.get_source(sid)["identifier"] == "uid-1"
        personas.delete_source(sid)
        assert personas.get_source(sid) is None

    def test_bound_platforms(self, db_path, pid):
        personas.add_source(pid, "zhihu", "uid-1")
        personas.add_source(pid, "weibo", "uid-2")
        assert personas.bound_platforms(pid) == {"zhihu", "weibo"}

    def test_mark_source_synced(self, db_path, pid):
        personas.add_source(pid, "zhihu", "uid-1")
        sid = personas.list_sources(pid)[0]["id"]
        personas.mark_source_synced(sid, 42, "samples/zhihu/1/x.json")
        source = personas.list_sources(pid)[0]
        assert source["last_synced_count"] == 42
        assert source["last_sample_path"] == "samples/zhihu/1/x.json"
        assert source["last_synced_at"] is not None
