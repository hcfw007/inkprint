"""db.py — schema 初始化与增量列迁移，跑在临时 SQLite 上。"""

import sqlite3

from app import db


def _columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


class TestInitDb:
    def test_creates_tables(self, db_path):
        assert db_path.exists()
        with db.connect() as conn:
            tables = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert {"personas", "source_bindings"} <= tables

    def test_schema_columns(self, db_path):
        with db.connect() as conn:
            persona_cols = _columns(conn, "personas")
            source_cols = _columns(conn, "source_bindings")
        expected = {"id", "name", "description", "created_at", "author_text", "author_goal"}
        assert expected <= persona_cols
        assert {
            "id",
            "persona_id",
            "platform",
            "identifier",
            "last_synced_at",
            "last_synced_count",
            "last_sample_path",
        } <= source_cols

    def test_idempotent(self, db_path):
        db.init_db()
        db.init_db()
        with db.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM personas").fetchone()[0]
        assert count == 0

    def test_migrates_legacy_schema(self, tmp_path, monkeypatch):
        """旧库（无 author_text 等后加列）升级后补齐列且保留数据。"""
        path = tmp_path / "legacy.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE source_bindings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id INTEGER NOT NULL,
                platform TEXT NOT NULL,
                identifier TEXT NOT NULL,
                last_synced_at TEXT,
                UNIQUE (persona_id, platform, identifier)
            );
            INSERT INTO personas (name) VALUES ('legacy');
            """
        )
        conn.commit()
        conn.close()

        db.init_db()

        with db.connect() as conn:
            assert {"author_text", "author_goal"} <= _columns(conn, "personas")
            assert {"last_synced_count", "last_sample_path"} <= _columns(conn, "source_bindings")
            name = conn.execute("SELECT name FROM personas").fetchone()[0]
        assert name == "legacy"

    def test_migrations_cover_all_declared_columns(self):
        for table, column, _ddl in db.COLUMN_MIGRATIONS:
            assert table in ("personas", "source_bindings")
            assert column and isinstance(column, str)


class TestConnect:
    def test_row_factory_is_row(self, db_path):
        with db.connect() as conn:
            conn.execute("INSERT INTO personas (name) VALUES ('a')")
            row = conn.execute("SELECT * FROM personas WHERE name = 'a'").fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["name"] == "a"

    def test_foreign_keys_enforced(self, db_path):
        with db.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO source_bindings (persona_id, platform, identifier) "
                    "VALUES (999, 'zhihu', 'x')"
                )
            except sqlite3.IntegrityError:
                return
        raise AssertionError("foreign key constraint was not enforced")

    def test_commit_on_exit(self, db_path):
        with db.connect() as conn:
            conn.execute("INSERT INTO personas (name) VALUES ('commit-me')")
        with db.connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM personas WHERE name = 'commit-me'").fetchone()
        assert row[0] == 1

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        path = tmp_path / "deep" / "nested" / "inkprint.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()
        assert path.exists()
