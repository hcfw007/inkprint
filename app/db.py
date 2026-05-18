"""SQLite layer for inkprint.

Single-file DB at data/inkprint.db. Pure sqlite3 stdlib, no ORM —
keeps the dependency surface small and the schema obvious.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inkprint.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS source_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    identifier TEXT NOT NULL,
    last_synced_at TEXT,
    last_synced_count INTEGER,
    last_sample_path TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE,
    UNIQUE (persona_id, platform, identifier)
);
"""

# Lightweight forward migrations for columns added after initial release.
# Each row: (table, column, DDL fragment). ALTER TABLE ... ADD COLUMN is
# idempotent-via-introspection: we skip if PRAGMA table_info already lists it.
COLUMN_MIGRATIONS = (
    ("source_bindings", "last_synced_count", "INTEGER"),
    ("source_bindings", "last_sample_path", "TEXT"),
)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        for table, column, ddl in COLUMN_MIGRATIONS:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
