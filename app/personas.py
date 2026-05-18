"""Persona repository — CRUD over personas + source_bindings."""

import sqlite3
from typing import Any

from .db import connect


def list_all() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT id, name, description, created_at FROM personas ORDER BY created_at DESC"
        ).fetchall()


def get(persona_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT id, name, description, created_at FROM personas WHERE id = ?",
            (persona_id,),
        ).fetchone()


def create(name: str, description: str | None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO personas (name, description) VALUES (?, ?)",
            (name.strip(), (description or "").strip() or None),
        )
        return int(cur.lastrowid or 0)


def list_sources(persona_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT id, platform, identifier, last_synced_at
            FROM source_bindings
            WHERE persona_id = ?
            ORDER BY id ASC
            """,
            (persona_id,),
        ).fetchall()


def add_source(persona_id: int, platform: str, identifier: str) -> dict[str, Any]:
    with connect() as conn:
        try:
            conn.execute(
                "INSERT INTO source_bindings (persona_id, platform, identifier) VALUES (?, ?, ?)",
                (persona_id, platform.strip(), identifier.strip()),
            )
            return {"ok": True}
        except sqlite3.IntegrityError as e:
            return {"ok": False, "error": str(e)}
