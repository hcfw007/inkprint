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
            "SELECT id, name, description, created_at, author_text, author_goal "
            "FROM personas WHERE id = ?",
            (persona_id,),
        ).fetchone()


def update_author(persona_id: int, author_text: str, author_goal: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE personas SET author_text = ?, author_goal = ? WHERE id = ?",
            (author_text.strip() or None, author_goal.strip() or None, persona_id),
        )


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
            SELECT id, platform, identifier, last_synced_at,
                   last_synced_count, last_sample_path
            FROM source_bindings
            WHERE persona_id = ?
            ORDER BY id ASC
            """,
            (persona_id,),
        ).fetchall()


def add_source(persona_id: int, platform: str, identifier: str) -> dict[str, Any]:
    platform = platform.strip()
    identifier = identifier.strip()
    with connect() as conn:
        existing = conn.execute(
            "SELECT 1 FROM source_bindings WHERE persona_id = ? AND platform = ?",
            (persona_id, platform),
        ).fetchone()
        if existing:
            return {"ok": False, "error": f"persona already has a {platform} source"}
        try:
            conn.execute(
                "INSERT INTO source_bindings (persona_id, platform, identifier) VALUES (?, ?, ?)",
                (persona_id, platform, identifier),
            )
            return {"ok": True}
        except sqlite3.IntegrityError as e:
            return {"ok": False, "error": str(e)}


def bound_platforms(persona_id: int) -> set[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT platform FROM source_bindings WHERE persona_id = ?",
            (persona_id,),
        ).fetchall()
    return {row["platform"] for row in rows}


def get_source(source_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT id, persona_id, platform, identifier, last_synced_at "
            "FROM source_bindings WHERE id = ?",
            (source_id,),
        ).fetchone()


def delete(persona_id: int) -> None:
    """Delete a persona. ON DELETE CASCADE handles source_bindings."""
    with connect() as conn:
        conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,))


def delete_source(source_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM source_bindings WHERE id = ?", (source_id,))


def mark_source_synced(source_id: int, item_count: int, sample_path: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE source_bindings
            SET last_synced_at = datetime('now'),
                last_synced_count = ?,
                last_sample_path = ?
            WHERE id = ?
            """,
            (item_count, sample_path, source_id),
        )
