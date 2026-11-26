"""Shared fixtures — every test runs offline against tmp dirs.

Isolation guarantees:
- LLM / Tavily env vars are cleared per-test (app/__init__ runs load_dotenv(),
  so a developer's local .env must not leak into assertions).
- Outbound network is blocked at the connect level: any un-mocked network
  call fails loudly. (socketpair() self-pipes must keep working so asyncio
  event loops — and FastAPI's TestClient — can start; nothing can reach a
  remote host without connect/sendto.)
- samples/ profiles/ data/ are redirected into tmp_path — the real gitignored
  runtime dirs are never touched.
"""

import json
import socket
from pathlib import Path

import pytest

from app import db, samples, voice_profile

ENV_VARS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_PROXY", "TAVILY_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Tripwire: real network I/O raises loudly; in-process ASGI stays usable.

    Block at connect/sendto level instead of replacing the socket()
    constructor: asyncio event loops build socketpair() self-pipes at
    startup, which a constructor-level block would break — that would make
    FastAPI's TestClient (and any async code) untestable.
    """

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is blocked in this test suite (all I/O must be mocked)")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket.socket, "sendto", _blocked)


@pytest.fixture()
def profiles_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    monkeypatch.setattr(voice_profile, "PROFILES_DIR", d)
    return d


@pytest.fixture()
def samples_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "samples"
    d.mkdir()
    monkeypatch.setattr(samples, "SAMPLES_DIR", d)
    return d


@pytest.fixture()
def db_path(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "data" / "inkprint.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


@pytest.fixture()
def dump_factory(samples_dir: Path):
    """Write a crawler-style JSON dump and return its path."""

    def _dump(platform: str, persona_id: int, items: list[dict], name: str = "dump.json") -> Path:
        pdir = samples_dir / platform / str(persona_id)
        pdir.mkdir(parents=True, exist_ok=True)
        path = pdir / name
        path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return path

    return _dump


def make_sample(i: int, platform: str = "zhihu", text: str | None = None, **extra) -> dict:
    """A minimal normalized sample item."""
    item = {
        "platform": platform,
        "content_id": f"{platform}-{i}",
        "content_text": (
            text
            if text is not None
            else f"这是第 {i} 条足够长的样本正文，用来通过最小长度门槛。再补一句凑够字数。"
        ),
        "updated_time": 1_700_000_000 + i,
    }
    item.update(extra)
    return item
