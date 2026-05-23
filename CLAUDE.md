# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

inkprint extracts a writer's style signature from their real social-media posts and emits a structured `VOICE PROFILE` markdown that any downstream LLM can read to imitate the voice. It also has a `compose` mode that uses that profile (plus optional web-search grounding) to draft new posts in the persona's voice.

## Commands

```bash
# Install / bootstrap (idempotent — re-run safely; clones the two crawlers, builds their venvs)
./setup.sh                  # macOS / Linux
python setup.py             # cross-platform

# Run the app (FastAPI + Jinja, http://127.0.0.1:8765/)
uv run uvicorn app.main:app --reload --port 8765

# Lint / format (Ruff config in pyproject.toml; crawlers/ is excluded)
uv run ruff check . --fix
uv run ruff format .
```

No test suite exists yet. There is no `pytest` configured — do not invent test commands.

## Runtime prerequisites the code assumes

- **Two Python venvs coexist:** the app uses Python 3.13 via `uv`, but `crawlers/weibo/` requires Python 3.11 (separate venv created by `setup.py`). Don't try to unify them.
- **Node.js** is required at runtime — `crawlers/zhihu/` uses `execjs` to run Zhihu's signing JS.
- **System Chrome** is driven via Playwright CDP (`browser_auth.py`). `playwright install` is intentionally NOT run; do not add it.
- **LLM config** comes from `.env` (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`) — any OpenAI-compatible endpoint. Optional `TAVILY_API_KEY` enables web-search tool-use in `compose`. Optional `LLM_PROXY` only when calling Anthropic/OpenAI from inside CN.

## Architecture

Single FastAPI process. All routes live in `app/main.py`; every other module under `app/` is a leaf called from there. SQLite (stdlib `sqlite3`, no ORM) at `data/inkprint.db`.

**The core data flow is layered, and the layer boundaries matter:**

```
HTTP route (main.py)
   ↓
domain module (personas / voice_profile / samples)
   ↓
external boundary (crawler_zhihu / crawler_weibo / browser_auth / llm / search)
   ↓
filesystem dump → samples/{platform}/{persona_id}/*.json
   ↓
samples.load_merged() unions across platforms, dedupes by (platform, content_id),
keeps newest by updated_time
   ↓
voice_profile picks a platform-balanced subset → LLM → profiles/{persona_id}/{ts}.md
```

Key invariants to preserve when editing:

- **Crawlers run as subprocesses under their own venv**, not as Python imports. `crawler_zhihu.py` / `crawler_weibo.py` rewrite the upstream config files, spawn the child, then harvest the resulting JSON dumps. Treat `crawlers/*` as third-party — gitignored, never edit directly, and re-clonable by `setup.py`.
- **Proxy env vars are scrubbed before crawler subprocess calls** (`PROXY_ENV_KEYS` in `crawler_zhihu.py`) because zhihu/weibo are CN sites and a clash-style auto-proxy will tunnel them overseas and time out. `llm.py` uses `httpx.Client(trust_env=False)` for the same reason — only respect `LLM_PROXY` when explicitly set.
- **Samples are normalized to a common shape across platforms** (`platform`, `content_id`, `content_text` at minimum — see `samples.COMMON_FIELDS`). Downstream code must not branch on `platform` outside the crawler boundary and a few labeled UI cases. New platforms add a normalizer in their crawler module, not in samples.
- **Voice profile generation samples are platform-balanced via round-robin** (`voice_profile._pick_representative`). The intent is that a chatty long-form platform does not drown out a terser short-form one; preserve that when changing selection.
- **Author overrides are display-time overlays, not file rewrites.** `apply_author_overrides` rewrites only the `Author:` line in-memory so historical profile snapshots on disk stay faithful to what the model produced.
- **Profile history is per-persona under `profiles/{id}/{timestamp}.md`.** `_migrate_legacy_flat` quietly migrates the older `profiles/{id}.md` flat layout on read — keep that path working.
- **DB schema migrations are additive via `COLUMN_MIGRATIONS` in `db.py`** (PRAGMA-introspecting ALTER TABLE ADD COLUMN). No migration tool; add new columns by appending to that tuple.
- **Compose uses tool-use loop when search is available**, falling back to a single-shot `llm.chat` otherwise. The `trace` field on `ComposeResult` powers the "执行轨迹" UI — preserve it when changing the compose path.
- **Sync is blocking** by design (no background workers). Each `POST /personas/{id}/sources/{sid}/sync` runs the crawler in the request and can take minutes. Don't refactor toward async tasks casually.

## Gitignored runtime directories

`crawlers/`, `samples/`, `profiles/`, `data/`, `.env` — recreated by setup or by running the app. Don't commit anything under them.

## User instruction overrides

Per global rules, before editing any function, grep all call sites first; before editing any file, read it first. Crawlers under `crawlers/` are explicitly out of scope for ruff and for edits.
