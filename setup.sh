#!/usr/bin/env bash
# One-shot bootstrapper for inkprint. Idempotent — re-runs do nothing
# expensive when the artifacts are already in place.
#
# Prerequisites (script does NOT install these):
#   - Python 3.13 — uv will fetch automatically if missing
#   - uv         — `brew install uv` or curl install script
#   - Node.js ≥ 16 — knowledge_zhihu signing JS runs under it
#   - Google Chrome — both crawlers drive system Chrome via CDP
#
# Usage:
#   ./setup.sh
#
# After it finishes, edit .env to fill in LLM_API_KEY etc., then:
#   uv run uvicorn app.main:app --reload --port 8765

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

step() { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }

# --- check tools ---
step "checking prerequisites"
for cmd in uv git node; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "  ✗ missing: $cmd"
    echo "    install it then re-run; see README for details"
    exit 1
  fi
done
ok "uv $(uv --version | awk '{print $2}') · git · node $(node --version)"
if [ ! -d "/Applications/Google Chrome.app" ] && [ ! -x "$(command -v google-chrome 2>/dev/null)" ]; then
  warn "Google Chrome not detected — crawlers will fail to launch their CDP session"
fi

# --- main app deps ---
step "installing main app deps"
uv sync --quiet
ok "uv sync done"

# --- zhihu crawler ---
step "setting up crawlers/zhihu (MediaCrawler_zhihu)"
ZHIHU_DIR="crawlers/zhihu"
if [ ! -d "$ZHIHU_DIR/.git" ]; then
  rm -rf "$ZHIHU_DIR"
  git clone --depth 1 https://github.com/xx-hub/MediaCrawler_zhihu.git "$ZHIHU_DIR"
  ok "cloned MediaCrawler_zhihu"
else
  ok "already cloned"
fi
( cd "$ZHIHU_DIR" && uv sync --quiet )
ok "uv sync inside zhihu"
for src in config/base_config.example.py config/zhihu_config.example.py; do
  dst="${src/.example/}"
  if [ ! -f "$ZHIHU_DIR/$dst" ]; then
    cp "$ZHIHU_DIR/$src" "$ZHIHU_DIR/$dst"
    ok "seeded $dst"
  fi
done

# --- weibo crawler ---
step "setting up crawlers/weibo (dataabc/weibo-crawler)"
WEIBO_DIR="crawlers/weibo"
if [ ! -d "$WEIBO_DIR/.git" ]; then
  rm -rf "$WEIBO_DIR"
  git clone --depth 1 https://github.com/dataabc/weibo-crawler.git "$WEIBO_DIR"
  ok "cloned weibo-crawler"
else
  ok "already cloned"
fi
# weibo-crawler ships with no pyproject.toml; we create a python 3.11 venv
# explicitly because its lxml/pin set isn't happy on 3.13.
if [ ! -x "$WEIBO_DIR/.venv/bin/python" ]; then
  ( cd "$WEIBO_DIR" && uv venv --python 3.11 .venv --quiet )
  ok "created venv (python 3.11)"
fi
( cd "$WEIBO_DIR" && uv pip install --python .venv/bin/python -r requirements.txt --quiet )
ok "pip install inside weibo"
# back up the pristine json5 config so crawler_weibo.py has a stable template
if [ ! -f "$WEIBO_DIR/config_default.json.bak" ] && [ -f "$WEIBO_DIR/config.json" ]; then
  cp "$WEIBO_DIR/config.json" "$WEIBO_DIR/config_default.json.bak"
  ok "snapshotted config template"
fi

# --- .env ---
step "setting up .env"
if [ ! -f .env ]; then
  cp .env.example .env
  ok ".env created from .env.example — open it and fill in LLM_API_KEY"
else
  ok ".env already exists, leaving alone"
fi

step "done"
cat <<'EOF'
next:
  1. edit .env so LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are set
     (optional: TAVILY_API_KEY for fact-grounded compose)
  2. uv run uvicorn app.main:app --reload --port 8765
  3. open http://127.0.0.1:8765/
EOF
