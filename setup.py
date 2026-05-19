#!/usr/bin/env python3
"""Cross-platform bootstrapper for inkprint.

Single source of truth for cold-start. Idempotent — re-running only
does the work that is actually missing. Designed to run on macOS,
Linux, and Windows without modification.

Prerequisites (NOT installed by this script):
  - uv          (https://github.com/astral-sh/uv) — brings Python with it
  - git
  - Node.js ≥ 16
  - Google Chrome (system install — both crawlers drive it via CDP)

Usage:
  python3 setup.py      # macOS / Linux
  python setup.py       # Windows
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZHIHU_REPO = "https://github.com/xx-hub/MediaCrawler_zhihu.git"
WEIBO_REPO = "https://github.com/dataabc/weibo-crawler.git"
WEIBO_PYTHON = "3.11"

USE_COLOR = sys.stdout.isatty() and platform.system() != "Windows"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def step(msg: str) -> None:
    print()
    print(_c("1;34", f"▶ {msg}"))


def ok(msg: str) -> None:
    print(f"  {_c('32', '✓')} {msg}")


def warn(msg: str) -> None:
    print(f"  {_c('33', '!')} {msg}")


def fail(msg: str) -> None:
    print(f"  {_c('31', '✗')} {msg}")
    sys.exit(1)


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    """Run a command, raising on non-zero exit. Returns stdout when capture=True."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=capture,
        )
        return result.stdout.strip() if capture else ""
    except subprocess.CalledProcessError as e:
        fail(f"command failed: {' '.join(cmd)}\n{e.stderr or e.stdout or ''}")
        return ""  # unreachable but keeps mypy happy


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_prerequisites() -> None:
    step("checking prerequisites")
    missing = [c for c in ("uv", "git", "node") if not have(c)]
    if missing:
        fail(f"missing tool(s): {', '.join(missing)}. install then re-run.")
    uv_ver = run(["uv", "--version"], capture=True).split()[1]
    node_ver = run(["node", "--version"], capture=True)
    ok(f"uv {uv_ver} · git · node {node_ver}")

    chrome_paths = [
        Path("/Applications/Google Chrome.app"),
        Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path(r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    ]
    if not any(p.exists() for p in chrome_paths) and not have("google-chrome"):
        warn("Google Chrome not detected — crawlers will fail to launch their CDP session")


def setup_main_app() -> None:
    step("installing main app deps")
    run(["uv", "sync", "--quiet"], cwd=ROOT)
    ok("uv sync done")


def ensure_clone(url: str, target: Path) -> None:
    if (target / ".git").exists():
        ok("already cloned")
        return
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(target)])
    ok(f"cloned {url.rsplit('/', 1)[-1]}")


def setup_zhihu() -> None:
    step("setting up crawlers/zhihu (MediaCrawler_zhihu)")
    zhihu = ROOT / "crawlers" / "zhihu"
    ensure_clone(ZHIHU_REPO, zhihu)
    run(["uv", "sync", "--quiet"], cwd=zhihu)
    ok("uv sync inside zhihu")
    for example in ("config/base_config.example.py", "config/zhihu_config.example.py"):
        src = zhihu / example
        dst = zhihu / example.replace(".example", "")
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            ok(f"seeded {dst.name}")


def setup_weibo() -> None:
    step("setting up crawlers/weibo (dataabc/weibo-crawler)")
    weibo = ROOT / "crawlers" / "weibo"
    ensure_clone(WEIBO_REPO, weibo)
    venv_python_unix = weibo / ".venv" / "bin" / "python"
    venv_python_win = weibo / ".venv" / "Scripts" / "python.exe"
    venv_python = venv_python_win if platform.system() == "Windows" else venv_python_unix
    if not venv_python.exists():
        run(["uv", "venv", "--python", WEIBO_PYTHON, ".venv", "--quiet"], cwd=weibo)
        ok(f"created venv (python {WEIBO_PYTHON})")
    run(
        ["uv", "pip", "install", "--python", str(venv_python), "-r", "requirements.txt", "--quiet"],
        cwd=weibo,
    )
    ok("pip install inside weibo")
    config = weibo / "config.json"
    backup = weibo / "config_default.json.bak"
    if not backup.exists() and config.exists():
        shutil.copy2(config, backup)
        ok("snapshotted config template")


def setup_env() -> None:
    step("setting up .env")
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy2(example, env)
        ok(".env created from .env.example — open it and fill in LLM_API_KEY")
    elif env.exists():
        ok(".env already exists, leaving alone")
    else:
        warn(".env.example missing — skipping")


def main() -> None:
    check_prerequisites()
    setup_main_app()
    setup_zhihu()
    setup_weibo()
    setup_env()

    step("done")
    print(
        "next:\n"
        "  1. edit .env so LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are set\n"
        "     (optional: TAVILY_API_KEY for fact-grounded compose)\n"
        "  2. uv run uvicorn app.main:app --reload --port 8765\n"
        "  3. open http://127.0.0.1:8765/"
    )


if __name__ == "__main__":
    main()
