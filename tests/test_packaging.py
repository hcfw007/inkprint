"""Packaging guardrails — dependencies the code imports must be declared explicitly.

app/__init__.py does `from dotenv import load_dotenv` at import time. That must
not rely on python-dotenv arriving transitively via uvicorn[standard]: dropping
or swapping that extra would break the app on a fresh install.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _declared_dependencies() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return list(tomllib.load(f)["project"]["dependencies"])


def test_python_dotenv_is_an_explicit_dependency():
    deps = [d.lower() for d in _declared_dependencies()]
    assert any(d.startswith("python-dotenv") for d in deps), (
        "app/__init__.py imports dotenv directly; declare python-dotenv in "
        "[project].dependencies instead of relying on uvicorn[standard]"
    )
