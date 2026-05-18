"""Browser-driven cookie acquisition.

Pops a real Chrome window via Playwright (CDP into system Chrome with a
dedicated persistent profile under data/browser_profile/{platform}/), lets
the user log in, then harvests the resulting cookies as a `Cookie:` header
string suitable for handing to a downstream HTTP-based crawler.

Persistent profile means a returning user with valid cookies skips the
login UI almost instantly.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "data" / "browser_profile"
COOKIES_DIR = ROOT / "data" / "cookies"

WEIBO_LOGIN_URL = "https://weibo.com"
WEIBO_LOGIN_MARKER_COOKIES = ("SUB", "SUBP")
LOGIN_POLL_INTERVAL_SECONDS = 1.5
LOGIN_TIMEOUT_SECONDS = 180


class BrowserAuthError(RuntimeError):
    """Cookie acquisition failed (timeout, browser crash, etc.)."""


async def _wait_for_login(context, marker_names: tuple[str, ...]) -> dict[str, str]:
    deadline = asyncio.get_event_loop().time() + LOGIN_TIMEOUT_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        cookies = await context.cookies()
        bag = {c["name"]: c["value"] for c in cookies}
        if all(name in bag for name in marker_names):
            return bag
        await asyncio.sleep(LOGIN_POLL_INTERVAL_SECONDS)
    raise BrowserAuthError(f"login timed out after {LOGIN_TIMEOUT_SECONDS}s")


def _format_cookie_header(bag: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in bag.items())


def _cookie_file(platform: str) -> Path:
    return COOKIES_DIR / f"{platform}.txt"


def load_cached(platform: str) -> str | None:
    path = _cookie_file(platform)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def save_cached(platform: str, cookie: str) -> None:
    COOKIES_DIR.mkdir(parents=True, exist_ok=True)
    _cookie_file(platform).write_text(cookie, encoding="utf-8")


def clear_cached(platform: str) -> None:
    _cookie_file(platform).unlink(missing_ok=True)


async def get_weibo_cookie() -> str:
    """Open weibo.com in a managed Chrome, wait for login, return Cookie header."""
    profile_dir = PROFILES_DIR / "weibo"
    profile_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel="chrome",
            headless=False,
        )
        try:
            page = await context.new_page()
            await page.goto(WEIBO_LOGIN_URL, wait_until="domcontentloaded")
            bag = await _wait_for_login(context, WEIBO_LOGIN_MARKER_COOKIES)
            return _format_cookie_header(bag)
        finally:
            await context.close()


async def ensure_weibo_cookie() -> str:
    """Return a usable weibo cookie — from cache if present, else prompt login."""
    cached = load_cached("weibo")
    if cached:
        return cached
    fresh = await get_weibo_cookie()
    save_cached("weibo", fresh)
    return fresh
