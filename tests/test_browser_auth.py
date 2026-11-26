"""browser_auth 编排层测试。

Playwright 边界全部 mock：不 launch 真实浏览器、不访问网络，
只验证编排逻辑——参数组装、cookie 解析、登录轮询（重试）、超时映射、缓存读写。
"""

import asyncio
from pathlib import Path

import pytest

from app import browser_auth as ba


class FakePage:
    def __init__(self, goto_error: BaseException | None = None):
        self.goto_error = goto_error
        self.goto_calls: list[tuple[str, str | None]] = []

    async def goto(self, url, wait_until=None):
        self.goto_calls.append((url, wait_until))
        if self.goto_error is not None:
            raise self.goto_error


class FakeContext:
    """按批次吐 cookie 的假 context；批次耗尽后重复最后一批。"""

    def __init__(self, cookie_batches: list[list[dict]], goto_error=None):
        self._batches = list(cookie_batches) or [[]]
        self.page = FakePage(goto_error)
        self.closed = False
        self.cookies_calls = 0

    async def new_page(self):
        return self.page

    async def cookies(self):
        self.cookies_calls += 1
        if len(self._batches) > 1:
            return self._batches.pop(0)
        return self._batches[0]

    async def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, context: FakeContext):
        self._context = context
        self.launch_kwargs: dict | None = None

    async def launch_persistent_context(self, **kwargs):
        self.launch_kwargs = kwargs
        return self._context


class FakePW:
    def __init__(self, context: FakeContext):
        self.chromium = FakeChromium(context)


class FakeAsyncPlaywright:
    """async_playwright() 的替身：async context manager。"""

    def __init__(self, pw: FakePW):
        self._pw = pw

    async def __aenter__(self):
        return self._pw

    async def __aexit__(self, *exc):
        return False


def install_playwright(monkeypatch, context: FakeContext) -> FakePW:
    pw = FakePW(context)
    monkeypatch.setattr(ba, "async_playwright", lambda: FakeAsyncPlaywright(pw))
    return pw


def cookies(*pairs: tuple[str, str]) -> list[dict]:
    return [{"name": name, "value": value} for name, value in pairs]


@pytest.fixture()
def browser_env(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    cookies_dir = tmp_path / "data" / "cookies"
    profiles_dir = tmp_path / "data" / "browser_profile"
    monkeypatch.setattr(ba, "COOKIES_DIR", cookies_dir)
    monkeypatch.setattr(ba, "PROFILES_DIR", profiles_dir)
    return {"cookies_dir": cookies_dir, "profiles_dir": profiles_dir}


class TestCookieCache:
    def test_save_and_load_roundtrip(self, browser_env):
        ba.save_cached("weibo", "SCF=x; SSOLoginState=y")
        assert (browser_env["cookies_dir"] / "weibo.txt").exists()
        assert ba.load_cached("weibo") == "SCF=x; SSOLoginState=y"

    def test_load_missing_returns_none(self, browser_env):
        assert ba.load_cached("weibo") is None

    def test_load_blank_returns_none(self, browser_env):
        ba.save_cached("weibo", "  \n")
        assert ba.load_cached("weibo") is None

    def test_clear_cached(self, browser_env):
        ba.save_cached("weibo", "a=1")
        ba.clear_cached("weibo")
        assert ba.load_cached("weibo") is None

    def test_clear_cached_missing_is_noop(self, browser_env):
        ba.clear_cached("weibo")  # 不抛异常


class TestFormatCookieHeader:
    def test_joins_in_order(self):
        assert ba._format_cookie_header({"a": "1", "b": "2"}) == "a=1; b=2"


class TestWaitForLogin:
    def test_returns_bag_when_markers_present(self, monkeypatch):
        ctx = FakeContext([cookies(("SCF", "x"), ("SSOLoginState", "y"))])
        bag = asyncio.run(ba._wait_for_login(ctx, ba.WEIBO_LOGIN_MARKER_COOKIES))
        assert bag == {"SCF": "x", "SSOLoginState": "y"}

    def test_polls_until_markers_appear(self, monkeypatch):
        monkeypatch.setattr(ba, "LOGIN_POLL_INTERVAL_SECONDS", 0)
        ctx = FakeContext(
            [
                cookies(("SUB", "anon")),
                cookies(("SCF", "x"), ("SSOLoginState", "y")),
            ]
        )
        bag = asyncio.run(ba._wait_for_login(ctx, ba.WEIBO_LOGIN_MARKER_COOKIES))
        assert bag["SCF"] == "x"
        assert ctx.cookies_calls == 2  # 第一轮缺失 → 重试

    def test_timeout_maps_to_browser_auth_error(self, monkeypatch):
        monkeypatch.setattr(ba, "LOGIN_TIMEOUT_SECONDS", 0)
        ctx = FakeContext([cookies(("SUB", "anon"))])
        with pytest.raises(ba.BrowserAuthError, match="login timed out after 0s"):
            asyncio.run(ba._wait_for_login(ctx, ba.WEIBO_LOGIN_MARKER_COOKIES))


class TestGetWeiboCookie:
    def test_launches_chrome_and_returns_cookie_header(self, browser_env, monkeypatch):
        ctx = FakeContext([cookies(("SCF", "x"), ("SSOLoginState", "y"))])
        pw = install_playwright(monkeypatch, ctx)
        result = asyncio.run(ba.get_weibo_cookie())
        assert result == "SCF=x; SSOLoginState=y"
        assert pw.chromium.launch_kwargs == {
            "user_data_dir": str(browser_env["profiles_dir"] / "weibo"),
            "channel": "chrome",
            "headless": False,
        }
        assert (browser_env["profiles_dir"] / "weibo").is_dir()
        assert ctx.page.goto_calls == [(ba.WEIBO_LOGIN_URL, "domcontentloaded")]
        assert ctx.closed  # finally 里必须 close

    def test_login_timeout_closes_context_and_raises(self, browser_env, monkeypatch):
        monkeypatch.setattr(ba, "LOGIN_TIMEOUT_SECONDS", 0)
        ctx = FakeContext([cookies(("SUB", "anon"))])
        install_playwright(monkeypatch, ctx)
        with pytest.raises(ba.BrowserAuthError):
            asyncio.run(ba.get_weibo_cookie())
        assert ctx.closed

    def test_goto_error_propagates_and_closes_context(self, browser_env, monkeypatch):
        ctx = FakeContext([cookies(("SCF", "x"))], goto_error=RuntimeError("chrome crash"))
        install_playwright(monkeypatch, ctx)
        with pytest.raises(RuntimeError, match="chrome crash"):
            asyncio.run(ba.get_weibo_cookie())
        assert ctx.closed


class TestEnsureWeiboCookie:
    def test_cache_hit_skips_browser(self, browser_env, monkeypatch):
        ba.save_cached("weibo", "cached=1")

        async def _fail():
            raise AssertionError("不应触发浏览器登录")

        monkeypatch.setattr(ba, "get_weibo_cookie", _fail)
        assert asyncio.run(ba.ensure_weibo_cookie()) == "cached=1"

    def test_cache_miss_logs_in_and_persists(self, browser_env, monkeypatch):

        async def _login():
            return "fresh=1"

        monkeypatch.setattr(ba, "get_weibo_cookie", _login)
        assert asyncio.run(ba.ensure_weibo_cookie()) == "fresh=1"
        assert ba.load_cached("weibo") == "fresh=1"
