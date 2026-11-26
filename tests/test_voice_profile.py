"""Pure-logic tests for voice_profile: selection, author handling, formatting."""

import pytest
from conftest import make_sample

from app import voice_profile


class TestPickRepresentative:
    def test_filters_short_and_non_str_text(self):
        items = [
            make_sample(1, text="短"),
            {**make_sample(2), "content_text": None},
            {**make_sample(3), "content_text": 12345},
            make_sample(4, text="这一条足够长，" * 10),
        ]
        picked = voice_profile._pick_representative(items)
        assert [p["content_id"] for p in picked] == ["zhihu-4"]

    def test_round_robin_interleaves_platforms(self):
        zhihu = [
            make_sample(i, platform="zhihu", text=f"知乎长文样本{'x' * (100 - i)}")
            for i in (1, 2, 3)
        ]
        weibo = [
            make_sample(i, platform="weibo", text=f"微博短样本{'y' * (80 - i)}") for i in (1, 2)
        ]
        picked = voice_profile._pick_representative([*weibo, *zhihu])
        order = [(p["platform"], p["content_id"]) for p in picked]
        # 每轮每个平台各出一条：zhihu 先（dict 插入顺序由输入决定，weibo 在前）
        assert order == [
            ("weibo", "weibo-1"),
            ("zhihu", "zhihu-1"),
            ("weibo", "weibo-2"),
            ("zhihu", "zhihu-2"),
            ("zhihu", "zhihu-3"),
        ]

    def test_global_limit_keeps_longest_first(self):
        items = [make_sample(i, text="x" * (100 + i)) for i in range(40)]
        picked = voice_profile._pick_representative(items)
        assert len(picked) == voice_profile.SAMPLE_PICK_LIMIT
        # 单平台 bucket 内按长度降序
        lengths = [len(p["content_text"]) for p in picked]
        assert lengths == sorted(lengths, reverse=True)

    def test_missing_platform_goes_to_unknown_bucket(self):
        item = {**make_sample(1), "platform": None}
        picked = voice_profile._pick_representative([item])
        assert picked == [item]

    def test_empty_input_returns_empty(self):
        assert voice_profile._pick_representative([]) == []

    def test_param_does_not_shadow_samples_module(self):
        """The parameter must not be named `samples`: voice_profile imports the
        app.samples module under that name, and shadowing it makes any future
        samples.load_merged() reference inside the function silently bind to the
        argument list instead."""
        import inspect

        params = list(inspect.signature(voice_profile._pick_representative).parameters)
        assert "samples" not in params
        assert params == ["items"]


class TestAuthorLine:
    def test_extract_author_ascii_colon(self):
        assert voice_profile.extract_author("VOICE PROFILE\nAuthor: 张三\nRhythm\n") == "张三"

    def test_extract_author_fullwidth_colon(self):
        assert voice_profile.extract_author("Author： 李四 \n") == "李四"

    def test_extract_author_missing(self):
        assert voice_profile.extract_author("没有作者行") == ""

    def test_strip_author_removes_line_and_newline(self):
        md = "Author: 张三\nRhythm\n- 快\n"
        assert voice_profile.strip_author(md) == "Rhythm\n- 快\n"

    def test_strip_author_only_first_match(self):
        md = "Author: 甲\n正文\nAuthor: 乙\n"
        out = voice_profile.strip_author(md)
        assert "Author: 乙" in out
        assert "Author: 甲" not in out


class TestFormatSample:
    def test_zhihu_with_question(self):
        item = make_sample(1, platform="zhihu", text="回答正文", question_title="问题是什么？")
        out = voice_profile._format_sample(1, item)
        assert out == "[样本 1 · 知乎] 问题：问题是什么？\n回答：回答正文"

    def test_zhihu_title_fallback(self):
        item = {**make_sample(1), "title": "旧标题"}
        out = voice_profile._format_sample(1, item)
        assert "问题：旧标题" in out

    def test_weibo_plain(self):
        item = make_sample(2, platform="weibo", text="一句话", title="不进标题")
        assert voice_profile._format_sample(2, item) == "[样本 2 · 微博] 一句话"

    def test_unknown_platform_uses_raw_name(self):
        item = make_sample(3, platform="douban", text="内容")
        assert voice_profile._format_sample(3, item).startswith("[样本 3 · douban] ")


class TestPlatformBreakdown:
    def test_counts_and_labels(self):
        used = [
            make_sample(1, platform="zhihu"),
            make_sample(2, platform="zhihu"),
            make_sample(3, platform="weibo"),
        ]
        out = voice_profile._platform_breakdown(used, total_count=9)
        # 按平台名字母序：weibo < zhihu
        assert out == "已选取最具代表性的 3 条（微博 1 条 + 知乎 2 条），来自共 9 条样本"

    def test_unknown_platform_label(self):
        out = voice_profile._platform_breakdown([make_sample(1, platform="x")], total_count=1)
        assert "x 1 条" in out


class TestBuildPrompt:
    def test_prompt_contains_schema_and_instructions(self):
        picked = [
            make_sample(1, platform="zhihu", text="长文" * 30, question_title="问题甲"),
            make_sample(2, platform="weibo", text="短评" * 20),
        ]
        messages = voice_profile._build_prompt(picked, total_count=5)
        assert [m["role"] for m in messages] == ["system", "user"]
        system, user = messages[0]["content"], messages[1]["content"]
        assert "VOICE PROFILE" in system and "Do NOT include any author bio" in system
        for key in ("Goal:", "Rhythm", "Banned Moves", "Channel Notes", "Confidence:"):
            assert key in user
        assert "问题：问题甲" in user and "短评" * 20 in user
        assert "已选取最具代表性的 2 条" in user
        assert "不要输出 Author" in user


class TestGenerate:
    def _dump_eligible(self, dump_factory):
        dump_factory(
            "zhihu",
            1,
            [make_sample(i, platform="zhihu", text=f"知乎正文{'。' * 40}{i}") for i in (1, 2)],
            name="a.json",
        )
        dump_factory(
            "weibo",
            1,
            [make_sample(9, platform="weibo", text=f"微博正文{'。' * 40}9")],
            name="b.json",
        )

    def test_generate_writes_profile_and_counts(self, dump_factory, profiles_dir, monkeypatch):
        self._dump_eligible(dump_factory)
        captured = {}

        def fake_chat(messages, temperature=0.3):
            captured["messages"] = messages
            return "VOICE PROFILE\n=============\nGoal: 分享"

        monkeypatch.setattr(voice_profile.llm, "chat", fake_chat)
        result = voice_profile.generate(1)
        assert result.sample_count_total == 3
        assert result.sample_count_used == 3  # 全部达标（2 知乎 + 1 微博，轮转全收）
        assert result.profile_path.parent == profiles_dir / "1"
        assert result.profile_path.read_text(encoding="utf-8").startswith("VOICE PROFILE")
        assert "知乎正文" in captured["messages"][1]["content"]

    def test_generate_no_samples_raises(self, samples_dir, profiles_dir):
        with pytest.raises(voice_profile.ProfileError, match="no samples"):
            voice_profile.generate(1)

    def test_generate_has_no_unreachable_empty_guard(self):
        """_load_samples() already raises ProfileError on an empty sample union,
        so any second empty-check inside generate() is dead code with a
        misleading message ("sample file is empty"). Keep the guard in exactly
        one place: _load_samples."""
        import inspect

        src = inspect.getsource(voice_profile.generate)
        assert "sample file is empty" not in src

    def test_generate_no_eligible_samples_raises(self, dump_factory, profiles_dir):
        dump_factory("zhihu", 1, [make_sample(1, text="太短")])
        with pytest.raises(voice_profile.ProfileError, match="minimum length"):
            voice_profile.generate(1)

    def test_generate_wraps_llm_error(self, dump_factory, profiles_dir, monkeypatch):
        self._dump_eligible(dump_factory)

        def boom(messages, temperature=0.3):
            raise voice_profile.llm.LLMError("api down")

        monkeypatch.setattr(voice_profile.llm, "chat", boom)
        with pytest.raises(voice_profile.ProfileError, match="api down"):
            voice_profile.generate(1)

    def test_generate_twice_in_same_second_keeps_both_versions(
        self, dump_factory, profiles_dir, monkeypatch
    ):
        """Two generates inside the same second must not overwrite each other."""
        from datetime import datetime

        self._dump_eligible(dump_factory)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 5, 18, 12, 34, 56, 123000, tzinfo=tz)

        monkeypatch.setattr(voice_profile, "datetime", FrozenDatetime)
        replies = iter(["VOICE PROFILE 甲", "VOICE PROFILE 乙"])
        monkeypatch.setattr(
            voice_profile.llm, "chat", lambda messages, temperature=0.3: next(replies)
        )

        first = voice_profile.generate(1)
        second = voice_profile.generate(1)

        assert first.profile_path != second.profile_path
        assert first.profile_path.read_text(encoding="utf-8") == "VOICE PROFILE 甲"
        assert second.profile_path.read_text(encoding="utf-8") == "VOICE PROFILE 乙"
        versions = voice_profile.list_versions(1)
        assert len(versions) == 2
        # 最新的（第二次生成）排最前
        assert versions[0].path == second.profile_path
        assert voice_profile.read_existing(1) == "VOICE PROFILE 乙"

    def test_generate_filename_carries_milliseconds(self, dump_factory, profiles_dir, monkeypatch):
        from datetime import datetime

        self._dump_eligible(dump_factory)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 5, 18, 12, 34, 56, 123000, tzinfo=tz)

        monkeypatch.setattr(voice_profile, "datetime", FrozenDatetime)
        monkeypatch.setattr(
            voice_profile.llm, "chat", lambda messages, temperature=0.3: "VOICE PROFILE"
        )
        result = voice_profile.generate(1)
        assert result.profile_path.stem == "20260518T123456.123Z"
        # 新 stem 仍能被 format_ts 解析
        assert voice_profile.format_ts(result.profile_path.stem) == "2026-05-18 12:34:56 UTC"


class TestFormatTs:
    def test_valid_stem(self):
        assert voice_profile.format_ts("20260518T123456Z") == "2026-05-18 12:34:56 UTC"

    def test_millisecond_stem(self):
        assert voice_profile.format_ts("20260518T123456.123Z") == "2026-05-18 12:34:56 UTC"

    def test_collision_suffix_stripped(self):
        assert voice_profile.format_ts("20260518T123456.123Z_001") == "2026-05-18 12:34:56 UTC"

    def test_invalid_stem_passthrough(self):
        assert voice_profile.format_ts("not-a-ts") == "not-a-ts"


class TestVersions:
    def _write(self, persona_dir, name, text="内容"):
        persona_dir.mkdir(parents=True, exist_ok=True)
        path = persona_dir / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_list_versions_sorted_newest_first(self, profiles_dir):
        self._write(profiles_dir / "7", "20260101T000000Z", "旧")
        self._write(profiles_dir / "7", "20260301T000000Z", "新")
        versions = voice_profile.list_versions(7)
        assert [v.name for v in versions] == ["20260301T000000Z", "20260101T000000Z"]
        assert versions[0].created_at == "2026-03-01 00:00:00 UTC"

    def test_list_versions_empty_dir(self, profiles_dir):
        assert voice_profile.list_versions(42) == []

    def test_migrate_legacy_flat_moves_file(self, profiles_dir, monkeypatch):
        import os

        legacy = profiles_dir / "7.md"
        legacy.write_text("旧版 profile", encoding="utf-8")
        os.utime(legacy, (1_700_000_000, 1_700_000_000))
        versions = voice_profile.list_versions(7)
        assert not legacy.exists()
        assert len(versions) == 1
        assert versions[0].path.read_text(encoding="utf-8") == "旧版 profile"
        assert versions[0].name == "20231114T221320Z"
        # 幂等：再跑一次无副作用
        assert len(voice_profile.list_versions(7)) == 1

    def test_migrate_legacy_flat_target_exists(self, profiles_dir):
        import os

        self._write(profiles_dir / "7", "20231114T221320Z", "已有")
        legacy = profiles_dir / "7.md"
        legacy.write_text("重复", encoding="utf-8")
        os.utime(legacy, (1_700_000_000, 1_700_000_000))
        versions = voice_profile.list_versions(7)
        assert not legacy.exists()
        assert [v.name for v in versions] == ["20231114T221320Z"]
        assert versions[0].path.read_text(encoding="utf-8") == "已有"

    def test_read_existing_prefers_newest(self, profiles_dir):
        self._write(profiles_dir / "7", "20260101T000000Z", "旧")
        self._write(profiles_dir / "7", "20260201T000000Z", "新")
        assert voice_profile.read_existing(7) == "新"

    def test_read_existing_none_when_missing(self, profiles_dir):
        assert voice_profile.read_existing(7) is None

    def test_read_version(self, profiles_dir):
        self._write(profiles_dir / "7", "20260101T000000Z", "内容甲")
        assert voice_profile.read_version(7, "20260101T000000Z") == "内容甲"
        assert voice_profile.read_version(7, "nope") is None
