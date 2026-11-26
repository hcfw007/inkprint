"""samples.py — 跨平台 dump 合并、去重、归一化解析逻辑。"""

from conftest import make_sample

from app import samples


class TestLoadMerged:
    def test_empty_when_nothing_on_disk(self, samples_dir):
        assert samples.load_merged(1) == []

    def test_empty_when_samples_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(samples, "SAMPLES_DIR", tmp_path / "not-there")
        assert samples.load_merged(1) == []

    def test_unions_across_platforms(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [make_sample(1, platform="zhihu")])
        dump_factory("weibo", 1, [make_sample(2, platform="weibo")])
        merged = samples.load_merged(1)
        assert {(m["platform"], m["content_id"]) for m in merged} == {
            ("zhihu", "zhihu-1"),
            ("weibo", "weibo-2"),
        }

    def test_dedupes_by_platform_and_content_id_keeps_newest(self, samples_dir, dump_factory):
        old = make_sample(1, text="旧版本", updated_time=100)
        new = make_sample(1, text="新版本", updated_time=200)
        dump_factory("zhihu", 1, [old], name="a.json")
        dump_factory("zhihu", 1, [new], name="b.json")
        merged = samples.load_merged(1)
        assert len(merged) == 1
        assert merged[0]["content_text"] == "新版本"

    def test_same_content_id_different_platform_kept(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [{**make_sample(1), "content_id": "same"}])
        dump_factory("weibo", 1, [{**make_sample(1, platform="weibo"), "content_id": "same"}])
        assert len(samples.load_merged(1)) == 2

    def test_platform_filter(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [make_sample(1, platform="zhihu")])
        dump_factory("weibo", 1, [make_sample(2, platform="weibo")])
        merged = samples.load_merged(1, platform="weibo")
        assert [m["platform"] for m in merged] == ["weibo"]

    def test_only_persona_own_files(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [make_sample(1)])
        dump_factory("zhihu", 2, [make_sample(2)])
        assert len(samples.load_merged(1)) == 1


class TestNormalize:
    def test_rejects_missing_or_bad_content_id(self, samples_dir, dump_factory):
        items = [
            {"content_text": "没有 id"},
            {"content_id": 123, "content_text": "id 非字符串"},
            {"content_id": "ok", "content_text": "正常"},
        ]
        dump_factory("zhihu", 1, items)
        merged = samples.load_merged(1)
        assert [m["content_id"] for m in merged] == ["ok"]

    def test_rejects_non_str_content_text(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [{"content_id": "x", "content_text": None}])
        assert samples.load_merged(1) == []

    def test_backfills_platform_from_dir(self, samples_dir, dump_factory):
        dump_factory("weibo", 1, [{"content_id": "x", "content_text": "无平台字段"}])
        (item,) = samples.load_merged(1)
        assert item["platform"] == "weibo"

    def test_explicit_platform_not_overwritten(self, samples_dir, dump_factory):
        dump_factory("weibo", 1, [{"content_id": "x", "content_text": "t", "platform": "zhihu"}])
        (item,) = samples.load_merged(1)
        assert item["platform"] == "zhihu"


class TestUpdatedTs:
    def test_prefers_updated_time(self):
        assert samples.updated_ts({"updated_time": 5, "created_time": 9}) == 5

    def test_falls_back_to_created_time(self):
        assert samples.updated_ts({"created_time": 9}) == 9

    def test_falls_back_to_zero(self):
        assert samples.updated_ts({}) == 0


class TestListFor:
    def test_sorted_by_votes_then_length(self, samples_dir, dump_factory):
        items = [
            make_sample(1, text="x" * 50, voteup_count=1),
            make_sample(2, text="y" * 90, voteup_count=5),
            make_sample(3, text="z" * 70, voteup_count=1),
        ]
        dump_factory("zhihu", 1, items)
        summaries = samples.list_for(1)
        assert [(s.voteup_count, s.char_count) for s in summaries] == [(5, 90), (1, 70), (1, 50)]

    def test_snippet_truncates_and_flattens(self, samples_dir, dump_factory):
        text = "行一\n" + "长" * 200
        dump_factory("zhihu", 1, [make_sample(1, text=text)])
        (summary,) = samples.list_for(1)
        assert summary.snippet.endswith("…")
        assert "\n" not in summary.snippet
        assert len(summary.snippet) == samples.SNIPPET_LEN + 1
        assert summary.char_count == len(text)

    def test_short_snippet_no_ellipsis(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [make_sample(1, text="短文本")])
        (summary,) = samples.list_for(1)
        assert summary.snippet == "短文本"

    def test_summary_fields(self, samples_dir, dump_factory):
        item = make_sample(
            1, content_type="answer", question_title="问题？", voteup_count=7, title="忽略"
        )
        dump_factory("zhihu", 1, [item])
        (summary,) = samples.list_for(1)
        assert summary.idx == 0
        assert summary.platform == "zhihu"
        assert summary.content_type == "answer"
        assert summary.question_title == "问题？"
        assert summary.voteup_count == 7


class TestGetFor:
    def test_get_existing(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [make_sample(1), make_sample(2)])
        item = samples.get_for(1, 0)
        assert item["content_id"] in {"zhihu-1", "zhihu-2"}

    def test_out_of_range_returns_none(self, samples_dir, dump_factory):
        dump_factory("zhihu", 1, [make_sample(1)])
        assert samples.get_for(1, 5) is None
        assert samples.get_for(1, -1) is None
