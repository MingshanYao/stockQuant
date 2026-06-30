"""测试 iwencai 模块：NL 语义搜索。"""

import pandas as pd
import pytest


class TestIwencaiSearch:
    """iwencai_search 测试。"""

    def test_returns_dataframe_without_key(self):
        """未设置 API Key 时返回空 DataFrame 不抛异常。"""
        from stockquant.signals.iwencai import iwencai_search

        df = iwencai_search("测试查询")
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.iwencai import iwencai_search

        df = iwencai_search("测试查询")
        expected = ["uid", "title", "publish_date", "score", "summary", "source"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"


class TestIwencaiQuery:
    """iwencai_query 测试。"""

    def test_returns_dataframe_without_key(self):
        from stockquant.signals.iwencai import iwencai_query

        df = iwencai_query("测试查询")
        assert isinstance(df, pd.DataFrame)


class TestDedupArticles:
    """dedup_articles 测试。"""

    def test_returns_list(self):
        from stockquant.signals.iwencai import dedup_articles

        result = dedup_articles([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_dedup_keeps_highest_score(self):
        from stockquant.signals.iwencai import dedup_articles

        articles = [
            {"uid": "a", "title": "A1", "score": 0.5, "publish_date": "2026-01-01"},
            {"uid": "a", "title": "A2", "score": 0.9, "publish_date": "2026-01-01"},
            {"uid": "b", "title": "B1", "score": 0.7, "publish_date": "2026-01-02"},
        ]
        result = dedup_articles(articles)
        assert len(result) == 2
        uids = [r["uid"] for r in result]
        assert "a" in uids
        assert "b" in uids
        a_entry = next(r for r in result if r["uid"] == "a")
        assert a_entry["score"] == 0.9

    def test_no_uid_falls_back_to_title_date(self):
        from stockquant.signals.iwencai import dedup_articles

        articles = [
            {"title": "T", "publish_date": "2026-01-01", "score": 0.5},
            {"title": "T", "publish_date": "2026-01-01", "score": 0.8},
        ]
        result = dedup_articles(articles)
        assert len(result) == 1
        assert result[0]["score"] == 0.8

    def test_sorted_by_date_desc(self):
        from stockquant.signals.iwencai import dedup_articles

        articles = [
            {"uid": "a", "title": "A", "score": 0.5, "publish_date": "2025-01-01"},
            {"uid": "b", "title": "B", "score": 0.7, "publish_date": "2026-01-01"},
        ]
        result = dedup_articles(articles)
        assert result[0]["publish_date"] >= result[-1]["publish_date"]
