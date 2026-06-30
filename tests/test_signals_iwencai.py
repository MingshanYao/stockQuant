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
