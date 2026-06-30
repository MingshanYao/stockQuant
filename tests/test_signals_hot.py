"""测试 hot 模块：同花顺当日强势股 + 题材归因。"""

import pandas as pd
import pytest


class TestGetHotStocks:
    """get_hot_stocks 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 含关键列。"""
        from stockquant.signals.hot import get_hot_stocks

        df = get_hot_stocks("2026-05-09")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "交易日应有强势股数据"
        key_cols = ["code", "name", "reason", "date", "market"]
        for col in key_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_reason_tags_are_non_empty(self):
        """reason 列应包含非空题材标签。"""
        from stockquant.signals.hot import get_hot_stocks

        df = get_hot_stocks("2026-05-09")
        non_empty = df[df["reason"].str.strip() != ""]
        assert len(non_empty) > 0, "至少部分股票应有题材归因"
        assert "AI" in non_empty["reason"].iloc[0] or \
            "+" in non_empty["reason"].iloc[0], \
            "reason 应为 + 分隔的标签格式"

    def test_non_trading_day_returns_empty(self):
        """非交易日返回空 DataFrame 不抛异常。"""
        from stockquant.signals.hot import get_hot_stocks

        df = get_hot_stocks("2020-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_default_date_works(self):
        """默认日期（今天）不抛异常。"""
        from stockquant.signals.hot import get_hot_stocks

        df = get_hot_stocks()
        assert isinstance(df, pd.DataFrame)
