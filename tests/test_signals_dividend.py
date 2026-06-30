"""测试 dividend 模块：分红送转历史。"""

import pandas as pd
import pytest


class TestGetDividendHistory:
    """get_dividend_history 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 含预期列。"""
        from stockquant.signals.dividend import get_dividend_history

        df = get_dividend_history("600519", page_size=10)
        assert isinstance(df, pd.DataFrame)
        expected = [
            "date", "bonus_rmb", "transfer_ratio",
            "bonus_ratio", "plan",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_is_datetime_when_not_empty(self):
        """非空时 date 列为 datetime。"""
        from stockquant.signals.dividend import get_dividend_history

        df = get_dividend_history("600519", page_size=5)
        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_bonus_is_numeric(self):
        """bonus_rmb 为数值类型。"""
        from stockquant.signals.dividend import get_dividend_history

        df = get_dividend_history("600519", page_size=5)
        if not df.empty:
            assert df["bonus_rmb"].dtype.kind in ("i", "f")

    @pytest.mark.parametrize("code_input", [
        "sh600519",
        "600519.SH",
        "SH600519",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sh 前缀和 .SH 后缀。"""
        from stockquant.signals.dividend import get_dividend_history

        df = get_dividend_history(code_input, page_size=3)
        assert isinstance(df, pd.DataFrame)

    def test_respects_page_size_parameter(self):
        """page_size 参数控制返回行数。"""
        from stockquant.signals.dividend import get_dividend_history

        df = get_dividend_history("600519", page_size=3)
        assert len(df) <= 3
