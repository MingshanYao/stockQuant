"""测试 block_trade 模块：大宗交易记录。"""

import pandas as pd
import pytest


class TestGetBlockTrade:
    """get_block_trade 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 含预期列。"""
        from stockquant.signals.block_trade import get_block_trade

        df = get_block_trade("600519", page_size=10)
        assert isinstance(df, pd.DataFrame)
        expected = [
            "date", "price", "close", "premium_pct",
            "vol", "amount", "buyer", "seller",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_is_datetime_when_not_empty(self):
        """非空时 date 列为 datetime。"""
        from stockquant.signals.block_trade import get_block_trade

        df = get_block_trade("600519", page_size=5)
        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_premium_is_numeric(self):
        """premium_pct 为数值类型。"""
        from stockquant.signals.block_trade import get_block_trade

        df = get_block_trade("600519", page_size=5)
        if not df.empty:
            assert df["premium_pct"].dtype.kind in ("i", "f")

    @pytest.mark.parametrize("code_input", [
        "sh600519",
        "600519.SH",
        "SH600519",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sh 前缀和 .SH 后缀。"""
        from stockquant.signals.block_trade import get_block_trade

        df = get_block_trade(code_input, page_size=3)
        assert isinstance(df, pd.DataFrame)

    def test_respects_page_size_parameter(self):
        """page_size 参数控制返回行数。"""
        from stockquant.signals.block_trade import get_block_trade

        df = get_block_trade("600519", page_size=3)
        assert len(df) <= 3
