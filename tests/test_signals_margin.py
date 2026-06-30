"""测试 margin 模块：融资融券日级明细。"""

import pandas as pd
import pytest


class TestGetMarginTrading:
    """get_margin_trading 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 且包含关键列。"""
        from stockquant.signals.margin import get_margin_trading

        df = get_margin_trading("600519", page_size=5)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "茅台应有融资融券数据"
        expected_cols = ["date", "rzye", "rzmre", "rzche",
                         "rqye", "rqmcl", "rqchl", "rzrqye"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_is_datetime(self):
        """date 列为 datetime 类型。"""
        from stockquant.signals.margin import get_margin_trading

        df = get_margin_trading("600519", page_size=3)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_numeric_columns(self):
        """融资余额等核心列为数值类型。"""
        from stockquant.signals.margin import get_margin_trading

        df = get_margin_trading("600519", page_size=3)
        assert df["rzye"].dtype.kind in ("i", "f"), \
            f"rzye 应为数值, 实际 {df['rzye'].dtype}"
