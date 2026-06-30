"""测试 holders 模块：股东户数变化（筹码集中度）。"""

import pandas as pd
import pytest


class TestGetHolderChanges:
    """get_holder_changes 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 且包含关键列。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=5)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "茅台应有股东户数数据"
        expected_cols = ["date", "holder_num", "change_num",
                         "change_ratio", "avg_shares"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_is_datetime(self):
        """date 列为 datetime 类型。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=3)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_holder_num_is_numeric(self):
        """holder_num 为数值类型。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=3)
        assert df["holder_num"].dtype.kind in ("i", "f"), \
            f"holder_num 应为数值, 实际 {df['holder_num'].dtype}"

    def test_respects_periods_parameter(self):
        """periods 参数控制返回行数。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=3)
        assert len(df) <= 3, f"请求3期, 实际返回{len(df)}行"
