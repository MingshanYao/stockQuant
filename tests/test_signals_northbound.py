"""测试 northbound 模块：北向资金实时分钟 + 历史日级缓存。"""

import pandas as pd
import pytest


class TestGetNorthboundRealtime:
    """get_northbound_realtime 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 含 time/hgt_yi/sgt_yi 列。"""
        from stockquant.signals.northbound import get_northbound_realtime

        df = get_northbound_realtime()

        assert isinstance(df, pd.DataFrame)
        expected_cols = ["time", "hgt_yi", "sgt_yi"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_hgt_sgt_are_numeric(self):
        """hgt_yi / sgt_yi 为数值列。"""
        from stockquant.signals.northbound import get_northbound_realtime

        df = get_northbound_realtime()
        if not df.empty:
            vals = df.dropna(subset=["hgt_yi"])
            assert vals["hgt_yi"].dtype.kind in ("i", "f")


class TestGetNorthboundHistory:
    """get_northbound_history 测试。"""

    def test_returns_dataframe(self):
        """返回 DataFrame（文件不存在时可能为空）。"""
        from stockquant.signals.northbound import get_northbound_history

        df = get_northbound_history(n=5)
        assert isinstance(df, pd.DataFrame)

    def test_respects_n_parameter(self):
        """n 参数控制返回行数。"""
        from stockquant.signals.northbound import get_northbound_history

        df = get_northbound_history(n=3)
        assert len(df) <= 3
