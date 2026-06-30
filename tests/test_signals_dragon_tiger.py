"""测试 dragon_tiger 模块：龙虎榜个股 + 全市场。"""

import pandas as pd
import pytest


class TestGetDragonTigerBoard:
    """get_dragon_tiger_board 测试。"""

    def test_returns_expected_dict_structure(self):
        """返回 dict 包含 records/seats/institution 三个键。"""
        from stockquant.signals.dragon_tiger import get_dragon_tiger_board

        result = get_dragon_tiger_board("002475", look_back=30)
        assert isinstance(result, dict)
        assert "records" in result
        assert "seats" in result
        assert "institution" in result
        assert isinstance(result["records"], pd.DataFrame)
        assert isinstance(result["seats"], dict)
        assert "buy" in result["seats"] and "sell" in result["seats"]
        assert isinstance(result["institution"], dict)

    def test_records_has_expected_columns(self):
        """records DataFrame 包含预期列。"""
        from stockquant.signals.dragon_tiger import get_dragon_tiger_board

        result = get_dragon_tiger_board("600519", look_back=360)
        df = result["records"]
        expected = ["date", "reason", "net_buy_wan", "turnover_pct"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    @pytest.mark.parametrize("code_input", [
        "sz002475",
        "002475.SZ",
        "SZ002475",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sz 前缀和 .SZ 后缀。"""
        from stockquant.signals.dragon_tiger import get_dragon_tiger_board

        result = get_dragon_tiger_board(code_input, look_back=30)
        assert isinstance(result, dict)
        assert isinstance(result["records"], pd.DataFrame)


class TestGetDailyDragonTiger:
    """get_daily_dragon_tiger 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 含预期列。"""
        from stockquant.signals.dragon_tiger import get_daily_dragon_tiger

        # 查一个有数据的交易日
        df = get_daily_dragon_tiger("2026-05-16")
        assert isinstance(df, pd.DataFrame)
        expected = [
            "code", "name", "reason", "close", "change_pct",
            "net_buy_wan", "buy_wan", "sell_wan", "turnover_pct",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_min_net_buy_filter(self):
        """min_net_buy 参数过滤净买入。"""
        from stockquant.signals.dragon_tiger import get_daily_dragon_tiger

        df_all = get_daily_dragon_tiger("2026-05-16")
        df_filtered = get_daily_dragon_tiger("2026-05-16", min_net_buy=5000)

        if not df_all.empty:
            assert len(df_filtered) <= len(df_all)

    def test_empty_for_weekend(self):
        """非交易日返回空 DataFrame 不抛异常。"""
        from stockquant.signals.dragon_tiger import get_daily_dragon_tiger

        df = get_daily_dragon_tiger("2020-01-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 9
