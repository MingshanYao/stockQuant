"""测试 finance 模块：个股信息 + 新浪财报三表。"""

import pandas as pd
import pytest


class TestGetStockInfo:
    """get_stock_info 测试。"""

    def test_returns_dict(self):
        from stockquant.signals.finance import get_stock_info

        info = get_stock_info("600519")
        assert isinstance(info, dict)

    def test_contains_expected_fields(self):
        from stockquant.signals.finance import get_stock_info

        info = get_stock_info("600519")
        if info:
            key_fields = ["code", "name", "industry", "mcap", "float_mcap"]
            for f in key_fields:
                assert f in info, f"缺少字段: {f}"

    def test_normalizes_code(self):
        from stockquant.signals.finance import get_stock_info

        info = get_stock_info("SH600519")
        assert isinstance(info, dict)


class TestGetSinaFinancials:
    """get_sina_financials 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.finance import get_sina_financials

        df = get_sina_financials("600519", "lrb", num=4)
        assert isinstance(df, pd.DataFrame)

    def test_balance_sheet_works(self):
        from stockquant.signals.finance import get_sina_financials

        df = get_sina_financials("600519", "fzb", num=3)
        assert isinstance(df, pd.DataFrame)

    def test_cashflow_works(self):
        from stockquant.signals.finance import get_sina_financials

        df = get_sina_financials("600519", "llb", num=3)
        assert isinstance(df, pd.DataFrame)


class TestGetFinanceSnapshot:
    """get_finance_snapshot 测试（mootdx 可选）。"""

    def test_returns_dict(self):
        from stockquant.signals.finance import get_finance_snapshot

        result = get_finance_snapshot("600519")
        assert isinstance(result, dict)


class TestGetF10Profile:
    """get_f10_profile 测试（mootdx 可选）。"""

    def test_returns_string_or_none(self):
        from stockquant.signals.finance import get_f10_profile

        result = get_f10_profile("600519", "公司概况")
        assert result is None or isinstance(result, str)
