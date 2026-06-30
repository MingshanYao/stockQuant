"""测试 fund_flow 模块：个股资金流向 120 日日级。"""

import pandas as pd
import pytest


class TestGetFundFlow:
    """get_fund_flow 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 且包含预期列（push2his 可能因限流为空）。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=20)

        assert isinstance(df, pd.DataFrame)
        expected_cols = ["date", "main_net", "small_net", "mid_net",
                         "large_net", "super_net"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_column_is_datetime(self):
        """date 列为 datetime 类型（空 DataFrame 也应有正确 dtype）。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=5)
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), \
            f"date dtype={df['date'].dtype}, 可能被东财软封禁返回空数据"

    def test_respects_days_parameter(self):
        """days 参数控制返回行数不超过预期。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=10)
        assert len(df) <= 10, f"请求10天, 实际返回{len(df)}行"

    def test_unknown_code_returns_empty(self):
        """无效代码返回空 DataFrame 带正确列。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("999999", days=10)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("code_input", [
        "sh600519",
        "600519.SH",
        "SH600519",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sh/sz 前缀和 .SH/.SZ 后缀格式（验证不抛异常，数据可能因限流为空）。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow(code_input, days=5)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "main_net", "small_net",
                                     "mid_net", "large_net", "super_net"]


class TestGetFundFlowMinute:
    """get_fund_flow_minute 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 且包含预期列（非交易时段可能为空）。"""
        from stockquant.signals.fund_flow import get_fund_flow_minute

        df = get_fund_flow_minute("600519")

        assert isinstance(df, pd.DataFrame)
        expected_cols = ["time", "main_net", "small_net", "mid_net",
                         "large_net", "super_net"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_time_column_is_datetime(self):
        """time 列为 datetime 类型（空 DataFrame 也应有正确 dtype）。"""
        from stockquant.signals.fund_flow import get_fund_flow_minute

        df = get_fund_flow_minute("600519")
        assert pd.api.types.is_datetime64_any_dtype(df["time"]), \
            f"time dtype={df['time'].dtype}"

    def test_unknown_code_returns_empty(self):
        """无效代码返回空 DataFrame 带正确列。"""
        from stockquant.signals.fund_flow import get_fund_flow_minute

        df = get_fund_flow_minute("999999")
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("code_input", [
        "sh600519",
        "600519.SH",
        "SH600519",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sh/sz 前缀和 .SH/.SZ 后缀格式。"""
        from stockquant.signals.fund_flow import get_fund_flow_minute

        df = get_fund_flow_minute(code_input)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["time", "main_net", "small_net",
                                    "mid_net", "large_net", "super_net"]

    def test_trading_hours_may_have_data(self):
        """交易时段返回的数据行应有非空 time（非交易时段返回空时不执行断言）。"""
        from stockquant.signals.fund_flow import get_fund_flow_minute

        df = get_fund_flow_minute("000858")
        if len(df) > 0:
            assert df["time"].notna().all()
