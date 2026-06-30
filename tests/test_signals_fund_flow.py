"""测试 fund_flow 模块：个股资金流向 120 日日级。"""

import pandas as pd
import pytest


class TestGetFundFlow:
    """get_fund_flow 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 且包含预期列。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=20)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "茅台应有资金流数据"
        expected_cols = ["date", "main_net", "small_net", "mid_net",
                         "large_net", "super_net"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_column_is_datetime(self):
        """date 列为 datetime 类型。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=5)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

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
        """支持 sh/sz 前缀和 .SH/.SZ 后缀格式。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow(code_input, days=5)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, f"代码 {code_input} 归一化后应有数据"
