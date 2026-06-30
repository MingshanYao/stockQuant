"""测试 lockup 模块：限售解禁日历。"""

import pandas as pd
import pytest


class TestGetLockupExpiry:
    """get_lockup_expiry 测试。"""

    def test_returns_dict_with_history_and_upcoming(self):
        """返回 dict 包含 history/upcoming 两个 DataFrame。"""
        from stockquant.signals.lockup import get_lockup_expiry

        result = get_lockup_expiry("002475")
        assert isinstance(result, dict)
        assert "history" in result
        assert "upcoming" in result
        assert isinstance(result["history"], pd.DataFrame)
        assert isinstance(result["upcoming"], pd.DataFrame)

    def test_history_has_expected_columns(self):
        """history DataFrame 包含预期列。"""
        from stockquant.signals.lockup import get_lockup_expiry

        result = get_lockup_expiry("600519")
        expected = ["date", "type", "shares", "ratio"]
        for col in expected:
            assert col in result["history"].columns, f"缺少列: {col}"

    def test_history_date_is_datetime_when_not_empty(self):
        """history 非空时 date 列为 datetime。"""
        from stockquant.signals.lockup import get_lockup_expiry

        result = get_lockup_expiry("002475")
        df = result["history"]
        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df["date"])

    @pytest.mark.parametrize("code_input", [
        "sz002475",
        "002475.SZ",
        "SZ002475",
    ])
    def test_normalizes_code_input(self, code_input):
        """支持 sz 前缀和 .SZ 后缀。"""
        from stockquant.signals.lockup import get_lockup_expiry

        result = get_lockup_expiry(code_input)
        assert isinstance(result["history"], pd.DataFrame)
        assert isinstance(result["upcoming"], pd.DataFrame)

    def test_forward_days_parameter(self):
        """forward_days 参数控制未来查询天数。"""
        from stockquant.signals.lockup import get_lockup_expiry

        result = get_lockup_expiry("002475", forward_days=30)
        upcoming = result["upcoming"]
        assert isinstance(upcoming, pd.DataFrame)
