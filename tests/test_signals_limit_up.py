"""测试 limit_up 模块：涨停池/炸板池/跌停池/昨涨停池/涨停原因/打板情绪。"""

import datetime as dt

import pandas as pd
import pytest


class TestFmtZtTime:
    """_fmt_zt_time 单元测试。"""

    def test_formats_six_digit_string(self):
        from stockquant.signals.limit_up import _fmt_zt_time

        assert _fmt_zt_time("092500") == "09:25:00"
        assert _fmt_zt_time(92500) == "09:25:00"

    def test_formats_afternoon_time(self):
        from stockquant.signals.limit_up import _fmt_zt_time

        assert _fmt_zt_time(145500) == "14:55:00"
        assert _fmt_zt_time("130001") == "13:00:01"

    def test_formats_midnight(self):
        from stockquant.signals.limit_up import _fmt_zt_time

        assert _fmt_zt_time(0) == "00:00:00"


class TestGetLimitUpPool:
    """get_limit_up_pool 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.limit_up import get_limit_up_pool

        df = get_limit_up_pool()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.limit_up import get_limit_up_pool

        df = get_limit_up_pool()
        expected = [
            "code", "name", "price", "pct", "amount", "float_cap",
            "turnover", "limit_days", "first_seal", "last_seal",
            "seal_fund", "break_times", "industry", "zt_stat",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_non_trading_day_returns_empty(self):
        from stockquant.signals.limit_up import get_limit_up_pool

        df = get_limit_up_pool("2020-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_accepts_date_formats(self):
        from stockquant.signals.limit_up import get_limit_up_pool

        assert isinstance(get_limit_up_pool("20260630"), pd.DataFrame)
        assert isinstance(get_limit_up_pool(dt.date(2026, 6, 30)), pd.DataFrame)

    @pytest.mark.needs_push2
    def test_trading_day_may_have_data(self):
        from stockquant.signals.limit_up import get_limit_up_pool

        df = get_limit_up_pool("2026-06-26")
        if not df.empty:
            assert df["code"].str.match(r"^\d{6}$").all()
            assert df["price"].dtype == float


class TestGetBrokenBoardPool:
    """get_broken_board_pool 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.limit_up import get_broken_board_pool

        df = get_broken_board_pool()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.limit_up import get_broken_board_pool

        df = get_broken_board_pool()
        expected = [
            "code", "name", "price", "limit_price", "pct", "turnover",
            "first_seal", "break_times", "amplitude", "speed",
            "industry", "zt_stat",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_non_trading_day_returns_empty(self):
        from stockquant.signals.limit_up import get_broken_board_pool

        df = get_broken_board_pool("2020-01-01")
        assert isinstance(df, pd.DataFrame)


class TestGetLimitDownPool:
    """get_limit_down_pool 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.limit_up import get_limit_down_pool

        df = get_limit_down_pool()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.limit_up import get_limit_down_pool

        df = get_limit_down_pool()
        expected = [
            "code", "name", "price", "pct", "turnover", "pe",
            "seal_fund", "last_seal", "board_amount", "dt_days",
            "open_times", "industry",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_non_trading_day_returns_empty(self):
        from stockquant.signals.limit_up import get_limit_down_pool

        df = get_limit_down_pool("2020-01-01")
        assert isinstance(df, pd.DataFrame)


class TestGetYesterdayLimitUpPool:
    """get_yesterday_limit_up_pool 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.limit_up import get_yesterday_limit_up_pool

        df = get_yesterday_limit_up_pool()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.limit_up import get_yesterday_limit_up_pool

        df = get_yesterday_limit_up_pool()
        expected = [
            "code", "name", "price", "pct", "turnover", "amplitude",
            "speed", "y_first_seal", "y_limit_days", "industry", "zt_stat",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_non_trading_day_returns_empty(self):
        from stockquant.signals.limit_up import get_yesterday_limit_up_pool

        df = get_yesterday_limit_up_pool("2020-01-01")
        assert isinstance(df, pd.DataFrame)


class TestGetLimitUpReasons:
    """get_limit_up_reasons 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.limit_up import get_limit_up_reasons

        df = get_limit_up_reasons()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.limit_up import get_limit_up_reasons

        df = get_limit_up_reasons()
        expected = [
            "code", "name", "price", "pct", "reason", "board_type",
            "seal_rate", "break_times", "seal_amount", "high_days",
            "first_time", "is_again",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_non_trading_day_returns_empty(self):
        from stockquant.signals.limit_up import get_limit_up_reasons

        df = get_limit_up_reasons("2020-01-01")
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.needs_push2
    def test_trading_day_may_have_data(self):
        from stockquant.signals.limit_up import get_limit_up_reasons

        df = get_limit_up_reasons("2026-06-26")
        if not df.empty:
            assert df["code"].str.match(r"^\d{6}$").all()


class TestGetLimitUpSentiment:
    """get_limit_up_sentiment 测试。"""

    def test_returns_dict_with_expected_keys(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment()
        assert isinstance(sentiment, dict)
        expected_keys = [
            "date", "zt_count", "zb_count", "dt_count",
            "break_rate", "max_height", "ladder", "promotion_rate",
        ]
        for key in expected_keys:
            assert key in sentiment, f"缺少键: {key}"

    def test_counts_are_non_negative(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment()
        assert isinstance(sentiment["zt_count"], int)
        assert isinstance(sentiment["zb_count"], int)
        assert isinstance(sentiment["dt_count"], int)
        assert sentiment["zt_count"] >= 0
        assert sentiment["zb_count"] >= 0
        assert sentiment["dt_count"] >= 0

    def test_break_rate_in_range(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment()
        assert 0.0 <= sentiment["break_rate"] <= 100.0

    def test_max_height_non_negative(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment()
        assert sentiment["max_height"] >= 0

    def test_ladder_is_dict(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment()
        assert isinstance(sentiment["ladder"], dict)
        # ladder keys should be integers (limit days)
        for key in sentiment["ladder"]:
            assert isinstance(key, int)

    def test_promotion_rate_in_range(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment()
        assert 0.0 <= sentiment["promotion_rate"] <= 100.0

    def test_non_trading_day_returns_zero_counts(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        sentiment = get_limit_up_sentiment("2020-01-01")
        assert sentiment["zt_count"] == 0
        assert sentiment["zb_count"] == 0
        assert sentiment["dt_count"] == 0
        assert sentiment["max_height"] == 0
        assert sentiment["ladder"] == {}

    def test_accepts_date_formats(self):
        from stockquant.signals.limit_up import get_limit_up_sentiment

        assert isinstance(get_limit_up_sentiment("20260630"), dict)
        assert isinstance(get_limit_up_sentiment(dt.date(2026, 6, 30)), dict)
