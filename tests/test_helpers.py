"""测试 - 工具函数。"""

import datetime as dt
import pytest

from stockquant.utils.helpers import (
    ensure_date,
    normalize_stock_code,
    get_market_prefix,
    split_list,
)


class TestEnsureDate:
    def test_string_dash(self):
        assert ensure_date("2024-01-15") == dt.date(2024, 1, 15)

    def test_string_compact(self):
        assert ensure_date("20240115") == dt.date(2024, 1, 15)

    def test_date_object(self):
        d = dt.date(2024, 1, 15)
        assert ensure_date(d) is d

    def test_datetime_object(self):
        d = dt.datetime(2024, 1, 15, 10, 30)
        assert ensure_date(d) == dt.date(2024, 1, 15)

    def test_none(self):
        assert ensure_date(None) is None

    def test_invalid(self):
        with pytest.raises(ValueError):
            ensure_date("not-a-date")


class TestNormalizeStockCode:
    def test_plain(self):
        assert normalize_stock_code("600000") == "600000"

    def test_with_prefix(self):
        assert normalize_stock_code("sh600000") == "600000"
        assert normalize_stock_code("sz000001") == "000001"

    def test_with_suffix(self):
        assert normalize_stock_code("600000.SH") == "600000"

    def test_padding(self):
        assert normalize_stock_code("1") == "000001"


class TestGetMarketPrefix:
    def test_shanghai(self):
        assert get_market_prefix("600000") == "sh"

    def test_shenzhen(self):
        assert get_market_prefix("000001") == "sz"
        assert get_market_prefix("300001") == "sz"

    def test_beijing(self):
        assert get_market_prefix("430001") == "bj"


class TestSplitList:
    def test_basic(self):
        result = split_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_exact(self):
        result = split_list([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]
