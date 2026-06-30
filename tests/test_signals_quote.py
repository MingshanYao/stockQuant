"""测试 quote 模块：腾讯行情 + 百度K线MA。"""

import pandas as pd
import pytest


class TestGetTencentQuotes:
    """get_tencent_quotes 测试。"""

    def test_returns_dict(self):
        from stockquant.signals.quote import get_tencent_quotes

        quotes = get_tencent_quotes(["600519", "000001"])
        assert isinstance(quotes, dict)

    def test_contains_expected_fields(self):
        from stockquant.signals.quote import get_tencent_quotes

        quotes = get_tencent_quotes(["600519"])
        if "600519" in quotes:
            q = quotes["600519"]
            key_fields = ["name", "price", "pe_ttm", "pb", "mcap_yi"]
            for f in key_fields:
                assert f in q, f"缺少字段: {f}"

    def test_index_codes_work(self):
        from stockquant.signals.quote import get_tencent_quotes

        quotes = get_tencent_quotes(["000001", "399006"])
        assert isinstance(quotes, dict)

    def test_etf_codes_work(self):
        from stockquant.signals.quote import get_tencent_quotes

        quotes = get_tencent_quotes(["510050"])
        assert isinstance(quotes, dict)


class TestGetBaiduKlineMa:
    """get_baidu_kline_ma 测试。"""

    def test_returns_dict_with_keys(self):
        from stockquant.signals.quote import get_baidu_kline_ma

        data = get_baidu_kline_ma("600519")
        assert isinstance(data, dict)
        assert "keys" in data
        assert "rows" in data
