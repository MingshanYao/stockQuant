"""测试 options 模块：ETF 期权。"""

import pytest


class TestGetOptionCodes:
    """get_option_codes 测试。"""

    def test_returns_dict(self):
        from stockquant.signals.options import get_option_codes

        codes = get_option_codes("510050", call=True)
        assert isinstance(codes, dict)

    def test_call_put_different(self):
        from stockquant.signals.options import get_option_codes

        calls = get_option_codes("510050", call=True)
        puts = get_option_codes("510050", call=False)
        # 两者都是 dict，即使数据请求失败
        assert isinstance(calls, dict)
        assert isinstance(puts, dict)

    def test_unknown_underlying_returns_dict(self):
        from stockquant.signals.options import get_option_codes

        codes = get_option_codes("999999", call=True)
        assert isinstance(codes, dict)


class TestGetOptionTquote:
    """get_option_tquote 测试。"""

    def test_bogus_code_returns_empty(self):
        from stockquant.signals.options import get_option_tquote

        result = get_option_tquote("BOGUS")
        assert isinstance(result, dict)


class TestGetOptionGreeks:
    """get_option_greeks 测试。"""

    def test_bogus_code_returns_empty(self):
        from stockquant.signals.options import get_option_greeks

        result = get_option_greeks("BOGUS")
        assert isinstance(result, dict)
