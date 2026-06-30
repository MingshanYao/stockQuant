"""测试 _sina / _cninfo / _mootdx 共享基础设施。"""

import pandas as pd
import pytest


class TestSinaOptList:
    """_sina 新浪行情解析测试。"""

    def test_returns_empty_on_bogus_param(self):
        from stockquant.signals._sina import sina_opt_list

        result = sina_opt_list("BOGUS_PARAM")
        assert isinstance(result, list)


class TestCninfoOrgid:
    """_cninfo orgId 映射测试。"""

    def test_shanghai_stock_fallback(self):
        from stockquant.signals._cninfo import _cninfo_orgid

        # 即使映射表加载失败，fallback 也不应为空
        org = _cninfo_orgid("600519")
        assert isinstance(org, str)
        assert len(org) > 0

    def test_shenzhen_stock_fallback(self):
        from stockquant.signals._cninfo import _cninfo_orgid

        org = _cninfo_orgid("000001")
        assert isinstance(org, str)
        assert len(org) > 0

    def test_ts_to_date_int(self):
        from stockquant.signals._cninfo import _cninfo_ts_to_date

        # 2026-01-01 00:00:00 UTC = 1767225600 * 1000 ms
        result = _cninfo_ts_to_date(1767225600000)
        assert "2026" in result

    def test_ts_to_date_none(self):
        from stockquant.signals._cninfo import _cninfo_ts_to_date

        assert _cninfo_ts_to_date(None) == ""
        assert _cninfo_ts_to_date("") == ""


class TestMootdxImport:
    """_mootdx 可选依赖测试（mootdx 未安装时行为）。"""

    def test_tdx_client_not_installed(self):
        import sys

        # 模拟 mootdx 未安装
        mootdx_in_modules = "mootdx" in sys.modules
        try:
            # 暂时移除 mootdx 从 sys.modules（如果已导入）
            mootdx_mod = sys.modules.pop("mootdx", None)
            mootdx_quotes = sys.modules.pop("mootdx.quotes", None)

            from stockquant.signals._mootdx import tdx_client

            with pytest.raises(ImportError, match="mootdx"):
                tdx_client()
        finally:
            # 恢复
            if mootdx_mod is not None:
                sys.modules["mootdx"] = mootdx_mod
            if mootdx_quotes is not None:
                sys.modules["mootdx.quotes"] = mootdx_quotes
