"""测试 _eastmoney 基础设施：em_get 限流 + em_datacenter 通用查询。"""

import time
import pytest


class TestEmGet:
    """em_get 限流和会话复用测试。"""

    def test_em_get_returns_200_for_public_eastmoney(self):
        """em_get 能正常请求东财公开页面（验证 session / UA 配置正确）。"""
        from stockquant.signals._eastmoney import em_get

        r = em_get("https://push2.eastmoney.com/api/qt/stock/get",
                   params={"secid": "1.600519", "fields": "f57,f58"},
                   headers={"Referer": "https://quote.eastmoney.com/"},
                   timeout=15)
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d = r.json()
        assert d.get("data", {}).get("f58") == "贵州茅台"

    def test_em_get_enforces_min_interval(self):
        """连续两次 em_get 调用间隔不小于 EM_MIN_INTERVAL。"""
        from stockquant.signals._eastmoney import em_get, EM_MIN_INTERVAL

        t0 = time.time()
        em_get("https://push2.eastmoney.com/api/qt/stock/get",
               params={"secid": "1.600519", "fields": "f57,f58"},
               headers={"Referer": "https://quote.eastmoney.com/"},
               timeout=15)
        t1 = time.time()
        em_get("https://push2.eastmoney.com/api/qt/stock/get",
               params={"secid": "0.000001", "fields": "f57,f58"},
               headers={"Referer": "https://quote.eastmoney.com/"},
               timeout=15)
        elapsed = t1 - t0
        assert elapsed >= EM_MIN_INTERVAL, \
            f"间隔 {elapsed:.2f}s < {EM_MIN_INTERVAL}s"


class TestEmDatacenter:
    """em_datacenter 通用查询测试。"""

    def test_em_datacenter_margin_returns_data(self):
        """用融资融券 RPT 验证 em_datacenter 模板查询可用。"""
        from stockquant.signals._eastmoney import em_datacenter

        data = em_datacenter(
            "RPTA_WEB_RZRQ_GGMX",
            filter_str='(SCODE="600519")',
            page_size=5,
            sort_columns="DATE",
            sort_types="-1",
        )
        assert isinstance(data, list)
        assert len(data) > 0, "茅台应有融资融券数据"
        assert "RZYE" in data[0], f"字段缺失, got: {list(data[0].keys())[:10]}"

    def test_em_datacenter_empty_for_bogus_rpt(self):
        """不存在的 RPT 名称返回空列表不抛异常。"""
        from stockquant.signals._eastmoney import em_datacenter

        data = em_datacenter("RPT_DOES_NOT_EXIST", filter_str="", page_size=5)
        assert data == []
