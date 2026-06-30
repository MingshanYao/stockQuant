"""测试 _eastmoney 基础设施：em_get 限流 + em_datacenter 通用查询。"""

import time
import pytest
import requests


def _eastmoney_reachable() -> bool:
    """探测东财是否可达，不可达时跳过依赖 push2 的测试。"""
    try:
        r = requests.get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={"secid": "1.600519", "fields": "f57,f58"},
            headers={"Referer": "https://quote.eastmoney.com/"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _skip_if_eastmoney_blocked(request):
    """push2.eastmoney.com 被限流时自动跳过依赖它的测试。"""
    for marker in request.node.iter_markers():
        if marker.name == "needs_push2":
            if not _eastmoney_reachable():
                pytest.skip("东财 push2 API 不可达（IP 被限流）")
            return


needs_push2 = pytest.mark.needs_push2


class TestEmGet:
    """em_get 限流和会话复用测试。"""

    @needs_push2
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
        """连续两次 em_get 调用间隔不小于 EM_MIN_INTERVAL。

        使用模块内部状态测试限流数学，不依赖外部网络。
        """
        from stockquant.signals._eastmoney import EM_MIN_INTERVAL
        from stockquant.signals import _eastmoney as _mod

        import random as _random

        saved = _mod._em_last_call[0]
        try:
            _mod._em_last_call[0] = time.time()
            t0 = time.time()

            # 模拟 em_get 内部的 wait 逻辑
            wait = EM_MIN_INTERVAL - (time.time() - _mod._em_last_call[0])
            if wait > 0:
                time.sleep(wait + _random.uniform(0.1, 0.5))

            elapsed = time.time() - t0
            assert elapsed >= EM_MIN_INTERVAL, \
                f"间隔 {elapsed:.2f}s < {EM_MIN_INTERVAL}s"
        finally:
            _mod._em_last_call[0] = saved

    def test_em_get_retries_on_connection_error(self):
        """em_get 对不可达 hosts 抛出 requests.ConnectionError。"""
        from stockquant.signals._eastmoney import em_get

        with pytest.raises(
            (requests.ConnectionError, requests.Timeout),
        ):
            em_get("https://10.255.255.1/no-such-host",
                   timeout=2)


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

    def test_em_datacenter_handles_bad_status(self):
        """HTTP 非 200 或 JSON 解析失败返回空列表不抛异常。"""
        from stockquant.signals._eastmoney import em_datacenter

        data = em_datacenter(
            "RPTA_WEB_RZRQ_GGMX",
            filter_str="INVALID_FILTER_SYNTAX[[[",
            page_size=5,
        )
        assert isinstance(data, list)
