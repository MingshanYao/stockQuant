"""测试 valuation 模块：估值分析辅助函数。"""

import math

import pytest


class TestForwardPe:
    """forward_pe 测试。"""

    def test_basic(self):
        from stockquant.signals.valuation import forward_pe

        assert forward_pe(100, 5) == 20.0

    def test_zero_eps(self):
        from stockquant.signals.valuation import forward_pe

        assert forward_pe(100, 0) == float("inf")

    def test_negative_eps(self):
        from stockquant.signals.valuation import forward_pe

        assert forward_pe(100, -1) == float("inf")


class TestPeDigestion:
    """pe_digestion 测试。"""

    def test_basic(self):
        from stockquant.signals.valuation import pe_digestion

        # PE 60 → 30, CAGR 30% → log(2)/log(1.3) ≈ 2.64 years
        result = pe_digestion(60, 0.30)
        assert 2.5 < result < 2.8

    def test_already_below_target(self):
        from stockquant.signals.valuation import pe_digestion

        assert pe_digestion(20, 0.30) == 0.0

    def test_no_growth(self):
        from stockquant.signals.valuation import pe_digestion

        assert pe_digestion(60, 0) == float("inf")


class TestCalcPeg:
    """calc_peg 测试。"""

    def test_basic(self):
        from stockquant.signals.valuation import calc_peg

        assert calc_peg(30, 0.30) == 1.0

    def test_expensive(self):
        from stockquant.signals.valuation import calc_peg

        assert calc_peg(30, 0.15) == 2.0

    def test_no_growth(self):
        from stockquant.signals.valuation import calc_peg

        assert calc_peg(30, 0) == float("inf")


class TestFullValuation:
    """full_valuation 测试。"""

    def test_returns_dict_with_expected_keys(self):
        """返回 dict 含估值指标键。"""
        from stockquant.signals.valuation import full_valuation

        result = full_valuation("600519")

        assert isinstance(result, dict)
        expected_keys = [
            "name", "price", "mcap_yi", "pe_ttm", "pb",
            "eps_cur", "eps_next", "pe_fwd", "cagr_pct",
            "peg", "digest_years", "analyst_count",
        ]
        for key in expected_keys:
            assert key in result, f"缺少键: {key}"

    def test_maotai_has_valuation(self):
        """茅台应有行情数据，EPS 覆盖按同花顺实际返回。"""
        from stockquant.signals.valuation import full_valuation

        result = full_valuation("600519")
        assert result["name"] is not None
        assert result["price"] > 0
        # analyst_count 在同花顺 HTML 解析成功时 > 0，
        # 但 HTML 结构可能变化，不强制断言

    def test_normalizes_code(self):
        """支持 sh/sz 前缀和 .SH/.SZ 后缀格式。"""
        from stockquant.signals.valuation import full_valuation

        result = full_valuation("sh600519")
        assert result["name"] is not None

    def test_unknown_code_returns_empty(self):
        """无效代码返回空估值。"""
        from stockquant.signals.valuation import full_valuation

        result = full_valuation("999999")
        assert result["name"] is None


class TestValuationPipeline:
    """端到端估值流水线测试。"""

    def test_peg_workflow(self):
        """完整 PEG 计算流程。"""
        from stockquant.signals.valuation import (
            calc_peg,
            forward_pe,
            full_valuation,
        )

        r = full_valuation("600519")
        if r["eps_cur"]:
            pe_fwd = forward_pe(r["price"], r["eps_cur"])
            assert pe_fwd > 0
            if r["cagr_pct"] and r["cagr_pct"] > 0:
                peg = calc_peg(pe_fwd, r["cagr_pct"] / 100)
                assert peg > 0
