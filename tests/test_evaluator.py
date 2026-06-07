"""测试 - 因子评价体系。"""

import numpy as np
import pandas as pd
import pytest

from stockquant.analysis.factor import FactorAnalyzer
from stockquant.analysis.evaluator import FactorEvaluator


# ======================================================================
# 辅助
# ======================================================================

def _make_panel(n_days: int = 60, n_stocks: int = 50, seed: int = 42) -> dict:
    """生成测试用因子面板和收益面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

    factor = pd.DataFrame(
        rng.standard_normal((n_days, n_stocks)),
        index=dates, columns=codes,
    )
    close = pd.DataFrame(
        100 + np.cumsum(rng.standard_normal((n_days, n_stocks)) * 0.5, axis=0),
        index=dates, columns=codes,
    )
    return {"factor": factor, "close": close}


# ======================================================================
# Task 1: calc_factor_return
# ======================================================================

class TestCalcFactorReturn:

    def test_positive_factor_return(self):
        np.random.seed(42)
        n = 100
        factor = pd.Series(np.random.randn(n))
        returns = factor * 0.05 + np.random.randn(n) * 0.01
        fr = FactorAnalyzer.calc_factor_return(factor, returns)
        assert fr > 0

    def test_zero_factor_return(self):
        np.random.seed(42)
        n = 200
        factor = pd.Series(np.random.randn(n))
        returns = pd.Series(np.random.randn(n) * 0.01)
        fr = FactorAnalyzer.calc_factor_return(factor, returns)
        assert abs(fr) < 0.05

    def test_insufficient_data(self):
        factor = pd.Series([1.0, 2.0])
        returns = pd.Series([0.01, 0.02])
        fr = FactorAnalyzer.calc_factor_return(factor, returns)
        assert fr == 0.0


# ======================================================================
# Task 2: FactorEvaluator
# ======================================================================

class TestFactorEvaluator:

    def test_evaluate_single_returns_all_metrics(self):
        data = _make_panel()
        ev = FactorEvaluator(close_panel=data["close"])
        result = ev.evaluate(data["factor"], forward_period=1)

        assert "ic_mean" in result
        assert "ic_ir" in result
        assert "fr_mean" in result
        assert "fr_ir" in result
        assert "ic_pos_ratio" in result
        assert "t_stat" in result
        assert "fr_annual" in result

    def test_ic_sign_for_predictive_factor(self):
        """构造有明确预测力的因子，IC 应非零。"""
        rng = np.random.default_rng(42)
        n_days, n_stocks = 60, 50
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

        close = pd.DataFrame(
            100 + np.cumsum(rng.standard_normal((n_days, n_stocks)) * 0.5, axis=0),
            index=dates, columns=codes,
        )
        fwd_ret = close.pct_change().shift(-1)
        factor = fwd_ret * 100 + pd.DataFrame(
            rng.standard_normal((n_days, n_stocks)) * 0.01,
            index=dates, columns=codes,
        )

        ev = FactorEvaluator(close_panel=close)
        result = ev.evaluate(factor, forward_period=1)
        assert abs(result["ic_mean"]) > 0.1

    def test_multi_horizon(self):
        data = _make_panel()
        ev = FactorEvaluator(close_panel=data["close"])
        results = ev.evaluate_multi_horizon(data["factor"], periods=[1, 2, 3])
        assert len(results) == 3
        assert all(p in results for p in [1, 2, 3])

    def test_neutralize_integration(self):
        data = _make_panel()
        rng = np.random.default_rng(99)
        codes = data["factor"].columns
        industry = pd.Series(
            rng.choice(["bank", "pharma", "tech", "consumer"], len(codes)),
            index=codes,
        )
        market_cap = pd.Series(
            rng.uniform(1e9, 1e11, len(codes)),
            index=codes,
        )

        ev = FactorEvaluator(
            close_panel=data["close"],
            industry=industry,
            market_cap=market_cap,
        )
        result = ev.evaluate(data["factor"], forward_period=1)
        assert "ic_mean" in result
        assert not np.isnan(result["ic_mean"])


# ======================================================================
# Task 3: Factor system analysis
# ======================================================================

class TestFactorSystem:

    def test_evaluate_system(self):
        data = _make_panel()
        rng = np.random.default_rng(123)
        n_days, n_stocks = data["factor"].shape
        factor2 = pd.DataFrame(
            rng.standard_normal((n_days, n_stocks)),
            index=data["factor"].index,
            columns=data["factor"].columns,
        )
        ev = FactorEvaluator(close_panel=data["close"])
        result = ev.evaluate_system(
            {"f1": data["factor"], "f2": factor2},
            forward_period=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "ic_mean" in result.columns

    def test_correlation_matrix(self):
        data = _make_panel()
        rng = np.random.default_rng(123)
        n_days, n_stocks = data["factor"].shape
        factor2 = pd.DataFrame(
            rng.standard_normal((n_days, n_stocks)),
            index=data["factor"].index,
            columns=data["factor"].columns,
        )
        ev = FactorEvaluator(close_panel=data["close"])
        corr = ev.factor_correlation_matrix(
            {"f1": data["factor"], "f2": factor2}
        )
        assert corr.shape == (2, 2)
        assert abs(corr.loc["f1", "f1"] - 1.0) < 0.01

    def test_correlated_factors_detected(self):
        data = _make_panel()
        factor_copy = data["factor"] * 1.5 + 0.1
        ev = FactorEvaluator(close_panel=data["close"])
        corr = ev.factor_correlation_matrix(
            {"original": data["factor"], "scaled": factor_copy}
        )
        assert corr.loc["original", "scaled"] > 0.9


# ======================================================================
# Task 4: Integration
# ======================================================================

class TestAnalysisExports:

    def test_evaluator_importable(self):
        from stockquant.analysis import FactorEvaluator as FE
        assert FE is not None


class TestAlphaResearcherIntegration:

    @pytest.fixture
    def researcher(self):
        from stockquant.data.universe import BacktestDataset
        from stockquant.research.alpha_researcher import AlphaResearcher

        n_days, n_stocks = 30, 10
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        rng = np.random.default_rng(42)

        stock_data = {}
        for code in codes:
            close = 10 + np.cumsum(rng.normal(0, 0.2, n_days))
            close = np.maximum(close, 1.0)
            stock_data[code] = pd.DataFrame({
                "code": code, "date": dates,
                "open": close * 0.99, "high": close * 1.02,
                "low": close * 0.98, "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n_days),
                "amount": close * rng.integers(1_000_000, 5_000_000, n_days),
            })
        benchmark = pd.DataFrame({
            "code": "000300", "date": dates,
            "open": 4000.0, "high": 4020.0, "low": 3980.0,
            "close": 4000 + np.cumsum(rng.normal(0, 5, n_days)),
            "volume": rng.integers(int(1e8), int(5e8), n_days),
            "amount": 4000 * rng.integers(int(1e8), int(5e8), n_days),
        })
        dataset = BacktestDataset(
            stock_data=stock_data, codes=codes,
            benchmark=benchmark, benchmark_code="000300",
            start_date="2024-01-02", end_date="2024-02-12",
        )
        return AlphaResearcher(dataset, max_positions=3, rebalance_freq=5)

    def test_evaluate_factor(self, researcher):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=30)
        codes = [f"{i:06d}" for i in range(1, 11)]
        panel = pd.DataFrame(
            rng.standard_normal((30, 10)),
            index=dates, columns=codes,
        )
        result = researcher.evaluate_factor(panel, forward_period=1)
        assert "ic_mean" in result
        assert "fr_annual" in result

    def test_evaluate_multi_horizon(self, researcher):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=30)
        codes = [f"{i:06d}" for i in range(1, 11)]
        panel = pd.DataFrame(
            rng.standard_normal((30, 10)),
            index=dates, columns=codes,
        )
        result = researcher.evaluate_multi_horizon(panel, periods=[1, 2])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
