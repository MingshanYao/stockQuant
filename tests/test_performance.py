"""测试 - 绩效分析模块。"""

import numpy as np
import pandas as pd
import pytest

from stockquant.analysis.performance import PerformanceAnalyzer


@pytest.fixture
def analyzer():
    n = 252
    dates = pd.date_range("2024-01-01", periods=n)
    np.random.seed(42)
    daily_ret = np.random.normal(0.0005, 0.015, n)
    values = [1_000_000]
    for r in daily_ret:
        values.append(values[-1] * (1 + r))
    values = values[1:]

    equity = pd.DataFrame({"date": dates, "total_value": values})
    trades = pd.DataFrame({
        "date": ["2024-03-01", "2024-06-01"],
        "code": ["600000", "600000"],
        "direction": ["buy", "sell"],
        "quantity": [1000, 1000],
        "price": [10.0, 11.0],
        "commission": [5.0, 5.0],
    })

    return PerformanceAnalyzer(
        equity_curve=equity,
        trade_log=trades,
        daily_returns=daily_ret.tolist(),
    )


class TestPerformanceAnalyzer:
    def test_total_return(self, analyzer):
        ret = analyzer.total_return()
        assert isinstance(ret, float)

    def test_annualized_return(self, analyzer):
        ret = analyzer.annualized_return()
        assert isinstance(ret, float)

    def test_max_drawdown(self, analyzer):
        mdd = analyzer.max_drawdown()
        assert 0 <= mdd <= 1

    def test_sharpe_ratio(self, analyzer):
        sharpe = analyzer.sharpe_ratio()
        assert isinstance(sharpe, float)

    def test_volatility(self, analyzer):
        vol = analyzer.volatility()
        assert vol >= 0

    def test_trade_statistics(self, analyzer):
        stats = analyzer.trade_statistics()
        assert stats["总交易次数"] == 2

    def test_full_report(self, analyzer):
        report = analyzer.full_report()
        assert "总收益率" in report
        assert "夏普比率" in report
