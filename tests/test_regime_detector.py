"""Test - MarketRegimeDetector."""
import numpy as np
import pandas as pd
import pytest
from stockquant.risk.regime_detector import MarketRegimeDetector, Regime


def make_prices(start, n, daily_ret, vol_scale=1.0):
    """Generate a price series with given daily return and volatility scaling."""
    np.random.seed(42)
    base_ret = np.full(n, daily_ret)
    noise = np.random.randn(n) * 0.02 * vol_scale
    returns = base_ret + noise
    prices = start * np.cumprod(1 + returns)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


class TestMarketRegimeDetector:
    def test_bull_market(self):
        """Strong uptrend + normal vol -> BULL."""
        prices = make_prices(100, 252, 0.002)  # ~50% annual return
        detector = MarketRegimeDetector()
        regime = detector.detect(prices)
        assert regime == Regime.BULL

    def test_crisis_market(self):
        """Sharp decline + high vol -> CRISIS."""
        normal = make_prices(100, 232, 0.001)
        # Build a volatile crash: all 30 days have high volatility
        crash_returns = np.random.randn(30) * 0.04 - 0.02  # avg -2%/day, high vol
        # Force even more extreme drop in last 5 days, still volatile
        crash_returns[-5:] = np.random.randn(5) * 0.04 - 0.04  # avg -4%/day, high vol
        crash_prices = normal.iloc[-1] * np.cumprod(1 + crash_returns)
        crash_series = pd.Series(
            crash_prices,
            index=pd.date_range(normal.index[-1] + pd.Timedelta(days=1), periods=len(crash_returns), freq="B")
        )
        prices = pd.concat([normal, crash_series])
        detector = MarketRegimeDetector()
        regime = detector.detect(prices)
        assert regime == Regime.CRISIS

    def test_regime_scale_bull(self):
        """BULL -> position_scale = 1.0."""
        detector = MarketRegimeDetector()
        assert detector.get_position_scale(Regime.BULL) == 1.0

    def test_regime_scale_crisis(self):
        """CRISIS -> position_scale = 0.2."""
        detector = MarketRegimeDetector()
        assert detector.get_position_scale(Regime.CRISIS) == 0.2

    def test_returns_scale(self):
        """Returns a float between 0 and 1."""
        detector = MarketRegimeDetector()
        scale = detector.compute_scale(pd.Series([100, 101, 102, 103]))
        assert 0.0 <= scale <= 1.0
