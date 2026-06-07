"""测试 - 指标注册表 & 策略注册表。"""

import pandas as pd
import pytest

from stockquant.indicators import IndicatorRegistry
from stockquant.indicators.base import BaseIndicator
from stockquant.indicators.trend import TrendIndicators
from stockquant.indicators.oscillator import OscillatorIndicators
from stockquant.indicators.volume import VolumeIndicators
from stockquant.indicators.volatility import VolatilityIndicators
from stockquant.indicators.alpha101 import Alpha101Indicators

from stockquant.strategy import StrategyRegistry
from stockquant.strategy.base_strategy import BaseStrategy
from stockquant.strategy.examples import DualMAStrategy
from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy


class TestIndicatorRegistry:

    def test_list_all_indicators(self):
        names = IndicatorRegistry.list()
        assert len(names) == 5
        assert set(names) == {"trend", "oscillator", "volume", "volatility", "alpha101"}

    def test_create_by_name(self):
        trend = IndicatorRegistry.create("trend")
        assert isinstance(trend, TrendIndicators)

        osc = IndicatorRegistry.create("Oscillator")
        assert isinstance(osc, OscillatorIndicators)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="未注册的指标"):
            IndicatorRegistry.create("nonexistent")

    def test_get_returns_class(self):
        cls = IndicatorRegistry.get("volume")
        assert cls is VolumeIndicators
        assert issubclass(cls, BaseIndicator)

    def test_compute_via_registry(self):
        indicator = IndicatorRegistry.create("volatility")
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=30),
            "open": range(30),
            "high": range(1, 31),
            "low": range(30),
            "close": range(30),
            "volume": [1000] * 30,
        })
        result = indicator.compute(df, atr=True)
        assert "atr14" in result.columns


class TestStrategyRegistry:

    def test_list_all_strategies(self):
        names = StrategyRegistry.list()
        assert len(names) == 2
        assert set(names) == {"dual_ma", "alpha_factor"}

    def test_create_by_name(self):
        strategy = StrategyRegistry.create("dual_ma")
        assert isinstance(strategy, DualMAStrategy)
        assert isinstance(strategy, BaseStrategy)

    def test_create_alpha_factor(self):
        strategy = StrategyRegistry.create("alpha_factor")
        assert isinstance(strategy, AlphaFactorStrategy)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="未注册的策略"):
            StrategyRegistry.create("nonexistent")

    def test_get_returns_class(self):
        cls = StrategyRegistry.get("dual_ma")
        assert cls is DualMAStrategy
