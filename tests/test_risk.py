"""测试 - 风控模块。"""

import pytest

from stockquant.risk.stop_loss import StopLossManager
from stockquant.risk.position_manager import PositionManager
from stockquant.risk.stock_filter import StockFilter
from stockquant.backtest.context import Context
from stockquant.strategy.base_strategy import Position
from stockquant.utils.config import Config

import pandas as pd


@pytest.fixture(autouse=True)
def reset():
    Config.reset()
    yield
    Config.reset()


class TestStopLoss:
    def test_fixed_stop_loss(self):
        mgr = StopLossManager()
        pos = Position(code="600000", quantity=1000, avg_cost=10.0)

        # 亏损 5% → 不触发
        result = mgr.check(pos, 9.5)
        assert not result.triggered

        # 亏损 9% → 触发
        result = mgr.check(pos, 9.1)
        assert result.triggered
        assert result.action == "stop_loss"

    def test_take_profit(self):
        mgr = StopLossManager()
        pos = Position(code="600000", quantity=1000, avg_cost=10.0)

        # 盈利 25% → 触发止盈
        result = mgr.check(pos, 12.5)
        assert result.triggered
        assert result.action == "take_profit"


class TestPositionManager:
    def test_max_single_position(self):
        mgr = PositionManager()
        ctx = Context(initial_capital=1_000_000.0)

        # 请求 300,000 但单只上限 25% = 250,000
        allowed = mgr.check_buy(ctx, "600000", 300_000)
        assert allowed <= 250_000


class TestStockFilter:
    def test_filter_st(self):
        flt = StockFilter()
        df = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "name": ["浦发银行", "*ST某某", "齐翔腾达"],
        })
        result = flt.filter(df)
        assert len(result) == 2
        assert "*ST某某" not in result["name"].values


class TestRiskMonitorGraduated:
    @pytest.fixture
    def monitor(self):
        from stockquant.risk.risk_monitor import RiskMonitor
        return RiskMonitor()

    def test_green_when_no_drawdown(self, monitor):
        monitor.update_value(100_000)
        monitor.update_value(101_000)
        from stockquant.risk.risk_monitor import AlertLevel
        level = monitor.update_value(100_500)
        assert level == AlertLevel.GREEN

    def test_yellow_at_5pct_drawdown(self, monitor):
        monitor.update_value(100_000)
        from stockquant.risk.risk_monitor import AlertLevel
        level = monitor.update_value(95_000)
        assert level == AlertLevel.YELLOW

    def test_orange_at_8pct_drawdown(self, monitor):
        monitor.update_value(100_000)
        from stockquant.risk.risk_monitor import AlertLevel
        level = monitor.update_value(92_000)
        assert level == AlertLevel.ORANGE

    def test_red_at_12pct_drawdown(self, monitor):
        monitor.update_value(100_000)
        from stockquant.risk.risk_monitor import AlertLevel
        level = monitor.update_value(88_000)
        assert level == AlertLevel.RED

    def test_peak_updates(self, monitor):
        monitor.update_value(100_000)
        monitor.update_value(110_000)
        monitor.update_value(105_000)
        assert monitor._peak_value == 110_000

    def test_returns_alert_level_not_bool(self, monitor):
        monitor.update_value(100_000)
        from stockquant.risk.risk_monitor import AlertLevel
        result = monitor.update_value(95_000)
        assert isinstance(result, AlertLevel)

    def test_backward_compat_update_accepts_context(self, monitor):
        """Old `update(context)` interface still works."""
        from stockquant.backtest.context import Context
        ctx = Context(initial_capital=100_000)
        monitor.update_value(100_000)  # set initial peak
        from stockquant.risk.risk_monitor import AlertLevel
        level = monitor.update(ctx)
        assert isinstance(level, AlertLevel)
