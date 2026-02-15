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
