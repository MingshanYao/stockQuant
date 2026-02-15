"""测试 - 回测引擎基础功能。"""

import numpy as np
import pandas as pd
import pytest

from stockquant.backtest.context import Context
from stockquant.backtest.broker import Broker
from stockquant.strategy.base_strategy import Order
from stockquant.utils.config import Config


@pytest.fixture(autouse=True)
def reset():
    Config.reset()
    yield
    Config.reset()


@pytest.fixture
def context():
    return Context(initial_capital=1_000_000.0)


@pytest.fixture
def broker(context):
    return Broker(context)


class TestContext:
    def test_initial_state(self, context):
        assert context.cash == 1_000_000.0
        assert context.portfolio_value == 1_000_000.0

    def test_update_price(self, context):
        context.update_price("600000", 10.0)
        assert context.get_current_price("600000") == 10.0

    def test_position(self, context):
        pos = context.get_position("600000")
        assert pos.quantity == 0


class TestBroker:
    def test_buy_order(self, broker, context):
        import datetime as dt
        context.current_date = dt.date(2024, 1, 15)
        context.update_price("600000", 10.0)

        order = Order(code="600000", direction="buy", quantity=1000, price=None)
        result = broker.process_order(order)

        assert result.status == "filled"
        assert context.get_position("600000").quantity == 1000
        assert context.cash < 1_000_000.0

    def test_sell_insufficient(self, broker, context):
        import datetime as dt
        context.current_date = dt.date(2024, 1, 15)
        context.update_price("600000", 10.0)

        order = Order(code="600000", direction="sell", quantity=1000, price=None)

    def test_t_plus_1(self, broker, context):
        import datetime as dt
        context.current_date = dt.date(2024, 1, 15)
        context.update_price("600000", 10.0)

        # 买入
        buy_order = Order(code="600000", direction="buy", quantity=1000, price=None)
        broker.process_order(buy_order)

        # 当日卖出 → 应被拒绝（T+1）
        sell_order = Order(code="600000", direction="sell", quantity=1000, price=None)
        result = broker.process_order(sell_order)
        assert result.status == "rejected"

        # 次日解冻后卖出
        broker.on_new_day()
        sell_order2 = Order(code="600000", direction="sell", quantity=1000, price=None)
        result2 = broker.process_order(sell_order2)
        assert result2.status == "filled"
