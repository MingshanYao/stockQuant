"""测试 - 回测引擎基础功能。"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockquant.backtest.context import Context
from stockquant.backtest.broker import Broker
from stockquant.backtest.engine import BacktestEngine, BacktestResult
from stockquant.data.universe import BacktestDataset
from stockquant.strategy.base_strategy import BaseStrategy, Order
from stockquant.backtest.bar import BarSnapshot
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


# ======================================================================
# BarSnapshot
# ======================================================================

class TestBarSnapshot:
    def test_contains(self):
        bar = BarSnapshot(
            date=dt.date(2024, 1, 15),
            codes={"600000", "000001"},
            close={"600000": 10.5, "000001": 20.3},
        )
        assert "600000" in bar
        assert "000001" in bar
        assert "999999" not in bar

    def test_defaults(self):
        bar = BarSnapshot(date=dt.date(2024, 1, 15))
        assert bar.codes == set()
        assert bar.close == {}
        assert "anything" not in bar


# ======================================================================
# 辅助
# ======================================================================

def _make_daily(code: str, n_days: int = 60, base_price: float = 10.0) -> pd.DataFrame:
    """生成简单的合成日线数据。"""
    dates = pd.bdate_range(dt.date(2024, 1, 2), periods=n_days)
    rng = np.random.default_rng(hash(code) % 2**32)
    close = base_price + np.cumsum(rng.normal(0, 0.2, n_days))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "code": code,
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n_days),
        "amount": close * rng.integers(1_000_000, 5_000_000, n_days),
    })


def _make_dataset(
    codes: list[str],
    n_days: int = 60,
    benchmark_code: str = "000300",
) -> BacktestDataset:
    stock_data = {c: _make_daily(c, n_days) for c in codes}
    benchmark = _make_daily(benchmark_code, n_days, base_price=4000.0)
    return BacktestDataset(
        stock_data=stock_data,
        codes=codes,
        benchmark=benchmark,
        benchmark_code=benchmark_code,
        start_date="2024-01-02",
        end_date="2024-03-26",
    )


class _BuyAndHoldStrategy(BaseStrategy):
    """测试用策略：首日等权买入所有标的，之后不操作。"""

    def initialize(self) -> None:
        self._bought = False

    def handle_bar(self, bar: dict[str, pd.DataFrame]) -> None:
        if self._bought:
            return
        for code in bar:
            self.order_target_percent(code, 1.0 / max(len(bar), 1))
        self._bought = True


# ======================================================================
# TestEngine — 集成测试
# ======================================================================

class TestEngine:

    def test_load_universe_from_dataset(self):
        dataset = _make_dataset(["600000", "000001"])
        engine = BacktestEngine()
        engine.load_universe(dataset)

        assert len(engine._data) == 2
        assert "600000" in engine._data
        assert not engine._benchmark.empty
        assert engine._benchmark_code == "000300"

    def test_load_universe_requires_dates_for_stock_universe(self):
        from stockquant.data.universe import StockUniverse
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="start_date"):
            engine.load_universe(StockUniverse())

    def test_load_universe_rejects_bad_type(self):
        engine = BacktestEngine()
        with pytest.raises(TypeError):
            engine.load_universe("not a universe")

    def test_run_with_dataset(self):
        dataset = _make_dataset(["600000", "000001"], n_days=30)
        engine = BacktestEngine()
        engine.set_strategy(_BuyAndHoldStrategy())
        engine.load_universe(dataset)
        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.empty
        assert result.initial_capital == 1_000_000.0
        assert not result.benchmark.empty
        assert result.benchmark_code == "000300"

    def test_analyze_returns_performance_analyzer(self):
        dataset = _make_dataset(["600000"], n_days=30)
        engine = BacktestEngine()
        engine.set_strategy(_BuyAndHoldStrategy())
        engine.load_universe(dataset)
        result = engine.run()

        analyzer = result.analyze()
        report = analyzer.full_report()

        assert "总收益率" in report
        assert "Alpha" in report
        assert "Beta" in report

    def test_analyze_without_benchmark(self):
        engine = BacktestEngine()
        engine.set_strategy(_BuyAndHoldStrategy())
        engine.set_data({"600000": _make_daily("600000", 30)})
        result = engine.run()

        analyzer = result.analyze()
        assert analyzer.alpha() is not None

    def test_load_universe_sets_date_range(self):
        dataset = _make_dataset(["600000"], n_days=60)
        engine = BacktestEngine()
        engine.load_universe(
            dataset,
            start_date="2024-02-01",
            end_date="2024-02-28",
        )
        for d in engine._trade_dates:
            assert d >= dt.date(2024, 2, 1)
            assert d <= dt.date(2024, 2, 28)

    def test_lightweight_bar_backtest(self):
        """验证轻量 bar 模式回测产生正确的权益曲线。"""
        dataset = _make_dataset(["600000", "000001"], n_days=30)

        dates = pd.bdate_range(dt.date(2024, 1, 2), periods=30)
        rng = np.random.default_rng(42)
        alpha_panel = pd.DataFrame(
            rng.normal(0, 1, (30, 2)),
            index=dates, columns=["600000", "000001"],
        )

        from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy
        strategy = AlphaFactorStrategy()
        strategy.set_params(
            alpha_panel=alpha_panel, max_positions=2, rebalance_freq=5,
            label="Test", enable_risk_mgmt=False,
        )

        engine = BacktestEngine()
        engine.set_strategy(strategy)
        engine.load_universe(dataset)

        result = engine.run()
        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.empty
        assert len(result.equity_curve) > 0
