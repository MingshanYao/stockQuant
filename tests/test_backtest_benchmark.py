"""性能基准测试 — 验证回测加速效果。

运行方式: python tests/test_backtest_benchmark.py
"""
import datetime as dt
import time

import numpy as np
import pandas as pd

from stockquant.backtest.engine import BacktestEngine
from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy


def _make_daily(code, n_days=500, base_price=10.0):
    dates = pd.bdate_range(dt.date(2020, 1, 2), periods=n_days)
    rng = np.random.default_rng(hash(code) % 2**32)
    close = base_price + np.cumsum(rng.normal(0, 0.2, n_days))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "code": code, "date": dates, "open": close * 0.99,
        "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": rng.integers(1000000, 5000000, n_days),
        "amount": close * rng.integers(1000000, 5000000, n_days),
    })


def main():
    N_STOCKS = 200
    N_DAYS = 500

    print(f"Benchmark: {N_STOCKS} stocks x {N_DAYS} bars...")
    codes = [f"{i:06d}" for i in range(1, N_STOCKS + 1)]
    stock_data = {c: _make_daily(c, N_DAYS) for c in codes}

    dates = pd.bdate_range(dt.date(2020, 1, 2), periods=N_DAYS)
    rng = np.random.default_rng(42)
    alpha_panel = pd.DataFrame(
        rng.normal(0, 1, (N_DAYS, N_STOCKS)),
        index=dates, columns=codes,
    )

    benchmark_df = _make_daily("000300", N_DAYS, base_price=4000.0)

    strategy = AlphaFactorStrategy()
    strategy.set_params(
        alpha_panel=alpha_panel, max_positions=50, rebalance_freq=5,
        label="Bench", enable_risk_mgmt=False,
    )

    engine = BacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data(stock_data)
    engine._benchmark = benchmark_df
    engine._benchmark_code = "000300"

    start = time.perf_counter()
    result = engine.run()
    elapsed = time.perf_counter() - start

    n_bars = len(engine._trade_dates)
    print(f"  Bars processed: {n_bars}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Per bar: {elapsed / n_bars * 1000:.2f}ms")
    print(f"  Final equity points: {len(result.equity_curve)}")
    assert not result.equity_curve.empty, "equity curve should not be empty"
    print("PASSED")


if __name__ == "__main__":
    main()
