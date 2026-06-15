# 回测引擎性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过预计算收盘价面板 + 轻量 BarSnapshot 接口，将 Alpha191 回测从 ~5 小时加速到 ~10-20 分钟（15-30x）。

**Architecture:** 在 `engine.set_data()` 中一次性构建 `close_panel (dates × codes)` DataFrame 和 `code_date_idx` 索引映射。`_build_bar()` 改用零拷贝 iloc 切片替代布尔掩码拷贝。新增 `BarSnapshot` dataclass，策略通过 `uses_lightweight_bar = True` 类属性声明即可接收轻量 bar，完全跳过 DataFrame 拷贝。

**Tech Stack:** Python 3.10+, pandas, numpy, pytest

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `stockquant/backtest/bar.py` | **Create** | BarSnapshot dataclass |
| `stockquant/backtest/engine.py` | **Modify** | close_panel 预计算、_build_bar 优化、轻量 bar 分发、价格更新优化 |
| `stockquant/strategy/base_strategy.py` | **Modify** | 新增 `uses_lightweight_bar` 类属性 |
| `stockquant/strategy/alpha_factor_strategy.py` | **Modify** | 迁移到轻量 bar 模式 |
| `tests/test_backtest.py` | **Modify** | 新增 close_panel 正确性、轻量 bar 回测、BarSnapshot 测试 |

---

### Task 1: BarSnapshot dataclass

**Files:**
- Create: `stockquant/backtest/bar.py`

- [ ] **Step 1: Create BarSnapshot dataclass**

```python
"""回测 bar 数据结构。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class BarSnapshot:
    """单个交易日的轻量 bar 快照。

    Attributes
    ----------
    date : dt.date
        交易日。
    codes : set[str]
        当日有数据的股票代码集合。
    close : dict[str, float]
        股票代码 → 当日收盘价。
    """

    date: dt.date
    codes: set[str] = field(default_factory=set)
    close: dict[str, float] = field(default_factory=dict)

    def __contains__(self, code: str) -> bool:
        """支持 ``code in bar`` 用法（兼容旧接口）。"""
        return code in self.codes
```

- [ ] **Step 2: Verify the module imports correctly**

Run: `python -c "from stockquant.backtest.bar import BarSnapshot; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add stockquant/backtest/bar.py
git commit -m "feat: add BarSnapshot dataclass for lightweight bar data"
```

---

### Task 2: BaseStrategy 轻量 bar 支持

**Files:**
- Modify: `stockquant/strategy/base_strategy.py`

- [ ] **Step 1: Add uses_lightweight_bar class attribute**

In `class BaseStrategy(ABC)`, add after the docstring and before `def __init__`:

```python
# 类级别标记：声明本策略使用轻量 BarSnapshot 接口
uses_lightweight_bar: bool = False
```

- [ ] **Step 2: Verify existing strategies still work**

Run: `python -c "from stockquant.strategy.base_strategy import BaseStrategy; print(BaseStrategy.uses_lightweight_bar)"`
Expected: `False`

Run: `python -c "from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy; print(AlphaFactorStrategy.uses_lightweight_bar)"`
Expected: `False`

- [ ] **Step 3: Commit**

```bash
git add stockquant/strategy/base_strategy.py
git commit -m "feat: add uses_lightweight_bar flag to BaseStrategy"
```

---

### Task 3: 收盘价面板预计算 + 日期索引

**Files:**
- Modify: `stockquant/backtest/engine.py`

- [ ] **Step 1: Add `_close_panel` and `_code_date_idx` initialization in `__init__`**

In `BacktestEngine.__init__`, add after `self._benchmark_code = ""`:

```python
self._close_panel: pd.DataFrame = pd.DataFrame()
self._code_date_idx: dict[str, dict[dt.date, int]] = {}
```

- [ ] **Step 2: Build close panel and date index in `set_data`**

In `BacktestEngine.set_data`, after the `self._data[code] = df` loop (after the existing data normalization block), add:

```python
        # ── 预计算收盘价面板 (dates × codes) 和日期索引 ──
        close_data: dict[str, pd.Series] = {}
        self._code_date_idx = {}
        for code, df in self._data.items():
            if "date" in df.columns and "close" in df.columns:
                close_data[code] = df.set_index("date")["close"]
            if "date" in df.columns:
                dates = df["date"]
                self._code_date_idx[code] = {
                    d.date(): i for i, d in enumerate(dates)
                }
        if close_data:
            self._close_panel = pd.DataFrame(close_data).sort_index()
        else:
            self._close_panel = pd.DataFrame()
```

- [ ] **Step 3: Verify close panel is built correctly in a quick script**

Create and run a test:

```python
# verify_close_panel.py
import datetime as dt
import numpy as np
import pandas as pd
from stockquant.backtest.engine import BacktestEngine
from stockquant.strategy.base_strategy import BaseStrategy

def _make_daily(code, n_days=60, base_price=10.0):
    dates = pd.bdate_range(dt.date(2024, 1, 2), periods=n_days)
    rng = np.random.default_rng(hash(code) % 2**32)
    close = base_price + np.cumsum(rng.normal(0, 0.2, n_days))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "code": code, "date": dates, "open": close * 0.99,
        "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, n_days),
        "amount": close * rng.integers(1_000_000, 5_000_000, n_days),
    })

class _TestStrategy(BaseStrategy):
    uses_lightweight_bar = False
    def initialize(self): pass
    def handle_bar(self, bar): pass

engine = BacktestEngine()
engine.set_strategy(_TestStrategy())
engine.set_data({"600000": _make_daily("600000", 60), "000001": _make_daily("000001", 60)})

assert not engine._close_panel.empty, "close panel should not be empty"
assert engine._close_panel.shape[1] == 2, f"Expected 2 columns, got {engine._close_panel.shape[1]}"
assert len(engine._code_date_idx) == 2, f"Expected 2 stocks in idx map, got {len(engine._code_date_idx)}"
assert engine._close_panel.index.is_monotonic_increasing, "panel index should be sorted"

# Verify date index accuracy: first date for each stock should map to position 0
for code in ["600000", "000001"]:
    first_date = engine._data[code]["date"].iloc[0].date()
    assert engine._code_date_idx[code][first_date] == 0, f"{code} first date should be idx 0"
    last_date = engine._data[code]["date"].iloc[-1].date()
    assert engine._code_date_idx[code][last_date] == len(engine._data[code]) - 1

print("All assertions passed!")
```

Run: `python verify_close_panel.py`
Expected: `All assertions passed!`

Delete `verify_close_panel.py` after verification.

- [ ] **Step 4: Commit**

```bash
git add stockquant/backtest/engine.py
git commit -m "feat: pre-compute close price panel and date index in engine.set_data()"
```

---

### Task 4: 优化 _build_bar + 基准价格切片

**Files:**
- Modify: `stockquant/backtest/engine.py`

- [ ] **Step 1: Replace _build_bar with index-based zero-copy version**

Replace the current `_build_bar` method (lines 217-226):

```python
    def _build_bar(self, trade_date: dt.date) -> dict[str, pd.DataFrame]:
        """使用预计算索引的零拷贝 bar 构建。

        对比旧版: 不再用 ``df[df["date"] <= trade_date]`` 布尔掩码拷贝，
        改用 ``df.iloc[:idx+1]`` 连续整数切片（零拷贝或低拷贝）。
        """
        bar: dict[str, pd.DataFrame] = {}
        for code, df in self._data.items():
            idx_map = self._code_date_idx.get(code)
            if idx_map is None:
                continue
            idx = idx_map.get(trade_date)
            if idx is not None:
                bar[code] = df.iloc[: idx + 1]
        return bar
```

- [ ] **Step 2: Add setup_trade_date helper method for per-bar state updates**

Add a new method to consolidate the per-bar setup that happens in `run()`:

```python
    def _setup_trade_date(self, trade_date: dt.date) -> None:
        """设置当日上下文状态: 基准价格 + 价格缓存更新。"""
        ts = pd.Timestamp(trade_date)

        # 基准价格（零拷贝 iloc 切片替代布尔掩码）
        if hasattr(self, "_bm_close") and self._bm_close is not None:
            if hasattr(self, "_bm_index") and self._bm_index is not None:
                pos = self._bm_index.searchsorted(ts, side="right")
                self.context.benchmark_prices = self._bm_close.iloc[:pos]
            else:
                self.context.benchmark_prices = None
        else:
            self.context.benchmark_prices = None

        # 价格缓存更新（从 close_panel 一次查找替代遍历 bar）
        if ts in self._close_panel.index:
            row = self._close_panel.loc[ts].dropna()
            for code, price in row.items():
                self.context.update_price(code, price)
```

- [ ] **Step 3: Update run() to use optimized slices**

In `BacktestEngine.run()`, replace the existing benchmark slicing and price update sections:

Replace the existing benchmark pre-extraction block (after `self.strategy.initialize()`):

Old:
```python
        bm_close = None
        if not self._benchmark.empty and "close" in self._benchmark.columns:
            bm = self._benchmark.sort_values("date").set_index("date")
            bm_close = bm["close"]
```

New:
```python
        self._bm_close = None
        self._bm_index = None
        if not self._benchmark.empty and "close" in self._benchmark.columns:
            bm = self._benchmark.sort_values("date").set_index("date")
            self._bm_close = bm["close"]
            self._bm_index = bm["close"].index
```

Replace the main loop body (lines 167-210). The per-bar benchmark slicing and price update loop:

Old lines 170-192:
```python
            if bm_close is not None:
                self.context.benchmark_prices = bm_close[bm_close.index <= pd.Timestamp(trade_date)]
            else:
                self.context.benchmark_prices = None

            # 1. 日切：解冻 T+1
            self.broker.on_new_day()

            # 2. 盘前回调
            self.strategy.before_trading()

            # 3. 构建当日 bar 数据
            bar = self._build_bar(trade_date)
            if not bar:
                continue

            # 4. 更新价格缓存
            for code, row in bar.items():
                if not row.empty:
                    price = row["close"].iloc[-1]
                    self.context.update_price(code, price)
```

New:
```python
            # 1. 日切：解冻 T+1
            self.broker.on_new_day()

            # 2. 盘前回调
            self.strategy.before_trading()

            # ── 轻量 bar 模式：完全跳过 _build_bar ──
            if getattr(self.strategy, "uses_lightweight_bar", False):
                self._setup_trade_date(trade_date)
                ts = pd.Timestamp(trade_date)
                available: set[str] = set()
                prices: dict[str, float] = {}
                if ts in self._close_panel.index:
                    row = self._close_panel.loc[ts].dropna()
                    available = set(row.index)
                    prices = row.to_dict()
                from stockquant.backtest.bar import BarSnapshot
                bar_snapshot = BarSnapshot(date=trade_date, codes=available, close=prices)
                self.strategy._orders.clear()
                self.strategy.handle_bar(bar_snapshot)
            else:
                # 传统模式：构建完整 bar（零拷贝 iloc 切片）
                self._setup_trade_date(trade_date)
                bar = self._build_bar(trade_date)
                if not bar:
                    continue
                self.strategy._orders.clear()
                self.strategy.handle_bar(bar)
```

Note: the order clearing (`self.strategy._orders.clear()`) needs to happen before handle_bar in both branches. Keep lines 194-210 (订单撮合, 权益记录, 收盘后回调) unchanged after the if/else block.

- [ ] **Step 4: Verify correctness — run existing backtest tests**

Run: `python -m pytest tests/test_backtest.py::TestEngine -v`
Expected: All tests in TestEngine pass (especially `test_run_with_dataset` and `test_analyze_returns_performance_analyzer`)

- [ ] **Step 5: Commit**

```bash
git add stockquant/backtest/engine.py
git commit -m "perf: optimize _build_bar with zero-copy iloc slicing and close panel price updates"
```

---

### Task 5: AlphaFactorStrategy 迁移到轻量 bar

**Files:**
- Modify: `stockquant/strategy/alpha_factor_strategy.py`

- [ ] **Step 1: Enable lightweight bar flag**

At the top of `class AlphaFactorStrategy(BaseStrategy):`, before `def initialize`, add:

```python
    uses_lightweight_bar = True
```

- [ ] **Step 2: Update handle_bar signature and bar usage**

Change `handle_bar` method:

Old (line 137):
```python
    def handle_bar(self, bar: dict) -> None:
```

New:
```python
    def handle_bar(self, bar) -> None:
```

Replace the availability check (line 179):

Old:
```python
        available = [c for c in today_alpha.index if c in bar and not bar[c].empty]
```

New:
```python
        available = [c for c in today_alpha.index if c in bar]
```

Note: `BarSnapshot.__contains__` handles the `c in bar` check by delegating to `bar.codes`.

- [ ] **Step 3: Verify AlphaFactorStrategy works with lightweight bar**

Create and run a quick integration test:

```python
# verify_lightweight.py
import datetime as dt
import numpy as np
import pandas as pd
from stockquant.backtest.engine import BacktestEngine
from stockquant.data.universe import BacktestDataset
from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy

def _make_daily(code, n_days=60, base_price=10.0):
    dates = pd.bdate_range(dt.date(2024, 1, 2), periods=n_days)
    rng = np.random.default_rng(hash(code) % 2**32)
    close = base_price + np.cumsum(rng.normal(0, 0.2, n_days))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "code": code, "date": dates, "open": close * 0.99,
        "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": rng.integers(1000000, 5000000, n_days),
        "amount": close * rng.integers(1000000, 5000000, n_days),
    })

codes = ["600000", "000001", "000002"]
stock_data = {c: _make_daily(c, 60) for c in codes}
benchmark = _make_daily("000300", 60, base_price=4000.0)
dataset = BacktestDataset(
    stock_data=stock_data, codes=codes,
    benchmark=benchmark, benchmark_code="000300",
    start_date="2024-01-02", end_date="2024-03-26",
)

# Build a simple alpha panel (random factor values)
dates = pd.bdate_range(dt.date(2024, 1, 2), periods=60)
rng = np.random.default_rng(42)
alpha_panel = pd.DataFrame(
    rng.normal(0, 1, (60, len(codes))),
    index=dates, columns=codes
)

strategy = AlphaFactorStrategy()
strategy.set_params(
    alpha_panel=alpha_panel, max_positions=2, rebalance_freq=5,
    label="Test", enable_risk_mgmt=False,
)

engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data(stock_data)
engine._benchmark = benchmark
engine._benchmark_code = "000300"
engine.set_date_range(start_date="2024-01-02", end_date="2024-03-26")

result = engine.run()
assert not result.equity_curve.empty, "equity curve should not be empty"
assert len(result.daily_returns) > 0, "should have daily returns"
print(f"Backtest passed: {len(result.equity_curve)} equity points, final value: {result.final_value:,.0f}")
```

Run: `python verify_lightweight.py`
Expected: `Backtest passed: ... equity points, final value: ...`

Delete `verify_lightweight.py` after verification.

- [ ] **Step 4: Commit**

```bash
git add stockquant/strategy/alpha_factor_strategy.py
git commit -m "perf: migrate AlphaFactorStrategy to lightweight BarSnapshot interface"
```

---

### Task 6: 回归测试 + 性能基准

**Files:**
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Add BarSnapshot unit test**

In `tests/test_backtest.py`, add after the existing imports:

```python
from stockquant.backtest.bar import BarSnapshot


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
```

- [ ] **Step 2: Add lightweight bar backtest test**

In `tests/test_backtest.py`, add to `class TestEngine`:

```python
    def test_lightweight_bar_backtest(self):
        """验证轻量 bar 模式回测产生正确的权益曲线。"""
        dataset = _make_dataset(["600000", "000001"], n_days=30)

        # Build a simple alpha panel
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
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: All tests pass (including existing `TestEngine` tests and new `TestBarSnapshot` / lightweight tests)

- [ ] **Step 4: Run all existing tests to ensure no regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_backtest.py
git commit -m "test: add BarSnapshot unit tests and lightweight bar backtest integration test"
```

---

### Task 7: 性能验证 — 端到端基准测试

**Files:**
- Create: `tests/test_backtest_benchmark.py`

- [ ] **Step 1: Create benchmark script**

```python
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

    print(f"Benchmark: {N_STOCKS} stocks × {N_DAYS} bars...")
    codes = [f"{i:06d}" for i in range(1, N_STOCKS + 1)]
    stock_data = {c: _make_daily(c, N_DAYS) for c in codes}

    # Alpha panel
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
```

- [ ] **Step 2: Run benchmark**

Run: `python tests/test_backtest_benchmark.py`
Expected: 200 stocks × 500 bars completes in under 30 seconds (was previously minutes for similar scale)

- [ ] **Step 3: Commit**

```bash
git add tests/test_backtest_benchmark.py
git commit -m "test: add backtest performance benchmark"
```

---

## Self-Review

**1. Spec coverage:**
- close panel 预计算: Task 3
- code_date_idx 索引映射: Task 3
- _build_bar iloc 零拷贝优化: Task 4
- 基准价格切片优化: Task 4 (in _setup_trade_date)
- 价格更新 loop 优化: Task 4 (via close_panel.loc lookup)
- BarSnapshot dataclass: Task 1
- uses_lightweight_bar flag: Task 2
- 轻量 bar 分发逻辑: Task 4 (in run() loop)
- AlphaFactorStrategy 迁移: Task 5
- 回归测试: Task 6
- 性能基准: Task 7

**2. Placeholder scan:** No TBD, TODO, or vague references found.

**3. Type consistency:** `BarSnapshot` defined in Task 1 with `date`, `codes`, `close`, `__contains__` — Task 5 uses `c in bar` which maps to `__contains__`. Engine in Task 4 imports `BarSnapshot` from `stockquant.backtest.bar` and constructs with matching field names. All consistent.
