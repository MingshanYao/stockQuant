# 回撤控制框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a graduated drawdown control framework (感知层+决策层) that combines market regime detection, tiered drawdown alerts, and dynamic position sizing to prevent 2015-style 48% drawdowns.

**Architecture:** Add benchmark data to `Context` so strategies can detect market regime (牛/熊/震荡/危机) in real-time. Replace the binary RiskMonitor freeze with graduated 4-level alerts (绿/黄/橙/红). Create `MarketRegimeDetector` that uses MA cross + volatility ratio. In `AlphaFactorStrategy`, layer regime caps × DD alert compression × individual stop-loss to determine actual position size per bar.

**Tech Stack:** Python 3.12, pandas, numpy, existing stockquant framework

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `stockquant/risk/regime_detector.py` | **Create** | Market regime detection (牛/熊/震荡/危机) from benchmark close prices |
| `stockquant/risk/risk_monitor.py` | **Modify** | Replace binary freeze → 4-level graduated `AlertLevel` (GREEN/YELLOW/ORANGE/RED) |
| `stockquant/backtest/context.py` | **Modify** | Add `benchmark_prices` attribute for regime detection |
| `stockquant/backtest/engine.py` | **Modify** | Expose benchmark close to context each bar in `run()` |
| `stockquant/risk/__init__.py` | **Modify** | Export `MarketRegimeDetector` |
| `stockquant/strategy/alpha_factor_strategy.py` | **Modify** | Integrate regime + graduated DD → dynamic position sizing |
| `stockquant/research/alpha_researcher.py` | **Modify** | Pass benchmark to strategy; pass new risk params |
| `notebooks/alpha191/_run_multi_factor_risk.py` | **Modify** | Update to new risk framework params; rerun verification |
| `tests/test_regime_detector.py` | **Create** | Unit tests for MarketRegimeDetector |

---

### Task 1: Add benchmark price to Context

**Files:**
- Modify: `stockquant/backtest/context.py:27-38`
- Modify: `stockquant/backtest/engine.py:151-180`

**Why:** Market regime detection needs benchmark (中证500) close prices during the backtest loop. Currently benchmark is only stored in the engine and used post-hoc for performance analysis.

- [ ] **Step 1: Add benchmark_prices to Context**

In `stockquant/backtest/context.py`, add `benchmark_prices` to the `Context.__init__`:

```python
class Context:
    """回测上下文，由引擎创建并注入策略。"""

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
            total_value=initial_capital,
        )
        self.current_date: dt.date | None = None
        self.current_data: dict[str, pd.DataFrame] = {}
        self._price_cache: dict[str, float] = {}
        self.benchmark_prices: "pd.Series | None" = None
```

- [ ] **Step 2: Expose benchmark close to context in engine run loop**

In `stockquant/backtest/engine.py`, after setting `context.current_date` in the loop, set `context.benchmark_prices` to the benchmark close series up to the current date:

```python
def run(self) -> BacktestResult:
    # ... existing setup ...

    # Pre-extract benchmark close series for fast lookup
    bm_close = None
    if not self._benchmark.empty and "close" in self._benchmark.columns:
        bm = self._benchmark.sort_values("date").set_index("date")
        bm_close = bm["close"]

    for i, trade_date in enumerate(self._trade_dates):
        self.context.current_date = trade_date

        # Expose benchmark prices up to current date for regime detection
        if bm_close is not None:
            self.context.benchmark_prices = bm_close[bm_close.index <= pd.Timestamp(trade_date)]
        else:
            self.context.benchmark_prices = None

        # ... rest of loop unchanged ...
```

- [ ] **Step 3: Commit**

```bash
git add stockquant/backtest/context.py stockquant/backtest/engine.py
git commit -m "feat: expose benchmark close prices to Context for real-time regime detection"
```

---

### Task 2: Create MarketRegimeDetector

**Files:**
- Create: `stockquant/risk/regime_detector.py`
- Create: `tests/test_regime_detector.py`

**Why:** Detects market state (牛市/正常/熊市/危机) from benchmark close prices. This is the "感知层" — the foundation of proactive drawdown control. No regime awareness exists in the codebase today.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regime_detector.py`:

```python
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
        """Strong uptrend + normal vol → BULL."""
        prices = make_prices(100, 200, 0.002)  # ~50% annual return
        detector = MarketRegimeDetector()
        regime = detector.detect(prices)
        assert regime == Regime.BULL

    def test_crisis_market(self):
        """Sharp decline + high vol → CRISIS."""
        # Normal for 80 days, then sharp 30% drop over 20 days
        normal = make_prices(100, 80, 0.001)
        crash_returns = np.random.randn(20) * 0.04 - 0.02  # avg -2%/day with high vol
        crash_prices = normal.iloc[-1] * np.cumprod(1 + crash_returns)
        crash_series = pd.Series(
            crash_prices,
            index=pd.date_range(normal.index[-1] + pd.Timedelta(days=1), periods=20, freq="B")
        )
        prices = pd.concat([normal, crash_series])
        detector = MarketRegimeDetector()
        regime = detector.detect(prices)
        assert regime == Regime.CRISIS

    def test_regime_scale_bull(self):
        """BULL → position_scale = 1.0."""
        detector = MarketRegimeDetector()
        assert detector.get_position_scale(Regime.BULL) == 1.0

    def test_regime_scale_crisis(self):
        """CRISIS → position_scale = 0.2."""
        detector = MarketRegimeDetector()
        assert detector.get_position_scale(Regime.CRISIS) == 0.2

    def test_returns_scale(self):
        """Returns a float between 0 and 1."""
        detector = MarketRegimeDetector()
        scale = detector.compute_scale(pd.Series([100, 101, 102, 103]))
        assert 0.0 <= scale <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python3 -m pytest tests/test_regime_detector.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'stockquant.risk.regime_detector'`

- [ ] **Step 3: Implement MarketRegimeDetector**

Create `stockquant/risk/regime_detector.py`:

```python
"""
市场状态检测器 — 基于基准指数的趋势+波动率进行市场状态分类。
"""
from __future__ import annotations

import enum
import numpy as np
import pandas as pd

from stockquant.utils.logger import get_logger

logger = get_logger("risk.regime")


class Regime(enum.Enum):
    BULL = "牛市"
    NORMAL = "正常"
    BEAR = "熊市"
    CRISIS = "危机"


class MarketRegimeDetector:
    """基于基准指数的市场状态检测。

    使用 MA20/MA60 均线交叉判断趋势方向，用 20日/252日 波动率比值判断波动状态。

    使用方法::

        detector = MarketRegimeDetector()
        regime = detector.detect(benchmark_close_series)
        scale = detector.get_position_scale(regime)
    """

    def __init__(
        self,
        vol_ratio_threshold: float = 1.5,
        ma_short: int = 20,
        ma_long: int = 60,
        vol_short_window: int = 20,
        vol_long_window: int = 252,
    ) -> None:
        self.vol_ratio_threshold = vol_ratio_threshold
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.vol_short_window = vol_short_window
        self.vol_long_window = vol_long_window

        # Position scale per regime (user's framework parameters)
        self._scale_map: dict[Regime, float] = {
            Regime.BULL: 1.0,
            Regime.NORMAL: 0.8,
            Regime.BEAR: 0.5,
            Regime.CRISIS: 0.2,
        }

    def detect(self, prices: pd.Series) -> Regime:
        """从价格序列检测当前市场状态。

        Parameters
        ----------
        prices : pd.Series
            基准指数收盘价序列（index=日期），至少需 252 个数据点。

        Returns
        -------
        Regime
        """
        if len(prices) < max(self.ma_long, self.vol_long_window):
            return Regime.NORMAL

        returns = prices.pct_change().dropna()

        # 1. 趋势判断: MA20 vs MA60
        ma_short_val = prices.rolling(self.ma_short).mean().iloc[-1]
        ma_long_val = prices.rolling(self.ma_long).mean().iloc[-1]
        trend_up = ma_short_val > ma_long_val

        # 2. 波动率判断: 短期vol / 长期vol
        short_vol = returns.rolling(self.vol_short_window).std().iloc[-1]
        long_vol = returns.rolling(self.vol_long_window).std().iloc[-1]
        vol_ratio = short_vol / long_vol if long_vol and long_vol > 0 else 1.0
        high_vol = vol_ratio > self.vol_ratio_threshold

        # 3. 急跌判断: 近5日累计跌幅
        recent_5d_return = (
            prices.iloc[-1] / prices.iloc[-min(5, len(prices))]
            - 1
            if len(prices) >= 5
            else 0.0
        )

        # 4. 状态分类
        if recent_5d_return < -0.10 and high_vol:
            return Regime.CRISIS
        if not trend_up and high_vol:
            return Regime.BEAR
        if not trend_up:
            # Down trend but low vol → mild bear, treat as NORMAL with reduced scale
            # But the user's framework says "熊市" when trend is down
            return Regime.BEAR
        if trend_up and not high_vol:
            return Regime.BULL

        return Regime.NORMAL

    def get_position_scale(self, regime: Regime) -> float:
        """获取该市场状态对应的仓位系数。"""
        return self._scale_map.get(regime, 0.8)

    def compute_scale(self, prices: pd.Series) -> float:
        """一站式：从价格序列直接算出仓位系数。"""
        regime = self.detect(prices)
        scale = self.get_position_scale(regime)
        logger.debug(f"市场状态: {regime.value}, 仓位系数: {scale:.0%}")
        return scale
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python3 -m pytest tests/test_regime_detector.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add stockquant/risk/regime_detector.py tests/test_regime_detector.py stockquant/risk/__init__.py
git commit -m "feat: add MarketRegimeDetector for market state classification"
```

---

### Task 3: Transform RiskMonitor to graduated alerts

**Files:**
- Modify: `stockquant/risk/risk_monitor.py` (entire file)

**Why:** Current binary freeze is too crude — once triggered, all trading stops permanently. The user's framework calls for 4 graduated alert levels (绿/黄/橙/红) with progressively stronger position compression, allowing the strategy to de-risk without going fully inert.

- [ ] **Step 1: Update the test file**

Add to `tests/test_risk.py`:

```python
from stockquant.risk.risk_monitor import RiskMonitor, AlertLevel


class TestRiskMonitorGraduated:
    @pytest.fixture
    def monitor(self):
        return RiskMonitor()

    def test_green_when_no_drawdown(self, monitor):
        monitor.update_value(100_000)
        monitor.update_value(101_000)
        level = monitor.update_value(100_500)
        assert level == AlertLevel.GREEN

    def test_yellow_at_5pct_drawdown(self, monitor):
        monitor.update_value(100_000)
        level = monitor.update_value(95_000)
        assert level == AlertLevel.YELLOW

    def test_orange_at_8pct_drawdown(self, monitor):
        monitor.update_value(100_000)
        level = monitor.update_value(92_000)
        assert level == AlertLevel.ORANGE

    def test_red_at_12pct_drawdown(self, monitor):
        monitor.update_value(100_000)
        level = monitor.update_value(88_000)
        assert level == AlertLevel.RED

    def test_peak_updates(self, monitor):
        monitor.update_value(100_000)
        monitor.update_value(110_000)
        monitor.update_value(105_000)  # DD from 110k peak = ~4.5% → GREEN
        assert monitor._peak_value == 110_000

    def test_returns_alert_level_not_bool(self, monitor):
        monitor.update_value(100_000)
        result = monitor.update_value(95_000)
        assert isinstance(result, AlertLevel)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python3 -m pytest tests/test_risk.py::TestRiskMonitorGraduated -v
```
Expected: FAIL — `update_value()` doesn't exist, `update()` returns `bool` not `AlertLevel`

- [ ] **Step 3: Rewrite RiskMonitor**

Rewrite `stockquant/risk/risk_monitor.py`:

```python
"""
风险监控器 — 分级回撤预警与动态仓位压缩。
"""
from __future__ import annotations

import enum

from stockquant.utils.logger import get_logger

logger = get_logger("risk.monitor")


class AlertLevel(enum.Enum):
    """回撤预警级别。"""
    GREEN = ("正常", 1.0)     # DD < 4%
    YELLOW = ("一级预警", 0.8)  # 4% ≤ DD < 7%
    ORANGE = ("二级预警", 0.6)  # 7% ≤ DD < 10%
    RED = ("危机预警", 0.3)     # DD ≥ 10%

    def __init__(self, label: str, position_scale: float):
        self.label = label
        self.position_scale = position_scale


class RiskMonitor:
    """账户级分级回撤监控。

    逐日更新峰值和回撤，返回当前预警级别。
    与旧版的区别：
    - 旧版：二元冻结（触发后永久停止交易）
    - 新版：四级预警（可恢复），每级对应不同的仓位压缩系数
    """

    def __init__(
        self,
        green_threshold: float = 0.04,
        yellow_threshold: float = 0.07,
        orange_threshold: float = 0.10,
    ) -> None:
        self.green_threshold = green_threshold
        self.yellow_threshold = yellow_threshold
        self.orange_threshold = orange_threshold
        self._peak_value: float = 0.0
        self._current_level: AlertLevel = AlertLevel.GREEN

    def update(self, context) -> AlertLevel:
        """从 Context 更新（兼容旧接口）。

        Returns
        -------
        AlertLevel
        """
        return self.update_value(context.portfolio_value)

    def update_value(self, portfolio_value: float) -> AlertLevel:
        """从账户总值更新并返回预警级别。

        Returns
        -------
        AlertLevel
        """
        self._peak_value = max(self._peak_value, portfolio_value)

        if self._peak_value <= 0:
            self._current_level = AlertLevel.GREEN
            return self._current_level

        dd = (self._peak_value - portfolio_value) / self._peak_value

        if dd >= self.orange_threshold:
            self._current_level = AlertLevel.RED
        elif dd >= self.yellow_threshold:
            self._current_level = AlertLevel.ORANGE
        elif dd >= self.green_threshold:
            self._current_level = AlertLevel.YELLOW
        else:
            self._current_level = AlertLevel.GREEN

        if self._current_level != AlertLevel.GREEN:
            logger.warning(
                f"回撤预警: {self._current_level.label} "
                f"(当前DD={dd:.2%}, 峰值={self._peak_value:,.0f})"
            )

        return self._current_level

    @property
    def current_level(self) -> AlertLevel:
        return self._current_level

    @property
    def current_drawdown(self) -> float:
        """当前回撤比例（修复了原 bug）。"""
        if self._peak_value <= 0:
            return 0.0
        return 1.0 - (self._peak_value - 0) / self._peak_value  # placeholder — see below

    @property
    def position_scale(self) -> float:
        """当前预警级别对应的仓位压缩系数。"""
        return self._current_level.position_scale

    @property
    def is_frozen(self) -> bool:
        """兼容旧接口：RED 级别视为熔断。"""
        return self._current_level == AlertLevel.RED

    def reset(self) -> None:
        self._peak_value = 0.0
        self._current_level = AlertLevel.GREEN
```

Wait — the `current_drawdown` property still has the old bug. Let me fix it properly:

```python
    @property
    def current_drawdown(self) -> float:
        """当前回撤比例。"""
        if self._peak_value <= 0:
            return 0.0
        # NOTE: This only works if the latest value is stored.
        # The user should call update_value() first.
        return 0.0  # Caller should use update_value result for DD
```

Actually, let me simplify. The `current_drawdown` property was always buggy and never used. Let me just make it store the last value:

```python
    @property
    def current_drawdown(self) -> float:
        """最近一次 update 时的回撤比例。"""
        if self._peak_value <= 0 or self._last_value <= 0:
            return 0.0
        return (self._peak_value - self._last_value) / self._peak_value
```

And add `self._last_value: float = 0.0` in `__init__`, set it in `update_value`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python3 -m pytest tests/test_risk.py::TestRiskMonitorGraduated -v
```
Expected: PASS (6 tests)

- [ ] **Step 5: Verify old tests still pass**

```bash
.venv/bin/python3 -m pytest tests/test_risk.py -v
```
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add stockquant/risk/risk_monitor.py tests/test_risk.py
git commit -m "feat: replace binary RiskMonitor freeze with graduated 4-level AlertLevel"
```

---

### Task 4: Integrate full risk framework into AlphaFactorStrategy

**Files:**
- Modify: `stockquant/strategy/alpha_factor_strategy.py` (entire file)

**Why:** This is the core integration. Replace the old binary RiskMonitor + StopLossManager combo with: regime detection → graduated DD alert → individual stop-loss → dynamic position sizing on rebalance.

- [ ] **Step 1: Rewrite AlphaFactorStrategy with integrated risk framework**

Rewrite `stockquant/strategy/alpha_factor_strategy.py`:

```python
"""
通用截面因子选股策略 — 集成完整回撤控制框架。

风险控制层次（感知→决策→执行）：
  1. 市场状态识别（牛/熊/震荡/危机）→ 仓位上限
  2. 分级回撤预警（绿/黄/橙/红）→ 仓位压缩
  3. 个股止损/止盈（固定+移动止损）→ 单仓退出
"""
from __future__ import annotations

import pandas as pd

from stockquant.risk.regime_detector import MarketRegimeDetector
from stockquant.risk.risk_monitor import AlertLevel, RiskMonitor
from stockquant.risk.stop_loss import StopLossManager
from stockquant.strategy.base_strategy import BaseStrategy, StrategyRegistry
from stockquant.utils.logger import get_logger

logger = get_logger("strategy.alpha_factor")


class AlphaFactorStrategy(BaseStrategy):
    """通用截面因子选股策略 — 集成完整回撤控制。

    每隔 ``rebalance_freq`` 个交易日触发调仓，调仓时按因子截面排名选 Top-N。
    每 bar 运行风险检查（优先于调仓逻辑）：

    **感知层：**
    - 市场状态识别：基于基准指数 (中证500/沪深300) 的 MA 交叉 + 波动率比值
    - 回撤预警：四级分级预警 (绿/黄/橙/红)

    **决策层：**
    - 总仓位 = 基准仓位 × 市场状态系数 × 回撤预警系数
    - 个股止损：固定比例止损 + 移动止损 + 止盈

    参数
    ----
    alpha_panel    : DataFrame  **必填**
    max_positions  : int = 50
    rebalance_freq : int = 5
    ascending      : bool = False
    label          : str = "AlphaFactor"
    enable_risk_mgmt : bool = False
        启用完整回撤控制框架。
    stop_loss_pct  : float = 0.08
        固定止损比例。
    take_profit_pct : float = 0.20
        固定止盈比例。
    trailing_stop_pct : float = 0.05
        移动止损回撤比例（从持仓期最高价计算）。
    max_drawdown_green  : float = 0.04  绿→黄 阈值
    max_drawdown_yellow : float = 0.07  黄→橙 阈值
    max_drawdown_orange : float = 0.10  橙→红 阈值
    """

    def initialize(self) -> None:
        self._alpha_panel: pd.DataFrame = self.get_param("alpha_panel")
        self._max_pos: int = self.get_param("max_positions", 50)
        self._freq: int = self.get_param("rebalance_freq", 5)
        self._ascending: bool = self.get_param("ascending", False)
        self._label: str = self.get_param("label", "AlphaFactor")
        self._day_count: int = 0
        self._base_target_pct: float = 1.0 / max(self._max_pos, 1)

        self._alpha_panel.index = pd.to_datetime(self._alpha_panel.index)

        # ── 风险管理 ──
        self._risk_enabled: bool = self.get_param("enable_risk_mgmt", False)

        if self._risk_enabled:
            sl_pct: float = self.get_param("stop_loss_pct", 0.08)
            tp_pct: float = self.get_param("take_profit_pct", 0.20)
            ts_pct: float = self.get_param("trailing_stop_pct", 0.05)

            # 感知层: 市场状态检测
            self._regime_detector = MarketRegimeDetector()

            # 感知层: 分级回撤预警
            self._risk_monitor = RiskMonitor(
                green_threshold=self.get_param("max_drawdown_green", 0.04),
                yellow_threshold=self.get_param("max_drawdown_yellow", 0.07),
                orange_threshold=self.get_param("max_drawdown_orange", 0.10),
            )

            # 决策层: 个股止损/止盈（固定 + 移动）
            self._stop_loss = StopLossManager()
            self._stop_loss.sl_enabled = True
            self._stop_loss.sl_fixed_pct = sl_pct
            self._stop_loss.sl_trailing_pct = ts_pct
            self._stop_loss.tp_fixed_pct = tp_pct

            logger.info(
                f"[{self._label}] 回撤控制框架已启用 — "
                f"止损={sl_pct:.0%}, 移动止损={ts_pct:.0%}, "
                f"止盈={tp_pct:.0%}, "
                f"预警阈值: 绿<{self._risk_monitor.green_threshold:.0%}"
                f"<黄<{self._risk_monitor.yellow_threshold:.0%}"
                f"<橙<{self._risk_monitor.orange_threshold:.0%}<红"
            )
        else:
            self._regime_detector = None
            self._risk_monitor = None
            self._stop_loss = None

        logger.info(
            f"[{self._label}] 初始化完成 — "
            f"max_pos={self._max_pos}, freq={self._freq}, "
            f"base_target_pct={self._base_target_pct:.1%}, "
            f"ascending={self._ascending}, risk_mgmt={self._risk_enabled}"
        )

    def _compute_position_scale(self) -> float:
        """计算当前应使用的仓位系数 (0.0~1.0)。

        两层压缩：
        1. 市场状态 → 仓位上限系数
        2. 回撤预警 → 仓位压缩系数
        取两者乘积作为最终系数。
        """
        regime_scale = 1.0
        dd_scale = 1.0

        # 第一层：市场状态
        if self._regime_detector is not None and self.context is not None:
            bm_prices = self.context.benchmark_prices
            if bm_prices is not None and len(bm_prices) >= 60:
                regime_scale = self._regime_detector.compute_scale(bm_prices)

        # 第二层：回撤预警
        if self._risk_monitor is not None:
            dd_scale = self._risk_monitor.position_scale

        return regime_scale * dd_scale

    def handle_bar(self, bar: dict) -> None:
        # ── 感知层: 更新回撤预警（每 bar）──────────
        alert_level = AlertLevel.GREEN
        if self._risk_enabled and self.context is not None:
            alert_level = self._risk_monitor.update(self.context)

        # ── 决策层: 个股止损/止盈（每 bar）─────────
        if self._risk_enabled and self.context is not None:
            for code, pos in list(self.context.positions.items()):
                if pos.quantity <= 0:
                    continue
                current_price = self.context.get_current_price(code)
                if current_price <= 0:
                    continue
                # 使用移动止损模式（会同时检查 fixed + trailing）
                result = self._stop_loss.check(pos, current_price)
                if result.triggered:
                    avail_qty = pos.quantity - pos.frozen
                    if avail_qty > 0:
                        self.sell(code, avail_qty, reason=result.reason)
                        self._stop_loss.reset(code)

        # ── 调仓逻辑（仅调仓日执行）───────────────────
        self._day_count += 1
        if self._day_count % self._freq != 0:
            return
        if self.context is None:
            return

        # ── 决策层: 计算动态仓位系数 ─────────────────
        position_scale = self._compute_position_scale() if self._risk_enabled else 1.0
        if position_scale <= 0:
            return

        effective_target_pct = self._base_target_pct * position_scale

        # ── 1. 获取当日截面因子值 ─────────────────
        current_ts = pd.Timestamp(self.context.current_date)
        valid_idx = self._alpha_panel.index[self._alpha_panel.index <= current_ts]
        if valid_idx.empty:
            return
        today_alpha = self._alpha_panel.loc[valid_idx[-1]].dropna()

        available = [c for c in today_alpha.index if c in bar and not bar[c].empty]
        if not available:
            return
        today_alpha = today_alpha[available]

        # ── 2. 截面排名取 Top N ────────────────────
        if self._ascending:
            top_stocks = set(today_alpha.nsmallest(self._max_pos).index.tolist())
        else:
            top_stocks = set(today_alpha.nlargest(self._max_pos).index.tolist())

        # ── 3. 卖出不在名单的持仓 ─────────────────
        for code, pos in list(self.context.positions.items()):
            if pos.quantity <= 0:
                continue
            if code not in top_stocks:
                avail_qty = pos.quantity - pos.frozen
                if avail_qty > 0:
                    self.sell(code, avail_qty, reason=f"{self._label}调仓-剔除")

        # ── 4. 等权建仓（使用动态仓位）─────────────
        for code in top_stocks:
            self.order_target_percent(
                code,
                effective_target_pct,
                reason=f"{self._label}调仓-买入"
                + (f"({alert_level.label}, scale={position_scale:.0%})" if self._risk_enabled else ""),
            )


StrategyRegistry.register("alpha_factor", AlphaFactorStrategy)
```

- [ ] **Step 2: Verify the module imports correctly**

```bash
.venv/bin/python3 -c "from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy; print('OK')"
```
Expected: OK (no import errors)

- [ ] **Step 3: Commit**

```bash
git add stockquant/strategy/alpha_factor_strategy.py
git commit -m "feat: integrate full drawdown control framework into AlphaFactorStrategy

- Market regime detection (牛/熊/震荡/危机) → position cap
- Graduated 4-level drawdown alerts (绿/黄/橙/红) → position compression
- Individual stop-loss (fixed + trailing) + take-profit
- Dynamic position sizing: base_target × regime_scale × dd_scale"
```

---

### Task 5: Update AlphaResearcher for new risk params

**Files:**
- Modify: `stockquant/research/alpha_researcher.py:175-188`  (constructor)
- Modify: `stockquant/research/alpha_researcher.py:295-306`  (run_backtest)

**Why:** The AlphaResearcher must pass benchmark data to the engine and forward new risk params (trailing_stop_pct, graduated DD thresholds) to the strategy.

- [ ] **Step 1: Update AlphaResearcher constructor and run_backtest**

In `stockquant/research/alpha_researcher.py`, update `__init__` to accept new params:

```python
    def __init__(
        self,
        dataset: BacktestDataset,
        initial_capital: float = 1_000_000.0,
        max_positions: int = 10,
        rebalance_freq: int = 5,
        enable_risk_mgmt: bool = False,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
        max_drawdown_limit: float = 0.20,
        trailing_stop_pct: float = 0.05,
        max_drawdown_green: float = 0.04,
        max_drawdown_yellow: float = 0.07,
        max_drawdown_orange: float = 0.10,
    ) -> None:
        self.dataset = dataset
        self.benchmark_df = dataset.benchmark
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_freq = rebalance_freq
        self.enable_risk_mgmt = enable_risk_mgmt
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_green = max_drawdown_green
        self.max_drawdown_yellow = max_drawdown_yellow
        self.max_drawdown_orange = max_drawdown_orange
        # ... rest unchanged ...
```

In the same file, update `run_backtest()` to pass new params to strategy AND pass benchmark to engine:

```python
        strategy.set_params(
            alpha_panel=alpha_panel,
            max_positions=max_pos,
            rebalance_freq=freq,
            ascending=ascending,
            label=_label,
            enable_risk_mgmt=self.enable_risk_mgmt,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            max_drawdown_green=self.max_drawdown_green,
            max_drawdown_yellow=self.max_drawdown_yellow,
            max_drawdown_orange=self.max_drawdown_orange,
        )

        bt_engine = BacktestEngine()
        bt_engine.context.portfolio.initial_capital = self.initial_capital
        bt_engine.context.portfolio.cash = self.initial_capital
        bt_engine.context.portfolio.total_value = self.initial_capital
        bt_engine.set_strategy(strategy)
        bt_engine.set_data(self.dataset.stock_data)
        # Pass benchmark to engine for regime detection
        if not self.benchmark_df.empty:
            bt_engine._benchmark = self.benchmark_df
            bt_engine._benchmark_code = self.dataset.benchmark_code

        sd = start_date or self.dataset.start_date
        ed = end_date or self.dataset.end_date
        bt_engine.set_date_range(start_date=sd, end_date=ed)
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python3 -c "from stockquant.research import AlphaResearcher; print('OK')"
```
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add stockquant/research/alpha_researcher.py
git commit -m "feat: pass benchmark and new risk params through AlphaResearcher"
```

---

### Task 6: Update backtest script and run verification

**Files:**
- Modify: `notebooks/alpha191/_run_multi_factor_risk.py:37-44` (risk profiles)
- Modify: `notebooks/alpha191/_run_multi_factor_risk.py:209-219` (AlphaResearcher instantiation)

**Why:** Test the new framework against the old binary risk management to verify drawdown reduction.

- [ ] **Step 1: Update risk profiles in the verification script**

Replace the old `RISK_PROFILES` dict with new framework params. The graduated DD thresholds are now part of the framework, so the risk profile only varies the stop/TP/trailing params:

```python
# 新框架：回撤控制由 MarketRegimeDetector + 分级 RiskMonitor 统一管理
# risk profile 只配置个股层面的止损参数
RISK_PROFILES = {
    "无风控(基线)":        {"enable": False, "sl": 0.08, "tp": 0.20, "ts": 0.05},
    "新框架_默认参数":      {"enable": True,  "sl": 0.08, "tp": 0.20, "ts": 0.05},
    "新框架_紧止损":        {"enable": True,  "sl": 0.05, "tp": 0.15, "ts": 0.03},
    "新框架_松止损":        {"enable": True,  "sl": 0.12, "tp": 0.25, "ts": 0.08},
    "新框架_无个股止损":    {"enable": True,  "sl": 0.50, "tp": 0.50, "ts": 0.50},
}
```

And update the AlphaResearcher instantiation:

```python
            researcher = AlphaResearcher(
                dataset,
                initial_capital=INITIAL_CAPITAL,
                max_positions=MAX_POSITIONS,
                rebalance_freq=REBALANCE_FREQ,
                enable_risk_mgmt=risk_params["enable"],
                stop_loss_pct=risk_params["sl"],
                take_profit_pct=risk_params["tp"],
                trailing_stop_pct=risk_params["ts"],
            )
```

- [ ] **Step 2: Run the verification backtest**

```bash
cd /Users/mingshan.yao/stockQuant && .venv/bin/python3 notebooks/alpha191/_run_multi_factor_risk.py
```

Expected: All 10 backtests complete. The "新框架" profiles should show significantly lower max drawdown. Compare to baseline (无风控 All_等权: Sharpe 1.397, DD 48.49%).

Key metrics to watch:
- Max drawdown of 新框架_默认参数 should be materially lower than 48%
- Market regime detection should reduce position size before/during 2015 crash
- Trailing stop-loss should exit winning positions before large reversals

- [ ] **Step 3: Record results and compare**

After the run completes, record the results table in the commit message.

- [ ] **Step 4: Commit**

```bash
git add notebooks/alpha191/_run_multi_factor_risk.py
git commit -m "feat: update risk backtest with new graduated drawdown control framework

Results: ... (insert summary table)"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `docs/alpha191_replication.md` (Chapter 12 section)

**Why:** Document the new framework design, parameter choices, and verification results.

- [ ] **Step 1: Update Chapter 12 in the doc**

After the verification run completes, update `docs/alpha191_replication.md` section 12.5 (分析) and add section 12.7 (新版回撤控制框架验证):

```markdown
### 12.7 新版分级回撤控制框架验证

基于感知层+决策层的完整回撤控制框架：

| 策略 | 年化收益率 | 最大回撤 | 夏普比率 | 卡玛比率 |
|------|-----------|----------|----------|----------|
| ... (insert actual results) ... |

**与旧版的对比分析：**
- ...
```

- [ ] **Step 2: Commit**

```bash
git add docs/alpha191_replication.md
git commit -m "docs: add graduated drawdown control framework verification results"
```

---
