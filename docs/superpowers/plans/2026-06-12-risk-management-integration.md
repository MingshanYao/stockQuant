# Risk Management Integration for AlphaFactorStrategy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate existing risk management modules (StopLossManager, RiskMonitor) into AlphaFactorStrategy, then re-run multi-factor backtests to verify drawdown control.

**Architecture:** Modify `AlphaFactorStrategy` to optionally enable risk management via params. Stop-loss/take-profit checks run every bar (not just rebalance days). Portfolio-level drawdown circuit breaker halts all trading when triggered. PositionManager is not needed since the factor strategy already sizes positions via `max_positions`.

**Tech Stack:** Python 3.12, pandas, existing stockquant framework

---

### Task 1: Add risk management to AlphaFactorStrategy

**Files:**
- Modify: `stockquant/strategy/alpha_factor_strategy.py` (entire file)

- [ ] **Step 1: Rewrite AlphaFactorStrategy with risk management**

Replace the contents of `stockquant/strategy/alpha_factor_strategy.py`:

```python
"""
通用截面因子选股策略 — 基于任意 Alpha 因子面板的等权轮动策略。

支持可选的风险管理:
  - 个股止损/止盈（每 bar 检查）
  - 账户级最大回撤熔断
"""

from __future__ import annotations

import pandas as pd

from stockquant.strategy.base_strategy import BaseStrategy, StrategyRegistry
from stockquant.risk.stop_loss import StopLossManager, StopResult
from stockquant.risk.risk_monitor import RiskMonitor
from stockquant.utils.logger import get_logger

logger = get_logger("strategy.alpha_factor")


class AlphaFactorStrategy(BaseStrategy):
    """通用截面因子选股策略。

    每隔 ``rebalance_freq`` 个交易日触发调仓：
    1. 检查账户级回撤熔断（若启用）
    2. 检查个股止盈止损（若启用，每 bar 执行）
    3. 获取当日截面因子值，按排名选 Top-N；
    4. 卖出不在名单的持仓；
    5. 对名单内股票等权调仓。

    参数（通过 :meth:`set_params` 设置）
    -------------------------------------
    alpha_panel    : DataFrame  **必填**  ``index=日期, columns=股票代码``
    max_positions  : int = 10   最多同时持仓只数
    rebalance_freq : int = 5    调仓频率（交易日数）
    ascending      : bool = False
        因子排序方向。``False`` 表示值越大越好，``True`` 表示值越小越好。
    label          : str = "AlphaFactor"
        策略标签。

    enable_risk_mgmt : bool = False
        是否启用风险管理（止损 + 回撤熔断）。
    stop_loss_pct    : float = 0.08
        固定止损比例（仅 enable_risk_mgmt=True 时生效）。
    take_profit_pct  : float = 0.20
        固定止盈比例。
    max_drawdown_limit : float = 0.20
        账户最大回撤熔断阈值。
    """

    def initialize(self) -> None:
        self._alpha_panel: pd.DataFrame = self.get_param("alpha_panel")
        self._max_pos: int = self.get_param("max_positions", 10)
        self._freq: int = self.get_param("rebalance_freq", 5)
        self._ascending: bool = self.get_param("ascending", False)
        self._label: str = self.get_param("label", "AlphaFactor")
        self._day_count: int = 0
        self._target_pct: float = 1.0 / max(self._max_pos, 1)

        # 风险管理
        self._risk_enabled: bool = self.get_param("enable_risk_mgmt", False)
        if self._risk_enabled:
            self._stop_loss = StopLossManager()
            self._stop_loss.sl_fixed_pct = self.get_param("stop_loss_pct", 0.08)
            self._stop_loss.tp_fixed_pct = self.get_param("take_profit_pct", 0.20)

            self._risk_monitor = RiskMonitor()
            self._risk_monitor.max_drawdown_limit = self.get_param(
                "max_drawdown_limit", 0.20
            )
            logger.info(
                f"[{self._label}] 风险管理已启用 — "
                f"止损={self._stop_loss.sl_fixed_pct:.0%}, "
                f"止盈={self._stop_loss.tp_fixed_pct:.0%}, "
                f"回撤熔断={self._risk_monitor.max_drawdown_limit:.0%}"
            )
        else:
            self._stop_loss = None
            self._risk_monitor = None

        self._alpha_panel.index = pd.to_datetime(self._alpha_panel.index)

        logger.info(
            f"[{self._label}] 初始化完成 — "
            f"max_pos={self._max_pos}, freq={self._freq}, "
            f"target_pct={self._target_pct:.1%}, ascending={self._ascending}"
        )

    def handle_bar(self, bar: dict) -> None:
        # ── 风险管理（每 bar 执行，优先于调仓逻辑） ──────────────
        if self._risk_enabled and self.context is not None:
            # 1. 账户级回撤熔断
            if self._risk_monitor.update(self.context):
                return

            # 2. 个股止损/止盈
            for code, pos in list(self.context.positions.items()):
                if pos.quantity <= 0:
                    continue
                current_price = self.context.get_current_price(code)
                if current_price <= 0:
                    continue
                result = self._stop_loss.check(pos, current_price)
                if result.triggered:
                    avail_qty = pos.quantity - pos.frozen
                    if avail_qty > 0:
                        self.sell(code, avail_qty, reason=result.reason)
                        logger.info(
                            f"[{self._label}] {result.reason} → 卖出 {code} "
                            f"({avail_qty}股)"
                        )

        # ── 调仓逻辑（仅在调仓日执行） ──────────────────────────
        self._day_count += 1
        if self._day_count % self._freq != 0:
            return

        if self.context is None:
            return

        # 1. 获取当日截面因子值
        current_ts = pd.Timestamp(self.context.current_date)
        valid_idx = self._alpha_panel.index[self._alpha_panel.index <= current_ts]
        if valid_idx.empty:
            return
        today_alpha = self._alpha_panel.loc[valid_idx[-1]].dropna()

        available = [c for c in today_alpha.index if c in bar and not bar[c].empty]
        if not available:
            return
        today_alpha = today_alpha[available]

        # 2. 截面排名取 Top N
        if self._ascending:
            top_stocks = set(today_alpha.nsmallest(self._max_pos).index.tolist())
        else:
            top_stocks = set(today_alpha.nlargest(self._max_pos).index.tolist())

        # 3. 卖出不在目标名单的持仓
        for code, pos in list(self.context.positions.items()):
            if pos.quantity <= 0:
                continue
            if code not in top_stocks:
                avail_qty = pos.quantity - pos.frozen
                if avail_qty > 0:
                    self.sell(code, avail_qty, reason=f"{self._label}调仓-剔除")

        # 4. 等权建仓
        for code in top_stocks:
            self.order_target_percent(
                code,
                self._target_pct,
                reason=f"{self._label}调仓-买入",
            )


StrategyRegistry.register("alpha_factor", AlphaFactorStrategy)
```

- [ ] **Step 2: Verify the modified file has no syntax errors**

```bash
.venv/bin/python3 -c "from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy; print('OK')"
```

Expected: prints `OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add stockquant/strategy/alpha_factor_strategy.py
git commit -m "feat: add optional risk management (stop-loss + circuit breaker) to AlphaFactorStrategy"
```

---

### Task 2: Add risk params passthrough to AlphaResearcher.run_backtest()

**Files:**
- Modify: `stockquant/research/alpha_researcher.py:287-294`

- [ ] **Step 1: Add risk params to strategy setup in run_backtest()**

Edit `stockquant/research/alpha_researcher.py`, replace lines 287-294 (the strategy.set_params call):

```python
        # 构建策略
        strategy = AlphaFactorStrategy()
        strategy.set_params(
            alpha_panel=alpha_panel,
            max_positions=max_pos,
            rebalance_freq=freq,
            ascending=ascending,
            label=_label,
            enable_risk_mgmt=self.enable_risk_mgmt,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_drawdown_limit=self.max_drawdown_limit,
        )
```

- [ ] **Step 2: Add risk params to AlphaResearcher.__init__()**

Edit `stockquant/research/alpha_researcher.py:169-175`. Current signature:

```python
    def __init__(
        self,
        dataset: "BacktestDataset",
        initial_capital: float = 1_000_000.0,
        max_positions: int = 10,
        rebalance_freq: int = 5,
    ) -> None:
```

Add 4 risk parameters after `rebalance_freq`:

```python
    def __init__(
        self,
        dataset: "BacktestDataset",
        initial_capital: float = 1_000_000.0,
        max_positions: int = 10,
        rebalance_freq: int = 5,
        enable_risk_mgmt: bool = False,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
        max_drawdown_limit: float = 0.20,
    ) -> None:
```

Then add after line 180 (`self.rebalance_freq = rebalance_freq`):

```python
        self.enable_risk_mgmt = enable_risk_mgmt
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
```

- [ ] **Step 3: Verify**

```bash
.venv/bin/python3 -c "from stockquant.research import AlphaResearcher; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add stockquant/research/alpha_researcher.py
git commit -m "feat: pass risk management params through AlphaResearcher to strategy"
```

---

### Task 3: Create multi-factor backtest script with risk management

**Files:**
- Create: `notebooks/alpha191/_run_multi_factor_risk.py`

- [ ] **Step 1: Create the risk-managed backtest script**

This is a copy of `_run_multi_factor.py` with risk management enabled and multiple risk parameter sets tested. Create `notebooks/alpha191/_run_multi_factor_risk.py`:

```python
"""多因子组合回测 — 含风险管理（止损 + 回撤熔断）。

在 _run_multi_factor.py 的基础上，测试不同风险参数下的回撤控制效果。
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stockquant.data.database import Database
from stockquant.data.universe import BacktestDataset
from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS
from stockquant.analysis.evaluator import FactorEvaluator
from stockquant.research import AlphaResearcher

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

START_DATE      = "2010-01-01"
END_DATE        = "2016-12-31"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS   = 50
REBALANCE_FREQ  = 5

ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS]
OUTPUT_DIR = "notebooks/alpha191/replication"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 风险参数组合（按激进程度排序）
RISK_PROFILES = {
    "无风控(基线)":    {"enable": False, "sl": 0.08, "tp": 0.20, "dd": 0.20},
    "宽松风控":        {"enable": True,  "sl": 0.15, "tp": 0.30, "dd": 0.25},
    "适度风控":        {"enable": True,  "sl": 0.10, "tp": 0.20, "dd": 0.15},
    "严格风控":        {"enable": True,  "sl": 0.08, "tp": 0.15, "dd": 0.10},
    "极严风控(5%回撤)": {"enable": True,  "sl": 0.05, "tp": 0.10, "dd": 0.05},
}

print("=" * 70)
print("Alpha191 多因子组合回测 — 风险管理测试")
print("=" * 70)
print(f"风险配置组数: {len(RISK_PROFILES)}")
for name, p in RISK_PROFILES.items():
    if p["enable"]:
        print(f"  {name}: 止损={p['sl']:.0%}, 止盈={p['tp']:.0%}, 回撤熔断={p['dd']:.0%}")
    else:
        print(f"  {name}")

# ======================================================================
# 1. 加载数据
# ======================================================================
print("\n第1步: 加载数据...")

db = Database()
all_bars = db.query(
    "SELECT * FROM daily_bars WHERE date >= ? AND date <= ? ORDER BY code, date",
    [START_DATE, END_DATE],
)
all_bars["date"] = pd.to_datetime(all_bars["date"])

all_codes = all_bars["code"].unique().tolist()
all_codes = [c for c in all_codes if not c.startswith(("688", "43", "83", "87"))]

stock_data = {}
for code in all_codes:
    stock_data[code] = all_bars[all_bars["code"] == code].reset_index(drop=True)
del all_bars

benchmark_df = db.query(
    "SELECT * FROM index_daily WHERE code = '000300' AND date >= ? AND date <= ? ORDER BY date",
    [START_DATE, END_DATE],
)
if benchmark_df.empty:
    benchmark_df = db.query(
        "SELECT * FROM index_daily WHERE code = '000905' AND date >= ? AND date <= ? ORDER BY date",
        [START_DATE, END_DATE],
    )
    benchmark_code = "000905"
else:
    benchmark_code = "000300"
benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

dataset = BacktestDataset(
    stock_data=stock_data,
    codes=list(stock_data.keys()),
    benchmark=benchmark_df,
    benchmark_code=benchmark_code,
    start_date=START_DATE,
    end_date=END_DATE,
)
print(f"  {len(stock_data)} 只股票, {len(benchmark_df)} 个交易日")

# ======================================================================
# 2. 计算因子 & 评价
# ======================================================================
print("\n第2步: 计算 Alpha191 因子...")
engine = Alpha191Indicators.from_dataset(dataset)
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
print(f"  计算完成: {len(all_factors)} 个因子")

MIN_COVERAGE = 0.3
valid_factors = {
    k: v for k, v in all_factors.items()
    if np.isfinite(v.values).mean() >= MIN_COVERAGE
}

evaluator = FactorEvaluator(close_panel=engine.close)
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}

print("  评价因子体系（IC / FR / T统计量）...")
system_eval = evaluator.evaluate_system(named_factors, forward_period=1, neutralize=False)

# ======================================================================
# 3. 筛选有效因子 & 去冗余 & 构建组合因子
# ======================================================================
print("\n第3步: 筛选有效因子 & 去冗余...")

ic_sig = system_eval["ic_ir"].abs() > 0.5
t_sig = system_eval["t_stat"].abs() > 2.0
effective_mask = ic_sig & t_sig
effective_names = system_eval[effective_mask].index.tolist()
print(f"  有效因子: {len(effective_names)} 个")

effective_dict = {n: named_factors[n] for n in effective_names}
corr_matrix = evaluator.factor_correlation_matrix(effective_dict, forward_period=1)

REDUNDANCY_THRESHOLD = 0.7
ic_ir_values = system_eval.loc[effective_names, "ic_ir"]
sorted_by_icir = ic_ir_values.abs().sort_values(ascending=False).index.tolist()
independent_factors = []
for name in sorted_by_icir:
    is_redundant = False
    for kept in independent_factors:
        if abs(corr_matrix.loc[name, kept]) > REDUNDANCY_THRESHOLD:
            is_redundant = True
            break
    if not is_redundant:
        independent_factors.append(name)

print(f"  去冗余后独立因子: {len(independent_factors)} 个")

# 方向统一 & 组合构建
aligned_panels = {}
for name in independent_factors:
    panel = named_factors[name].copy()
    ic_val = system_eval.loc[name, "ic_mean"]
    aligned_panels[name] = -panel if ic_val < 0 else panel

def zscore_panel(panel):
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)

def combine_equal(panels):
    zscored = [zscore_panel(p) for p in panels.values()]
    stacked = np.stack([z.values for z in zscored], axis=0)
    n_valid = np.sum(np.isfinite(stacked), axis=0)
    min_required = max(len(zscored) // 2, 1)
    composite = np.nanmean(stacked, axis=0)
    composite[n_valid < min_required] = np.nan
    return pd.DataFrame(composite, index=zscored[0].index, columns=zscored[0].columns)

def combine_icir_weighted(panels, icir):
    names = list(panels.keys())
    weights = np.array([abs(icir[n]) for n in names])
    weights = weights / weights.sum()
    zscored = [zscore_panel(panels[n]) for n in names]
    stacked = np.stack([z.values for z in zscored], axis=0)
    n_valid = np.sum(np.isfinite(stacked), axis=0)
    min_required = max(len(zscored) // 2, 1)
    weighted = np.nansum(stacked * weights[:, None, None], axis=0)
    weighted[n_valid < min_required] = np.nan
    return pd.DataFrame(weighted, index=zscored[0].index, columns=zscored[0].columns)

icir_dict = {n: system_eval.loc[n, "ic_ir"] for n in independent_factors}
sorted_independent = sorted(independent_factors, key=lambda n: abs(icir_dict[n]), reverse=True)

# 使用 All_ICIR加权 和 All_等权（之前验证最优的两个组合）
all_names = sorted_independent
all_subset = {n: aligned_panels[n] for n in all_names}
all_subset_icir = {n: icir_dict[n] for n in all_names}

combo_panels = {
    "All_ICIR加权": combine_icir_weighted(all_subset, all_subset_icir),
    "All_等权": combine_equal(all_subset),
}
print(f"  测试组合: {list(combo_panels.keys())}")

# ======================================================================
# 4. 回测 — 每个风险配置 × 每个组合
# ======================================================================
print("\n第4步: 回测（2 组合 × 5 风控配置 = 10 个回测）...")
print("  注意: 每个回测约需 15-30 分钟")

all_results = {}

for profile_name, risk_params in RISK_PROFILES.items():
    for combo_name, combo_panel in combo_panels.items():
        label = f"{combo_name}_{profile_name}"
        print(f"\n  回测 {label}...", end=" ", flush=True)

        try:
            researcher = AlphaResearcher(
                dataset,
                initial_capital=INITIAL_CAPITAL,
                max_positions=MAX_POSITIONS,
                rebalance_freq=REBALANCE_FREQ,
                enable_risk_mgmt=risk_params["enable"],
                stop_loss_pct=risk_params["sl"],
                take_profit_pct=risk_params["tp"],
                max_drawdown_limit=risk_params["dd"],
            )
            r = researcher.run_backtest(alpha_panel=combo_panel, label=label)
            all_results[label] = r
            az = r.get_analyzer()
            dd = az.max_drawdown()
            print(f"年化: {az.annualized_return():+.2%}  "
                  f"夏普: {az.sharpe_ratio():.3f}  "
                  f"最大回撤: {dd:+.2%}")
        except Exception as e:
            print(f"失败: {e}")

# ======================================================================
# 5. 绩效对比
# ======================================================================
print("\n" + "=" * 70)
print("风险管理回测 — 绩效对比表")
print("=" * 70)

if all_results:
    # 构建手动汇总表（因为不同 researcher 实例不能共用 metrics_table）
    rows = []
    for label, result in all_results.items():
        az = result.get_analyzer()
        rows.append({
            "策略": label,
            "年化收益率": az.annualized_return(),
            "最大回撤": az.max_drawdown(),
            "夏普比率": az.sharpe_ratio(),
            "卡玛比率": az.calmar_ratio(),
            "Alpha": az.alpha(),
        })
    metrics = pd.DataFrame(rows).set_index("策略")
    metrics = metrics.sort_values("夏普比率", ascending=False)

    print(metrics.to_string(float_format=lambda x: f"{x:+.4f}"))

    # 回撤对比突出显示
    print("\n回撤控制对比:")
    for idx, row in metrics.iterrows():
        dd = row["最大回撤"]
        status = "✓ 达标" if abs(dd) <= 0.05 else ("△ 接近" if abs(dd) <= 0.10 else "✗ 超标")
        print(f"  {idx}: 回撤={dd:+.2%}  {status}")

# ======================================================================
# 6. 净值曲线
# ======================================================================
print("\n第6步: 绘制净值曲线对比...")

if len(all_results) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for combo_idx, combo_name in enumerate(combo_panels.keys()):
        ax = axes[combo_idx]
        combo_results = {k: v for k, v in all_results.items() if combo_name in k}

        colors = plt.cm.tab10(np.linspace(0, 1, len(combo_results)))
        for (name, result), color in zip(combo_results.items(), colors):
            az = result.get_analyzer()
            equity = az.equity
            if equity is not None and len(equity) > 0:
                values = equity["total_value"].values
                normalized = values / values[0]
                label = name.replace(f"{combo_name}_", "")
                ax.plot(equity.index, normalized, label=label,
                        color=color, linewidth=1.5)

        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("净值（归一化）", fontsize=11)
        ax.set_title(f"{combo_name} — 不同风控配置净值对比", fontsize=13)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/risk_management_equity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  已保存: risk_management_equity.png")

print("\n风险管理回测完成！")
```

- [ ] **Step 2: Verify syntax**

```bash
.venv/bin/python3 -c "compile(open('notebooks/alpha191/_run_multi_factor_risk.py').read(), '_run_multi_factor_risk.py', 'exec'); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/alpha191/_run_multi_factor_risk.py
git commit -m "feat: add risk-managed multi-factor backtest script with 5 risk profiles"
```

---

### Task 4: Run risk-managed backtest and collect results

- [ ] **Step 1: Run the script**

```bash
.venv/bin/python3 notebooks/alpha191/_run_multi_factor_risk.py 2>&1 | tee /tmp/risk_backtest_output.txt
```

Note: This runs 10 backtests (2 combos × 5 risk profiles). Expect ~5-8 hours total runtime.

- [ ] **Step 2: Review the output**

Check `/tmp/risk_backtest_output.txt` for:
- Drawdown under each risk profile
- Whether "极严风控(5%回撤)" achieves <5% max drawdown
- Trade-off between drawdown control and returns (Sharpe, annualized return)

- [ ] **Step 3: Document findings**

Based on the results, note:
- Which risk profile achieves <5% max drawdown
- What the return penalty is for tight risk controls
- Whether the 5% circuit breaker fires frequently during the 2015 crash
- Any unexpected behavior (e.g., stop-loss cascades)

---

### Task 5: Update doc with risk management findings

**Files:**
- Modify: `docs/alpha191_replication.md` (append new chapter)

- [ ] **Step 1: Add Chapter 12 "风险控制分析" to the doc**

Append after Chapter 11. Content template (fill in actual results after Task 4):

```markdown
---

## 12. 风险控制分析

### 12.1 回撤来源诊断

第 9-11 章回测显示，无论是单因子还是多因子策略，最大回撤均在 47-50% 区间。回撤来源分析：

| 来源 | 贡献度 | 说明 |
|------|--------|------|
| 系统性风险（市场β） | ~60% | 2010-2016 包含 2015 年股灾，全A市场最大回撤超过 45% |
| 策略层面 | ~40% | 无止损、无回撤熔断、始终满仓 |

策略风险敞口分析：
- **无个股止损：** 持仓股下跌 30-50% 也会一直持有到调仓日（最多 5 天）
- **无回撤熔断：** 账户可任由亏损扩大至 50%
- **始终满仓：** 熊市中没有减仓或空仓机制

代码层面根因：项目已有完整的风险管理模块（`stockquant/risk/`），但未接入 `AlphaFactorStrategy`。
`StopLossManager`、`RiskMonitor`、`PositionManager` 均已实现，仅 `DualMAStrategy` 示例策略集成了 `RiskMonitor`。

### 12.2 风控集成方案

修改 `AlphaFactorStrategy`，添加可选风控参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_risk_mgmt` | False | 是否启用风险管理 |
| `stop_loss_pct` | 0.08 | 固定止损比例 |
| `take_profit_pct` | 0.20 | 固定止盈比例 |
| `max_drawdown_limit` | 0.20 | 账户回撤熔断阈值 |

执行逻辑：每 bar 先检查回撤熔断 → 再检查个股止盈止损 → 最后在调仓日执行因子轮动。

### 12.3 回测结果

测试 2 个最优组合因子（All_ICIR加权、All_等权）× 5 种风控配置：

| 策略 | 风控配置 | 年化收益 | 最大回撤 | 夏普比率 |
|------|----------|----------|----------|----------|
| ... | 无风控(基线) | ... | ... | ... |
| ... | 宽松(15%SL/25%DD) | ... | ... | ... |
| ... | 适度(10%SL/15%DD) | ... | ... | ... |
| ... | 严格(8%SL/10%DD) | ... | ... | ... |
| ... | 极严(5%SL/5%DD) | ... | ... | ... |

*(填入实际回测结果)*

### 12.4 分析结论

1. 回撤控制效果：...
2. 收益-风控权衡：...
3. 5% 回撤目标的可行性：...
4. 建议的风控参数：...
```

- [ ] **Step 2: Commit**

```bash
git add docs/alpha191_replication.md
git commit -m "docs: add risk management analysis chapter"
```

---

## Execution Notes

- Tasks 1-3 can be done without running the backtest (pure code changes)
- Task 4 is the long-running backtest (5-8 hours)
- Task 5 updates the doc with actual results from Task 4
- The risk module APIs used:
  - `StopLossManager().check(position, current_price)` → `StopResult(triggered, action, code, reason)`
  - `RiskMonitor().update(context)` → `bool` (True = circuit breaker triggered)
- Stop-loss check happens EVERY bar (not just rebalance days) — this is critical for timely risk control
- The `PositionManager` is NOT integrated since the factor strategy already limits per-position size via `max_positions`
