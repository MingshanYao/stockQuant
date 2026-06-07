# Alpha191 因子沪深300研究 Notebook 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个 Jupyter Notebook，系统性研究 Alpha191 因子在沪深300上的表现，实践国泰君安研究报告中的完整因子评价体系。

**Architecture:** Notebook 分为 8 个章节：数据加载 → 因子计算 → IC/ICIR 分析 → Factor Return 分析 → 多周期检验 → 因子体系评价 → 因子去冗余（相关矩阵）→ 回测验证。直接使用 `Alpha191Indicators.from_dataset()` 构建面板引擎，用 `FactorEvaluator` 做因子评价，用 `AlphaResearcher` 做回测和可视化。

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, seaborn, stockquant (Alpha191Engine, FactorEvaluator, AlphaResearcher, StockUniverse)

---

## File Structure

| 文件 | 职责 |
|------|------|
| `notebooks/alpha191/alpha191_hs300_research.ipynb` | **新建** — Alpha191 沪深300 研究 Notebook |

只新建一个文件。所有分析逻辑使用框架已有的 `FactorEvaluator`、`AlphaResearcher`、`Alpha191Engine`。

## 关键 API 说明

**Alpha191Engine** (`stockquant/indicators/alpha191/alpha191.py`):
- `Alpha191Indicators.from_dataset(dataset)` → 返回 `Alpha191Engine`
- `engine.compute_factor(alpha_id)` → 单因子面板 `DataFrame`（行=日期，列=股票）
- `engine.compute_factors([1,2,...])` → `dict[int, DataFrame]`
- `engine.compute_all()` → `dict[int, DataFrame]` 全部 190 个因子
- 常量：`SKIP_ALPHAS={30}`，`BENCHMARK_ALPHAS={75,149,150,181,182,190}`

**FactorEvaluator** (`stockquant/analysis/evaluator.py`):
- `FactorEvaluator(close_panel, industry=None, market_cap=None)`
- `evaluate(factor, forward_period=1)` → `dict` (ic_mean, ic_ir, fr_mean, fr_ir, t_stat, ...)
- `evaluate_multi_horizon(factor, periods=[1,2,3,4,5])` → `dict[int, dict]`
- `evaluate_system(factors_dict, forward_period=1)` → `DataFrame` 汇总表
- `factor_correlation_matrix(factors_dict)` → `DataFrame` 相关矩阵

**AlphaResearcher** (`stockquant/research/alpha_researcher.py`):
- 注意：`alpha_engine` 属性只创建 Alpha101Engine，不支持 Alpha191
- 但 `run_backtest(alpha_panel=...)` 接受自定义因子面板，可传入 Alpha191 因子面板
- `evaluate_factor(factor_panel)` 和 `evaluate_multi_horizon(factor_panel)` 也接受任意面板
- 可视化方法：`plot_equity()`、`plot_monthly_heatmap()`、`full_analysis()`

---

### Task 1: 创建 Notebook 目录和基础框架

**Files:**
- Create: `notebooks/alpha191/alpha191_hs300_research.ipynb`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p notebooks/alpha191
```

- [ ] **Step 2: 创建 Notebook — 标题 + 导入 + 全局配置**

创建 `notebooks/alpha191/alpha191_hs300_research.ipynb`，包含以下 cells：

**Cell 0 (markdown):**
```markdown
# Alpha191 因子系统研究 — 沪深300 因子评价体系

基于国泰君安 (2017) 《基于短周期价量特征的多因子选股体系》研究报告，
系统性研究 Alpha191 因子在沪深300上的表现，实践报告中的完整因子评价方法论：

1. **数据准备** — 加载沪深300日线数据 + 构建 Alpha191 面板引擎
2. **因子批量计算** — 一次性计算全部 190 个因子面板
3. **IC/ICIR 分析** — Rank IC 衡量因子截面预测能力
4. **Factor Return 分析** — 截面回归因子收益率及 IR
5. **多周期检验** — 扫描 T+1 ~ T+5 找有效预测周期
6. **因子体系评价** — 全量因子四指标汇总表
7. **因子去冗余** — Factor Return 相关矩阵，识别冗余因子对
8. **优质因子回测验证** — 选取 Top 因子做多空回测
```

**Cell 1 (markdown):**
```markdown
## 1. 导入库与配置
```

**Cell 2 (code):**
```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stockquant.data.universe import Pool, StockUniverse
from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS, BENCHMARK_ALPHAS
from stockquant.analysis.evaluator import FactorEvaluator
from stockquant.research import AlphaResearcher

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

# ── 全局参数 ──────────────────────────────────────────────
START_DATE      = "2022-01-01"
END_DATE        = "2024-12-31"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS   = 10
REBALANCE_FREQ  = 5

# ── 因子范围 ──────────────────────────────────────────────
ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS]

print("✅ 模块导入成功")
print(f"   研究因子: {len(ALL_ALPHA_IDS)} 个 Alpha191 因子（排除 Alpha030）")
print(f"   回测区间: {START_DATE} ~ {END_DATE}")
```

- [ ] **Step 3: 运行 Cell 2 验证导入成功**

```bash
cd notebooks/alpha191 && ../../.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 alpha191_hs300_research.ipynb --output /dev/null 2>&1 | tail -5
```

或在 Jupyter 中直接运行。

- [ ] **Step 4: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add alpha191 hs300 research notebook skeleton"
```

---

### Task 2: 数据加载 + Alpha191 引擎构建

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加数据加载 cells**

**Cell 3 (markdown):**
```markdown
## 2. 加载沪深300数据
```

**Cell 4 (code):**
```python
dataset = (
    StockUniverse()
    .scope(Pool.CSI300)
    .exclude(Pool.STAR, Pool.CHINEXT, Pool.BSE)
    .load(START_DATE, END_DATE, benchmark=Pool.CSI300)
)

print(dataset.summary())
print(f"\n基准指数: {dataset.benchmark_code}  |  {len(dataset.benchmark)} 条日线")
```

- [ ] **Step 2: 添加 Alpha191 引擎构建 cells**

**Cell 5 (markdown):**
```markdown
## 3. 构建 Alpha191 面板引擎 & 批量计算因子

使用 `Alpha191Indicators.from_dataset()` 构建面板引擎，一次性计算全部 190 个因子。
```

**Cell 6 (code):**
```python
print("构建 Alpha191 面板引擎...")
engine = Alpha191Indicators.from_dataset(dataset)
print(f"✅ 引擎构建完成: {engine.close.shape[0]} 个交易日 × {engine.close.shape[1]} 只股票")

print("\n批量计算全部因子面板...")
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
print(f"✅ 完成: 成功计算 {len(all_factors)} 个因子")

# 查看一个因子的面板概览
sample_id = 14
print(f"\nAlpha{sample_id:03d} 面板预览:")
all_factors[sample_id].tail(3).iloc[:, :6]
```

- [ ] **Step 3: 添加因子覆盖率概览 cell**

**Cell 7 (code):**
```python
coverage = {}
for alpha_id, panel in all_factors.items():
    finite_ratio = np.isfinite(panel.values).mean()
    coverage[alpha_id] = finite_ratio

coverage_s = pd.Series(coverage).sort_values()
print(f"因子覆盖率统计:")
print(f"  最低: Alpha{coverage_s.index[0]:03d} = {coverage_s.iloc[0]:.1%}")
print(f"  中位: {coverage_s.median():.1%}")
print(f"  最高: Alpha{coverage_s.index[-1]:03d} = {coverage_s.iloc[-1]:.1%}")
print(f"  覆盖率 < 50% 的因子: {(coverage_s < 0.5).sum()} 个")

# 过滤低覆盖率因子
MIN_COVERAGE = 0.3
valid_factors = {k: v for k, v in all_factors.items() if coverage.get(k, 0) >= MIN_COVERAGE}
print(f"\n有效因子（覆盖率 >= {MIN_COVERAGE:.0%}）: {len(valid_factors)} / {len(all_factors)} 个")
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add data loading and alpha191 engine construction"
```

---

### Task 3: IC/ICIR 分析

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加 IC/ICIR 分析 cells**

**Cell 8 (markdown):**
```markdown
## 4. IC / ICIR 分析

**IC (Information Coefficient)**：每日截面因子值与次日收益率的 Rank 相关系数，衡量因子的截面预测能力。

**ICIR (IC Information Ratio)**：IC 均值 / IC 标准差，衡量 IC 的稳定性。|ICIR| > 0.5 通常被认为有效。
```

**Cell 9 (code):**
```python
evaluator = FactorEvaluator(close_panel=engine.close)

# 批量计算 IC/ICIR（T+1 前向收益）
factor_names = {k: f"Alpha{k:03d}" for k in valid_factors}
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}

print(f"计算 {len(named_factors)} 个因子的 IC/ICIR (T+1)...")
system_eval = evaluator.evaluate_system(named_factors, forward_period=1, neutralize=False)
print("✅ 完成")

# IC 汇总表（按 |IC| 降序，已在 evaluate_system 中排好序）
ic_cols = ["ic_mean", "ic_std", "ic_ir", "ic_pos_ratio", "n_periods"]
ic_summary = system_eval[ic_cols].copy()
ic_summary.columns = ["IC均值", "IC标准差", "ICIR", "IC>0占比", "有效天数"]

print(f"\n{'='*60}")
print(f"Top 20 因子 — IC/ICIR 排名（按 |IC均值| 降序）")
print(f"{'='*60}")
ic_summary.head(20).style.format({
    "IC均值": "{:.4f}", "IC标准差": "{:.4f}", "ICIR": "{:.3f}", "IC>0占比": "{:.1%}",
})
```

- [ ] **Step 2: 添加 IC 分布可视化 cell**

**Cell 10 (code):**
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) IC 均值分布直方图
ax = axes[0]
ic_values = system_eval["ic_mean"].dropna()
ax.hist(ic_values, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("IC 均值", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 IC 均值分布", fontsize=13)
ax.grid(True, alpha=0.3)

# (b) ICIR 分布直方图
ax = axes[1]
icir_values = system_eval["ic_ir"].dropna()
ax.hist(icir_values, bins=40, color="#e74c3c", alpha=0.7, edgecolor="white")
ax.axvline(0.5, color="green", linestyle="--", linewidth=1, label="ICIR=0.5")
ax.axvline(-0.5, color="green", linestyle="--", linewidth=1)
ax.set_xlabel("ICIR", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 ICIR 分布", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 统计显著因子数量
sig_pos = (system_eval["ic_ir"] > 0.5).sum()
sig_neg = (system_eval["ic_ir"] < -0.5).sum()
print(f"显著正向因子（ICIR > 0.5）: {sig_pos} 个")
print(f"显著负向因子（ICIR < -0.5）: {sig_neg} 个")
print(f"显著因子合计: {sig_pos + sig_neg} / {len(system_eval)} 个")
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add IC/ICIR analysis section"
```

---

### Task 4: Factor Return 分析

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加 Factor Return 分析 cells**

**Cell 11 (markdown):**
```markdown
## 5. Factor Return 分析

**Factor Return (因子收益率)**：将因子值截面标准化后，对截面收益率做 OLS 回归，回归系数即因子收益率。

**FR IR (Factor Return Information Ratio)**：FR 均值 / FR 标准差，衡量因子收益的稳定性。
```

**Cell 12 (code):**
```python
fr_cols = ["fr_mean", "fr_std", "fr_ir", "fr_annual", "t_stat"]
fr_summary = system_eval[fr_cols].copy()
fr_summary.columns = ["FR均值", "FR标准差", "FR_IR", "年化FR", "T统计量"]
fr_summary = fr_summary.reindex(fr_summary["T统计量"].abs().sort_values(ascending=False).index)

print(f"{'='*60}")
print(f"Top 20 因子 — Factor Return 排名（按 |T统计量| 降序）")
print(f"{'='*60}")
fr_summary.head(20)
```

- [ ] **Step 2: 添加 T 统计量可视化 cell**

**Cell 13 (code):**
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) T 统计量分布
ax = axes[0]
t_values = system_eval["t_stat"].dropna()
ax.hist(t_values, bins=40, color="#2ecc71", alpha=0.7, edgecolor="white")
ax.axvline(2.0, color="red", linestyle="--", linewidth=1, label="|T|=2.0")
ax.axvline(-2.0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("T 统计量", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 T 统计量分布", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 年化 Factor Return 分布
ax = axes[1]
fr_annual = system_eval["fr_annual"].dropna()
ax.hist(fr_annual * 100, bins=40, color="#9b59b6", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("年化 Factor Return (%)", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子年化 FR 分布", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

sig_t = (system_eval["t_stat"].abs() > 2.0).sum()
print(f"T统计量显著（|T| > 2.0）的因子: {sig_t} / {len(system_eval)} 个")
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add factor return analysis section"
```

---

### Task 5: 多周期检验 (T+1 ~ T+5)

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加多周期检验 cells**

**Cell 14 (markdown):**
```markdown
## 6. 多周期检验 (T+1 ~ T+5)

扫描不同预测周期（T+1, T+2, ..., T+5），检验因子在不同时间尺度上的预测能力。
短周期因子（价量特征）通常在 T+1 ~ T+3 表现最好，随后衰减。
```

**Cell 15 (code):**
```python
# 选取 IC 最显著的 Top 20 因子做多周期检验
top_n = 20
top_factors_by_ic = system_eval["ic_mean"].abs().nlargest(top_n).index.tolist()
print(f"选取 Top {top_n} 因子（按 |IC|）做多周期检验:")
print(f"  {', '.join(top_factors_by_ic[:10])}")
print(f"  {', '.join(top_factors_by_ic[10:])}")

multi_horizon_results = {}
for i, name in enumerate(top_factors_by_ic, 1):
    print(f"  [{i}/{top_n}] 多周期评价: {name}...", end="\r")
    result = evaluator.evaluate_multi_horizon(named_factors[name], periods=[1, 2, 3, 4, 5], neutralize=False)
    multi_horizon_results[name] = result
print(f"✅ 完成 {top_n} 个因子多周期评价              ")
```

- [ ] **Step 2: 添加多周期 IC 衰减热力图 cell**

**Cell 16 (code):**
```python
# 构建多周期 IC 衰减矩阵
ic_decay = pd.DataFrame({
    name: {f"T+{d}": metrics["ic_mean"] for d, metrics in horizons.items()}
    for name, horizons in multi_horizon_results.items()
}).T

fig, ax = plt.subplots(figsize=(10, max(6, len(ic_decay) * 0.35)))
vals = ic_decay.values[np.isfinite(ic_decay.values)]
vmax = max(abs(vals).max(), 0.01) if len(vals) else 0.05
sns.heatmap(
    ic_decay.astype(float), annot=True, fmt=".4f", cmap="RdYlGn",
    center=0, vmin=-vmax, vmax=vmax,
    linewidths=0.5, ax=ax,
)
ax.set_title(f"Top {top_n} 因子多周期 IC 衰减矩阵", fontsize=13)
ax.set_xlabel("预测周期", fontsize=11)
ax.set_ylabel("因子", fontsize=11)
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: 添加多周期 ICIR 衰减矩阵 cell**

**Cell 17 (code):**
```python
icir_decay = pd.DataFrame({
    name: {f"T+{d}": metrics["ic_ir"] for d, metrics in horizons.items()}
    for name, horizons in multi_horizon_results.items()
}).T

fig, ax = plt.subplots(figsize=(10, max(6, len(icir_decay) * 0.35)))
vals = icir_decay.values[np.isfinite(icir_decay.values)]
vmax = max(abs(vals).max(), 0.1) if len(vals) else 0.5
sns.heatmap(
    icir_decay.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
    center=0, vmin=-vmax, vmax=vmax,
    linewidths=0.5, ax=ax,
)
ax.set_title(f"Top {top_n} 因子多周期 ICIR 衰减矩阵", fontsize=13)
ax.set_xlabel("预测周期", fontsize=11)
ax.set_ylabel("因子", fontsize=11)
plt.tight_layout()
plt.show()

print("IC 衰减模式分析:")
for name in ic_decay.index[:5]:
    row = ic_decay.loc[name].astype(float)
    peak = row.abs().idxmax()
    print(f"  {name}: 最强周期 {peak} (IC={row[peak]:.4f})")
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add multi-horizon testing section (T+1~T+5)"
```

---

### Task 6: 因子体系评价 — 完整四指标汇总

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加因子体系评价 cells**

**Cell 18 (markdown):**
```markdown
## 7. 因子体系评价 — 完整四指标汇总

综合 IC、ICIR、Factor Return、FR IR、T统计量，给出全量因子的系统性评价。
参照国泰君安报告的评价标准：
- **IC均值 > 0.03** 且 **ICIR > 0.5**：有效因子
- **|T统计量| > 2.0**：统计显著
- 结合 FR_IR 和年化 FR 做综合判断
```

**Cell 19 (code):**
```python
# 完整四指标汇总表
full_eval = system_eval.copy()
full_eval.columns = ["IC均值", "IC标准差", "ICIR", "IC>0占比", "FR均值", "FR标准差", "FR_IR", "年化FR", "T统计量", "有效天数"]

# 添加因子有效性标签
full_eval["IC显著"] = full_eval["ICIR"].abs() > 0.5
full_eval["T显著"] = full_eval["T统计量"].abs() > 2.0
full_eval["有效因子"] = full_eval["IC显著"] & full_eval["T显著"]

n_effective = full_eval["有效因子"].sum()
print(f"{'='*70}")
print(f"Alpha191 因子体系评价汇总")
print(f"{'='*70}")
print(f"总因子数:         {len(full_eval)}")
print(f"IC 显著 (|ICIR|>0.5):  {full_eval['IC显著'].sum()}")
print(f"T 显著 (|T|>2.0):      {full_eval['T显著'].sum()}")
print(f"有效因子 (两项均满足):  {n_effective}")
print(f"{'='*70}")
print(f"\n有效因子列表:")
effective = full_eval[full_eval["有效因子"]].sort_values("ICIR", key=abs, ascending=False)
effective[["IC均值", "ICIR", "FR_IR", "年化FR", "T统计量"]]
```

- [ ] **Step 2: 添加四象限散点图 cell**

**Cell 20 (code):**
```python
fig, ax = plt.subplots(figsize=(12, 8))
x = full_eval["ICIR"].values
y = full_eval["T统计量"].values
colors = np.where(full_eval["有效因子"].values, "#e74c3c", "#95a5a6")
sizes = np.where(full_eval["有效因子"].values, 60, 20)

ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, edgecolors="white", linewidth=0.5)

# 标注有效因子名称
for name, row in effective.iterrows():
    ax.annotate(
        name, (row["ICIR"], row["T统计量"]),
        fontsize=7, alpha=0.8,
        textcoords="offset points", xytext=(5, 5),
    )

ax.axhline(2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(-2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

ax.set_xlabel("ICIR", fontsize=12)
ax.set_ylabel("T 统计量", fontsize=12)
ax.set_title("Alpha191 因子四象限图（ICIR vs T统计量）", fontsize=14)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add factor system evaluation section"
```

---

### Task 7: 因子去冗余 — Factor Return 相关矩阵

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加因子去冗余 cells**

**Cell 21 (markdown):**
```markdown
## 8. 因子去冗余 — Factor Return 相关矩阵

计算有效因子之间的 Factor Return 相关性，识别高度相关（冗余）的因子对。
相关系数 > 0.7 的因子对提供的信息高度重叠，实际使用时只需保留其中一个。
```

**Cell 22 (code):**
```python
# 选取有效因子做相关性分析
if len(effective) >= 2:
    effective_factor_dict = {name: named_factors[name] for name in effective.index if name in named_factors}
    print(f"计算 {len(effective_factor_dict)} 个有效因子的 FR 相关矩阵...")
    corr_matrix = evaluator.factor_correlation_matrix(effective_factor_dict, forward_period=1)
    print("✅ 完成")

    fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix) * 0.5), max(6, len(corr_matrix) * 0.4)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=len(corr_matrix) <= 20,
        fmt=".2f" if len(corr_matrix) <= 20 else "",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, square=True,
    )
    ax.set_title("有效因子 Factor Return 相关矩阵", fontsize=13)
    plt.tight_layout()
    plt.show()

    # 找出高相关因子对
    HIGH_CORR_THRESHOLD = 0.7
    pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > HIGH_CORR_THRESHOLD:
                pairs.append((corr_matrix.index[i], corr_matrix.columns[j], r))

    if pairs:
        print(f"\n高相关因子对（|r| > {HIGH_CORR_THRESHOLD}）:")
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {a} ↔ {b}: r = {r:.3f}")
    else:
        print(f"\n无高相关因子对（阈值 |r| > {HIGH_CORR_THRESHOLD}），因子间独立性较好")
else:
    print("⚠️ 有效因子不足 2 个，跳过相关性分析")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add factor redundancy analysis section"
```

---

### Task 8: 优质因子回测验证

**Files:**
- Modify: `notebooks/alpha191/alpha191_hs300_research.ipynb` (追加 cells)

- [ ] **Step 1: 添加回测验证 cells**

**Cell 23 (markdown):**
```markdown
## 9. 优质因子回测验证

选取因子评价中表现最好的因子，做 Top-N 多头等权回测，验证因子的实际选股能力。
使用 `AlphaResearcher` 的回测框架，直接传入 Alpha191 因子面板。
```

**Cell 24 (code):**
```python
researcher = AlphaResearcher(
    dataset,
    initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_POSITIONS,
    rebalance_freq=REBALANCE_FREQ,
)
print(f"✅ AlphaResearcher 创建完成")

# 选取最优因子做回测
if len(effective) > 0:
    top_backtest_ids = effective.index[:min(10, len(effective))].tolist()
else:
    top_backtest_ids = system_eval["ic_mean"].abs().nlargest(5).index.tolist()

print(f"回测因子: {', '.join(top_backtest_ids)}")
```

**Cell 25 (code):**
```python
backtest_results = {}
for i, name in enumerate(top_backtest_ids, 1):
    print(f"[{i}/{len(top_backtest_ids)}] 回测 {name}...", end="")
    try:
        alpha_id = int(name.replace("Alpha", ""))
        panel = all_factors[alpha_id]
        r = researcher.run_backtest(alpha_panel=panel, label=name)
        backtest_results[name] = r
        az = r.get_analyzer()
        print(f"  总收益: {az.total_return():+.2%}  夏普: {az.sharpe_ratio():.3f}")
    except Exception as e:
        print(f"  ⚠️ 失败: {e}")

print(f"\n✅ 完成 {len(backtest_results)}/{len(top_backtest_ids)} 个因子回测")
```

- [ ] **Step 2: 添加绩效对比 cell**

**Cell 26 (code):**
```python
if backtest_results:
    compare_df = researcher.compare_factors(backtest_results)
    display(compare_df)
```

- [ ] **Step 3: 添加最优因子深度分析 cell**

**Cell 27 (code):**
```python
if backtest_results:
    metrics = researcher.metrics_table(backtest_results)
    best_name = metrics["夏普比率"].idxmax()
    best_result = backtest_results[best_name]
    print(f"最优因子: {best_name}  夏普比率={metrics.loc[best_name, '夏普比率']:.3f}")
    print()
    researcher.full_analysis(best_result)
```

- [ ] **Step 4: 添加研究结论 cell**

**Cell 28 (markdown):**
```markdown
## 10. 研究结论

### 评价方法论总结

本研究实践了国泰君安 (2017) 报告中的完整因子评价体系：

| 维度 | 方法 | 判断标准 |
|------|------|----------|
| IC 分析 | Rank IC (Spearman) | IC均值, ICIR > 0.5 |
| Factor Return | 截面回归 OLS | FR_IR, 年化 FR |
| 统计检验 | T 统计量 | \|T\| > 2.0 |
| 多周期 | T+1 ~ T+5 扫描 | IC 衰减速度 |
| 去冗余 | FR 相关矩阵 | \|r\| > 0.7 标记冗余 |

### 发现

（运行 Notebook 后根据实际结果填写）
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: add backtest verification and conclusion sections"
```

---

### Task 9: 完整运行验证

- [ ] **Step 1: 运行全量 Notebook 确认无报错**

由于 Notebook 需要真实市场数据和较长计算时间，在 Jupyter 环境中打开并逐 cell 运行，确认：
- 数据加载成功
- 因子计算完成
- 所有图表正确渲染
- 无运行时错误

如果没有真实数据连接，确认 Notebook 代码语法正确、import 无误即可。

- [ ] **Step 2: 运行现有测试确认无回归**

```bash
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_hs300_daily.py
```

Expected: 全部 PASSED，无回归。

- [ ] **Step 3: Final commit**

```bash
git add notebooks/alpha191/alpha191_hs300_research.ipynb
git commit -m "feat: complete alpha191 hs300 research notebook"
```
