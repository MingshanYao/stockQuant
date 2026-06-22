"""复现国泰君安 (2017) Alpha191 因子研究 — 全A股 2010-2016。

与 _run_notebook.py (CSI300 2022-2024) 对比，验证因子效果差异来源。
"""
import gc
import warnings
warnings.filterwarnings("ignore")

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from stockquant.data.universe import Pool, StockUniverse
from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS
from stockquant.analysis.evaluator import FactorEvaluator
from stockquant.research import AlphaResearcher

# 模块级变量用于 fork 模式回测并行
_backtest_dataset = None
_backtest_params = None


def _run_single_backtest(args: tuple):
    """模块级函数：运行单个因子回测（用于 ProcessPoolExecutor fork）。"""
    alpha_id, panel, label = args
    researcher = AlphaResearcher(_backtest_dataset, **_backtest_params)
    r = researcher.run_backtest(alpha_panel=panel, label=label)
    return label, r

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

print("=" * 70)
print("Alpha191 国泰君安复现研究 — 全A股 2010-2016")
print("=" * 70)
print(f"研究因子: {len(ALL_ALPHA_IDS)} 个 Alpha191 因子")
print(f"回测区间: {START_DATE} ~ {END_DATE}")
print(f"股票池:   全A股（排除科创板/北交所）")

# ======================================================================
# 1. 加载全A股数据（直接从本地DB读取，不走远程fetch）
# ======================================================================
print("\n" + "=" * 70)
print("第1章: 加载全A股数据 (2010-2016)")
print("=" * 70)

from stockquant.data.universe import Pool, StockUniverse, BacktestDataset

dataset = (
    StockUniverse()
    .scope(Pool.ALL_A)
    .exclude(Pool.STAR, Pool.BSE)
    .load(START_DATE, END_DATE, benchmark="000300")
)

print(dataset.summary())
print(f"\n基准指数: {dataset.benchmark_code}  |  {len(dataset.benchmark)} 条日线")

# ======================================================================
# 2. 构建 Alpha191 面板引擎 & 批量计算因子
# ======================================================================
print("\n" + "=" * 70)
print("第2章: 构建 Alpha191 面板引擎 & 批量计算因子")
print("=" * 70)

print("构建 Alpha191 面板引擎...")
engine = Alpha191Indicators.from_dataset(dataset)
print(f"✅ 引擎构建完成: {engine.close.shape[0]} 个交易日 × {engine.close.shape[1]} 只股票")

print("\n批量计算全部因子面板...")
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
print(f"✅ 完成: 成功计算 {len(all_factors)} 个因子")

coverage = {}
for alpha_id, panel in all_factors.items():
    finite_ratio = np.isfinite(panel.values).mean()
    coverage[alpha_id] = finite_ratio

coverage_s = pd.Series(coverage).sort_values()
print(f"\n因子覆盖率统计:")
print(f"  最低: Alpha{coverage_s.index[0]:03d} = {coverage_s.iloc[0]:.1%}")
print(f"  中位: {coverage_s.median():.1%}")
print(f"  最高: Alpha{coverage_s.index[-1]:03d} = {coverage_s.iloc[-1]:.1%}")

MIN_COVERAGE = 0.3
valid_factors = {k: v for k, v in all_factors.items() if coverage.get(k, 0) >= MIN_COVERAGE}
print(f"\n有效因子（覆盖率 >= {MIN_COVERAGE:.0%}）: {len(valid_factors)} / {len(all_factors)} 个")

# ======================================================================
# 3. IC / ICIR 分析
# ======================================================================
print("\n" + "=" * 70)
print("第3章: IC / ICIR 分析")
print("=" * 70)

# ── 加载行业 & 市值数据（来自 DataManager）──
from stockquant.data.data_manager import DataManager
dm = DataManager()
stock_info_df = dm.get_stock_info()
if not stock_info_df.empty:
    industry_map = stock_info_df.set_index("code")["industry"]
    market_cap = stock_info_df.set_index("code")["float_cap"]
    print(f"  行业映射: {industry_map.nunique()} 个行业, {len(industry_map)} 只股票")
else:
    industry_map = None
    market_cap = None
    print("  ⚠️ stock_info 为空，行业/市值因子将不可用")

# ── 加载基准收益（用于 Beta 计算）──
benchmark_bars_eval = dm.fetch_index_daily("000905", start_date=START_DATE, end_date=END_DATE)
benchmark_bars_eval["date"] = pd.to_datetime(benchmark_bars_eval["date"])
benchmark_returns = benchmark_bars_eval.set_index("date")["close"].pct_change()
print(f"  基准收益: {len(benchmark_returns.dropna())} 个交易日")

evaluator = FactorEvaluator(
    close_panel=engine.close,
    industry=industry_map,
    market_cap=market_cap,
    benchmark_returns=benchmark_returns,
)
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}

print(f"计算 {len(named_factors)} 个因子的 IC/ICIR (T+1)...")
system_eval = evaluator.evaluate_system(named_factors, forward_period=1)
print("✅ 完成")

ic_cols = ["ic_mean", "ic_std", "ic_ir", "ic_pos_ratio", "n_periods"]
ic_summary = system_eval[ic_cols].copy()
ic_summary.columns = ["IC均值", "IC标准差", "ICIR", "IC>0占比", "有效天数"]

print(f"\nTop 20 因子 — IC/ICIR 排名（按 |IC均值| 降序）")
print(ic_summary.head(20).to_string(float_format=lambda x: f"{x:.4f}"))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
ic_values = system_eval["ic_mean"].dropna()
ax.hist(ic_values, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("IC 均值", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 IC 均值分布 (全A 2010-2016)", fontsize=13)
ax.grid(True, alpha=0.3)

ax = axes[1]
icir_values = system_eval["ic_ir"].dropna()
ax.hist(icir_values, bins=40, color="#e74c3c", alpha=0.7, edgecolor="white")
ax.axvline(0.5, color="green", linestyle="--", linewidth=1, label="ICIR=0.5")
ax.axvline(-0.5, color="green", linestyle="--", linewidth=1)
ax.set_xlabel("ICIR", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 ICIR 分布 (全A 2010-2016)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ic_icir_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: ic_icir_distribution.png")

sig_pos = (system_eval["ic_ir"] > 0.5).sum()
sig_neg = (system_eval["ic_ir"] < -0.5).sum()
print(f"\n显著正向因子（ICIR > 0.5）: {sig_pos} 个")
print(f"显著负向因子（ICIR < -0.5）: {sig_neg} 个")
print(f"显著因子合计: {sig_pos + sig_neg} / {len(system_eval)} 个")

# ======================================================================
# 4. Factor Return 分析
# ======================================================================
print("\n" + "=" * 70)
print("第4章: Factor Return 分析")
print("=" * 70)

fr_cols = ["fr_mean", "fr_std", "fr_ir", "fr_annual", "t_stat"]
fr_summary = system_eval[fr_cols].copy()
fr_summary.columns = ["FR均值", "FR标准差", "FR_IR", "年化FR", "T统计量"]
fr_summary = fr_summary.reindex(fr_summary["T统计量"].abs().sort_values(ascending=False).index)

print(f"Top 20 因子 — Factor Return 排名（按 |T统计量| 降序）")
print(fr_summary.head(20).to_string(float_format=lambda x: f"{x:.4f}"))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
t_values = system_eval["t_stat"].dropna()
ax.hist(t_values, bins=40, color="#2ecc71", alpha=0.7, edgecolor="white")
ax.axvline(2.0, color="red", linestyle="--", linewidth=1, label="|T|=2.0")
ax.axvline(-2.0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("T 统计量", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 T 统计量分布 (全A 2010-2016)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
fr_annual = system_eval["fr_annual"].dropna()
ax.hist(fr_annual * 100, bins=40, color="#9b59b6", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("年化 Factor Return (%)", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子年化 FR 分布 (全A 2010-2016)", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/t_stat_fr_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: t_stat_fr_distribution.png")

sig_t = (system_eval["t_stat"].abs() > 2.0).sum()
print(f"\nT统计量显著（|T| > 2.0）的因子: {sig_t} / {len(system_eval)} 个")

# ======================================================================
# 5. 多周期检验 (T+1 ~ T+5)
# ======================================================================
print("\n" + "=" * 70)
print("第5章: 多周期检验 (T+1 ~ T+5)")
print("=" * 70)

top_n = 20
top_factors_by_ic = system_eval["ic_mean"].abs().nlargest(top_n).index.tolist()
print(f"选取 Top {top_n} 因子（按 |IC|）做多周期检验:")
print(f"  {', '.join(top_factors_by_ic[:10])}")
print(f"  {', '.join(top_factors_by_ic[10:])}")

# ── 预中性化因子 + 各周期 forward returns，消除重复 neutralize ──
periods = [1, 2, 3, 4, 5]
print(f"  预中性化 {top_n} 个因子 + {len(periods)} 周期 forward returns...", flush=True)
top_neut = {n: evaluator._neutralize_panel(named_factors[n]) for n in top_factors_by_ic}
fwd_neut_period = {p: evaluator._neutralize_panel(evaluator._forward_returns(p)) for p in periods}
# 预排名 fwd（Spearman 专用）——每个周期 rank 一次，所有因子复用
fwd_ranked_period = {p: panel.rank(axis=1) for p, panel in fwd_neut_period.items()}

# 并行 evaluate（IC 走 numpy 快速路径，无 pandas corrwith GIL 竞争）
def _eval_one(name, p):
    return name, p, evaluator.evaluate(
        named_factors[name], forward_period=p,
        factor_neutral=top_neut[name], fwd_neutral=fwd_neut_period[p],
        fwd_ranked=fwd_ranked_period[p],
    )

multi_horizon_results = {}
with ThreadPoolExecutor(max_workers=min(12, top_n * len(periods))) as ex:
    futures = {}
    for name in top_factors_by_ic:
        for p in periods:
            futures[ex.submit(_eval_one, name, p)] = (name, p)
    for i, future in enumerate(as_completed(futures), 1):
        name, p, result = future.result()
        if name not in multi_horizon_results:
            multi_horizon_results[name] = {}
        multi_horizon_results[name][p] = result
        print(f"  [{i}/{top_n * len(periods)}] {name} T+{p} IC={result['ic_mean']:.4f}", flush=True)
print(f"✅ 完成 {top_n} 个因子多周期评价")

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
ax.set_title(f"Top {top_n} 因子多周期 IC 衰减矩阵 (全A 2010-2016)", fontsize=13)
ax.set_xlabel("预测周期", fontsize=11)
ax.set_ylabel("因子", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ic_decay_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: ic_decay_heatmap.png")

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
ax.set_title(f"Top {top_n} 因子多周期 ICIR 衰减矩阵 (全A 2010-2016)", fontsize=13)
ax.set_xlabel("预测周期", fontsize=11)
ax.set_ylabel("因子", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/icir_decay_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: icir_decay_heatmap.png")

print("\nIC 衰减模式分析:")
for name in ic_decay.index[:5]:
    row = ic_decay.loc[name].astype(float)
    peak = row.abs().idxmax()
    print(f"  {name}: 最强周期 {peak} (IC={row[peak]:.4f})")

# ======================================================================
# 6. 因子体系评价 — 完整四指标汇总
# ======================================================================
print("\n" + "=" * 70)
print("第6章: 因子体系评价 — 完整四指标汇总")
print("=" * 70)

full_eval = system_eval.copy()
full_eval.columns = ["IC均值", "IC标准差", "ICIR", "IC>0占比", "FR均值", "FR标准差", "FR_IR", "年化FR", "T统计量", "有效天数", "覆盖率"]

full_eval["IC显著"] = full_eval["ICIR"].abs() > 0.5
full_eval["T显著"] = full_eval["T统计量"].abs() > 2.0
full_eval["有效因子"] = full_eval["IC显著"] & full_eval["T显著"]

n_effective = full_eval["有效因子"].sum()
print(f"总因子数:         {len(full_eval)}")
print(f"IC 显著 (|ICIR|>0.5):  {full_eval['IC显著'].sum()}")
print(f"T 显著 (|T|>2.0):      {full_eval['T显著'].sum()}")
print(f"有效因子 (两项均满足):  {n_effective}")

effective = full_eval[full_eval["有效因子"]].sort_values("ICIR", key=abs, ascending=False)
if len(effective) > 0:
    print(f"\n有效因子列表:")
    print(effective[["IC均值", "ICIR", "FR_IR", "年化FR", "T统计量"]].to_string(float_format=lambda x: f"{x:.4f}"))
else:
    print("\n⚠️ 无同时满足两项标准的有效因子，放宽标准查看:")
    relaxed = full_eval[full_eval["IC显著"] | full_eval["T显著"]].sort_values("ICIR", key=abs, ascending=False)
    print(relaxed[["IC均值", "ICIR", "FR_IR", "年化FR", "T统计量"]].head(20).to_string(float_format=lambda x: f"{x:.4f}"))

fig, ax = plt.subplots(figsize=(12, 8))
x = full_eval["ICIR"].values
y = full_eval["T统计量"].values
is_effective = full_eval["有效因子"].values
colors = np.where(is_effective, "#e74c3c", "#95a5a6")
sizes = np.where(is_effective, 60, 20)

ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, edgecolors="white", linewidth=0.5)

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
ax.set_title("Alpha191 因子四象限图 (全A 2010-2016)", fontsize=14)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/quadrant_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: quadrant_scatter.png")

# ======================================================================
# 6.5. 因子体系预测力 & 数量边际分析 & 覆盖率报告
# ======================================================================
print("\n" + "=" * 70)
print("第6.5章: 因子体系预测力评估 & 数量边际分析 & 覆盖率报告")
print("=" * 70)

# ── 预中性化：收集所有候选因子 + forward returns，仅中性化一次 ──
candidates = full_eval[full_eval["IC显著"] | full_eval["T显著"]]
candidate_dict = {name: named_factors[name] for name in candidates.index if name in named_factors}
print(f"\n预中性化 {len(candidate_dict)} 个候选因子 + forward returns（复用至 Ch7）...", flush=True)

fwd_neut_cache = evaluator._neutralize_panel(evaluator._forward_returns(1))
fwd_ranked_cache = fwd_neut_cache.rank(axis=1)  # 预排名供 Spearman 快速路径

# 批量中性化——一次池提交处理所有因子，消除 O(F×C) 池任务洪水
print(f"  批量中性化 {len(candidate_dict)} 个因子...", flush=True)
neut_factors_cache = evaluator._neutralize_panels_batch(candidate_dict)
gc.collect()
print(f"✅ 预中性化完成: {len(neut_factors_cache)} 个因子\n", flush=True)

# ── 6.5a 因子体系整体预测力（复用预中性化数据）──
print("--- 因子体系整体预测力 (Model Predictive Power) ---")
if neut_factors_cache:
    model_ic = evaluator.evaluate_model_predictive_power(
        candidate_dict,
        forward_period=1,
        neutralized_factors=neut_factors_cache,
        fwd_neutral=fwd_neut_cache,
        fwd_ranked=fwd_ranked_cache,
    )
    print(f"合成 Alpha 预测 IC: "
          f"均值={model_ic.get('ic_mean', np.nan):.4f}, "
          f"IR={model_ic.get('ic_ir', np.nan):.3f}, "
          f"T={model_ic.get('t_stat', np.nan):.2f}, "
          f"胜率={model_ic.get('ic_pos_ratio', np.nan):.1%}", flush=True)

# ── 6.5b 因子数量边际分析（复用预中性化数据）──
print("\n--- 因子数量边际分析 (Factor Count Analysis) ---")
if len(neut_factors_cache) >= 2:
    print("按 FR IR 降序分析因子数量边际贡献...", flush=True)
    # 从已有 system_eval 获取 FR IR，跳过 O(F·d·s·k) 的 Phase 2 重算
    fr_ir_map = system_eval["fr_ir"].abs().to_dict()
    count_df = evaluator.factor_count_analysis(
        candidate_dict,
        forward_period=1,
        neutralized_factors=neut_factors_cache,
        fwd_neutral=fwd_neut_cache,
        fr_ir_map=fr_ir_map,
        fwd_ranked=fwd_ranked_cache,
    )
    print(count_df[["ic_mean", "ic_ir", "added_factor"]].to_string(
        float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(count_df.index, count_df["ic_ir"].values, "o-", color="#3498db", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("因子数量", fontsize=11)
    ax.set_ylabel("合成 ICIR", fontsize=11)
    ax.set_title("因子数量边际分析 — 按 FR IR 降序逐步增加", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/factor_count_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 已保存: factor_count_analysis.png")

# ── 6.5c 覆盖率报告 ──
print("\n--- 因子覆盖率报告 ---")
cov_report = evaluator.coverage_report(named_factors)
print(f"最低覆盖率: {cov_report['min_coverage']:.1%}")
print(f"中位覆盖率: {cov_report['median_coverage']:.1%}")
print(f"最高覆盖率: {cov_report['max_coverage']:.1%}")
low_factors = cov_report.get('low_coverage_factors', [])
if low_factors:
    print(f"低覆盖因子 (<30%): {len(low_factors)} 个")
    for name, cov in low_factors[:10]:
        print(f"  {name}: {cov:.1%}")

# ======================================================================
# 7. 因子去冗余 — Factor Return 相关矩阵（复用预中性化数据）
# ======================================================================
print("\n" + "=" * 70)
print("第7章: 因子去冗余 — Factor Return 相关矩阵")
print("=" * 70)

if len(candidate_dict) >= 2:
    print(f"计算 {len(candidate_dict)} 个显著因子的 FR 相关矩阵...")
    corr_matrix = evaluator.factor_correlation_matrix(
        candidate_dict, forward_period=1,
        neutralized_factors=neut_factors_cache,
        fwd_neutral=fwd_neut_cache,
    )
    print("✅ 完成")

    fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix) * 0.5), max(6, len(corr_matrix) * 0.4)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=len(corr_matrix) <= 20,
        fmt=".2f" if len(corr_matrix) <= 20 else "",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, square=True,
    )
    ax.set_title("显著因子 Factor Return 相关矩阵 (全A 2010-2016)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fr_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 已保存: fr_correlation_matrix.png")

    HIGH_CORR_THRESHOLD = 0.7
    pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > HIGH_CORR_THRESHOLD:
                pairs.append((corr_matrix.index[i], corr_matrix.columns[j], r))

    if pairs:
        print(f"\n高相关因子对（|r| > {HIGH_CORR_THRESHOLD}）:")
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2]))[:20]:
            print(f"  {a} ↔ {b}: r = {r:.3f}")
    else:
        print(f"\n无高相关因子对（阈值 |r| > {HIGH_CORR_THRESHOLD}），因子间独立性较好")
else:
    print("⚠️ 显著因子不足 2 个，跳过相关性分析")

# ── 释放预中性化内存 + 关闭进程池，为 Ch8 回测腾出 RAM ──
from stockquant.analysis.evaluator import close_pool
del neut_factors_cache, fwd_neut_cache, fwd_ranked_cache
close_pool()
gc.collect()
print("\n📦 已释放预中性化缓存 + 进程池\n")

# ======================================================================
# 8. 优质因子回测验证
# ======================================================================
print("\n" + "=" * 70)
print("第8章: 优质因子回测验证")
print("=" * 70)

researcher = AlphaResearcher(
    dataset,
    initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_POSITIONS,
    rebalance_freq=REBALANCE_FREQ,
)
print(f"✅ AlphaResearcher 创建完成")

if len(effective) > 0:
    top_backtest_ids = effective.index[:min(10, len(effective))].tolist()
elif len(candidates) > 0:
    top_backtest_ids = candidates.index[:min(10, len(candidates))].tolist()
else:
    top_backtest_ids = system_eval["ic_mean"].abs().nlargest(5).index.tolist()

print(f"回测因子: {', '.join(top_backtest_ids)}")

# 设置模块级变量用于 fork 并行
_backtest_dataset = dataset
_backtest_params = {
    "initial_capital": INITIAL_CAPITAL,
    "max_positions": MAX_POSITIONS,
    "rebalance_freq": REBALANCE_FREQ,
}

backtest_items = []
for name in top_backtest_ids:
    alpha_id = int(name.replace("Alpha", ""))
    panel = all_factors[alpha_id]
    backtest_items.append((alpha_id, panel, name))

backtest_results = {}
max_w = min(max(os.cpu_count() - 2, 1), len(backtest_items))
ctx = multiprocessing.get_context("fork")
with ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as executor:
    futures = {executor.submit(_run_single_backtest, item): item[2] for item in backtest_items}
    for i, future in enumerate(as_completed(futures), 1):
        try:
            label, r = future.result()
            backtest_results[label] = r
            az = r.get_analyzer()
            print(f"[{i}/{len(backtest_items)}] {label}: 总收益={az.total_return():+.2%}  夏普={az.sharpe_ratio():.3f}")
        except Exception as e:
            name = futures[future]
            print(f"[{i}/{len(backtest_items)}] {name}: ⚠️ 失败: {e}")

print(f"\n✅ 完成 {len(backtest_results)}/{len(backtest_items)} 个因子回测")

if backtest_results:
    metrics = researcher.metrics_table(backtest_results)
    print(f"\n绩效对比表:")
    print(metrics[["年化收益率", "最大回撤", "夏普比率", "卡玛比率", "Alpha"]].to_string(
        float_format=lambda x: f"{x:+.4f}"
    ))

    best_name = metrics["夏普比率"].idxmax()
    print(f"\n最优因子: {best_name}  夏普比率={metrics.loc[best_name, '夏普比率']:.3f}")

# ======================================================================
# 9. 研究结论
# ======================================================================
print("\n" + "=" * 70)
print("第9章: 研究结论 — 与 CSI300 2022-2024 对比")
print("=" * 70)
print(f"""
研究设计:
  本次: 全A股 (排除科创/北交) × 2010-2016
  对照: 沪深300 × 2022-2024

评价方法论:
  IC 分析:     Rank IC (Spearman) — IC均值, ICIR > 0.5
  Factor Return: 截面回归 OLS — FR_IR, 年化 FR
  统计检验:     T 统计量 — |T| > 2.0
  多周期:       T+1 ~ T+5 扫描 — IC 衰减速度
  去冗余:       FR 相关矩阵 — |r| > 0.7 标记冗余

复现结果:
  有效因子 (IC+T 均显著): {n_effective} / {len(full_eval)} 个
  IC 显著因子:            {full_eval['IC显著'].sum()} 个
  T 显著因子:             {full_eval['T显著'].sum()} 个
""")

print("✅ Alpha191 国泰君安复现研究完成！")
print(f"📊 图表已保存到 {OUTPUT_DIR}/ 目录")
