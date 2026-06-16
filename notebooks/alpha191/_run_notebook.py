"""运行 alpha191_hs300_research notebook 的等效脚本。"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
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

START_DATE      = "2022-01-01"
END_DATE        = "2024-12-31"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS   = 10
REBALANCE_FREQ  = 5

ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS]

print("=" * 70)
print("Alpha191 因子系统研究 — 沪深300 因子评价体系")
print("=" * 70)
print(f"研究因子: {len(ALL_ALPHA_IDS)} 个 Alpha191 因子（排除 Alpha030）")
print(f"回测区间: {START_DATE} ~ {END_DATE}")

# ======================================================================
# 2. 加载沪深300数据
# ======================================================================
print("\n" + "=" * 70)
print("第2章: 加载沪深300数据")
print("=" * 70)

dataset = (
    StockUniverse()
    .scope(Pool.CSI300)
    .exclude(Pool.STAR, Pool.CHINEXT, Pool.BSE)
    .load(START_DATE, END_DATE, benchmark=Pool.CSI300)
)

print(dataset.summary())
print(f"\n基准指数: {dataset.benchmark_code}  |  {len(dataset.benchmark)} 条日线")

# ======================================================================
# 3. 构建 Alpha191 面板引擎 & 批量计算因子
# ======================================================================
print("\n" + "=" * 70)
print("第3章: 构建 Alpha191 面板引擎 & 批量计算因子")
print("=" * 70)

print("构建 Alpha191 面板引擎...")
engine = Alpha191Indicators.from_dataset(dataset)
print(f"✅ 引擎构建完成: {engine.close.shape[0]} 个交易日 × {engine.close.shape[1]} 只股票")

print("\n批量计算全部因子面板...")
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
print(f"✅ 完成: 成功计算 {len(all_factors)} 个因子")

sample_id = 14
print(f"\nAlpha{sample_id:03d} 面板预览:")
print(all_factors[sample_id].tail(3).iloc[:, :6])

# 因子覆盖率统计
coverage = {}
for alpha_id, panel in all_factors.items():
    finite_ratio = np.isfinite(panel.values).mean()
    coverage[alpha_id] = finite_ratio

coverage_s = pd.Series(coverage).sort_values()
print(f"\n因子覆盖率统计:")
print(f"  最低: Alpha{coverage_s.index[0]:03d} = {coverage_s.iloc[0]:.1%}")
print(f"  中位: {coverage_s.median():.1%}")
print(f"  最高: Alpha{coverage_s.index[-1]:03d} = {coverage_s.iloc[-1]:.1%}")
print(f"  覆盖率 < 50% 的因子: {(coverage_s < 0.5).sum()} 个")

MIN_COVERAGE = 0.3
valid_factors = {k: v for k, v in all_factors.items() if coverage.get(k, 0) >= MIN_COVERAGE}
print(f"\n有效因子（覆盖率 >= {MIN_COVERAGE:.0%}）: {len(valid_factors)} / {len(all_factors)} 个")

# ======================================================================
# 4. IC / ICIR 分析
# ======================================================================
print("\n" + "=" * 70)
print("第4章: IC / ICIR 分析")
print("=" * 70)

evaluator = FactorEvaluator(close_panel=engine.close, style_factors="auto")
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}

print(f"计算 {len(named_factors)} 个因子的 IC/ICIR (T+1)...")
system_eval = evaluator.evaluate_system(named_factors, forward_period=1)
print("✅ 完成")

ic_cols = ["ic_mean", "ic_std", "ic_ir", "ic_pos_ratio", "n_periods"]
ic_summary = system_eval[ic_cols].copy()
ic_summary.columns = ["IC均值", "IC标准差", "ICIR", "IC>0占比", "有效天数"]

print(f"\nTop 20 因子 — IC/ICIR 排名（按 |IC均值| 降序）")
print(ic_summary.head(20).to_string(float_format=lambda x: f"{x:.4f}"))

# IC/ICIR 分布图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
ic_values = system_eval["ic_mean"].dropna()
ax.hist(ic_values, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("IC 均值", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子 IC 均值分布", fontsize=13)
ax.grid(True, alpha=0.3)

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
plt.savefig("notebooks/alpha191/ic_icir_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: ic_icir_distribution.png")

sig_pos = (system_eval["ic_ir"] > 0.5).sum()
sig_neg = (system_eval["ic_ir"] < -0.5).sum()
print(f"\n显著正向因子（ICIR > 0.5）: {sig_pos} 个")
print(f"显著负向因子（ICIR < -0.5）: {sig_neg} 个")
print(f"显著因子合计: {sig_pos + sig_neg} / {len(system_eval)} 个")

# ======================================================================
# 5. Factor Return 分析
# ======================================================================
print("\n" + "=" * 70)
print("第5章: Factor Return 分析")
print("=" * 70)

fr_cols = ["fr_mean", "fr_std", "fr_ir", "fr_annual", "t_stat"]
fr_summary = system_eval[fr_cols].copy()
fr_summary.columns = ["FR均值", "FR标准差", "FR_IR", "年化FR", "T统计量"]
fr_summary = fr_summary.reindex(fr_summary["T统计量"].abs().sort_values(ascending=False).index)

print(f"Top 20 因子 — Factor Return 排名（按 |T统计量| 降序）")
print(fr_summary.head(20).to_string(float_format=lambda x: f"{x:.4f}"))

# T统计量 + 年化FR 分布图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
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

ax = axes[1]
fr_annual = system_eval["fr_annual"].dropna()
ax.hist(fr_annual * 100, bins=40, color="#9b59b6", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("年化 Factor Return (%)", fontsize=11)
ax.set_ylabel("因子数量", fontsize=11)
ax.set_title("Alpha191 因子年化 FR 分布", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/alpha191/t_stat_fr_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: t_stat_fr_distribution.png")

sig_t = (system_eval["t_stat"].abs() > 2.0).sum()
print(f"\nT统计量显著（|T| > 2.0）的因子: {sig_t} / {len(system_eval)} 个")

# ======================================================================
# 6. 多周期检验 (T+1 ~ T+5)
# ======================================================================
print("\n" + "=" * 70)
print("第6章: 多周期检验 (T+1 ~ T+5)")
print("=" * 70)

top_n = 20
top_factors_by_ic = system_eval["ic_mean"].abs().nlargest(top_n).index.tolist()
print(f"选取 Top {top_n} 因子（按 |IC|）做多周期检验:")
print(f"  {', '.join(top_factors_by_ic[:10])}")
print(f"  {', '.join(top_factors_by_ic[10:])}")

multi_horizon_results = {}
for i, name in enumerate(top_factors_by_ic, 1):
    print(f"  [{i}/{top_n}] 多周期评价: {name}...", end="\r")
    result = evaluator.evaluate_multi_horizon(named_factors[name], periods=[1, 2, 3, 4, 5])
    multi_horizon_results[name] = result
print(f"✅ 完成 {top_n} 个因子多周期评价              ")

# IC 衰减热力图
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
plt.savefig("notebooks/alpha191/ic_decay_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: ic_decay_heatmap.png")

# ICIR 衰减热力图
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
plt.savefig("notebooks/alpha191/icir_decay_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: icir_decay_heatmap.png")

print("\nIC 衰减模式分析:")
for name in ic_decay.index[:5]:
    row = ic_decay.loc[name].astype(float)
    peak = row.abs().idxmax()
    print(f"  {name}: 最强周期 {peak} (IC={row[peak]:.4f})")

# ======================================================================
# 7. 因子体系评价 — 完整四指标汇总
# ======================================================================
print("\n" + "=" * 70)
print("第7章: 因子体系评价 — 完整四指标汇总")
print("=" * 70)

full_eval = system_eval.copy()
full_eval.columns = ["IC均值", "IC标准差", "ICIR", "IC>0占比", "FR均值", "FR标准差", "FR_IR", "年化FR", "T统计量", "有效天数"]

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

# 四象限散点图
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
ax.set_title("Alpha191 因子四象限图（ICIR vs T统计量）", fontsize=14)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("notebooks/alpha191/quadrant_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 已保存: quadrant_scatter.png")

# ======================================================================
# 8. 因子去冗余 — Factor Return 相关矩阵
# ======================================================================
print("\n" + "=" * 70)
print("第8章: 因子去冗余 — Factor Return 相关矩阵")
print("=" * 70)

# 用 IC 或 T 显著的因子做相关性分析
candidates = full_eval[full_eval["IC显著"] | full_eval["T显著"]]
if len(candidates) >= 2:
    candidate_dict = {name: named_factors[name] for name in candidates.index if name in named_factors}
    print(f"计算 {len(candidate_dict)} 个显著因子的 FR 相关矩阵...")
    corr_matrix = evaluator.factor_correlation_matrix(candidate_dict, forward_period=1)
    print("✅ 完成")

    fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix) * 0.5), max(6, len(corr_matrix) * 0.4)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=len(corr_matrix) <= 20,
        fmt=".2f" if len(corr_matrix) <= 20 else "",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, square=True,
    )
    ax.set_title("显著因子 Factor Return 相关矩阵", fontsize=13)
    plt.tight_layout()
    plt.savefig("notebooks/alpha191/fr_correlation_matrix.png", dpi=150, bbox_inches="tight")
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
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {a} ↔ {b}: r = {r:.3f}")
    else:
        print(f"\n无高相关因子对（阈值 |r| > {HIGH_CORR_THRESHOLD}），因子间独立性较好")
else:
    print("⚠️ 显著因子不足 2 个，跳过相关性分析")

# ======================================================================
# 9. 优质因子回测验证
# ======================================================================
print("\n" + "=" * 70)
print("第9章: 优质因子回测验证")
print("=" * 70)

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
elif len(candidates) > 0:
    top_backtest_ids = candidates.index[:min(10, len(candidates))].tolist()
else:
    top_backtest_ids = system_eval["ic_mean"].abs().nlargest(5).index.tolist()

print(f"回测因子: {', '.join(top_backtest_ids)}")

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

if backtest_results:
    metrics = researcher.metrics_table(backtest_results)
    print(f"\n绩效对比表:")
    print(metrics[["年化收益率", "最大回撤", "夏普比率", "卡玛比率", "Alpha"]].to_string(
        float_format=lambda x: f"{x:+.4f}"
    ))

    best_name = metrics["夏普比率"].idxmax()
    print(f"\n最优因子: {best_name}  夏普比率={metrics.loc[best_name, '夏普比率']:.3f}")

# ======================================================================
# 10. 研究结论
# ======================================================================
print("\n" + "=" * 70)
print("第10章: 研究结论")
print("=" * 70)
print(f"""
评价方法论总结:
  IC 分析:     Rank IC (Spearman) — IC均值, ICIR > 0.5
  Factor Return: 截面回归 OLS — FR_IR, 年化 FR
  统计检验:     T 统计量 — |T| > 2.0
  多周期:       T+1 ~ T+5 扫描 — IC 衰减速度
  去冗余:       FR 相关矩阵 — |r| > 0.7 标记冗余

研究发现:
  有效因子 (IC+T 均显著): {n_effective} / {len(full_eval)} 个
  IC 显著因子:            {full_eval['IC显著'].sum()} 个
  T 显著因子:             {full_eval['T显著'].sum()} 个
""")

print("✅ Alpha191 因子评价体系研究完成！")
print(f"📊 图表已保存到 notebooks/alpha191/ 目录")
