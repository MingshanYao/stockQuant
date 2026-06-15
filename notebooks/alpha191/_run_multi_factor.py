"""多因子组合回测 — 基于 Alpha191 全A股 2010-2016 复现结果。

在单因子评价的基础上，测试多因子组合策略：
  1. 去冗余：从 18 个有效因子中去除高相关因子对（|r| > 0.7）
  2. 方向统一：IC < 0 的因子翻转方向
  3. 组合方法：等权合成 / ICIR 加权 / 排名合成
  4. 组合规模：Top3 / Top5 / 全部独立因子
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

print("=" * 70)
print("Alpha191 多因子组合回测 — 全A股 2010-2016")
print("=" * 70)

# ======================================================================
# 1. 加载数据（与单因子复现相同）
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
# 3. 筛选有效因子 & 去冗余
# ======================================================================
print("\n第3步: 筛选有效因子 & 去冗余...")

ic_significant = system_eval["ic_ir"].abs() > 0.5
t_significant = system_eval["t_stat"].abs() > 2.0
effective_mask = ic_significant & t_significant
effective_names = system_eval[effective_mask].index.tolist()
print(f"  有效因子（IC+T 均显著）: {len(effective_names)} 个")
print(f"  {', '.join(effective_names)}")

effective_dict = {n: named_factors[n] for n in effective_names}
corr_matrix = evaluator.factor_correlation_matrix(effective_dict, forward_period=1)

REDUNDANCY_THRESHOLD = 0.7

ic_ir_values = system_eval.loc[effective_names, "ic_ir"]

# 贪心去冗余：按 |ICIR| 降序逐个加入，跳过与已加入因子高相关的
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

print(f"\n  去冗余后独立因子: {len(independent_factors)} 个")
for i, name in enumerate(independent_factors, 1):
    ic_val = system_eval.loc[name, "ic_mean"]
    icir_val = system_eval.loc[name, "ic_ir"]
    direction = "+" if ic_val > 0 else "-"
    print(f"    {i}. {name}  IC={ic_val:+.4f}  ICIR={icir_val:+.4f}  方向={direction}")

# ======================================================================
# 4. 因子方向统一 & 面板准备
# ======================================================================
print("\n第4步: 统一因子方向（IC<0 的因子翻转）...")

aligned_panels = {}
for name in independent_factors:
    panel = named_factors[name].copy()
    ic_val = system_eval.loc[name, "ic_mean"]
    if ic_val < 0:
        aligned_panels[name] = -panel
        print(f"  {name}: IC={ic_val:+.4f} → 翻转 (×-1)")
    else:
        aligned_panels[name] = panel
        print(f"  {name}: IC={ic_val:+.4f} → 保持原方向")

# ======================================================================
# 5. 构建多因子组合
# ======================================================================
print("\n第5步: 构建多因子组合...")

def zscore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日截面标准化。"""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    std = std.replace(0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)

def rank_pct_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日截面排名百分位。"""
    return panel.rank(axis=1, pct=True)

def combine_equal(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """等权合成：标准化后简单平均，要求至少半数因子有值。"""
    zscored = [zscore_panel(p) for p in panels.values()]
    stacked = np.stack([z.values for z in zscored], axis=0)
    n_valid = np.sum(np.isfinite(stacked), axis=0)
    min_required = max(len(zscored) // 2, 1)
    composite = np.nanmean(stacked, axis=0)
    composite[n_valid < min_required] = np.nan
    return pd.DataFrame(composite, index=zscored[0].index, columns=zscored[0].columns)

def combine_icir_weighted(panels: dict[str, pd.DataFrame], icir: dict[str, float]) -> pd.DataFrame:
    """ICIR 加权合成，要求至少半数因子有值。"""
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

def combine_rank(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """排名合成：截面排名百分位后平均，要求至少半数因子有值。"""
    ranked = [rank_pct_panel(p) for p in panels.values()]
    stacked = np.stack([r.values for r in ranked], axis=0)
    n_valid = np.sum(np.isfinite(stacked), axis=0)
    min_required = max(len(ranked) // 2, 1)
    composite = np.nanmean(stacked, axis=0)
    composite[n_valid < min_required] = np.nan
    return pd.DataFrame(composite, index=ranked[0].index, columns=ranked[0].columns)

icir_dict = {n: system_eval.loc[n, "ic_ir"] for n in independent_factors}

# 按 |ICIR| 排序确定 Top3 / Top5
sorted_independent = sorted(independent_factors, key=lambda n: abs(icir_dict[n]), reverse=True)
top3_names = sorted_independent[:3]
top5_names = sorted_independent[:5]
all_names = sorted_independent

print(f"  Top 3: {', '.join(top3_names)}")
print(f"  Top 5: {', '.join(top5_names)}")
print(f"  全部:  {', '.join(all_names)}")

# 构建 9 个组合
combos = {}
for size_label, factor_names in [("Top3", top3_names), ("Top5", top5_names), ("All", all_names)]:
    subset = {n: aligned_panels[n] for n in factor_names}
    subset_icir = {n: icir_dict[n] for n in factor_names}

    combos[f"{size_label}_等权"] = combine_equal(subset)
    combos[f"{size_label}_ICIR加权"] = combine_icir_weighted(subset, subset_icir)
    combos[f"{size_label}_排名"] = combine_rank(subset)
    print(f"  {size_label}: 3 种组合方法已构建")

print(f"\n  共 {len(combos)} 个组合策略")

# ======================================================================
# 6. 评价组合因子 IC/ICIR
# ======================================================================
print("\n第6步: 评价组合因子 IC / ICIR...")
combo_eval = evaluator.evaluate_system(combos, forward_period=1, neutralize=False)

ic_cols = ["ic_mean", "ic_ir", "fr_annual", "t_stat"]
combo_summary = combo_eval[ic_cols].copy()
combo_summary.columns = ["IC均值", "ICIR", "年化FR", "T统计量"]
combo_summary = combo_summary.sort_values("ICIR", ascending=False)
print(combo_summary.to_string(float_format=lambda x: f"{x:+.4f}"))

# 加入最优单因子 Alpha032 作为基线
alpha032_eval = system_eval.loc["Alpha032", ic_cols]
alpha032_eval.index = ["IC均值", "ICIR", "年化FR", "T统计量"]
print(f"\n基线 Alpha032:  IC={alpha032_eval['IC均值']:+.4f}  ICIR={alpha032_eval['ICIR']:+.4f}  T={alpha032_eval['T统计量']:+.1f}")

# ======================================================================
# 7. 回测
# ======================================================================
print("\n第7步: 回测（9 个组合 + 1 个基线）...")
print("  注意: 每个回测约需 15-30 分钟（2255 只股票 × 1700 天）")

researcher = AlphaResearcher(
    dataset,
    initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_POSITIONS,
    rebalance_freq=REBALANCE_FREQ,
)

backtest_results = {}

# 基线：Alpha032（方向已统一，IC>0 直接用原值）
print(f"\n[0/10] 回测基线 Alpha032...", end="")
try:
    r = researcher.run_backtest(alpha_panel=all_factors[32], label="Alpha032_单因子")
    backtest_results["Alpha032_单因子"] = r
    az = r.get_analyzer()
    print(f"  年化: {az.annualized_return():+.2%}  夏普: {az.sharpe_ratio():.3f}")
except Exception as e:
    print(f"  失败: {e}")

# 9 个组合
for i, (name, panel) in enumerate(combos.items(), 1):
    print(f"[{i}/10] 回测 {name}...", end="")
    try:
        r = researcher.run_backtest(alpha_panel=panel, label=name)
        backtest_results[name] = r
        az = r.get_analyzer()
        print(f"  年化: {az.annualized_return():+.2%}  夏普: {az.sharpe_ratio():.3f}")
    except Exception as e:
        print(f"  失败: {e}")

# ======================================================================
# 8. 绩效对比
# ======================================================================
print("\n" + "=" * 70)
print("第8步: 绩效对比表")
print("=" * 70)

if backtest_results:
    metrics = researcher.metrics_table(backtest_results)
    display_cols = ["年化收益率", "最大回撤", "夏普比率", "卡玛比率", "Alpha"]
    available_cols = [c for c in display_cols if c in metrics.columns]
    print(metrics[available_cols].to_string(float_format=lambda x: f"{x:+.4f}"))

    best = metrics["夏普比率"].idxmax()
    print(f"\n最优策略: {best}  夏普={metrics.loc[best, '夏普比率']:.3f}")

    # 对比提升
    if "Alpha032_单因子" in metrics.index:
        baseline_sharpe = metrics.loc["Alpha032_单因子", "夏普比率"]
        best_sharpe = metrics.loc[best, "夏普比率"]
        print(f"相对 Alpha032 单因子基线 (夏普={baseline_sharpe:.3f}):")
        for name in metrics.index:
            if name == "Alpha032_单因子":
                continue
            s = metrics.loc[name, "夏普比率"]
            diff = s - baseline_sharpe
            print(f"  {name}: 夏普={s:.3f} ({diff:+.3f})")

# ======================================================================
# 9. 净值曲线图
# ======================================================================
print("\n第9步: 绘制净值曲线对比图...")

if len(backtest_results) >= 2:
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(backtest_results)))

    for (name, result), color in zip(backtest_results.items(), colors):
        az = result.get_analyzer()
        equity = az.equity
        if equity is not None and len(equity) > 0:
            values = equity["total_value"].values
            normalized = values / values[0]
            linewidth = 2.5 if "单因子" in name else 1.5
            linestyle = "--" if "单因子" in name else "-"
            ax.plot(equity.index, normalized,
                    label=name, color=color, linewidth=linewidth, linestyle=linestyle)

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("净值（归一化）", fontsize=12)
    ax.set_title("多因子组合 vs 单因子基线 净值曲线对比 (全A 2010-2016)", fontsize=14)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/multi_factor_equity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  已保存: multi_factor_equity.png")

# ======================================================================
# 10. 总结
# ======================================================================
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(f"""
多因子组合回测完成:
  独立因子池: {len(independent_factors)} 个（去冗余后）
  组合方法:   等权合成 / ICIR加权 / 排名合成
  组合规模:   Top3 / Top5 / 全部
  回测策略:   {len(backtest_results)} 个（含基线）

图表已保存到 {OUTPUT_DIR}/ 目录
""")

print("多因子组合回测完成！")
