"""Alpha191 回测复现 — 跳过 BARRA 中性化，快速出结果。

与 _run_replication.py 的区别：
- 不做 FactorEvaluator (BARRA风格因子 + 行业中性化 IC)，该阶段 CPU 极重
- 直接：数据加载 → 因子计算 → 简单 IC 排名 → 回测
"""
import gc
import os
import warnings
warnings.filterwarnings("ignore")

import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from stockquant.data.universe import Pool, StockUniverse, BacktestDataset
from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS
from stockquant.research import AlphaResearcher

# 模块级变量用于 fork 模式回测并行
_backtest_dataset = None
_backtest_params = None


def _run_single_backtest(args: tuple):
    """模块级函数：运行单个因子回测。"""
    alpha_id, panel, label = args
    researcher = AlphaResearcher(_backtest_dataset, **_backtest_params)
    r = researcher.run_backtest(alpha_panel=panel, label=label)
    return label, r


START_DATE = "2010-01-01"
END_DATE = "2016-12-31"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 50
REBALANCE_FREQ = 5
TOP_N_BACKTEST = 50  # 回测 Top N 因子（按 |IC| 排名）

ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS]

OUTPUT_DIR = "notebooks/alpha191/replication"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Alpha191 回测复现 — 全A股 2010-2016 (快速模式)")
print("=" * 70)
print(f"因子数量: {len(ALL_ALPHA_IDS)} (SKIP_ALPHAS={SKIP_ALPHAS})")
print(f"回测区间: {START_DATE} ~ {END_DATE}")

# ======================================================================
# 1. 加载全A股数据
# ======================================================================
print("\n" + "=" * 70)
print("第1章: 加载全A股数据")
print("=" * 70)

dataset = (
    StockUniverse()
    .scope(Pool.ALL_A)
    .exclude(Pool.STAR, Pool.BSE)
    .load(START_DATE, END_DATE, benchmark="000300")
)
print(dataset.summary())

# ======================================================================
# 2. 计算全部因子
# ======================================================================
print("\n" + "=" * 70)
print("第2章: 批量计算因子")
print("=" * 70)

engine = Alpha191Indicators.from_dataset(dataset)
print(f"引擎: {engine.close.shape[0]} 天 × {engine.close.shape[1]} 股")

all_factors = engine.compute_factors(ALL_ALPHA_IDS)
print(f"完成: {len(all_factors)} 个因子")

# 覆盖率
coverage = {}
for alpha_id, panel in all_factors.items():
    coverage[alpha_id] = np.isfinite(panel.values).mean()

cov_s = pd.Series(coverage).sort_values()
print(f"覆盖率: 最低 {cov_s.index[0]:03d}={cov_s.iloc[0]:.1%}  中位={cov_s.median():.1%}  最高 {cov_s.index[-1]:03d}={cov_s.iloc[-1]:.1%}")

# ======================================================================
# 3. 简单 IC 排名 (无中性化，纯截面 Spearman)
# ======================================================================
print("\n" + "=" * 70)
print("第3章: 快速 IC 排名 (截面 Rank IC，无风格中性化)")
print("=" * 70)

fwd_ret = engine.close.pct_change().shift(-1)  # T+1 forward return
# 统一日期交集
common_idx = engine.close.index.intersection(fwd_ret.dropna(how="all").index)
print(f"  有效截面: {len(common_idx)} 天")

ic_results = {}
for alpha_id, panel in all_factors.items():
    if coverage.get(alpha_id, 0) < 0.3:
        continue
    # 截面 Rank IC: spearman corr per date
    f_aligned = panel.loc[panel.index.intersection(common_idx)]
    r_aligned = fwd_ret.loc[fwd_ret.index.intersection(common_idx)]
    ic_series = f_aligned.corrwith(r_aligned, axis=1, method="spearman").dropna()
    ic_results[alpha_id] = {
        "ic_mean": ic_series.mean(),
        "ic_ir": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan,
    }

ic_df = pd.DataFrame(ic_results).T
ic_df = ic_df.sort_values("ic_mean", key=abs, ascending=False)
print(f"Top 10 |IC| 因子:")
for aid, row in ic_df.head(10).iterrows():
    print(f"  Alpha{aid:03d}: IC={row['ic_mean']:+.4f}, ICIR={row['ic_ir']:+.3f}")

# ======================================================================
# 4. 回测 Top N 因子
# ======================================================================
print("\n" + "=" * 70)
print(f"第4章: 回测 Top {TOP_N_BACKTEST} 因子")
print("=" * 70)

top_ids = ic_df.index[:TOP_N_BACKTEST].tolist()
print(f"回测因子: {[f'Alpha{aid:03d}' for aid in top_ids]}")

researcher = AlphaResearcher(
    dataset,
    initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_POSITIONS,
    rebalance_freq=REBALANCE_FREQ,
)

_backtest_dataset = dataset
_backtest_params = {
    "initial_capital": INITIAL_CAPITAL,
    "max_positions": MAX_POSITIONS,
    "rebalance_freq": REBALANCE_FREQ,
}

backtest_items = [(aid, all_factors[aid], f"Alpha{aid:03d}") for aid in top_ids]
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
            print(f"[{i}/{len(backtest_items)}] {label}: "
                  f"收益={az.total_return():+.2%}  "
                  f"年化={az.annualized_return():+.2%}  "
                  f"夏普={az.sharpe_ratio():.3f}  "
                  f"最大回撤={az.max_drawdown():.2%}  "
                  f"卡玛={az.calmar_ratio():.3f}")
        except Exception as e:
            name = futures[future]
            print(f"[{i}/{len(backtest_items)}] {name}: ⚠️ 失败: {e}")

print(f"\n完成: {len(backtest_results)}/{len(backtest_items)} 个因子回测")

# ======================================================================
# 5. 绩效汇总
# ======================================================================
print("\n" + "=" * 70)
print("绩效汇总")
print("=" * 70)

if backtest_results:
    metrics = researcher.metrics_table(backtest_results)
    cols = ["年化收益率", "最大回撤", "夏普比率", "卡玛比率", "Alpha", "Beta"]
    available = [c for c in cols if c in metrics.columns]
    print(metrics[available].to_string(float_format=lambda x: f"{x:+.4f}"))

    best_name = metrics["夏普比率"].idxmax()
    print(f"\n最优因子: {best_name}  夏普比率={metrics.loc[best_name, '夏普比率']:.3f}")
    print(f"总收益最强: {metrics['年化收益率'].idxmax()}  年化={metrics['年化收益率'].max():+.4f}")

# 保存 CSV
ic_df.to_csv(f"{OUTPUT_DIR}/ic_ranking.csv")
if backtest_results:
    metrics.to_csv(f"{OUTPUT_DIR}/backtest_metrics.csv")

print(f"\n✅ 完成！结果保存到 {OUTPUT_DIR}/")
print("  - ic_ranking.csv")
print("  - backtest_metrics.csv")
