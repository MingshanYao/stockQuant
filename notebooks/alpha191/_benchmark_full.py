"""端到端性能基准：1 个组合 × 1 个风控配置，全量 2255 股 × 1700 天"""
import warnings, os, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from stockquant.data.database import Database
from stockquant.data.universe import BacktestDataset
from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS
from stockquant.analysis.evaluator import FactorEvaluator
from stockquant.research import AlphaResearcher

START_DATE = "2010-01-01"
END_DATE   = "2016-12-31"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS   = 50
REBALANCE_FREQ  = 5
ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS]

t0 = time.perf_counter()

# 1. 加载数据
print("1. 加载数据...", flush=True)
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
    stock_data=stock_data, codes=list(stock_data.keys()),
    benchmark=benchmark_df, benchmark_code=benchmark_code,
    start_date=START_DATE, end_date=END_DATE,
)
t1 = time.perf_counter()
print(f"   {len(stock_data)} 只股票, {len(benchmark_df)} 个交易日 ({t1-t0:.1f}s)", flush=True)

# 2. 计算因子
print("2. 计算 Alpha191 因子...", flush=True)
engine = Alpha191Indicators.from_dataset(dataset)
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
t2 = time.perf_counter()
print(f"   {len(all_factors)} 个因子 ({t2-t1:.1f}s)", flush=True)

# 3. 筛选 & 组合
print("3. 筛选有效因子 & 构建组合...", flush=True)
MIN_COVERAGE = 0.3
valid_factors = {k: v for k, v in all_factors.items() if np.isfinite(v.values).mean() >= MIN_COVERAGE}
evaluator = FactorEvaluator(close_panel=engine.close)
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}
system_eval = evaluator.evaluate_system(named_factors, forward_period=1, neutralize=False)

ic_sig = system_eval["ic_ir"].abs() > 0.5
t_sig = system_eval["t_stat"].abs() > 2.0
effective_names = system_eval[ic_sig & t_sig].index.tolist()
print(f"   有效因子: {len(effective_names)} 个", flush=True)

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
print(f"   去冗余后: {len(independent_factors)} 个", flush=True)

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

icir_dict = {n: system_eval.loc[n, "ic_ir"] for n in independent_factors}
sorted_independent = sorted(independent_factors, key=lambda n: abs(icir_dict[n]), reverse=True)
all_names = sorted_independent
all_subset = {n: aligned_panels[n] for n in all_names}
combo_panel = combine_equal(all_subset)

t3 = time.perf_counter()
print(f"   ({t3-t2:.1f}s)", flush=True)

# 4. 回测（1 个风控配置）
print("4. 回测（All_等权 + 新框架_默认参数）...", flush=True)
researcher = AlphaResearcher(
    dataset, initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_POSITIONS, rebalance_freq=REBALANCE_FREQ,
    enable_risk_mgmt=True, stop_loss_pct=0.08,
    take_profit_pct=0.20, trailing_stop_pct=0.05,
)
r = researcher.run_backtest(alpha_panel=combo_panel, label="Bench")
t4 = time.perf_counter()

az = r.get_analyzer()
dd = az.max_drawdown()
print(f"   年化: {az.annualized_return():+.2%}  夏普: {az.sharpe_ratio():.3f}  最大回撤: {dd:+.2%}", flush=True)

total = t4 - t0
print(f"\n总耗时: {total:.1f}s ({total/60:.1f}min)")
print(f"  数据加载: {t1-t0:.1f}s")
print(f"  因子计算: {t2-t1:.1f}s")
print(f"  因子筛选: {t3-t2:.1f}s")
print(f"  回测执行: {t4-t3:.1f}s  <-- 原瓶颈")
