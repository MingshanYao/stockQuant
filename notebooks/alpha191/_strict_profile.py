"""严格 Wall-Clock 计时 — 测量全链路真实耗时，不留计时盲区。

每个阶段用 time.monotonic() 记录起始和结束，累计误差。
"""
import time
import sys

# ── 记录脚本启动的绝对时间 ──
SCRIPT_START = time.monotonic()

def wall_clock(label=""):
    """返回从脚本启动到当前的 wall-clock 时间，并打印阶段标记。"""
    elapsed = time.monotonic() - SCRIPT_START
    if label:
        print(f"[T+{elapsed:6.1f}s] {label}", flush=True)
    return elapsed

wall_clock("脚本开始")

import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

wall_clock("import 完成")

from stockquant.data.database import Database
from stockquant.data.universe import BacktestDataset
wall_clock("stockquant.data 导入完成")

from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS
wall_clock("alpha191 导入完成")

from stockquant.analysis.evaluator import FactorEvaluator
wall_clock("evaluator 导入完成")

# ── 配置 ──
START_DATE = "2015-01-01"
END_DATE   = "2016-12-31"
N_STOCKS   = 200
N_FACTORS  = 60
TOP_N_MULTI = 10

# ═══════════════════════════════════════════════════════════════
# Phase 0: 数据加载
# ═══════════════════════════════════════════════════════════════
t0 = time.monotonic()
db = Database()
t1 = time.monotonic()
print(f"\n  数据库连接: {t1 - t0:.2f}s")

bars = db.query(
    "SELECT * FROM daily_bars WHERE date >= ? AND date <= ? ORDER BY code, date",
    [START_DATE, END_DATE],
)
bars["date"] = pd.to_datetime(bars["date"])
t2 = time.monotonic()
print(f"  数据查询: {t2 - t1:.2f}s, {len(bars)} rows")

codes = bars["code"].unique().tolist()
codes = [c for c in codes if not c.startswith(("688", "43", "83", "87"))][:N_STOCKS]

stock_data = {}
for code in codes:
    df = bars[bars["code"] == code].reset_index(drop=True)
    if len(df) >= 60:
        stock_data[code] = df
del bars
codes = list(stock_data.keys())

benchmark_df = db.query(
    "SELECT * FROM index_daily WHERE code = '000905' AND date >= ? AND date <= ? ORDER BY date",
    [START_DATE, END_DATE],
)
benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

dataset = BacktestDataset(
    stock_data=stock_data, codes=codes, benchmark=benchmark_df,
    benchmark_code="000905", start_date=START_DATE, end_date=END_DATE,
)
elapsed = time.monotonic() - t0
wall_clock(f"Phase 0 完成: 数据加载 {elapsed:.2f}s ({len(codes)} stocks × {len(dataset.benchmark)} days)")

# ═══════════════════════════════════════════════════════════════
# Phase 1: 因子计算（含引擎构建）
# ═══════════════════════════════════════════════════════════════
t0 = time.monotonic()
engine = Alpha191Indicators.from_dataset(dataset)
t1 = time.monotonic()
print(f"  引擎构建: {t1 - t0:.2f}s  ({engine.close.shape[0]}d × {engine.close.shape[1]}s)")

ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS][:N_FACTORS]
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
t2 = time.monotonic()
print(f"  因子计算: {t2 - t1:.2f}s ({len(ALL_ALPHA_IDS)} factors)")

# 过滤低覆盖
coverage = {aid: np.isfinite(p.values).mean() for aid, p in all_factors.items()}
valid_factors = {k: v for k, v in all_factors.items() if coverage.get(k, 0) >= 0.3}
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}
print(f"  有效因子: {len(valid_factors)}/{len(all_factors)}")
wall_clock(f"Phase 1 完成: 因子计算 {time.monotonic() - t0:.2f}s")

# ═══════════════════════════════════════════════════════════════
# Phase 2: evaluate_system (IC + FR)
# ═══════════════════════════════════════════════════════════════
t0 = time.monotonic()
evaluator = FactorEvaluator(close_panel=engine.close)
t_ctor = time.monotonic() - t0
print(f"  FactorEvaluator 构造: {t_ctor:.2f}s (含 BARRA 风格因子计算)")

t1 = time.monotonic()
system_eval = evaluator.evaluate_system(named_factors, forward_period=1)
t_sys = time.monotonic() - t1
wall_clock(f"Phase 2 完成: eval_sys ctor={t_ctor:.1f}s + eval={t_sys:.1f}s = {time.monotonic() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# Phase 3: 多周期检验 (新并行方式)
# ═══════════════════════════════════════════════════════════════
from concurrent.futures import ThreadPoolExecutor, as_completed

top_names = system_eval["ic_mean"].abs().nlargest(TOP_N_MULTI).index.tolist()
periods = [1, 2, 3, 4, 5]

t0 = time.monotonic()

# 预中性化
top_neut = {}
for i, name in enumerate(top_names):
    t_f = time.monotonic()
    top_neut[name] = evaluator._neutralize_panel(named_factors[name])
    print(f"  中性化因子 [{i+1}/{len(top_names)}] {name}: {time.monotonic() - t_f:.2f}s", flush=True)

fwd_neut_period = {}
for p in periods:
    t_f = time.monotonic()
    fwd_neut_period[p] = evaluator._neutralize_panel(evaluator._forward_returns(p))
    print(f"  中性化 fwd T+{p}: {time.monotonic() - t_f:.2f}s", flush=True)

# 预排名 fwd —— 每个周期 rank 一次，所有因子复用
fwd_ranked_period = {}
for p, panel in fwd_neut_period.items():
    t_f = time.monotonic()
    fwd_ranked_period[p] = panel.rank(axis=1)
    print(f"  排名 fwd T+{p}: {time.monotonic() - t_f:.2f}s", flush=True)

t_prep = time.monotonic() - t0
print(f"  预中性化总计: {t_prep:.2f}s")

# 并行 evaluate（IC 走 numpy 快速路径）
def _eval_one(tup):
    name, p = tup
    return name, p, evaluator.evaluate(
        named_factors[name], forward_period=p,
        factor_neutral=top_neut[name], fwd_neutral=fwd_neut_period[p],
        fwd_ranked=fwd_ranked_period[p],
    )

t_eval_start = time.monotonic()
items = [(n, p) for n in top_names for p in periods]
with ThreadPoolExecutor(max_workers=min(12, len(items))) as ex:
    futures = {ex.submit(_eval_one, item): item for item in items}
    for future in as_completed(futures):
        name, p, r = future.result()

t_eval = time.monotonic() - t_eval_start
print(f"  并行 evaluate: {t_eval:.2f}s ({len(items)} calls)")
wall_clock(f"Phase 3 完成: multi_horizon prep={t_prep:.1f}s + eval={t_eval:.1f}s = {time.monotonic() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# Phase 4: _neutralize_panel 微基准
# ═══════════════════════════════════════════════════════════════
sample_panel = list(named_factors.values())[0]
fwd_returns = evaluator._forward_returns(1)

# 首次调用（含 pool 创建）
t0 = time.monotonic()
_ = evaluator._neutralize_panel(fwd_returns)
t_first = time.monotonic() - t0
print(f"\n  首次 neutralize (含 pool 创建): {t_first:.2f}s")

# 后续调用（pool 复用）
t0 = time.monotonic()
for _ in range(10):
    _ = evaluator._neutralize_panel(sample_panel)
t_reuse = (time.monotonic() - t0) / 10
print(f"  复用 pool neutralize (avg 10): {t_reuse:.2f}s/call")

# 复杂度测试
print(f"\n  面板大小 → 耗时 (avg 3 runs):")
for nr, nc in [(100, 50), (250, 100), (500, 200), (1000, 200)]:
    tp = pd.DataFrame(np.random.randn(nr, nc),
                      index=[f"D{i}" for i in range(nr)],
                      columns=[f"S{i}" for i in range(nc)])
    t0 = time.monotonic()
    for _ in range(3):
        _ = evaluator._neutralize_panel(tp)
    et = (time.monotonic() - t0) / 3
    print(f"    {nr:5d}×{nc:4d} = {nr*nc:8d} cells → {et:.3f}s  ({nr*nc/et:.0f} cells/s)")

# ═══════════════════════════════════════════════════════════════
# Phase 5: 预中性化 批量 (并行)
# ═══════════════════════════════════════════════════════════════
full_eval = system_eval.copy()
full_eval["IC显著"] = full_eval["ic_ir"].abs() > 0.5
full_eval["T显著"] = full_eval["t_stat"].abs() > 2.0
candidates = full_eval[full_eval["IC显著"] | full_eval["T显著"]]
candidate_dict = {name: named_factors[name]
                  for name in candidates.index if name in named_factors}
print(f"\n  候选因子: {len(candidate_dict)} 个")

t0 = time.monotonic()
print(f"  批量中性化 {len(candidate_dict)} 个因子...", flush=True)
neut_cache = evaluator._neutralize_panels_batch(candidate_dict)
gc.collect()

elapsed_batch = time.monotonic() - t0
wall_clock(f"Phase 5 完成: 批量预中性化 {elapsed_batch:.2f}s ({len(neut_cache)} factors)")

# ═══════════════════════════════════════════════════════════════
# Phase 6: 因子体系评价
# ═══════════════════════════════════════════════════════════════
fwd_neut_cache = evaluator._neutralize_panel(fwd_returns)
fwd_ranked_cache = fwd_neut_cache.rank(axis=1)  # 预排名供 Spearman 快速路径

t6_start = time.monotonic()

# 6a
t0 = time.monotonic()
model_ic = evaluator.evaluate_model_predictive_power(
    candidate_dict, forward_period=1,
    neutralized_factors=neut_cache, fwd_neutral=fwd_neut_cache,
    fwd_ranked=fwd_ranked_cache,
)
t_mp = time.monotonic() - t0
print(f"\n  model_predictive_power: {t_mp:.2f}s  IC={model_ic.get('ic_mean', np.nan):.4f}")

# 6b — 使用 system_eval 的 FR IR 跳过 Phase 2 重算
t0 = time.monotonic()
fr_ir_map = system_eval["fr_ir"].abs().to_dict()
count_df = evaluator.factor_count_analysis(
    candidate_dict, forward_period=1,
    neutralized_factors=neut_cache, fwd_neutral=fwd_neut_cache,
    fr_ir_map=fr_ir_map,
    fwd_ranked=fwd_ranked_cache,
)
t_fca = time.monotonic() - t0
print(f"  factor_count_analysis (skip Phase2): {t_fca:.2f}s  ({len(count_df)} steps, final ICIR={count_df['ic_ir'].iloc[-1]:.4f})")

# 6c
t0 = time.monotonic()
corr_matrix = evaluator.factor_correlation_matrix(
    candidate_dict, forward_period=1,
    neutralized_factors=neut_cache, fwd_neutral=fwd_neut_cache,
)
t_corr = time.monotonic() - t0
print(f"  factor_correlation_matrix: {t_corr:.2f}s  ({len(corr_matrix)}×{len(corr_matrix)})")

wall_clock(f"Phase 6 完成: {time.monotonic() - t6_start:.1f}s total")

# ═══════════════════════════════════════════════════════════════
# Phase 7: 回测
# ═══════════════════════════════════════════════════════════════
from stockquant.research import AlphaResearcher
from stockquant.analysis.evaluator import close_pool

researcher = AlphaResearcher(dataset, initial_capital=1_000_000, max_positions=50, rebalance_freq=5)
sample_name, _ = list(candidate_dict.items())[0]
alpha_id = int(sample_name.replace("Alpha", ""))

t0 = time.monotonic()
result = researcher.run_backtest(alpha_panel=all_factors[alpha_id], label=sample_name)
t_bt = time.monotonic() - t0
az = result.get_analyzer()
print(f"\n  回测: {t_bt:.2f}s  {sample_name}: 收益={az.total_return():+.2%} 夏普={az.sharpe_ratio():.3f}")

# 清理
del neut_cache, fwd_neut_cache
close_pool()
gc.collect()

# ═══════════════════════════════════════════════════════════════
# 最终汇总
# ═══════════════════════════════════════════════════════════════
TOTAL_WALL = time.monotonic() - SCRIPT_START
print(f"\n{'='*60}")
print(f"  TOTAL WALL-CLOCK: {TOTAL_WALL:.1f}s ({TOTAL_WALL/60:.1f}min)")
print(f"{'='*60}")

# 打印关键时间点
print(f"""
  时序分解:
    导入 + 初始化:     ~{SCRIPT_START - SCRIPT_START:.0f}s (计入 TOTAL)
    Phase 0 数据加载:  已计入
    Phase 1 因子计算:  已计入
    Phase 2 eval_system: 已计入
    Phase 3 multi_horizon: 已计入
    Phase 5 预中性化:  已计入
    Phase 6 体系评价:  已计入
    Phase 7 回测:      已计入
    ─────────────────────
    TOTAL WALL:        {TOTAL_WALL:.1f}s
""")
