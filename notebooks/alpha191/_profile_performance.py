"""全链路性能 Profiling — 30 个 Alpha191 因子 × 1 年数据。

分析每个阶段的耗时、调用次数、算法复杂度，定位瓶颈。
"""
import gc
import time
import tracemalloc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from stockquant.data.database import Database
from stockquant.data.universe import BacktestDataset
from stockquant.indicators.alpha191 import Alpha191Indicators, SKIP_ALPHAS
from stockquant.analysis.evaluator import FactorEvaluator

# ── 配置 ──
START_DATE = "2015-01-01"
END_DATE   = "2015-12-31"
N_STOCKS   = 200        # 股票数
N_FACTORS  = 30         # 因子数
TOP_N_MULTI = 10        # 多周期因子数

OUTPUT = []
def record(phase, elapsed, calls=1, note=""):
    OUTPUT.append((phase, elapsed, calls, note))
    print(f"  {phase:40s} {elapsed:8.2f}s  ×{calls:4d}  {note}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
# Phase 0: 数据加载
# ═══════════════════════════════════════════════════════════════
section("Phase 0: 数据加载")
tracemalloc.start()

t0 = time.time()
db = Database()
bars = db.query(
    "SELECT * FROM daily_bars WHERE date >= ? AND date <= ? ORDER BY code, date",
    [START_DATE, END_DATE],
)
bars["date"] = pd.to_datetime(bars["date"])

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
elapsed = time.time() - t0
_, peak_mem = tracemalloc.get_traced_memory()
record("DB查询+数据集构建", elapsed, 1,
       f"{len(codes)} stocks × {len(dataset.benchmark)} days, peak_mem={peak_mem/1024**2:.0f}MB")
tracemalloc.stop()

# ═══════════════════════════════════════════════════════════════
# Phase 1: 因子计算
# ═══════════════════════════════════════════════════════════════
section("Phase 1: 因子计算")

t0 = time.time()
engine = Alpha191Indicators.from_dataset(dataset)
elapsed_engine = time.time() - t0
print(f"  引擎构建: {elapsed_engine:.1f}s  ({engine.close.shape[0]} days × {engine.close.shape[1]} stocks)")

ALL_ALPHA_IDS = [i for i in range(1, 192) if i not in SKIP_ALPHAS][:N_FACTORS]
print(f"  计算 {len(ALL_ALPHA_IDS)} 个因子...")

t0 = time.time()
all_factors = engine.compute_factors(ALL_ALPHA_IDS)
elapsed_factors = time.time() - t0

# 单因子耗时分布
times = []
for aid, panel in all_factors.items():
    times.append(panel.shape)
elapsed_total = elapsed_engine + elapsed_factors
record("引擎+因子计算", elapsed_total, len(ALL_ALPHA_IDS),
       f"avg={elapsed_factors/len(ALL_ALPHA_IDS):.1f}s/factor, "
       f"shape=({engine.close.shape[0]},{engine.close.shape[1]})")

# 过滤低覆盖因子
coverage = {}
for alpha_id, panel in all_factors.items():
    coverage[alpha_id] = np.isfinite(panel.values).mean()
MIN_COVERAGE = 0.3
valid_factors = {k: v for k, v in all_factors.items() if coverage.get(k, 0) >= MIN_COVERAGE}
named_factors = {f"Alpha{k:03d}": v for k, v in valid_factors.items()}
print(f"  有效因子: {len(valid_factors)}/{len(all_factors)} (coverage >= {MIN_COVERAGE:.0%})")

# ═══════════════════════════════════════════════════════════════
# Phase 2: IC/ICIR + Factor Return (evaluate_system)
# ═══════════════════════════════════════════════════════════════
section("Phase 2: evaluate_system (IC + FR)")
evaluator = FactorEvaluator(close_panel=engine.close)

t0 = time.time()
system_eval = evaluator.evaluate_system(named_factors, forward_period=1)
elapsed_sys = time.time() - t0

record("evaluate_system", elapsed_sys, len(named_factors),
       f"avg={elapsed_sys/len(named_factors):.2f}s/factor, "
       f"calls ~{len(named_factors)*2} _neutralize_panel")

# ═══════════════════════════════════════════════════════════════
# Phase 3: 多周期检验
# ═══════════════════════════════════════════════════════════════
section("Phase 3: 多周期检验 (multi_horizon)")

top_factors = (system_eval["ic_mean"].abs()
               .sort_values(ascending=False)
               .head(TOP_N_MULTI).index.tolist())
top_dict = {n: named_factors[n] for n in top_factors}
periods = [1, 2, 3, 4, 5]

# 预中性化 + 并行 evaluate（匹配新版 replication script）
multi_periods = [1, 2, 3, 4, 5]
multi_names = top_factors[:TOP_N_MULTI]
multi_factors = {n: named_factors[n] for n in multi_names}

t0_prep = time.time()
multi_top_neut = {n: evaluator.neutralize_panel(named_factors[n]) for n in multi_names}
multi_fwd_neut = {p: evaluator.neutralize_panel(evaluator.forward_returns(p)) for p in multi_periods}
elapsed_prep = time.time() - t0_prep
print(f"  预中性化: {elapsed_prep:.2f}s ({len(multi_names)} 因子 + {len(multi_periods)} 周期)")

def _multi_eval_one(tup):
    name, p = tup
    return name, p, evaluator.evaluate(
        named_factors[name], forward_period=p,
        factor_neutral=multi_top_neut[name], fwd_neutral=multi_fwd_neut[p],
    )

t0 = time.time()
items = [(n, p) for n in multi_names for p in multi_periods]
with ThreadPoolExecutor(max_workers=min(12, len(items))) as ex:
    futures = {ex.submit(_multi_eval_one, item): item for item in items}
    for future in as_completed(futures):
        name, p, r = future.result()
elapsed_multi = time.time() - t0

record("multi_horizon (并行+pool复用)", elapsed_multi, TOP_N_MULTI * len(periods),
       f"prep={elapsed_prep:.1f}s, eval={elapsed_multi:.1f}s, "
       f"avg={elapsed_multi/len(items):.2f}s/eval")

# ═══════════════════════════════════════════════════════════════
# Phase 4: _neutralize_panel 单次耗时 & 复杂度
# ═══════════════════════════════════════════════════════════════
section("Phase 4: _neutralize_panel 微基准")

fwd_returns = evaluator.forward_returns(1)
sample_panel = list(named_factors.values())[0]

# 4a. 单次 neutralize
t0 = time.time()
_ = evaluator.neutralize_panel(fwd_returns)
elapsed_fwd = time.time() - t0

# 4b. 连续 neutralize N 次 (simulate sequential calls)
N_bench = 20
t0 = time.time()
for i in range(N_bench):
    _ = evaluator.neutralize_panel(sample_panel)
elapsed_seq = time.time() - t0

# 4c. 不同面板大小的复杂度
sizes = [(100, 50), (250, 100), (500, 200), (1000, 200)]
size_results = []
for n_rows, n_cols in sizes:
    test_panel = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        index=[f"D{i}" for i in range(n_rows)],
        columns=[f"S{i}" for i in range(n_cols)],
    )
    t0 = time.time()
    for _ in range(5):
        _ = evaluator.neutralize_panel(test_panel)
    elapsed = (time.time() - t0) / 5
    size_results.append((n_rows, n_cols, elapsed))

record("forward_returns neutral.", elapsed_fwd, 1, f"shape={fwd_returns.shape}")
record(f"neutralize ×{N_bench} (seq)", elapsed_seq, N_bench,
       f"avg={elapsed_seq/N_bench:.2f}s/call, "
       f"fork={elapsed_seq/N_bench - elapsed_fwd:.2f}s overhead")

print("\n  复杂度 vs 面板大小 (avg 5 runs):")
for nr, nc, et in size_results:
    cells = nr * nc
    print(f"    {nr:5d}×{nc:4d} = {cells:8d} cells → {et:.3f}s  ({cells/et:.0f} cells/s)")

# ═══════════════════════════════════════════════════════════════
# Phase 5: 预中性化 (simulate Ch6.5 batch)
# ═══════════════════════════════════════════════════════════════
section("Phase 5: 预中性化 批量 (simulate Ch6.5)")

# 取显著因子模拟
full_eval = system_eval.copy()
full_eval["IC显著"] = full_eval["ic_ir"].abs() > 0.5
full_eval["T显著"] = full_eval["t_stat"].abs() > 2.0
candidates = full_eval[full_eval["IC显著"] | full_eval["T显著"]]
candidate_dict = {name: named_factors[name]
                  for name in candidates.index if name in named_factors}
print(f"  候选因子: {len(candidate_dict)} 个")

t0_fwd = time.time()
fwd_neut_cache = evaluator.neutralize_panel(fwd_returns)
elapsed_fwd_neut = time.time() - t0_fwd

t0 = time.time()
neut_cache: dict = {}
items = list(candidate_dict.items())
total = len(items)

def _par_neut(name_panel):
    name, panel = name_panel
    try:
        return name, evaluator.neutralize_panel(panel)
    except Exception:
        return name, None

with ThreadPoolExecutor(max_workers=min(12, total)) as ex:
    futures = [ex.submit(_par_neut, item) for item in items]
    for future in as_completed(futures):
        name, neut = future.result()
        if neut is not None:
            neut_cache[name] = neut
gc.collect()

elapsed_batch = time.time() - t0

record("fwd neutral.", elapsed_fwd_neut, 1, f"shape={fwd_returns.shape}")
record(f"batch neutralize ×{total} (并行)", elapsed_batch, total,
       f"avg={elapsed_batch/total:.2f}s/factor, "
       f"parallel={min(12, total)} threads + pool reuse")

# ═══════════════════════════════════════════════════════════════
# Phase 6: 模型预测力 + 数量边际 + 相关矩阵
# ═══════════════════════════════════════════════════════════════
section("Phase 6: 因子体系评价 (复用预中性化)")

# 6a. model_predictive_power
t0 = time.time()
model_ic = evaluator.evaluate_model_predictive_power(
    candidate_dict, forward_period=1,
    neutralized_factors=neut_cache, fwd_neutral=fwd_neut_cache,
)
elapsed_mp = time.time() - t0
record("evaluate_model_predictive_power", elapsed_mp, len(candidate_dict),
       f"IC={model_ic.get('ic_mean', np.nan):.4f}, "
       f"每因子 avg={elapsed_mp/len(candidate_dict)*1000:.0f}ms")

# 6b. factor_count_analysis (包含内部 Phase2 FR排名 + Phase3 边际)
t0 = time.time()
count_df = evaluator.factor_count_analysis(
    candidate_dict, forward_period=1,
    neutralized_factors=neut_cache, fwd_neutral=fwd_neut_cache,
)
elapsed_fca = time.time() - t0
record("factor_count_analysis", elapsed_fca, len(candidate_dict),
       f"{len(count_df)} steps, "
       f"avg={elapsed_fca/len(candidate_dict):.2f}s/step, "
       f"final ICIR={count_df['ic_ir'].iloc[-1]:.4f}")

# 6c. factor_correlation_matrix
t0 = time.time()
corr_matrix = evaluator.factor_correlation_matrix(
    candidate_dict, forward_period=1,
    neutralized_factors=neut_cache, fwd_neutral=fwd_neut_cache,
)
elapsed_corr = time.time() - t0
n_pairs = len(corr_matrix) * (len(corr_matrix) - 1) // 2
record("factor_correlation_matrix", elapsed_corr, n_pairs,
       f"{len(corr_matrix)}×{len(corr_matrix)} matrix, "
       f"avg={elapsed_corr/max(n_pairs,1)*1000:.1f}ms/pair")

# ═══════════════════════════════════════════════════════════════
# Phase 7: 回测 (单因子)
# ═══════════════════════════════════════════════════════════════
section("Phase 7: 回测 (单因子)")

from stockquant.research import AlphaResearcher

researcher = AlphaResearcher(
    dataset,
    initial_capital=1_000_000.0,
    max_positions=50,
    rebalance_freq=5,
)

# 只测 1 个因子
sample_name, sample_panel = list(candidate_dict.items())[0]
alpha_id = int(sample_name.replace("Alpha", ""))

t0 = time.time()
result = researcher.run_backtest(
    alpha_panel=all_factors[alpha_id], label=sample_name,
)
elapsed_bt = time.time() - t0
az = result.get_analyzer()
record("单因子回测", elapsed_bt, 1,
       f"{sample_name}: 总收益={az.total_return():+.2%}, "
       f"夏普={az.sharpe_ratio():.3f}")

# ═══════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════
section("汇总: 全链路耗时分解")

print(f"\n{'阶段':<40s} {'耗时(s)':>8s}  {'调用':>6s}  {'占比':>7s}  {'备注'}")
print("-" * 100)
total_time = sum(r[1] for r in OUTPUT)
for phase, elapsed, calls, note in OUTPUT:
    pct = elapsed / total_time * 100
    print(f"{phase:<40s} {elapsed:8.2f}  {calls:6d}  {pct:6.1f}%  {note}")
print("-" * 100)
print(f"{'TOTAL':<40s} {total_time:8.2f}")

# 复杂度汇总
print(f"\n{'='*60}")
print("  算法复杂度分析")
print(f"{'='*60}")
print(f"""
  _neutralize_panel:       O(d·s·k) — d=dates, s=stocks, k=style factors
    每次 spawn ProcessPoolExecutor (fork overhead ~2s)

  evaluate_system:         O(F · d·s·k) — F 因子并行，每人 2 次 neutralize

  multi_horizon:           O(H · F · d·s·k) — H 周期 × F 因子，sequential!

  evaluate_model_predictive_power:  O(F · d·s) — 每个因子 z-score + add
    (F 次 z-score, F 次 add, 1 次 corrwith)

  factor_count_analysis:
    Phase 2 (FR rank):     O(F · d·s·k) — 每个因子 1 次 evaluate
    Phase 3 (marginal):    O(F · H_corr) — F 次 corrwith, 维护运行和

  factor_correlation_matrix:  O(F² · d) — 两两因子截面相关

  关键瓶颈:
    1. _neutralize_panel 每次 fork ProcessPoolExecutor (~2s)
    2. multi_horizon sequential 调用 × H 周期
    3. factor_count_analysis Phase 2 重新计算 FR (O(F·d·s·k))
""")

# 内存释放
del neut_cache, fwd_neut_cache
gc.collect()
print("✅ Profiling 完成")
