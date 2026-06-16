"""小规模测试 — 验证预中性化生命周期的正确性和性能。

使用合成数据快速跑通所有优化方法，不依赖 Alpha191 因子计算。
"""
import gc
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from stockquant.data.database import Database
from stockquant.data.universe import BacktestDataset
from stockquant.analysis.evaluator import FactorEvaluator

# ── 加载少量数据 ──
print("=" * 60)
print("小规模测试: 30只股 × 126天 (2015H2)")
print("=" * 60)

db = Database()
bars = db.query(
    "SELECT * FROM daily_bars WHERE date >= ? AND date <= ? ORDER BY code, date",
    ["2015-07-01", "2015-12-31"],
)
bars["date"] = pd.to_datetime(bars["date"])

codes = bars["code"].unique().tolist()
codes = [c for c in codes if not c.startswith(("688", "43", "83", "87"))][:30]

stock_data = {}
for code in codes:
    df = bars[bars["code"] == code].reset_index(drop=True)
    if len(df) >= 60:
        stock_data[code] = df
del bars
codes = list(stock_data.keys())

benchmark_df = db.query(
    "SELECT * FROM daily_bars WHERE code = ? AND date >= ? AND date <= ? ORDER BY date",
    ["000905", "2015-07-01", "2015-12-31"],
)
benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

dataset = BacktestDataset(
    stock_data=stock_data,
    codes=codes,
    benchmark=benchmark_df,
    benchmark_code="000905",
    start_date="2015-07-01",
    end_date="2015-12-31",
)
print(f"数据: {len(codes)} 只股票, {len(dataset.benchmark)} 条基准日线")

# ── 构造 close_panel（日期 × 股票，FactorEvaluator 需要）──
dates = sorted(dataset.benchmark["date"].tolist())
close_data = {}
for code in codes:
    df = stock_data[code].set_index("date").sort_index()
    close_data[code] = df["close"]
close_panel = pd.DataFrame(close_data).reindex(dates).sort_index()
close_panel = close_panel.ffill()

evaluator = FactorEvaluator(close_panel=close_panel)
del close_panel  # evaluator keeps its own reference

# ── 构造 5 个合成因子面板（形状: 日期 × 股票）──
date_strs = [d.strftime("%Y-%m-%d") for d in dates]
rng = np.random.RandomState(42)

factor_names = ["F1", "F2", "F3", "F4", "F5"]
factor_panels = {}
for name in factor_names:
    panel = pd.DataFrame(
        rng.randn(len(date_strs), len(codes)) + rng.randn() * 0.1,
        index=date_strs, columns=codes,
    )
    factor_panels[name] = panel

print(f"合成因子: {len(factor_panels)} 个, shape={factor_panels['F1'].shape}")
print(f"close_panel: {evaluator.close.shape}\n")

# ═══ 测试 1: evaluate() 基准 vs 预中性化 ═══
print("─── 测试1: evaluate() 基准 vs 预中性化 ───")
factor_raw = factor_panels["F1"]

t0 = time.time()
r1 = evaluator.evaluate(factor_raw, forward_period=1)
t_baseline = time.time() - t0
print(f"  基准: {t_baseline:.1f}s  IC={r1['ic_mean']:.4f}  T={r1['t_stat']:.2f}")

fwd_neut = evaluator._neutralize_panel(evaluator._forward_returns(1))
factor_neut = evaluator._neutralize_panel(factor_raw)

t0 = time.time()
r2 = evaluator.evaluate(factor_raw, forward_period=1,
                        factor_neutral=factor_neut, fwd_neutral=fwd_neut)
t_cached = time.time() - t0
print(f"  复用: {t_cached:.1f}s  IC={r2['ic_mean']:.4f}  T={r2['t_stat']:.2f}")
assert np.allclose(r1['ic_mean'], r2['ic_mean'], equal_nan=True), f"IC mismatch: {r1['ic_mean']} vs {r2['ic_mean']}"
assert np.allclose(r1['t_stat'], r2['t_stat'], equal_nan=True), f"T mismatch: {r1['t_stat']} vs {r2['t_stat']}"
print(f"  ✅ 结果一致 (加速 {t_baseline/max(t_cached,0.001):.1f}x)")

# ═══ 测试 2: evaluate_model_predictive_power ═══
print("\n─── 测试2: evaluate_model_predictive_power 基准 vs 预中性化 ───")

t0 = time.time()
r1 = evaluator.evaluate_model_predictive_power(factor_panels, forward_period=1)
t_baseline = time.time() - t0
print(f"  基准: {t_baseline:.1f}s  IC={r1['ic_mean']:.4f}")

all_neut = {k: evaluator._neutralize_panel(v) for k, v in factor_panels.items()}

t0 = time.time()
r2 = evaluator.evaluate_model_predictive_power(
    factor_panels, forward_period=1,
    neutralized_factors=all_neut, fwd_neutral=fwd_neut)
t_cached = time.time() - t0
print(f"  复用: {t_cached:.1f}s  IC={r2['ic_mean']:.4f}")
assert np.allclose(r1['ic_mean'], r2['ic_mean'], equal_nan=True), \
    f"IC mismatch: {r1['ic_mean']} vs {r2['ic_mean']}"
print(f"  ✅ 结果一致 (加速 {t_baseline/max(t_cached,0.001):.1f}x)")

# ═══ 测试 3: factor_count_analysis ═══
print("\n─── 测试3: factor_count_analysis 基准 vs 预中性化 ───")

t0 = time.time()
df1 = evaluator.factor_count_analysis(factor_panels, forward_period=1)
t_baseline = time.time() - t0
print(f"  基准: {t_baseline:.1f}s  {len(df1)} 步")

t0 = time.time()
df2 = evaluator.factor_count_analysis(
    factor_panels, forward_period=1,
    neutralized_factors=all_neut, fwd_neutral=fwd_neut)
t_cached = time.time() - t0
print(f"  复用: {t_cached:.1f}s  {len(df2)} 步")
assert np.allclose(df1["ic_mean"].iloc[-1], df2["ic_mean"].iloc[-1], equal_nan=True)
assert len(df1) == len(df2)
print(f"  ✅ 结果一致 (加速 {t_baseline/max(t_cached,0.001):.1f}x)")

# ═══ 测试 4: factor_correlation_matrix ═══
print("\n─── 测试4: factor_correlation_matrix 基准 vs 预中性化 ───")

t0 = time.time()
c1 = evaluator.factor_correlation_matrix(factor_panels, forward_period=1)
t_baseline = time.time() - t0
print(f"  基准: {t_baseline:.1f}s  shape={c1.shape}")

t0 = time.time()
c2 = evaluator.factor_correlation_matrix(
    factor_panels, forward_period=1,
    neutralized_factors=all_neut, fwd_neutral=fwd_neut)
t_cached = time.time() - t0
print(f"  复用: {t_cached:.1f}s  shape={c2.shape}")
assert np.allclose(c1.values, c2.values, equal_nan=True)
print(f"  ✅ 结果一致 (加速 {t_baseline/max(t_cached,0.001):.1f}x)")

# ═══ 测试 5: 内存释放 ═══
print("\n─── 测试5: 内存释放 ───")
del all_neut, fwd_neut, factor_neut
gc.collect()
print("  ✅ 缓存已释放")

# ═══ 测试 6: 向后兼容性 — 旧调用签名不带新参数 ═══
print("\n─── 测试6: 向后兼容性 ───")
r = evaluator.evaluate(factor_raw, forward_period=1)
assert isinstance(r, dict) and 'ic_mean' in r
print(f"  ✅ evaluate() 旧签名 OK: IC={r['ic_mean']:.4f}")

r = evaluator.evaluate_model_predictive_power({"F1": factor_raw}, forward_period=1)
assert isinstance(r, dict) and 'ic_mean' in r
print(f"  ✅ evaluate_model_predictive_power() 旧签名 OK: IC={r['ic_mean']:.4f}")

df_fc = evaluator.factor_count_analysis({"F1": factor_raw, "F2": factor_panels["F2"]})
assert isinstance(df_fc, pd.DataFrame)
print(f"  ✅ factor_count_analysis() 旧签名 OK: {len(df_fc)} 步")

r = evaluator.factor_correlation_matrix({"F1": factor_raw, "F2": factor_panels["F2"]})
assert isinstance(r, pd.DataFrame)
print(f"  ✅ factor_correlation_matrix() 旧签名 OK: shape={r.shape}")

# ═══ 测试 7: evaluate_system 兼容性 ═══
print("\n─── 测试7: evaluate_system 仍正常运行 ───")
sys_eval = evaluator.evaluate_system(factor_panels, forward_period=1)
assert len(sys_eval) == 5
assert "ic_mean" in sys_eval.columns
print(f"  ✅ evaluate_system OK: {len(sys_eval)} 因子, "
      f"IC 范围 [{sys_eval['ic_mean'].min():.3f}, {sys_eval['ic_mean'].max():.3f}]")

print("\n" + "=" * 60)
print("所有 7 项测试通过 ✅")
print("=" * 60)
