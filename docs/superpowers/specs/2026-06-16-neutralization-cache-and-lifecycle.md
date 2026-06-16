# 风格中性化缓存与生命周期管理

## Problem

`_neutralize_panel` 被重复调用 ~674 次执行一次完整回测，每次 spawn 新的 ProcessPoolExecutor。同一 forward returns 和因子面板被中性化多次，导致：

- Ch6.5-7 单独占 ~96 次调用（sequential，约 30-60 分钟）
- 峰值内存 12.5GB（反复复制 30MB DataFrame）
- 每次 ProcessPoolExecutor fork/teardown 开销 ~2s

## Solution

**Explicit lifecycle**: replication script 在 Ch6 后预中性化所有候选因子一次，通过新增可选参数传递给 downstream 方法。Ch3 和 Ch5 不变（它们有自己的并行策略且调用量可控）。

### Method signature additions

All additions are optional `=None` params — fully backward compatible.

- `evaluate(factor_neutral=None, fwd_neutral=None)` — skip internal neutralization when both provided
- `evaluate_model_predictive_power(neutralized_factors=None, fwd_neutral=None)` — skip per-factor neutralization
- `factor_count_analysis(neutralized_factors=None, fwd_neutral=None)` — skip pre-neutralization phase
- `factor_correlation_matrix(neutralized_factors=None, fwd_neutral=None)` — skip per-factor neutralization

### Script lifecycle

```
Ch1-2: Load + compute (no neutralization)
Ch3:   evaluate_system(189) — 378 calls, parallel, unchanged
Ch4:   Factor Return — no neutralization
Ch5:   multi_horizon(20×5) — 200 calls, sequential, unchanged
Ch6:   System eval — no neutralization
---- MEMORY GATE ----
       Pre-neutralize fwd + 181 candidate factors (once)
       ~5.4GB peak
Ch6.5a: model_predictive_power — 0 calls (reuse)
Ch6.5b: factor_count_analysis — 0 calls (reuse)
Ch6.5c: coverage_report — no neutralization
Ch7:   factor_correlation_matrix — 0 calls (reuse)
---- MEMORY RELEASE ----
       del neut_factors, fwd_neut; gc.collect()
Ch8:   Parallel backtests — full RAM available
Ch9:   Conclusions
```

### Error resilience

Individual factor neutralization failures are caught and skipped (with print warning), so one bad factor doesn't kill the entire pre-neutralization loop.

### Net effect

- Ch3: unchanged (378 calls, parallel)
- Ch5: unchanged (200 calls, sequential but limited to 20 factors)
- Ch6.5-7: 96 calls → 0
- Peak memory: ~5.4GB (controlled, released before backtest)
- Total neutralization calls: 578 → 578 (no Ch6.5-7 regression)

## Files changed

| File | Lines | Nature |
|------|-------|--------|
| `stockquant/analysis/evaluator.py` | ~120 | Add optional params to 4 methods; `_calc_ic_fr_from_neutralized` already exists |
| `notebooks/alpha191/_run_replication.py` | ~60 | Pre-neutralization block + pass-through + memory gate |

## Verification

1. Run replication script end-to-end; must complete all 9 chapters
2. Spot-check: Ch3 IC values unchanged from pre-optimization run
3. Ch6.5a model predictive power IC ~ -0.037 (same magnitude, may shift slightly due to 181 vs 31 candidates)
4. No OOM during Ch6.5-7
5. Ch8 backtests complete successfully
