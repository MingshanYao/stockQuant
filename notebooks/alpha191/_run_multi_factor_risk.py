"""多因子组合回测 — 含风险管理（止损 + 回撤熔断）。

在 _run_multi_factor.py 的基础上，测试不同风险参数下的回撤控制效果。
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

# 风险参数组合（按回撤容忍度排序）
# 新框架：回撤控制由 MarketRegimeDetector + 分级 RiskMonitor 统一管理
# 每个 profile 配置个股层面的止损/止盈/移动止损参数
# 市场状态仓位系数（牛100%/正常80%/熊50%/危机20%）和
# 分级DD预警（绿100%/黄80%/橙60%/红30%）是框架内置的，无需外部配置
RISK_PROFILES = {
    "无风控(基线)":        {"enable": False, "sl": 0.08, "tp": 0.20, "ts": 0.05},
    "新框架_默认参数":      {"enable": True,  "sl": 0.08, "tp": 0.20, "ts": 0.05},
    "新框架_紧止损":        {"enable": True,  "sl": 0.05, "tp": 0.15, "ts": 0.03},
    "新框架_松止损":        {"enable": True,  "sl": 0.12, "tp": 0.25, "ts": 0.08},
    "新框架_仅靠框架(无个股止损)": {"enable": True,  "sl": 0.50, "tp": 0.50, "ts": 0.50},
}

print("=" * 70)
print("Alpha191 多因子组合回测 — 风险管理测试")
print("=" * 70)
print(f"风险配置组数: {len(RISK_PROFILES)}")
for name, p in RISK_PROFILES.items():
    if p["enable"]:
        print(f"  {name}: 止损={p['sl']:.0%}, 止盈={p['tp']:.0%}, 移动止损={p['ts']:.0%}")
    else:
        print(f"  {name}")

# ======================================================================
# 1. 加载数据
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
system_eval = evaluator.evaluate_system(named_factors, forward_period=1)

# ======================================================================
# 3. 筛选有效因子 & 去冗余 & 构建组合因子
# ======================================================================
print("\n第3步: 筛选有效因子 & 去冗余...")

ic_sig = system_eval["ic_ir"].abs() > 0.5
t_sig = system_eval["t_stat"].abs() > 2.0
effective_mask = ic_sig & t_sig
effective_names = system_eval[effective_mask].index.tolist()
print(f"  有效因子: {len(effective_names)} 个")

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

print(f"  去冗余后独立因子: {len(independent_factors)} 个")

# 方向统一
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

def combine_icir_weighted(panels, icir):
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

icir_dict = {n: system_eval.loc[n, "ic_ir"] for n in independent_factors}
sorted_independent = sorted(independent_factors, key=lambda n: abs(icir_dict[n]), reverse=True)

# 使用 All_ICIR加权 和 All_等权（之前验证最优的两个组合）
all_names = sorted_independent
all_subset = {n: aligned_panels[n] for n in all_names}
all_subset_icir = {n: icir_dict[n] for n in all_names}

combo_panels = {
    "All_ICIR加权": combine_icir_weighted(all_subset, all_subset_icir),
    "All_等权": combine_equal(all_subset),
}
print(f"  测试组合: {list(combo_panels.keys())}")

# ======================================================================
# 4. 回测 — 每个风险配置 × 每个组合
# ======================================================================
print("\n第4步: 回测（2 组合 × 5 风控配置 = 10 个回测）...")
print("  注意: 每个回测约需 15-30 分钟")

all_results = {}

for profile_name, risk_params in RISK_PROFILES.items():
    for combo_name, combo_panel in combo_panels.items():
        label = f"{combo_name}_{profile_name}"
        print(f"\n  回测 {label}...", end=" ", flush=True)

        try:
            researcher = AlphaResearcher(
                dataset,
                initial_capital=INITIAL_CAPITAL,
                max_positions=MAX_POSITIONS,
                rebalance_freq=REBALANCE_FREQ,
                enable_risk_mgmt=risk_params["enable"],
                stop_loss_pct=risk_params["sl"],
                take_profit_pct=risk_params["tp"],
                trailing_stop_pct=risk_params["ts"],
            )
            r = researcher.run_backtest(alpha_panel=combo_panel, label=label)
            all_results[label] = r
            az = r.get_analyzer()
            dd = az.max_drawdown()
            print(f"年化: {az.annualized_return():+.2%}  "
                  f"夏普: {az.sharpe_ratio():.3f}  "
                  f"最大回撤: {dd:+.2%}")
        except Exception as e:
            print(f"失败: {e}")

# ======================================================================
# 5. 绩效对比
# ======================================================================
print("\n" + "=" * 70)
print("风险管理回测 — 绩效对比表")
print("=" * 70)

if all_results:
    rows = []
    for label, result in all_results.items():
        az = result.get_analyzer()
        rows.append({
            "策略": label,
            "年化收益率": az.annualized_return(),
            "最大回撤": az.max_drawdown(),
            "夏普比率": az.sharpe_ratio(),
            "卡玛比率": az.calmar_ratio(),
            "Alpha": az.alpha(),
        })
    metrics = pd.DataFrame(rows).set_index("策略")
    metrics = metrics.sort_values("夏普比率", ascending=False)

    print(metrics.to_string(float_format=lambda x: f"{x:+.4f}"))

    # 回撤对比突出显示
    print("\n回撤控制对比:")
    for idx, row in metrics.iterrows():
        dd = row["最大回撤"]
        status = "✓ 达标" if abs(dd) <= 0.05 else ("△ 接近" if abs(dd) <= 0.10 else "✗ 超标")
        print(f"  {idx}: 回撤={dd:+.2%}  {status}")

# ======================================================================
# 6. 净值曲线
# ======================================================================
print("\n第6步: 绘制净值曲线对比...")

if len(all_results) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for combo_idx, combo_name in enumerate(combo_panels.keys()):
        ax = axes[combo_idx]
        combo_results = {k: v for k, v in all_results.items() if combo_name in k}

        colors = plt.cm.tab10(np.linspace(0, 1, len(combo_results)))
        for (name, result), color in zip(combo_results.items(), colors):
            az = result.get_analyzer()
            equity = az.equity
            if equity is not None and len(equity) > 0:
                values = equity["total_value"].values
                normalized = values / values[0]
                label = name.replace(f"{combo_name}_", "")
                ax.plot(equity.index, normalized, label=label,
                        color=color, linewidth=1.5)

        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("净值（归一化）", fontsize=11)
        ax.set_title(f"{combo_name} — 不同风控配置净值对比", fontsize=13)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/risk_management_equity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  已保存: risk_management_equity.png")

print("\n风险管理回测完成！")
