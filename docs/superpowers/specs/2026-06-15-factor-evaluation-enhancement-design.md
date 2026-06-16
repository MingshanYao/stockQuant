# Factor Evaluation 体系增强设计

**日期：** 2026-06-15
**状态：** 设计已完成，待实现

---

## 1. 动机与目标

### 1.1 问题

在 Alpha191 因子复现的回测中，观察到以下现象：

- **策略净值曲线呈现「漫长亏损 + 剧烈拉升」的锯齿形态** — 策略长时间不赚钱甚至亏钱，但在少数时间段内斜率非常陡峭
- **现有评价体系无法区分「市场 Beta 驱动」和「选股 Alpha」** — 策略的收益可能主要来自对市场风格的暴露，而非真正的选股能力
- **交易层面的统计缺失** — 没有胜率、盈亏比、Profit Factor 等核心交易指标

### 1.2 核心目标

参照国泰君安(2017)《基于短周期价量特征的多因子选股体系》中的方法论：

> **股票收益率 = 风格收益 + 阿尔法收益**
> 评价阿尔法模型的标准应是计算其对阿尔法收益的预测是否可靠
> 预测与组合实现的目标必须一致

具体目标：

| # | 目标 | 对应需求 |
|---|------|----------|
| 1 | 风格中性化的因子 IC 计算 | 剔除市场影响评估因子 |
| 2 | 组合构建时的行业/市值中性约束 | 确保预测与组合目标一致 |
| 3 | 超额收益曲线（策略-基准） | 剔除市场影响 #1 |
| 4 | 剥离 Beta 后的纯 Alpha 曲线 | 剔除市场影响 #2 |
| 5 | 逐笔交易的 Alpha 盈亏 | 剔除市场影响 #4 |
| 6 | 交易胜率统计（胜率/盈亏比/Profit Factor/最大连败等） | 交易胜率需求 |

---

## 2. 整体架构

```
                                  ┌─────────────────────────┐
                                  │    FactorEvaluator      │ ← 扩展风格中性化
                                  │   (evaluator.py)        │    行业/市值/Beta/波动/动量
                                  └──────────┬──────────────┘
                                             │ 评价结果
                                             ▼
                                  ┌─────────────────────────┐
                                  │  AlphaFactorStrategy     │ ← 优化器约束：行业±5%, 市值±0.1σ
                                  │   (alpha_factor_strategy)│    
                                  └──────────┬──────────────┘
                                             │ 回测结果
                                             ▼
                                  ┌─────────────────────────┐
                                  │  PerformanceAnalyzer     │ ← 新增胜率统计 + Alpha收益
                                  │   (performance.py)       │    逐笔Alpha盈亏
                                  └──────────┬──────────────┘
                                             │ 绩效数据
                                             ▼
                                  ┌─────────────────────────┐
                                  │     PlotEngine           │ ← 三面板图：净值/超额/Alpha
                                  │   (plot.py)              │    交易仪表盘
                                  └─────────────────────────┘
```

**架构原则：** 各模块职责清晰——FactorEvaluator 负责评价、Strategy 负责执行、PerformanceAnalyzer 负责分析、PlotEngine 负责可视化。

---

## 3. FactorEvaluator — 风格中性化 IC 计算

### 3.1 当前状态

- `FactorEvaluator.__init__` 接受 `close_panel`, `industry`, `market_cap`
- `_neutralize_panel()` 仅支持行业和市值两因子做截面回归去残差
- `neutralize` 参数默认 `True`，但可为 `False`

### 3.2 改造方案

#### 3.2.1 风格因子数据源扩展

新增参数 `style_factor_data: dict[str, pd.DataFrame]`，键为风格因子名称（`beta`, `volatility`, `momentum`, `turnover`），值为 DataFrame（行=日期，列=股票代码），存储每只股票每日的风格因子暴露值。

若用户未提供，系统自动从 `close_panel` 计算：

| 风格因子 | 计算方式 | 窗口 |
|----------|----------|------|
| Beta | 滚动回归股票收益 vs 基准收益 | 60 日 |
| Volatility | 滚动标准差 | 20 日 |
| Momentum | 过去收益率 | 20 日 |
| Turnover | 日换手率（需外部传入） | — |

#### 3.2.2 截面回归中性化（增强版）

对每日截面做多元线性回归，取残差作为纯 Alpha 因子值：

```
factor_value = β_0 + β_ind·X_ind + β_size·X_size + β_beta·X_beta
              + β_vol·X_vol + β_mom·X_mom + ε
```

#### 3.2.3 evaluate() 流程

```
输入: factor_panel, forward_period=1
  │
  ├─ 1. 风格中性化因子值（回归取残差） → 纯Alpha预测值 E{ε}
  │
  ├─ 2. 对前向收益做同样的风格中性化 → 实际Alpha收益 α
  │
  ├─ 3. IC = corr(E{ε_t+1}, α_t+1)   # Rank IC (Spearman)
  │
  ├─ 4. Factor Return: 截面回归 R = β_factor × E{ε} + ε'
  │
  └─ 5. T统计量: IC_mean / (IC_std / √N)
```

关键变化：**前向收益也做风格中性化**——这是 alpha191 方法论的核心，只有实际 Alpha 收益才应该用于评价 Alpha 预测能力。

#### 3.2.4 接口变更

```python
class FactorEvaluator:
    def __init__(
        self,
        close_panel: pd.DataFrame,
        industry: pd.Series | None = None,
        market_cap: pd.Series | pd.DataFrame | None = None,
        style_factors: dict[str, pd.DataFrame] | None = None,
        benchmark_returns: pd.Series | None = None,
    ):
```

`neutralize` 参数从 `evaluate()` 签名中移除（始终强制中性化）。

---

## 4. AlphaFactorStrategy — 组合构建风格中性化

### 4.1 当前状态

当前策略在每个调仓日，简单取 Top-N 等权买入，没有行业/市值约束。

### 4.2 改造方案

使用**线性规划优化器 (cvxpy)** 构建组合：

#### 4.2.1 约束条件

| 约束 | 定义 |
|------|------|
| 行业中性 | 组合中各行业的权重与基准的行业权重偏差 ≤ ±5% |
| 市值中性 | 组合的市值Z-score与基准的市值Z-score偏差 ≤ ±0.1 |
| 满仓投资 | 总权重 = 1（或 max_positions 只） |
| 个股上限 | 单只股票权重 ≤ 2%（50只等权时的上限） |
| 个股下限 | 单只股票 ≥ 0（不允许卖空） |

#### 4.2.2 目标函数

```
maximize: Σ w_i × factor_score_i
```

其中 `factor_score_i` 是每只股票的因子值（已风格中性化）。

#### 4.2.3 依赖项

新增依赖：`cvxpy`（线性规划求解器）

#### 4.2.4 备用方案

若 cvxpy 不可用，提供 `_build_skip_neutral` 维持当前简单 Top-N 行为。

---

## 5. PerformanceAnalyzer — 交易胜率 + Alpha 收益

### 5.1 交易胜率统计

当前 `trade_statistics()` 仅返回总交易次数、买入次数、卖出次数、总佣金。

扩展为：

| 字段 | 类型 | 计算方式 |
|------|------|----------|
| 总交易次数 | int | len(trades) |
| 盈利交易次数 | int | sum(PnL > 0) |
| 亏损交易次数 | int | sum(PnL < 0) |
| **胜率** | float | 盈利交易数 / 总交易数 |
| **平均盈利** | float | 盈利PnL总和 / 盈利交易数 |
| **平均亏损** | float | 亏损PnL总和 / 亏损交易数 |
| **盈亏比** | float | 平均盈利 / 平均亏损 |
| **Profit Factor** | float | 盈利PnL总和 / 亏损PnL总和 |
| 最大连胜 | int | 连续盈利交易的最长序列 |
| 最大连败 | int | 连续亏损交易的最长序列 |
| 平均持仓天数（盈利） | float | 盈利交易的平均持仓 |
| 平均持仓天数（亏损） | float | 亏损交易的平均持仓 |
| 总佣金 | float | 保持现有 |

计算方式：
- 交易 PnL 从 `trade_log` 获取（`commission`、`price`、`quantity` 字段）
- 每笔交易的 PnL = 卖出金额 - 买入金额 - 佣金
- 持仓天数 = 卖出日期 - 买入日期
- 连续盈亏通过顺序扫描交易日志计算

### 5.2 逐笔 Alpha 盈亏

```python
def per_trade_alpha_pnl(self, benchmark: pd.Series) -> pd.DataFrame:
    """计算每笔交易的 Alpha 盈亏。

    对每笔交易：
    1. 计算持仓期间的市场收益率 r_market
    2. 计算该股票的 Beta（60日滚动回归）
    3. Alpha PnL = 实际PnL - (Beta × r_market × 投入资金)

    返回：trade_log + alpha_pnl 列
    """
```

### 5.3 超额收益与 Alpha 收益序列

```python
def excess_returns(self) -> pd.Series:
    """策略日收益率 - 基准日收益率"""

def alpha_returns(self, window: int = 60) -> pd.Series:
    """策略日收益率 - (滚动Beta × 基准日收益率)
    Beta = cov(R_strategy, R_benchmark) / var(R_benchmark)
    使用滚动窗口计算时变 Beta。
    """

def cumulative_alpha(self) -> pd.Series:
    """累计Alpha收益曲线（从1.0开始）"""
```

---

## 6. PlotEngine — 可视化增强

### 6.1 三面板性能图

新增 `plot_performance_dashboard(result, save_path=None)`：

| 面板 | 内容 | 数据来源 |
|------|------|----------|
| 顶部 | 策略净值 + 基准净值 + 超额收益曲线 | equity_curve, benchmark |
| 中部 | 纯 Alpha 累计收益曲线（剥离Beta） | cumulative_alpha() |
| 底部 | 月度 Alpha 收益热力图 | alpha_returns() resample |

曲线均从 1.0 归一化开始，便于横截面比较。

### 6.2 交易仪表盘

新增 `plot_trade_dashboard(analyzer, save_path=None)`：

```
┌─────────────────────────────────────┬─────────────────────────┐
│  胜率卡片                           │  盈亏分布直方图          │
│  - Win Rate: X%                    │  - 绿色：盈利交易        │
│  - Profit Factor                   │  - 红色：亏损交易        │
│  - 最大连胜/连败                   │  - 虚线：零线            │
│  - 平均持仓（盈利/亏损区分）        │                         │
└─────────────────────────────────────┴─────────────────────────┘
```

### 6.3 与现有接口兼容

- 所有新增方法默认 `save_path=None`，仅在提供时保存
- Matplotlib 和 Plotly 两种后端都支持（优先实现 Matplotlib）
- 不修改现有 `plot_equity_curve()` 行为

---

## 7. 数据依赖 & 外部接入

### 7.1 行业分类数据

用于行业中性化，需要外部数据源提供股票到行业的映射。

- 格式：`pd.Series(index=股票代码, values=行业名称)`
- 来源：BaoStock 每日数据中可提取
- 行业权重基准：**回测股票截面的行业分布**（截面所有股票在各行业的占比），而非某个指数的行业权重

### 7.2 市值数据

从数据源每日获取个股的流通市值或总市值。

- 格式：`pd.DataFrame(index=日期, columns=股票代码, values=市值)`
- 用于风格中性化回归和组合构建中的市值约束

### 7.3 基准指数

- 用于计算 Beta 和超额收益
- 当前已有基准数据（沪深300/中证500）

---

## 8. 实现顺序

| 阶段 | 内容 | 涉及文件 |
|------|------|----------|
| Phase 1 | PerformanceAnalyzer 扩展（胜率+Alpha收益） | performance.py |
| Phase 2 | PlotEngine 可视化增强 | plot.py |
| Phase 3 | FactorEvaluator 风格中性化 | evaluator.py |
| Phase 4 | AlphaFactorStrategy 优化器组合构建 | alpha_factor_strategy.py |
| Phase 5 | 集成测试 & notebooks 更新 | _run_notebook.py 等 |

---

## 9. 测试策略

| 测试维度 | 方法 |
|----------|------|
| 风格中性化 IC | 模拟一个已知有风格偏向的因子，验证中性化后 IC 是否改变 |
| 优化器组合 | 验证组合的行业/市值约束在阈值范围内 |
| 胜率统计 | 使用已知交易日志验证胜率/盈亏比正确性 |
| Alpha 收益曲线 | 当策略=基准时验证 Alpha=0 |

---

## 10. 性能考虑

- **风格中性化回归**：每日截面做 OLS，因子数量约200、股票数量约2255，每个因子需约1700次回归，总计约340,000次 OLS。建议使用 numpy 向量化或并行计算（当前 `evaluate_system` 已有 ProcessPoolExecutor）
- **优化器求解**：每次调仓求解一次，回测期内约340次求解，每次求解2255个变量，cvxpy 可秒级完成
