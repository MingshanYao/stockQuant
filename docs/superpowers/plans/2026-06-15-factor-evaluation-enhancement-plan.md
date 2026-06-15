# 因子评价体系增强 — 实现计划

**参考 Spec：** `docs/superpowers/specs/2026-06-15-factor-evaluation-enhancement-design.md`

---

## Phase 1: PerformanceAnalyzer 扩展

**文件：** `stockquant/analysis/performance.py`

### 1.1 交易胜率统计

将 `trade_statistics()` 从基础统计扩展为完整胜率分析：

```python
def trade_statistics(self) -> dict[str, Any]:
    """交易统计摘要（含胜率/盈亏比/Profit Factor/最大连败等）。"""
```

改动细节：

1. 从 `self.trades` 计算每笔交易的 PnL：
   - 需要配对买入/卖出记录
   - 买入记录：`direction == "buy"`，记录 `price`, `quantity`, `commission`, `date`
   - 卖出记录：`direction == "sell"`，配对到对应的买入记录
   - 每笔完整交易的 PnL = 卖出金额 - 买入金额 - 两笔的 commission

2. 新增统计字段：
   - `盈利交易次数` / `亏损交易次数` / `胜率`
   - `平均盈利` / `平均亏损` / `盈亏比`
   - `Profit Factor`
   - `最大连胜` / `最大连败`
   - `平均持仓天数（盈利）` / `平均持仓天数（亏损）`

3. `full_report()` 自动包含这些字段。

### 1.2 逐笔交易的 Alpha 盈亏

```python
def per_trade_alpha_pnl(self, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """计算每笔交易的 Alpha 盈亏。

    Parameters
    ----------
    benchmark : pd.Series, optional
        基准每日价格序列（index=日期）。若为 None，使用 self.benchmark_returns。

    Returns
    -------
    pd.DataFrame
        原始交易日志 + alpha_pnl / alpha_return 列。
    """
```

对每笔交易：
1. 定位该交易持仓的起始和结束日期
2. 计算持仓期间的基准累计收益率
3. 计算该股票的 Beta（使用持仓期前的 60 个交易日数据）：
   - 股票日收益率 vs 基准日收益率的协方差/方差
4. Alpha PnL = 实际 PnL - Beta × (投入资金 × 基准累计收益)

### 1.3 超额收益与 Alpha 收益序列

```python
def excess_returns(self) -> pd.Series:
    """策略日收益率 - 基准日收益率。"""

def alpha_returns(self, window: int = 60) -> pd.Series:
    """策略日收益率 - (滚动Beta × 基准日收益率)。

    使用滚动窗口计算时变 Beta。
    """

def cumulative_alpha(self) -> pd.Series:
    """累计纯 Alpha 收益曲线（从 1.0 开始）。"""
```

注意：`alpha_returns` 使用的滚动 Beta 是**策略组合 vs 基准**的 Beta，不是个股 Beta。

### 1.4 修改 `__init__`

如果 `benchmark_returns` 为 None，后续调用 `excess_returns()`、`alpha_returns()`、`per_trade_alpha_pnl()` 应优雅地返回 None 或空数据。

---

## Phase 2: PlotEngine 可视化增强

**文件：** `stockquant/visualization/plot.py`

### 2.1 三面板性能图

```python
def plot_performance_dashboard(
    self,
    result: AlphaBacktestResult | BacktestResult,
    title: str = "策略绩效看板",
    save_path: str | None = None,
) -> None:
    """三面板性能图：

    面板 1（顶部）：策略净值 + 基准净值 + 累计超额收益
    面板 2（中部）：纯 Alpha 累计收益曲线（剥离 Beta）
    面板 3（底部）：月度 Alpha 收益热力图
    """
```

实现细节：

1. **面板 1（顶部）：**
   - 策略净值 = equity_curve.total_value / total_value[0]
   - 基准净值 = benchmark.close / benchmark.close[0]
   - 超额收益 = (1 + cumulative_excess_return)，从 excess_returns() 累积
   - 三条曲线共享 y 轴

2. **面板 2（中部）：**
   - 使用 PerformanceAnalyzer.cumulative_alpha() 绘制纯 Alpha 曲线
   - 水平参考线 y=1

3. **面板 3（底部）：**
   - 使用 PerformanceAnalyzer.alpha_returns() 按月 resample
   - 热力图色阶：红（正）→ 白（零）→ 绿（负）

### 2.2 交易仪表盘

```python
def plot_trade_dashboard(
    self,
    analyzer: PerformanceAnalyzer,
    title: str = "交易分析仪表盘",
    save_path: str | None = None,
) -> None:
    """交易仪表盘：

    左侧：胜率卡片（胜率、盈亏比、Profit Factor、最大连败/连胜）
    右侧：逐笔交易盈亏分布直方图（绿=盈利，红=亏损）
    """
```

实现细节：

1. **左侧（文本卡片区）：** 使用 matplotlib 的 `ax.text()` / `ax.table()` 展示
2. **右侧（直方图）：** 使用 `ax.hist()` 分两部分（盈利/亏损分开着色）

---

## Phase 3: FactorEvaluator 风格中性化

**文件：** `stockquant/analysis/evaluator.py`

### 3.1 接口变更

```python
class FactorEvaluator:
    def __init__(
        self,
        close_panel: pd.DataFrame,
        industry: pd.Series | None = None,                       # code → industry
        market_cap: pd.Series | pd.DataFrame | None = None,      # code→cap, 或 dates×codes
        style_factors: dict[str, pd.DataFrame] | None = None,     # {name: dates×codes}
        benchmark_returns: pd.Series | None = None,              # 用于计算Beta
    ):
        self.close = close_panel
        self.industry = industry
        self.market_cap = market_cap
        self.style_factors = style_factors or {}
        self.benchmark_returns = benchmark_returns
```

关键变更：
- `neutralize` 参数从 `evaluate()` 签名中移除
- `evaluate()` 始终做风格中性化（论文要求）

### 3.2 风格因子自动计算

```python
def _compute_style_factors(self) -> dict[str, pd.DataFrame]:
    """从 close_panel 自动计算风格因子。

    返回：
        beta: 60日滚动Beta（需要 benchmark_returns）
        volatility: 20日滚动波动率
        momentum: 过去20日收益率
    """
```

### 3.3 截面回归中性化（增强版）

```python
def _neutralize_panel(self, factor: pd.DataFrame) -> pd.DataFrame:
    """对每日截面做多元线性回归，取残差作为纯 Alpha 因子值。

    回归模型：
        factor_value = β_0 + β_ind·X_ind + β_size·X_size
                      + β_beta·X_beta + β_vol·X_vol + β_mom·X_mom + ε

    返回：ε 残差截面（仅包含因子值，不含行业/风格信息）
    """
```

实现流程：

```
对每个日期 d（因子面板的每个索引）：
  1. 提取截面因子值 f_d，去 NaN
  2. 构建设计矩阵 X：
     a. 行业哑变量：对 industry.unique() 每个行业一列（drop_first=True）
     b. 市值：market_cap 的 Z-score 归一化
     c. Beta：直接从 close_panel 计算或从 style_factors 获取
     d. Volatility：同上
     e. Momentum：同上
  3. 对齐 f_d 和 X 的股票代码
  4. OLS：regress f_d ~ X，取残差 ε_d
  5. 填入 result.loc[d, codes] = ε_d
```

注意：
- 使用 `numpy.linalg.lstsq` 或 `statsmodels.OLS`（推荐 `np.linalg.lstsq`，无额外依赖）
- 需要处理 NaN：只对当前日期有有效因子值和风格数据的股票做回归
- 风格因子数据本身也需要 Z-score 标准化后使用

### 3.4 evaluate() 流程调整

```python
def evaluate(
    self,
    factor: pd.DataFrame,
    forward_period: int = 1,
    method: str = "spearman",
) -> dict[str, float]:
    """单因子完整评价 — 始终做风格中性化。"""
    # 1. 对因子值做风格中性化 → 纯 Alpha 预测值 E{ε}
    factor_neutral = self._neutralize_panel(factor)

    # 2. 计算前向收益
    fwd = self._forward_returns(forward_period)

    # 3. 对前向收益做同样的风格中性化 → 实际 Alpha 收益
    fwd_neutral = self._neutralize_panel(fwd)

    # 4. 取交集日期+股票
    common_idx = factor_neutral.index.intersection(fwd_neutral.index)
    common_cols = factor_neutral.columns.intersection(fwd_neutral.columns)
    f = factor_neutral.loc[common_idx, common_cols]
    r = fwd_neutral.loc[common_idx, common_cols]

    # 5. IC = Rank IC between 预测Alpha and 实际Alpha
    ic_series = f.corrwith(r, axis=1, method=method).dropna()

    # 6. Factor Return（略）
    ...

    # 7. T 统计量
    ...
```

关键变化对比：

| 旧逻辑 | 新逻辑 |
|--------|--------|
| 因子值可选中性化 | 因子值**强制**中性化 |
| 不对前向收益做中性化 | **前向收益也做**风格中性化 |
| IC = corr(因子值, 原始收益) | IC = corr(纯Alpha预测值, 实际Alpha收益) |
| neutral_ize 参数存在 | neutralize 参数移除 |

---

## Phase 4: AlphaFactorStrategy 优化器组合构建

**文件：** `stockquant/strategy/alpha_factor_strategy.py`

### 4.1 新增参数

```python
class AlphaFactorStrategy(BaseStrategy):
    def __init__(
        self,
        ...
        enable_style_neutral: bool = True,
        industry_exposure_limit: float = 0.05,   # ±5%
        size_exposure_limit: float = 0.1,         # ±0.1σ
    ):
```

### 4.2 新增优化器方法

```python
def _build_optimized_portfolio(
    self,
    factor_panel: pd.DataFrame,
) -> list[str]:
    """使用 cvxpy 构建行业/市值中性的优化组合。

    输入：
        factor_panel: dates × codes，因子值

    约束：
        - 行业权重偏差 ≤ industry_exposure_limit (相对比例)
        - 市值 Z-score 偏差 ≤ size_exposure_limit
        - 满仓投资
        - 个股上限 = 2% (50只等权)
        - 个股下限 = 0

    目标：
        maximize Σ w_i × factor_score_i

    返回：
        选择的股票代码列表
    """
```

实现流程：

```
对每个调仓日：
  1. 获取该日所有股票的因子值，去掉 NaN
  2. 该日截面的行业归属
  3. 计算各行业在截面中的占比（行业权重基准）
  4. 该日截面的市值，做 Z-score 归一化
  5. 构建 cvxpy 问题：
     n = 股票数量
     w = cp.Variable(n, nonneg=True)        # 权重变量，非负
     factor_scores = 当前因子值向量
     
     objective = cp.Maximize(factor_scores.T @ w)
     
     constraints = [
         cp.sum(w) <= 1,                      # 满仓（或允许不满仓）
         w <= max_weight_per_stock,            # 个股上限
         # 行业中性约束
         industry_exposure @ w <= industry_upper,
         industry_exposure @ w >= industry_lower,
         # 市值中性约束
         |size_zscore @ w| <= size_exposure_limit,
     ]
     
     problem = cp.Problem(objective, constraints)
     problem.solve()
  6. 取 w.value > 0 的股票
```

### 4.3 备用方案

```python
def _build_skip_neutral(
    self,
    factor_panel: pd.DataFrame,
) -> list[str]:
    """旧逻辑：简单取 Top-N，无行业/市值约束。"""
```

### 4.4 整合到 handle_bar

```python
def handle_bar(self, bar):
    ...
    # 调仓日
    if self._is_rebalance_day():
        factor = self.compute_factors(bar)
        if self.enable_style_neutral and HAS_CVXPY:
            selected = self._build_optimized_portfolio(factor)
        else:
            selected = self._build_skip_neutral(factor)
        self.rebalance(selected)
    ...
```

### 4.5 cvxpy 模块导入

```python
# 文件顶部
HAS_CVXPY = False
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    pass
```

---

## Phase 5: 集成测试 & Notebooks 更新

### 5.1 单元测试

**文件：** `tests/test_performance.py` — 新增：

```python
def test_trade_statistics_extended():
    """验证完整交易统计包含胜率/盈亏比等字段。"""

def test_excess_returns():
    """验证超额收益 = 策略收益 - 基准收益。"""

def test_alpha_returns():
    """验证 Alpha 收益曲线正确计算。"""

def test_per_trade_alpha_pnl():
    """验证逐笔 Alpha 盈亏计算。"""
```

**文件：** `tests/test_evaluator.py` — 新增：

```python
def test_neutralize_with_style_factors():
    """验证新增风格因子后的中性化结果。"""

def test_forward_return_also_neutralized():
    """验证前向收益也被中性化。"""
```

### 5.2 Notebooks 更新

**文件：** `notebooks/alpha191/_run_replication.py` — 更新：

```python
# 创建 FactorEvaluator 时传入完整风格因子
industry_map = ...  # 从数据源获取行业映射
market_cap_panel = ...  # 每日市值数据

evaluator = FactorEvaluator(
    close_panel=engine.close,
    industry=industry_map,
    market_cap=market_cap_panel,
    style_factors={
        "beta": ... ,
        "volatility": ... ,
        "momentum": ... ,
    },
)

# 之后调用 eveluate_system 和以前一样，但内部已做完整中性化
```

### 5.3 PlotEngine 集成

更新各 notebook 中调用 plot 的部分，新增性能看板和交易仪表盘：

```python
plotter = PlotEngine()
plotter.plot_performance_dashboard(result, save_path="...")
plotter.plot_trade_dashboard(az, save_path="...")
```

---

## 依赖 & 安装

| 依赖 | 用途 | Phase |
|------|------|-------|
| cvxpy | 组合优化器 | Phase 4 |

```bash
pip install cvxpy
```

---

## 执行顺序 & 依赖关系

```
Phase 1 (performance.py)  ← 无外部依赖
     ↓
Phase 2 (plot.py)         ← 依赖 Phase 1 的新接口
     ↓
Phase 3 (evaluator.py)    ← 无外部依赖
     ↓
Phase 4 (strategy.py)     ← 依赖 Phase 3 的中性化因子
     ↓
Phase 5 (tests/notebooks) ← 依赖全部前序 Phase
```

---

## 验收标准

| # | 验收项 | 如何验证 |
|---|--------|----------|
| 1 | trade_statistics 返回胜率/盈亏比/Profit Factor | `test_trade_statistics_extended` |
| 2 | per_trade_alpha_pnl 正确计算 Alpha PnL | `test_per_trade_alpha_pnl` |
| 3 | excess_returns 等于策略-基准 | `test_excess_returns` |
| 4 | cumulative_alpha 从1.0开始 | `test_alpha_returns` |
| 5 | 三面板性能图正确渲染 | 手动验证生成图片 |
| 6 | 交易仪表盘正确渲染 | 手动验证生成图片 |
| 7 | 风格中性化后的 IC 与原始 IC 不同 | `test_neutralize_with_style_factors` |
| 8 | 前向收益也被中性化 | `test_forward_return_also_neutralized` |
| 9 | 优化器构建的组合满足行业/市值约束 | 日志输出 + 断言 |
| 10 | cvxpy 不可用时优雅降级 | 模拟导入失败 |
