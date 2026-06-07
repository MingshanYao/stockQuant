# 因子评价体系框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一套完整的因子评价体系，借鉴国泰君安 Alpha191 报告的方法论，从单因子检验到因子体系分析，系统性评估因子的预测能力。

**Architecture:** 新建 `FactorEvaluator` 类作为评价体系核心，整合已有的 `FactorAnalyzer`（中性化/IC）能力，新增 factor return 计算、多周期检验、因子相关性分析。`FactorEvaluator` 接收因子面板 + 收益面板 + 可选的风格数据，输出完整的评价报告。最终集成到 `AlphaResearcher` 中替换现有的 `ic_summary()`。

**Tech Stack:** Python, pandas, numpy, scipy.stats (t-test)

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `stockquant/analysis/factor.py` | 扩展 `FactorAnalyzer`：新增 `calc_factor_return()` 静态方法 |
| `stockquant/analysis/evaluator.py` | **新建** `FactorEvaluator` 类：单因子完整评价 + 因子体系分析 |
| `stockquant/analysis/__init__.py` | 导出 `FactorEvaluator` |
| `stockquant/research/alpha_researcher.py` | 新增 `evaluate()` / `evaluate_system()` 方法，集成 `FactorEvaluator` |
| `tests/test_evaluator.py` | **新建** 完整测试 |

## 核心概念说明

**报告中的评价体系分三层：**

### 第一层：单因子四指标检验

对每个因子，在每个截面（交易日）上：
1. 用行业 + ln(市值) 对因子做截面回归，取残差 → **风格正交化后的因子值**
2. 用正交化后的因子值与未来 d 天收益做截面相关 → **IC**（信息系数）
3. 用正交化后的因子值对未来收益做截面回归，得到因子收益率 → **Factor Return**
4. 汇总时间序列：`IC_mean`、`IC_std`、`ICIR = IC_mean/IC_std`、`FR_mean`、`FR_IR = FR_mean/FR_std`、`IC_pos_ratio`（胜率）、T 统计量

### 第二层：多周期有效性检验

对每个因子，分别用 d=1,2,3,4,5 天的前向收益重复第一层 → 得到因子在不同预测周期下的表现，找到有效周期极限。

### 第三层：因子体系分析

对所有因子的 factor return 时间序列做两两相关 → 因子相关性矩阵，用于去冗余、评估边际贡献。

---

## Task 1: `calc_factor_return` — 截面回归因子收益率

**Files:**
- Modify: `stockquant/analysis/factor.py`
- Test: `tests/test_evaluator.py`

因子收益率 = 在截面上，用因子值对未来收益做回归的回归系数。这是除 IC 外最重要的因子评价指标。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evaluator.py
"""测试 - 因子评价体系。"""

import numpy as np
import pandas as pd
import pytest

from stockquant.analysis.factor import FactorAnalyzer


class TestCalcFactorReturn:

    def test_positive_factor_return(self):
        """因子值与收益正相关时，factor return 应为正。"""
        np.random.seed(42)
        n = 100
        factor = pd.Series(np.random.randn(n))
        returns = factor * 0.05 + np.random.randn(n) * 0.01
        fr = FactorAnalyzer.calc_factor_return(factor, returns)
        assert fr > 0

    def test_zero_factor_return(self):
        """因子值与收益无关时，factor return 应接近零。"""
        np.random.seed(42)
        n = 200
        factor = pd.Series(np.random.randn(n))
        returns = pd.Series(np.random.randn(n) * 0.01)
        fr = FactorAnalyzer.calc_factor_return(factor, returns)
        assert abs(fr) < 0.05

    def test_insufficient_data(self):
        """数据不足时返回 0。"""
        factor = pd.Series([1.0, 2.0])
        returns = pd.Series([0.01, 0.02])
        fr = FactorAnalyzer.calc_factor_return(factor, returns)
        assert fr == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::TestCalcFactorReturn -v`
Expected: FAIL — `AttributeError: type object 'FactorAnalyzer' has no attribute 'calc_factor_return'`

- [ ] **Step 3: 实现 `calc_factor_return`**

在 `stockquant/analysis/factor.py` 的 `FactorAnalyzer` 类中，在 `calc_ir` 方法后添加：

```python
@staticmethod
def calc_factor_return(
    factor: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """截面回归因子收益率。

    在截面上对因子值做标准化后，用 OLS 回归未来收益，
    回归系数即为因子收益率（factor return）。
    """
    valid = pd.concat(
        [factor.rename("f"), forward_returns.rename("r")], axis=1
    ).dropna()
    if len(valid) < 10:
        return 0.0

    f = valid["f"].values
    r = valid["r"].values

    # 标准化因子
    f_std = f.std()
    if f_std == 0:
        return 0.0
    f_norm = (f - f.mean()) / f_std

    X = np.column_stack([f_norm, np.ones(len(f_norm))])
    beta = np.linalg.lstsq(X, r, rcond=None)[0]
    return float(beta[0])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::TestCalcFactorReturn -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add stockquant/analysis/factor.py tests/test_evaluator.py
git commit -m "feat: add calc_factor_return to FactorAnalyzer"
```

---

## Task 2: `FactorEvaluator` — 单因子完整评价

**Files:**
- Create: `stockquant/analysis/evaluator.py`
- Test: `tests/test_evaluator.py`

核心类：接收因子面板 + 收益面板 + 可选风格数据，输出单因子的完整四指标评价。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evaluator.py — 追加

from stockquant.analysis.evaluator import FactorEvaluator


def _make_panel(n_days: int = 60, n_stocks: int = 50, seed: int = 42) -> dict:
    """生成测试用因子面板和收益面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

    factor = pd.DataFrame(
        rng.standard_normal((n_days, n_stocks)),
        index=dates, columns=codes,
    )
    # 构造有预测力的收益：factor * 0.02 + noise
    returns = factor * 0.02 + pd.DataFrame(
        rng.standard_normal((n_days, n_stocks)) * 0.03,
        index=dates, columns=codes,
    )
    close = pd.DataFrame(
        100 + np.cumsum(rng.standard_normal((n_days, n_stocks)) * 0.5, axis=0),
        index=dates, columns=codes,
    )
    return {"factor": factor, "returns": returns, "close": close}


class TestFactorEvaluator:

    def test_evaluate_single_returns_all_metrics(self):
        """单因子评价应返回完整指标集。"""
        data = _make_panel()
        ev = FactorEvaluator(close_panel=data["close"])
        result = ev.evaluate(data["factor"], forward_period=1)

        assert "ic_mean" in result
        assert "ic_ir" in result
        assert "fr_mean" in result
        assert "fr_ir" in result
        assert "ic_pos_ratio" in result
        assert "t_stat" in result

    def test_ic_positive_for_predictive_factor(self):
        """有预测力的因子 IC 应为正。"""
        data = _make_panel()
        ev = FactorEvaluator(close_panel=data["close"])
        result = ev.evaluate(data["factor"], forward_period=1)
        assert result["ic_mean"] > 0

    def test_multi_horizon(self):
        """多周期评价应返回每个周期的结果。"""
        data = _make_panel()
        ev = FactorEvaluator(close_panel=data["close"])
        results = ev.evaluate_multi_horizon(
            data["factor"], periods=[1, 2, 3]
        )
        assert len(results) == 3
        assert all(p in results for p in [1, 2, 3])

    def test_neutralize_integration(self):
        """传入行业/市值后应能完成中性化评价。"""
        data = _make_panel()
        rng = np.random.default_rng(99)
        n_stocks = 50
        codes = data["factor"].columns
        industry = pd.Series(
            rng.choice(["银行", "医药", "科技", "消费"], n_stocks),
            index=codes,
        )
        market_cap = pd.Series(
            rng.uniform(1e9, 1e11, n_stocks),
            index=codes,
        )

        ev = FactorEvaluator(
            close_panel=data["close"],
            industry=industry,
            market_cap=market_cap,
        )
        result = ev.evaluate(data["factor"], forward_period=1)
        assert "ic_mean" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::TestFactorEvaluator -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockquant.analysis.evaluator'`

- [ ] **Step 3: 实现 `FactorEvaluator`**

创建 `stockquant/analysis/evaluator.py`：

```python
"""
因子评价器 — 单因子四指标检验 + 多周期分析 + 因子体系评价。

借鉴国泰君安 Alpha191 报告的评价方法论：
  风格正交化 → IC / Factor Return / IR / T统计量 → 多周期扫描 → 体系相关性
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockquant.analysis.factor import FactorAnalyzer
from stockquant.utils.logger import get_logger

logger = get_logger("analysis.evaluator")

TRADING_DAYS_PER_YEAR = 252


class FactorEvaluator:
    """因子评价器。

    Parameters
    ----------
    close_panel : DataFrame
        收盘价面板（行=日期，列=股票代码），用于计算前向收益。
    industry : Series, optional
        股票代码 → 行业分类的映射，用于风格正交化。
    market_cap : Series, optional
        股票代码 → 市值的映射，用于风格正交化。
        注：当前使用静态市值（截面快照），后续可扩展为面板。
    """

    def __init__(
        self,
        close_panel: pd.DataFrame,
        industry: pd.Series | None = None,
        market_cap: pd.Series | None = None,
    ) -> None:
        self.close = close_panel
        self.industry = industry
        self.market_cap = market_cap

    def _forward_returns(self, period: int) -> pd.DataFrame:
        """计算 T+period 的前向收益率面板。"""
        return self.close.pct_change(periods=period).shift(-period)

    def _neutralize_panel(self, factor: pd.DataFrame) -> pd.DataFrame:
        """对因子面板逐日做截面风格正交化。"""
        if self.industry is None and self.market_cap is None:
            return factor

        result = factor.copy()
        for date in factor.index:
            row = factor.loc[date].dropna()
            if len(row) < 10:
                continue
            codes = row.index
            ind = self.industry.reindex(codes) if self.industry is not None else None
            cap = self.market_cap.reindex(codes) if self.market_cap is not None else None
            neutralized = FactorAnalyzer.neutralize(row, market_cap=cap, industry=ind)
            result.loc[date, neutralized.index] = neutralized.values
        return result

    def evaluate(
        self,
        factor: pd.DataFrame,
        forward_period: int = 1,
        method: str = "spearman",
        neutralize: bool = True,
    ) -> dict[str, float]:
        """单因子完整评价 — 四指标体系。

        Parameters
        ----------
        factor : DataFrame
            因子面板（行=日期，列=股票代码）。
        forward_period : int
            前向收益天数，默认 T+1。
        method : str
            IC 计算方法，``spearman`` 或 ``pearson``。
        neutralize : bool
            是否做风格正交化（需要初始化时传入 industry/market_cap）。

        Returns
        -------
        dict
            ic_mean, ic_std, ic_ir, ic_pos_ratio,
            fr_mean, fr_std, fr_ir,
            fr_annual (年化因子收益率), t_stat
        """
        fwd = self._forward_returns(forward_period)

        if neutralize and (self.industry is not None or self.market_cap is not None):
            factor = self._neutralize_panel(factor)

        common_idx = factor.index.intersection(fwd.index)
        common_cols = factor.columns.intersection(fwd.columns)
        f = factor.loc[common_idx, common_cols]
        r = fwd.loc[common_idx, common_cols]

        # 逐日截面 IC
        ic_series = f.corrwith(r, axis=1, method=method).dropna()

        # 逐日截面 Factor Return
        fr_list = []
        for date in common_idx:
            f_row = f.loc[date].dropna()
            r_row = r.loc[date].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) >= 10:
                fr = FactorAnalyzer.calc_factor_return(f_row[common], r_row[common])
                fr_list.append(fr)
        fr_series = pd.Series(fr_list)

        ic_mean = ic_series.mean() if len(ic_series) else np.nan
        ic_std = ic_series.std() if len(ic_series) else np.nan
        ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        ic_pos = (ic_series > 0).mean() if len(ic_series) else np.nan

        fr_mean = fr_series.mean() if len(fr_series) else np.nan
        fr_std = fr_series.std() if len(fr_series) else np.nan
        fr_ir = fr_mean / fr_std if fr_std and fr_std > 0 else np.nan

        # 年化因子收益率 & T统计量（参考报告公式）
        d = forward_period
        n = len(fr_series)
        fr_annual = fr_mean * (TRADING_DAYS_PER_YEAR / d) if not np.isnan(fr_mean) else np.nan
        t_stat = (
            ic_mean / (ic_std / np.sqrt(n))
            if n > 0 and ic_std and ic_std > 0
            else np.nan
        )

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_pos_ratio": ic_pos,
            "fr_mean": fr_mean,
            "fr_std": fr_std,
            "fr_ir": fr_ir,
            "fr_annual": fr_annual,
            "t_stat": t_stat,
            "n_periods": n,
        }

    def evaluate_multi_horizon(
        self,
        factor: pd.DataFrame,
        periods: list[int] | None = None,
        **kwargs,
    ) -> dict[int, dict[str, float]]:
        """多周期因子评价 — 扫描不同预测周期的因子表现。

        Parameters
        ----------
        factor : DataFrame
            因子面板。
        periods : list[int]
            预测周期列表，默认 [1, 2, 3, 4, 5]。

        Returns
        -------
        dict[int, dict]
            {period: evaluate() 结果}
        """
        if periods is None:
            periods = [1, 2, 3, 4, 5]

        results = {}
        for d in periods:
            results[d] = self.evaluate(factor, forward_period=d, **kwargs)
        return results

    def evaluate_system(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """因子体系批量评价 — 返回所有因子的四指标汇总表。

        Parameters
        ----------
        factors : dict[str, DataFrame]
            {因子名: 因子面板}
        forward_period : int
            预测周期。

        Returns
        -------
        DataFrame
            行=因子名，列=各评价指标，按 |ic_mean| 降序。
        """
        rows = []
        total = len(factors)
        for i, (name, panel) in enumerate(factors.items(), 1):
            logger.info(f"[{i}/{total}] 评价因子: {name}")
            metrics = self.evaluate(panel, forward_period=forward_period, **kwargs)
            metrics["factor"] = name
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("factor")
        return df.reindex(df["ic_mean"].abs().sort_values(ascending=False).index)

    def factor_correlation_matrix(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
    ) -> pd.DataFrame:
        """因子收益率相关性矩阵 — 用于去冗余。

        计算所有因子对的 factor return 时间序列的 Pearson 相关系数。

        Parameters
        ----------
        factors : dict[str, DataFrame]
            {因子名: 因子面板}

        Returns
        -------
        DataFrame
            对称相关矩阵，行列均为因子名。
        """
        fwd = self._forward_returns(forward_period)
        fr_dict: dict[str, list[float]] = {}

        for name, panel in factors.items():
            common_idx = panel.index.intersection(fwd.index)
            common_cols = panel.columns.intersection(fwd.columns)
            f = panel.loc[common_idx, common_cols]
            r = fwd.loc[common_idx, common_cols]

            fr_list = []
            for date in common_idx:
                f_row = f.loc[date].dropna()
                r_row = r.loc[date].dropna()
                common = f_row.index.intersection(r_row.index)
                if len(common) >= 10:
                    fr = FactorAnalyzer.calc_factor_return(f_row[common], r_row[common])
                    fr_list.append(fr)
                else:
                    fr_list.append(np.nan)
            fr_dict[name] = fr_list

        fr_df = pd.DataFrame(fr_dict)
        return fr_df.corr()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::TestFactorEvaluator -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add stockquant/analysis/evaluator.py tests/test_evaluator.py
git commit -m "feat: add FactorEvaluator with full factor evaluation framework"
```

---

## Task 3: 因子体系分析测试

**Files:**
- Modify: `tests/test_evaluator.py`
- Modify: `stockquant/analysis/evaluator.py` (如有 bug)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evaluator.py — 追加

class TestFactorSystem:

    def test_evaluate_system(self):
        """批量评价多个因子。"""
        data = _make_panel()
        rng = np.random.default_rng(123)
        n_days, n_stocks = data["factor"].shape
        factor2 = pd.DataFrame(
            rng.standard_normal((n_days, n_stocks)),
            index=data["factor"].index,
            columns=data["factor"].columns,
        )
        ev = FactorEvaluator(close_panel=data["close"])
        result = ev.evaluate_system(
            {"f1": data["factor"], "f2": factor2},
            forward_period=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "ic_mean" in result.columns

    def test_correlation_matrix(self):
        """因子相关矩阵应为对称方阵。"""
        data = _make_panel()
        rng = np.random.default_rng(123)
        n_days, n_stocks = data["factor"].shape
        factor2 = pd.DataFrame(
            rng.standard_normal((n_days, n_stocks)),
            index=data["factor"].index,
            columns=data["factor"].columns,
        )
        ev = FactorEvaluator(close_panel=data["close"])
        corr = ev.factor_correlation_matrix(
            {"f1": data["factor"], "f2": factor2}
        )
        assert corr.shape == (2, 2)
        assert abs(corr.loc["f1", "f1"] - 1.0) < 0.01

    def test_correlated_factors_detected(self):
        """高相关因子的相关系数应接近 1。"""
        data = _make_panel()
        factor_copy = data["factor"] * 1.5 + 0.1  # 线性变换
        ev = FactorEvaluator(close_panel=data["close"])
        corr = ev.factor_correlation_matrix(
            {"original": data["factor"], "scaled": factor_copy}
        )
        assert corr.loc["original", "scaled"] > 0.9
```

- [ ] **Step 2: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::TestFactorSystem -v`
Expected: 3 passed

- [ ] **Step 3: 提交**

```bash
git add tests/test_evaluator.py
git commit -m "test: add factor system evaluation tests"
```

---

## Task 4: `__init__.py` 导出 + `AlphaResearcher` 集成

**Files:**
- Modify: `stockquant/analysis/__init__.py`
- Modify: `stockquant/research/alpha_researcher.py`
- Test: `tests/test_evaluator.py`

将 `FactorEvaluator` 集成到 `AlphaResearcher`，新增 `evaluate()` 和 `evaluate_system()` 方法。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_evaluator.py — 追加

from stockquant.analysis import FactorEvaluator


class TestAnalysisExports:

    def test_evaluator_importable(self):
        """FactorEvaluator 可从 analysis 包导入。"""
        assert FactorEvaluator is not None


class TestAlphaResearcherIntegration:

    @pytest.fixture
    def researcher(self):
        from stockquant.data.universe import BacktestDataset
        from stockquant.research.alpha_researcher import AlphaResearcher

        # 构造最简 dataset
        n_days, n_stocks = 30, 10
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        rng = np.random.default_rng(42)

        stock_data = {}
        for code in codes:
            close = 10 + np.cumsum(rng.normal(0, 0.2, n_days))
            stock_data[code] = pd.DataFrame({
                "code": code, "date": dates,
                "open": close * 0.99, "high": close * 1.02,
                "low": close * 0.98, "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n_days),
                "amount": close * rng.integers(1_000_000, 5_000_000, n_days),
            })
        benchmark = pd.DataFrame({
            "code": "000300", "date": dates,
            "open": 4000.0, "high": 4020.0, "low": 3980.0,
            "close": 4000 + np.cumsum(rng.normal(0, 5, n_days)),
            "volume": rng.integers(1e8, 5e8, n_days),
            "amount": 4000 * rng.integers(1e8, 5e8, n_days),
        })
        dataset = BacktestDataset(
            stock_data=stock_data, codes=codes,
            benchmark=benchmark, benchmark_code="000300",
            start_date="2024-01-02", end_date="2024-02-12",
        )
        return AlphaResearcher(dataset, max_positions=3, rebalance_freq=5)

    def test_evaluate_factor(self, researcher):
        """AlphaResearcher.evaluate_factor() 返回完整指标。"""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=30)
        codes = [f"{i:06d}" for i in range(1, 11)]
        panel = pd.DataFrame(
            rng.standard_normal((30, 10)),
            index=dates, columns=codes,
        )
        result = researcher.evaluate_factor(panel, forward_period=1)
        assert "ic_mean" in result
        assert "fr_annual" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::TestAlphaResearcherIntegration -v`
Expected: FAIL — `AttributeError: 'AlphaResearcher' has no attribute 'evaluate_factor'`

- [ ] **Step 3: 更新 `analysis/__init__.py`**

读取当前 `stockquant/analysis/__init__.py` 内容，追加 `FactorEvaluator` 导出：

```python
"""因子分析模块"""

from .performance import PerformanceAnalyzer
from .factor import FactorAnalyzer
from .evaluator import FactorEvaluator

__all__ = ["PerformanceAnalyzer", "FactorAnalyzer", "FactorEvaluator"]
```

- [ ] **Step 4: 在 `AlphaResearcher` 中添加集成方法**

在 `stockquant/research/alpha_researcher.py` 中，在 `ic_summary` 方法之后添加：

```python
def evaluate_factor(
    self,
    factor_panel: pd.DataFrame,
    forward_period: int = 1,
    neutralize: bool = False,
    **kwargs,
) -> dict[str, float]:
    """使用 FactorEvaluator 对因子面板做完整评价。

    Parameters
    ----------
    factor_panel : DataFrame
        因子面板（行=日期，列=股票代码）。
    forward_period : int
        前向收益天数，默认 1。
    neutralize : bool
        是否做风格正交化，默认 False（需要 stock_info 数据支持）。

    Returns
    -------
    dict
        包含 ic_mean, ic_ir, fr_mean, fr_ir, fr_annual, t_stat 等。
    """
    from stockquant.analysis.evaluator import FactorEvaluator

    close = self.alpha_engine.close
    ev = FactorEvaluator(close_panel=close)
    return ev.evaluate(
        factor_panel,
        forward_period=forward_period,
        neutralize=neutralize,
        **kwargs,
    )

def evaluate_multi_horizon(
    self,
    factor_panel: pd.DataFrame,
    periods: list[int] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """多周期因子评价，返回 DataFrame 便于对比。

    Returns
    -------
    DataFrame
        行=预测周期，列=评价指标。
    """
    from stockquant.analysis.evaluator import FactorEvaluator

    close = self.alpha_engine.close
    ev = FactorEvaluator(close_panel=close)
    results = ev.evaluate_multi_horizon(factor_panel, periods=periods, **kwargs)
    return pd.DataFrame(results).T.rename_axis("period")
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py -v`
Expected: ALL passed

- [ ] **Step 6: 提交**

```bash
git add stockquant/analysis/__init__.py stockquant/analysis/evaluator.py stockquant/research/alpha_researcher.py tests/test_evaluator.py
git commit -m "feat: integrate FactorEvaluator into AlphaResearcher"
```

---

## Task 5: 全量回归测试

**Files:** 无修改，仅验证

- [ ] **Step 1: 运行全量测试**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_hs300_daily.py`
Expected: ALL passed，无回归

- [ ] **Step 2: 快速冒烟测试 — 在 Python 中验证 API 可用**

```python
.venv/bin/python -c "
from stockquant.analysis import FactorEvaluator, FactorAnalyzer
import pandas as pd, numpy as np

# 快速验证
rng = np.random.default_rng(42)
close = pd.DataFrame(rng.standard_normal((60, 20)), index=pd.bdate_range('2024-01-02', periods=60))
factor = pd.DataFrame(rng.standard_normal((60, 20)), index=close.index, columns=close.columns)

ev = FactorEvaluator(close_panel=close)
r = ev.evaluate(factor, forward_period=1)
print('Single:', {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})

mh = ev.evaluate_multi_horizon(factor, periods=[1,2,3])
print(f'Multi-horizon: {len(mh)} periods')
print('OK')
"
```

Expected: 输出指标值和 "OK"

---

## 验证

```bash
# 1. 单元测试全通过
.venv/bin/python -m pytest tests/test_evaluator.py -v

# 2. 全量回归测试
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_hs300_daily.py

# 3. API 冒烟测试
.venv/bin/python -c "from stockquant.analysis import FactorEvaluator; print('import OK')"
```

## 后续扩展（不在本次范围）

1. **动态市值面板**：当前 `market_cap` 为静态 Series，后续可扩展为 DataFrame（每日市值）
2. **分组回测层**：按因子分 5/10 组，计算多空组合净值曲线
3. **因子衰减分析**：IC 的半衰期计算
4. **可视化**：IC 时间序列图、因子收益率累计图、相关矩阵热力图
