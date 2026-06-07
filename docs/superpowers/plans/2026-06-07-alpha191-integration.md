# Alpha191 因子集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将国泰君安191个短周期价量Alpha因子集成到现有指标框架中，复用Alpha101的架构模式和算子库。

**Architecture:** 在 `stockquant/indicators/` 下新建 `alpha191/` 包，包含 `operators.py`（新增算子）、`alpha191.py`（Alpha191Engine + Alpha191Indicators）。Alpha191Engine 复用 Alpha101 的面板数据结构（open/high/low/close/volume/vwap，行=日期，列=股票代码），新增 benchmark 面板支持（约6个因子需要基准指数数据）。通过 IndicatorRegistry 注册，与 AlphaResearcher 集成。

**Tech Stack:** Python 3.12, pandas, numpy, 现有 `stockquant.indicators` 框架

---

## 改动文件总览

| 文件 | 改动 |
|------|------|
| `stockquant/indicators/alpha191/__init__.py` | **新建** — 包导出 |
| `stockquant/indicators/alpha191/operators.py` | **新建** — Alpha191 专用算子 |
| `stockquant/indicators/alpha191/alpha191.py` | **新建** — Alpha191Engine + Alpha191Indicators |
| `stockquant/indicators/__init__.py` | 新增 `Alpha191Indicators` 导出 |
| `tests/test_alpha191.py` | **新建** — 算子 + 引擎 + 因子测试 |

## Alpha191 相对 Alpha101 的新增算子

Alpha191 使用的大部分算子与 Alpha101 相同（RANK, DELAY, DELTA, CORR, SMA/MEAN, TSRANK, TSMIN, TSMAX, DECAYLINEAR, SIGN, STD, SUM, PROD, LOG, ABS）。以下是 Alpha191 **独有**的算子，需要在 `alpha191/operators.py` 中实现：

| 算子 | 含义 | 使用因子（举例） |
|------|------|--------------|
| `SMA(x, n, m)` | 指数加权移动平均: Y[t] = (x[t]*m + Y[t-1]*(n-m)) / n | Alpha9,22,23,40,47,63,67... (~30个) |
| `WMA(x, n)` | 加权移动平均（权重递减） | Alpha27, Alpha30 |
| `REGBETA(y, x, n)` | 滚动窗口 OLS 回归斜率 | Alpha21, Alpha116, Alpha147 |
| `COUNT(condition, n)` | 滚动窗口内 True 的计数 | Alpha53, Alpha75, Alpha144 |
| `SUMIF(x, n, condition)` | 条件滚动求和 | Alpha144, Alpha190 |
| `HIGHDAY(x, n)` | 滚动窗口内最高值距今天数 | Alpha133 |
| `LOWDAY(x, n)` | 滚动窗口内最低值距今天数 | Alpha103, Alpha133 |
| `SEQUENCE(n)` | 返回 1, 2, ..., n 序列 | Alpha21, Alpha116, Alpha147 (配合REGBETA) |
| `FILTER(x, condition)` | 条件过滤（非条件行置 NaN） | Alpha149, Alpha150 |
| `SUMAC(x)` | 累计求和 | Alpha165, Alpha183 |

## 需要基准指数数据的因子

以下因子引用 `BANCHMARKINDEXCLOSE` / `BANCHMARKINDEXOPEN`，Alpha191Engine 需要接受 benchmark 面板参数：

- Alpha75, Alpha149, Alpha150, Alpha181, Alpha182, Alpha190

## 需要特殊处理的因子

- **Alpha30**: 使用 `REGRESI(ret, MKT, SMB, HML, 60)` — 三因子模型回归残差，需要 MKT/SMB/HML 因子面板，MVP 阶段跳过
- **Alpha69**: 使用 DTM/DBM 中间变量
- **Alpha143**: 使用 SELF 引用（递归），需要迭代计算
- **Alpha144**: SUMIF + COUNT 组合
- **Alpha146**: 复杂嵌套 SMA
- **Alpha166**: 三阶矩（偏度）计算
- **Alpha172, Alpha186**: 使用 LD/HD/TR（方向运动指标）

---

### Task 1: Alpha191 新增算子 + 测试

**Files:**
- Create: `stockquant/indicators/alpha191/__init__.py`
- Create: `stockquant/indicators/alpha191/operators.py`
- Create: `tests/test_alpha191.py`

- [ ] **Step 1: 创建包目录和空 `__init__.py`**

```bash
mkdir -p stockquant/indicators/alpha191
touch stockquant/indicators/alpha191/__init__.py
```

- [ ] **Step 2: 写算子的失败测试**

```python
# tests/test_alpha191.py
"""测试 - Alpha191 因子指标。"""

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# Alpha191 专用算子测试
# ======================================================================

class TestAlpha191Operators:

    @pytest.fixture
    def sample_series(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.standard_normal(100))

    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=60)
        codes = [f"{i:06d}" for i in range(1, 11)]
        return pd.DataFrame(
            rng.standard_normal((60, 10)),
            index=dates, columns=codes,
        )

    def test_ema_sma_basic(self, sample_series):
        from stockquant.indicators.alpha191.operators import ema_sma
        result = ema_sma(sample_series, n=10, m=2)
        assert len(result) == len(sample_series)
        assert result.isna().sum() < len(sample_series)

    def test_ema_sma_dataframe(self, sample_df):
        from stockquant.indicators.alpha191.operators import ema_sma
        result = ema_sma(sample_df, n=10, m=2)
        assert result.shape == sample_df.shape

    def test_wma(self, sample_series):
        from stockquant.indicators.alpha191.operators import wma
        result = wma(sample_series, n=10)
        assert len(result) == len(sample_series)

    def test_regbeta(self, sample_series):
        from stockquant.indicators.alpha191.operators import regbeta
        x = pd.Series(range(len(sample_series)), dtype=float)
        result = regbeta(sample_series, x, n=20)
        assert len(result) == len(sample_series)

    def test_count(self, sample_series):
        from stockquant.indicators.alpha191.operators import count
        cond = sample_series > 0
        result = count(cond, n=10)
        assert result.max() <= 10
        assert result.min() >= 0

    def test_highday(self, sample_series):
        from stockquant.indicators.alpha191.operators import highday
        result = highday(sample_series, n=10)
        assert result.max() < 10

    def test_lowday(self, sample_series):
        from stockquant.indicators.alpha191.operators import lowday
        result = lowday(sample_series, n=10)
        assert result.max() < 10

    def test_sumif(self, sample_series):
        from stockquant.indicators.alpha191.operators import sumif
        cond = sample_series > 0
        result = sumif(sample_series, n=10, condition=cond)
        assert len(result) == len(sample_series)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Operators -v`
Expected: FAIL — ImportError

- [ ] **Step 4: 实现算子**

```python
# stockquant/indicators/alpha191/operators.py
"""
Alpha191 专用算子 — 补充 Alpha101 算子库中不包含的运算。

Alpha191 与 Alpha101 共享的算子直接从 alpha101.operators 导入使用。
本模块仅包含 Alpha191 独有的运算符。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema_sma(
    x: pd.DataFrame | pd.Series, n: int, m: int,
) -> pd.DataFrame | pd.Series:
    """指数加权移动平均: Y[t] = (x[t]*m + Y[t-1]*(n-m)) / n。

    对应 Alpha191 公式中的 SMA(X, N, M) 运算符。
    注意：这不是简单移动平均，而是指数平滑。
    """
    alpha = m / n
    return x.ewm(alpha=alpha, adjust=False).mean()


def wma(
    x: pd.DataFrame | pd.Series, n: int,
) -> pd.DataFrame | pd.Series:
    """加权移动平均 (权重递减: n, n-1, ..., 1)。"""
    weights = np.arange(n, 0, -1, dtype=float)
    weights = weights / weights.sum()

    def _wma(arr: np.ndarray) -> float:
        k = len(arr)
        if k < n:
            w = np.arange(k, 0, -1, dtype=float)
            w /= w.sum()
            return float(np.dot(arr, w))
        return float(np.dot(arr, weights))

    return x.rolling(n, min_periods=max(1, n // 2)).apply(_wma, raw=True)


def regbeta(
    y: pd.DataFrame | pd.Series,
    x: pd.DataFrame | pd.Series,
    n: int,
) -> pd.DataFrame | pd.Series:
    """滚动窗口 OLS 回归斜率 beta。

    REGBETA(Y, X, N) = 过去 N 期 Y 对 X 的 OLS 回归系数。
    当 X 为 SEQUENCE(N) 时，等价于趋势斜率。
    """
    def _beta(y_arr: np.ndarray, x_arr: np.ndarray) -> float:
        if len(y_arr) < 3:
            return np.nan
        x_dm = x_arr - x_arr.mean()
        denom = np.dot(x_dm, x_dm)
        if denom == 0:
            return np.nan
        return float(np.dot(x_dm, y_arr - y_arr.mean()) / denom)

    if isinstance(y, pd.DataFrame):
        result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
        for col in y.columns:
            yc = y[col].values
            xc = x[col].values if isinstance(x, pd.DataFrame) else x.values
            vals = np.full(len(yc), np.nan)
            for i in range(n - 1, len(yc)):
                vals[i] = _beta(yc[i - n + 1: i + 1], xc[i - n + 1: i + 1])
            result[col] = vals
        return result
    else:
        x_vals = x.values if isinstance(x, pd.Series) else x
        vals = np.full(len(y), np.nan)
        y_arr = y.values
        for i in range(n - 1, len(y_arr)):
            vals[i] = _beta(y_arr[i - n + 1: i + 1], x_vals[i - n + 1: i + 1])
        return pd.Series(vals, index=y.index)


def sequence(n: int) -> np.ndarray:
    """返回 1, 2, ..., n 的序列。配合 REGBETA 使用。"""
    return np.arange(1, n + 1, dtype=float)


def count(
    condition: pd.DataFrame | pd.Series, n: int,
) -> pd.DataFrame | pd.Series:
    """滚动窗口内 True 的计数。"""
    return condition.astype(float).rolling(n, min_periods=1).sum()


def sumif(
    x: pd.DataFrame | pd.Series,
    n: int,
    condition: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """条件滚动求和: 仅对 condition 为 True 的值累加。"""
    masked = x.where(condition, 0.0)
    return masked.rolling(n, min_periods=1).sum()


def highday(
    x: pd.DataFrame | pd.Series, n: int,
) -> pd.DataFrame | pd.Series:
    """滚动窗口内最高值距今天数 (0 = 今天就是最高)。"""
    def _hd(arr: np.ndarray) -> float:
        return float(len(arr) - 1 - np.argmax(arr))

    return x.rolling(n, min_periods=max(1, n // 2)).apply(_hd, raw=True)


def lowday(
    x: pd.DataFrame | pd.Series, n: int,
) -> pd.DataFrame | pd.Series:
    """滚动窗口内最低值距今天数 (0 = 今天就是最低)。"""
    def _ld(arr: np.ndarray) -> float:
        return float(len(arr) - 1 - np.argmin(arr))

    return x.rolling(n, min_periods=max(1, n // 2)).apply(_ld, raw=True)


def filter_cond(
    x: pd.DataFrame | pd.Series,
    condition: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """条件过滤: 非 condition 的行置 NaN。"""
    return x.where(condition)


def sumac(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """累计求和。"""
    return x.cumsum()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Operators -v`
Expected: 8 PASSED

- [ ] **Step 6: Commit**

```bash
git add stockquant/indicators/alpha191/__init__.py stockquant/indicators/alpha191/operators.py tests/test_alpha191.py
git commit -m "feat: add Alpha191 operator library (ema_sma, wma, regbeta, count, sumif, highday, lowday)"
```

---

### Task 2: Alpha191Engine 骨架 + Alpha191Indicators 包装器

**Files:**
- Create: `stockquant/indicators/alpha191/alpha191.py`
- Modify: `stockquant/indicators/alpha191/__init__.py`
- Test: `tests/test_alpha191.py`

- [ ] **Step 1: 写引擎骨架的失败测试**

在 `tests/test_alpha191.py` 中追加：

```python
# ======================================================================
# 辅助 — 构建测试面板
# ======================================================================

def _make_test_dataset(n_days: int = 60, n_stocks: int = 10, seed: int = 42):
    """创建测试用面板数据 (open, high, low, close, volume, amount)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

    close = pd.DataFrame(
        10 + np.cumsum(rng.standard_normal((n_days, n_stocks)) * 0.2, axis=0),
        index=dates, columns=codes,
    )
    close = close.clip(lower=1.0)
    open_ = close * (1 + rng.uniform(-0.01, 0.01, (n_days, n_stocks)))
    high = pd.DataFrame(
        np.maximum(open_.values, close.values) * (1 + rng.uniform(0, 0.02, (n_days, n_stocks))),
        index=dates, columns=codes,
    )
    low = pd.DataFrame(
        np.minimum(open_.values, close.values) * (1 - rng.uniform(0, 0.02, (n_days, n_stocks))),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(100_000, 1_000_000, (n_days, n_stocks)),
        index=dates, columns=codes, dtype=float,
    )
    amount = close * volume * 100

    return {
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "amount": amount,
    }


# ======================================================================
# Alpha191Engine 基础测试
# ======================================================================

class TestAlpha191Engine:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset()
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    def test_panel_shapes(self, engine):
        assert engine.open.shape == engine.close.shape
        assert engine.vwap.shape == engine.close.shape
        assert engine.returns.shape == engine.close.shape

    def test_compute_factor_invalid_raises(self, engine):
        with pytest.raises(ValueError, match="未实现"):
            engine.compute_factor(999)

    def test_compute_factor_returns_dataframe(self, engine):
        result = engine.compute_factor(2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape


class TestAlpha191Indicators:

    def test_name_property(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Indicators
        ind = Alpha191Indicators()
        assert ind.name == "Alpha191"

    def test_compute_single_stock(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Indicators
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame({
            "open": 10 + rng.standard_normal(n) * 0.1,
            "high": 10.2 + rng.standard_normal(n) * 0.1,
            "low": 9.8 + rng.standard_normal(n) * 0.1,
            "close": 10 + rng.standard_normal(n) * 0.1,
            "volume": rng.integers(100_000, 1_000_000, n),
        })
        ind = Alpha191Indicators(alphas=[2, 13, 14])
        result = ind.compute(df)
        assert "alpha002" in result.columns
        assert "alpha013" in result.columns
        assert "alpha014" in result.columns

    def test_from_dataset(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Indicators
        from stockquant.data.universe import BacktestDataset

        rng = np.random.default_rng(42)
        n_days, n_stocks = 30, 5
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        stock_data = {}
        for code in codes:
            close = 10 + np.cumsum(rng.normal(0, 0.2, n_days))
            close = np.maximum(close, 1.0)
            stock_data[code] = pd.DataFrame({
                "code": code, "date": dates,
                "open": close * 0.99, "high": close * 1.02,
                "low": close * 0.98, "close": close,
                "volume": rng.integers(100_000, 500_000, n_days),
                "amount": close * rng.integers(100_000, 500_000, n_days),
            })
        benchmark = pd.DataFrame({
            "code": "000300", "date": dates,
            "open": 4000.0, "high": 4020.0, "low": 3980.0,
            "close": 4000 + np.cumsum(rng.normal(0, 5, n_days)),
            "volume": rng.integers(int(1e8), int(5e8), n_days),
            "amount": 4000 * rng.integers(int(1e8), int(5e8), n_days),
        })
        dataset = BacktestDataset(
            stock_data=stock_data, codes=codes,
            benchmark=benchmark, benchmark_code="000300",
            start_date="2024-01-02", end_date="2024-02-12",
        )
        engine = Alpha191Indicators.from_dataset(dataset)
        result = engine.compute_factor(14)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == n_stocks
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Engine -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现 Alpha191Engine + Alpha191Indicators**

```python
# stockquant/indicators/alpha191/alpha191.py
"""
国泰君安 Alpha191 因子指标 — 基于短周期价量特征的多因子选股体系。

基于国泰君安 (2017) 《基于短周期价量特征的多因子选股体系》研究报告，
实现 191 个短周期交易型 Alpha 因子。

推荐用法（Dataset 模式）
------------------------
>>> from stockquant.indicators import Alpha191Indicators
>>> engine = Alpha191Indicators.from_dataset(dataset)
>>> alpha001 = engine.alpha001()
>>> all_factors = engine.compute_all()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stockquant.data.universe import BacktestDataset

from stockquant.indicators.base import BaseIndicator, IndicatorRegistry
from stockquant.indicators.alpha101.operators import (
    adv,
    decay_linear,
    delay,
    delta,
    log,
    rank,
    scale,
    sign,
    signedpower,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_max,
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
)
from stockquant.indicators.alpha191.operators import (
    count,
    ema_sma,
    filter_cond,
    highday,
    lowday,
    regbeta,
    sequence,
    sumac,
    sumif,
    wma,
)
from stockquant.utils.logger import get_logger

logger = get_logger("indicators.alpha191")

# 需要基准指数数据的 Alpha 编号
BENCHMARK_ALPHAS: set[int] = {75, 149, 150, 181, 182, 190}

# 需要特殊数据（三因子模型等）的 Alpha 编号 — MVP 阶段跳过
SKIP_ALPHAS: set[int] = {30}

TOTAL_ALPHAS = 191


def _sma(x, d):
    """简单移动平均 — 复用 alpha101 的 sma 算子。"""
    from stockquant.indicators.alpha101.operators import sma
    return sma(x, d)


def _std(x, d):
    """滚动标准差 — 复用 alpha101 的 ts_stddev。"""
    return ts_stddev(x, d)


class Alpha191Indicators(BaseIndicator):
    """国泰君安 Alpha191 因子指标。

    继承 :class:`BaseIndicator`，通过 ``compute(df)`` 方法可对单只股票
    OHLCV DataFrame 计算 Alpha 因子并附加为新列。

    同时通过 ``panel()`` / ``from_dataset()`` 工厂方法支持多股票面板计算。
    """

    def __init__(self, alphas: Sequence[int] | None = None) -> None:
        self._alphas = list(alphas) if alphas else list(range(1, TOTAL_ALPHAS + 1))

    @property
    def name(self) -> str:
        return "Alpha191"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        alphas = kwargs.get("alphas", self._alphas)
        engine = self._build_engine_from_single(df)

        for alpha_id in alphas:
            if alpha_id in SKIP_ALPHAS:
                continue
            method = getattr(engine, f"alpha{alpha_id:03d}", None)
            if method is None:
                continue
            try:
                result = method()
                col_name = f"alpha{alpha_id:03d}"
                if isinstance(result, pd.DataFrame):
                    df[col_name] = result.iloc[:, 0].values
                else:
                    df[col_name] = result.values
            except Exception as e:
                logger.warning(f"Alpha191#{alpha_id} 计算失败: {e}")
                df[f"alpha{alpha_id:03d}"] = np.nan
        return df

    @classmethod
    def panel(
        cls,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame | None = None,
        amount: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        benchmark_close: pd.Series | None = None,
        benchmark_open: pd.Series | None = None,
    ) -> "Alpha191Engine":
        return Alpha191Engine(
            open_=open_, high=high, low=low, close=close,
            volume=volume, vwap=vwap, amount=amount, returns=returns,
            benchmark_close=benchmark_close,
            benchmark_open=benchmark_open,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: "BacktestDataset",
    ) -> "Alpha191Engine":
        from stockquant.indicators.alpha101.alpha101 import Alpha101Indicators
        stacked = Alpha101Indicators._stack_dataset(dataset)
        stacked = stacked.copy()
        if not isinstance(stacked.index, pd.MultiIndex):
            stacked = stacked.set_index(["date", "code"])

        def _pivot(col: str) -> pd.DataFrame | None:
            if col in stacked.columns:
                return stacked[col].unstack()
            return None

        bm_close = None
        bm_open = None
        if dataset.benchmark is not None and not dataset.benchmark.empty:
            bm = dataset.benchmark.copy()
            bm["date"] = pd.to_datetime(bm["date"])
            bm = bm.set_index("date").sort_index()
            if "close" in bm.columns:
                bm_close = bm["close"]
            if "open" in bm.columns:
                bm_open = bm["open"]

        return Alpha191Engine(
            open_=_pivot("open"),
            high=_pivot("high"),
            low=_pivot("low"),
            close=_pivot("close"),
            volume=_pivot("volume"),
            amount=_pivot("amount"),
            benchmark_close=bm_close,
            benchmark_open=bm_open,
        )

    @staticmethod
    def _build_engine_from_single(df: pd.DataFrame) -> "Alpha191Engine":
        code = df.get("code", pd.Series(["_stock"])).iloc[0] if "code" in df.columns else "_stock"

        def _col(col_name: str) -> pd.DataFrame | None:
            if col_name in df.columns:
                return df[[col_name]].rename(columns={col_name: code})
            return None

        return Alpha191Engine(
            open_=df[["open"]].rename(columns={"open": code}),
            high=df[["high"]].rename(columns={"high": code}),
            low=df[["low"]].rename(columns={"low": code}),
            close=df[["close"]].rename(columns={"close": code}),
            volume=df[["volume"]].rename(columns={"volume": code}),
            amount=_col("amount"),
        )


class Alpha191Engine:
    """Alpha191 面板计算引擎。

    承载全部 191 个因子的具体计算逻辑。面板数据格式：行=日期，列=股票代码。

    Parameters
    ----------
    open_, high, low, close, volume : DataFrame
        OHLCV 面板数据。
    vwap : DataFrame, optional
        VWAP 面板。缺失时用 amount / (volume * 100) 或 (high+low+close)/3。
    amount : DataFrame, optional
        成交额面板。
    returns : DataFrame, optional
        收益率面板。缺失时用 close.pct_change()。
    benchmark_close, benchmark_open : Series, optional
        基准指数收盘/开盘价序列（index=日期）。约6个因子需要。
    """

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame | None = None,
        amount: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        benchmark_close: pd.Series | None = None,
        benchmark_open: pd.Series | None = None,
    ) -> None:
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.amount = amount if amount is not None else close * volume * 100

        if vwap is not None:
            self.vwap = vwap
        elif amount is not None:
            self.vwap = amount / (volume * 100 + 1e-10)
        else:
            self.vwap = (high + low + close) / 3

        self.returns = returns if returns is not None else close.pct_change()
        self.benchmark_close = benchmark_close
        self.benchmark_open = benchmark_open
        self._cache: dict[str, Any] = {}

    # ==================================================================
    # 内部工具
    # ==================================================================

    @staticmethod
    def _clean(result: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return result.replace([np.inf, -np.inf], np.nan)

    def _where(self, condition, x, y) -> pd.DataFrame:
        return pd.DataFrame(
            np.where(condition, x, y),
            index=self.close.index,
            columns=self.close.columns,
        )

    def _adv(self, d: int) -> pd.DataFrame:
        key = f"adv{d}"
        if key not in self._cache:
            self._cache[key] = adv(self.volume, d)
        return self._cache[key]

    def _bm_close_panel(self) -> pd.DataFrame:
        """将基准指数 close Series 扩展为与股票面板同结构的 DataFrame。"""
        if self.benchmark_close is None:
            return self.close * 0
        return pd.DataFrame(
            np.tile(self.benchmark_close.reindex(self.close.index).values[:, None], (1, len(self.close.columns))),
            index=self.close.index,
            columns=self.close.columns,
        )

    def _bm_open_panel(self) -> pd.DataFrame:
        if self.benchmark_open is None:
            return self.open * 0
        return pd.DataFrame(
            np.tile(self.benchmark_open.reindex(self.close.index).values[:, None], (1, len(self.close.columns))),
            index=self.close.index,
            columns=self.close.columns,
        )

    # ==================================================================
    # 批量计算
    # ==================================================================

    def compute_factor(self, alpha_id: int) -> pd.DataFrame:
        method = getattr(self, f"alpha{alpha_id:03d}", None)
        if method is None:
            raise ValueError(f"Alpha191#{alpha_id} 未实现")
        try:
            return self._clean(method())
        except Exception as e:
            logger.warning(f"Alpha191#{alpha_id} 计算失败: {e}")
            return pd.DataFrame(
                index=self.close.index, columns=self.close.columns, dtype=float
            )

    def compute_factors(self, alpha_ids: Sequence[int]) -> dict[int, pd.DataFrame]:
        results: dict[int, pd.DataFrame] = {}
        for i in alpha_ids:
            if i in SKIP_ALPHAS:
                continue
            results[i] = self.compute_factor(i)
        logger.info(f"计算 {len(results)} 个指定 Alpha191 因子")
        return results

    def compute_all(self) -> dict[int, pd.DataFrame]:
        results: dict[int, pd.DataFrame] = {}
        for i in range(1, TOTAL_ALPHAS + 1):
            if i in SKIP_ALPHAS:
                continue
            method = getattr(self, f"alpha{i:03d}", None)
            if method is None:
                continue
            try:
                results[i] = self._clean(method())
            except Exception as e:
                logger.warning(f"Alpha191#{i} 计算失败: {e}")
        logger.info(f"成功计算 {len(results)}/{TOTAL_ALPHAS} 个 Alpha191 因子")
        return results

    # ==================================================================
    # Alpha #001 – #010 (第一批因子，在 Task 3 中实现)
    # ==================================================================
```

- [ ] **Step 4: 更新 `__init__.py`**

```python
# stockquant/indicators/alpha191/__init__.py
"""
国泰君安 Alpha191 因子指标包。

主要入口
--------
- :class:`Alpha191Indicators` — 继承 BaseIndicator，标准 compute(df) 接口
- :class:`Alpha191Engine` — 面板计算引擎（多股票截面计算）
"""

from stockquant.indicators.alpha191.alpha191 import (
    Alpha191Engine,
    Alpha191Indicators,
    BENCHMARK_ALPHAS,
    SKIP_ALPHAS,
    TOTAL_ALPHAS,
)

__all__ = [
    "Alpha191Indicators",
    "Alpha191Engine",
    "BENCHMARK_ALPHAS",
    "SKIP_ALPHAS",
    "TOTAL_ALPHAS",
]
```

- [ ] **Step 5: 运行测试**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py -v`
Expected: TestAlpha191Operators (8 PASSED), TestAlpha191Engine (3 tests — panel_shapes PASS, invalid_raises PASS, compute_factor FAIL because alpha002 not yet implemented)

注意：`compute_factor(2)` 会在 Task 3 实现因子后才通过。此处 `test_compute_factor_returns_dataframe` 将暂时失败（ValueError: Alpha191#2 未实现），这是预期的 — 在 Task 3 实现因子后此测试会通过。

先临时调整测试为 alpha_id=14（在 Task 3 最先实现），或在 Step 2 中将 `test_compute_factor_returns_dataframe` 使用的 alpha_id 改为任意一个在 Task 3 开头就实现的编号。

- [ ] **Step 6: Commit**

```bash
git add stockquant/indicators/alpha191/alpha191.py stockquant/indicators/alpha191/__init__.py tests/test_alpha191.py
git commit -m "feat: add Alpha191Engine skeleton and Alpha191Indicators wrapper"
```

---

### Task 3: 实现 Alpha191 因子 #001 – #050

**Files:**
- Modify: `stockquant/indicators/alpha191/alpha191.py`
- Modify: `tests/test_alpha191.py`

以下展示代表性因子的实现模式。每个 alpha 方法是 `Alpha191Engine` 上的实例方法，返回 `pd.DataFrame`。

- [ ] **Step 1: 追加因子测试**

在 `tests/test_alpha191.py` 中追加：

```python
class TestAlpha191Factors001to050:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [1, 2, 5, 7, 10, 13, 14, 15, 20, 25, 33, 38, 41, 42, 46, 48, 50])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha014_is_5day_return(self, engine):
        """Alpha14 = CLOSE - DELAY(CLOSE, 5)，验证公式正确性。"""
        result = engine.alpha014()
        expected = engine.close - delay(engine.close, 5)
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_alpha015_is_open_close_ratio(self, engine):
        """Alpha15 = OPEN/DELAY(CLOSE,1) - 1"""
        result = engine.alpha015()
        expected = engine.open / delay(engine.close, 1) - 1
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_alpha013_formula(self, engine):
        """Alpha13 = (HIGH*LOW)^0.5 - VWAP"""
        result = engine.alpha013()
        expected = (engine.high * engine.low) ** 0.5 - engine.vwap
        pd.testing.assert_frame_equal(result, expected, check_names=False)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors001to050 -v`
Expected: FAIL — alpha methods not defined

- [ ] **Step 3: 实现 Alpha #001 – #050**

在 `Alpha191Engine` 类中追加以下方法。展示关键的代表性因子：

```python
    # ==================================================================
    # Alpha #001 – #010
    # ==================================================================

    def alpha001(self) -> pd.DataFrame:
        """(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))"""
        return -1 * ts_corr(
            rank(delta(log(self.volume), 1)),
            rank((self.close - self.open) / (self.open + 1e-10)),
            6,
        )

    def alpha002(self) -> pd.DataFrame:
        """(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))"""
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10)
        return -1 * delta(inner, 1)

    def alpha003(self) -> pd.DataFrame:
        """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)"""
        d1 = delay(self.close, 1)
        cond_eq = self.close == d1
        cond_gt = self.close > d1
        inner_gt = self.close - pd.DataFrame(
            np.minimum(self.low.values, d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner_lt = self.close - pd.DataFrame(
            np.maximum(self.high.values, d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner = self._where(cond_eq, 0.0, self._where(cond_gt, inner_gt, inner_lt))
        return ts_sum(inner, 6)

    def alpha004(self) -> pd.DataFrame:
        """均线/波动率条件 + 量比条件复合因子 (同 Alpha101#21 结构)"""
        sma8 = _sma(self.close, 8)
        std8 = _std(self.close, 8)
        sma2 = _sma(self.close, 2)
        vol_ratio = self.volume / (_sma(self.volume, 20) + 1e-10)
        cond1 = (sma8 + std8) < sma2
        cond2 = sma2 < (sma8 - std8)
        cond3 = vol_ratio >= 1
        return self._where(
            cond1, -1.0, self._where(cond2, 1.0, self._where(cond3, 1.0, -1.0))
        )

    def alpha005(self) -> pd.DataFrame:
        """(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))"""
        return -1 * ts_max(
            ts_corr(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3
        )

    def alpha006(self) -> pd.DataFrame:
        """(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4))) * -1)"""
        return rank(sign(delta(self.open * 0.85 + self.high * 0.15, 4))) * -1

    def alpha007(self) -> pd.DataFrame:
        """((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))"""
        diff = self.vwap - self.close
        return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(self.volume, 3))

    def alpha008(self) -> pd.DataFrame:
        """RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)"""
        return rank(delta((self.high + self.low) / 2 * 0.2 + self.vwap * 0.8, 4) * -1)

    def alpha009(self) -> pd.DataFrame:
        """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)"""
        inner = (
            ((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2)
            * (self.high - self.low) / (self.volume + 1e-10)
        )
        return ema_sma(inner, n=7, m=2)

    def alpha010(self) -> pd.DataFrame:
        """(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2), 5))"""
        inner = self._where(self.returns < 0, _std(self.returns, 20), self.close)
        return rank(ts_max(inner ** 2, 5))

    # ==================================================================
    # Alpha #011 – #020
    # ==================================================================

    def alpha011(self) -> pd.DataFrame:
        """SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)"""
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10) * self.volume
        return ts_sum(inner, 6)

    def alpha012(self) -> pd.DataFrame:
        """(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))"""
        return rank(self.open - ts_sum(self.vwap, 10) / 10) * (-1 * rank((self.close - self.vwap).abs()))

    def alpha013(self) -> pd.DataFrame:
        """(((HIGH * LOW)^0.5) - VWAP)"""
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha014(self) -> pd.DataFrame:
        """CLOSE - DELAY(CLOSE, 5)"""
        return self.close - delay(self.close, 5)

    def alpha015(self) -> pd.DataFrame:
        """OPEN / DELAY(CLOSE, 1) - 1"""
        return self.open / (delay(self.close, 1) + 1e-10) - 1

    def alpha016(self) -> pd.DataFrame:
        """(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))"""
        return -1 * ts_max(rank(ts_corr(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha017(self) -> pd.DataFrame:
        """RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)"""
        return signedpower(rank(self.vwap - ts_max(self.vwap, 15)), delta(self.close, 5))

    def alpha018(self) -> pd.DataFrame:
        """CLOSE / DELAY(CLOSE, 5)"""
        return self.close / (delay(self.close, 5) + 1e-10)

    def alpha019(self) -> pd.DataFrame:
        """条件收益率因子"""
        d5 = delay(self.close, 5)
        cond_lt = self.close < d5
        cond_eq = self.close == d5
        ret_neg = (self.close - d5) / (d5 + 1e-10)
        ret_pos = (self.close - d5) / (self.close + 1e-10)
        return self._where(cond_lt, ret_neg, self._where(cond_eq, 0.0, ret_pos))

    def alpha020(self) -> pd.DataFrame:
        """(CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100"""
        d6 = delay(self.close, 6)
        return (self.close - d6) / (d6 + 1e-10) * 100

    # ==================================================================
    # Alpha #021 – #030
    # ==================================================================

    def alpha021(self) -> pd.DataFrame:
        """REGBETA(MEAN(CLOSE,6), SEQUENCE(6))"""
        mean6 = _sma(self.close, 6)
        seq = sequence(6)
        if isinstance(mean6, pd.DataFrame):
            result = pd.DataFrame(index=mean6.index, columns=mean6.columns, dtype=float)
            for col in mean6.columns:
                vals = mean6[col].values
                out = np.full(len(vals), np.nan)
                for i in range(5, len(vals)):
                    y = vals[i - 5: i + 1]
                    if np.any(np.isnan(y)):
                        continue
                    x_dm = seq - seq.mean()
                    denom = np.dot(x_dm, x_dm)
                    if denom == 0:
                        continue
                    out[i] = float(np.dot(x_dm, y - y.mean()) / denom)
                result[col] = out
            return result
        return regbeta(mean6, pd.Series(seq), 6)

    def alpha022(self) -> pd.DataFrame:
        """SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)"""
        mean6 = _sma(self.close, 6)
        inner = (self.close - mean6) / (mean6 + 1e-10)
        return ema_sma(inner - delay(inner, 3), n=12, m=1)

    def alpha023(self) -> pd.DataFrame:
        """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (...) * 100"""
        d1 = delay(self.close, 1)
        cond = self.close > d1
        std20 = _std(self.close, 20)
        upper = ema_sma(self._where(cond, std20, 0.0), n=20, m=1)
        lower_pos = ema_sma(self._where(cond, std20, 0.0), n=20, m=1)
        lower_neg = ema_sma(self._where(~cond, std20, 0.0), n=20, m=1)
        return upper / (lower_pos + lower_neg + 1e-10) * 100

    def alpha024(self) -> pd.DataFrame:
        """SMA(CLOSE-DELAY(CLOSE,5),5,1)"""
        return ema_sma(self.close - delay(self.close, 5), n=5, m=1)

    def alpha025(self) -> pd.DataFrame:
        """((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))"""
        return (
            -1 * rank(
                delta(self.close, 7)
                * (1 - rank(decay_linear(self.volume / (_sma(self.volume, 20) + 1e-10), 9)))
            )
            * (1 + rank(ts_sum(self.returns, 250)))
        )

    def alpha026(self) -> pd.DataFrame:
        """((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))"""
        return (ts_sum(self.close, 7) / 7 - self.close) + ts_corr(self.vwap, delay(self.close, 5), 230)

    def alpha027(self) -> pd.DataFrame:
        """WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)"""
        d3 = delay(self.close, 3)
        d6 = delay(self.close, 6)
        inner = (self.close - d3) / (d3 + 1e-10) * 100 + (self.close - d6) / (d6 + 1e-10) * 100
        return wma(inner, 12)

    def alpha028(self) -> pd.DataFrame:
        """3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA(...,3,1),3,1)"""
        inner = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9) + 1e-10) * 100
        sma1 = ema_sma(inner, n=3, m=1)
        return 3 * sma1 - 2 * ema_sma(sma1, n=3, m=1)

    def alpha029(self) -> pd.DataFrame:
        """(CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME"""
        d6 = delay(self.close, 6)
        return (self.close - d6) / (d6 + 1e-10) * self.volume

    # Alpha030 需要三因子模型 (MKT/SMB/HML)，MVP 阶段跳过

    # ==================================================================
    # Alpha #031 – #040
    # ==================================================================

    def alpha031(self) -> pd.DataFrame:
        """(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100"""
        mean12 = _sma(self.close, 12)
        return (self.close - mean12) / (mean12 + 1e-10) * 100

    def alpha032(self) -> pd.DataFrame:
        """(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))"""
        return -1 * ts_sum(rank(ts_corr(rank(self.high), rank(self.volume), 3)), 3)

    def alpha033(self) -> pd.DataFrame:
        """((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))"""
        part1 = -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)
        part2 = rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        return part1 * part2 * ts_rank(self.volume, 5)

    def alpha034(self) -> pd.DataFrame:
        """MEAN(CLOSE,12)/CLOSE"""
        return _sma(self.close, 12) / (self.close + 1e-10)

    def alpha035(self) -> pd.DataFrame:
        """(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)"""
        part1 = rank(decay_linear(delta(self.open, 1), 15))
        part2 = rank(decay_linear(ts_corr(self.volume, self.open, 17), 7))
        return pd.DataFrame(
            np.minimum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        ) * -1

    def alpha036(self) -> pd.DataFrame:
        """RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)"""
        return rank(ts_sum(ts_corr(rank(self.volume), rank(self.vwap), 6), 2))

    def alpha037(self) -> pd.DataFrame:
        """(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))"""
        x = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -1 * rank(x - delay(x, 10))

    def alpha038(self) -> pd.DataFrame:
        """(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)"""
        cond = ts_sum(self.high, 20) / 20 < self.high
        return self._where(cond, -1 * delta(self.high, 2), 0.0)

    def alpha039(self) -> pd.DataFrame:
        """((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)"""
        part1 = rank(decay_linear(delta(self.close, 2), 8))
        inner = self.vwap * 0.3 + self.open * 0.7
        part2 = rank(decay_linear(ts_corr(inner, ts_sum(_sma(self.volume, 180), 37), 14), 12))
        return (part1 - part2) * -1

    def alpha040(self) -> pd.DataFrame:
        """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100"""
        d1 = delay(self.close, 1)
        cond = self.close > d1
        upper = ts_sum(self._where(cond, self.volume, 0.0), 26)
        lower = ts_sum(self._where(~cond, self.volume, 0.0), 26)
        return upper / (lower + 1e-10) * 100

    # ==================================================================
    # Alpha #041 – #050
    # ==================================================================

    def alpha041(self) -> pd.DataFrame:
        """(RANK(MAX(DELTA((VWAP), 3), 5))* -1)"""
        return rank(ts_max(delta(self.vwap, 3), 5)) * -1

    def alpha042(self) -> pd.DataFrame:
        """((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))"""
        return -1 * rank(_std(self.high, 10)) * ts_corr(self.high, self.volume, 10)

    def alpha043(self) -> pd.DataFrame:
        """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)"""
        d1 = delay(self.close, 1)
        cond_gt = self.close > d1
        cond_lt = self.close < d1
        inner = self._where(cond_gt, self.volume, self._where(cond_lt, -self.volume, 0.0))
        return ts_sum(inner, 6)

    def alpha044(self) -> pd.DataFrame:
        """(TSRANK(DECAYLINEAR(CORR(((LOW)), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))"""
        part1 = ts_rank(decay_linear(ts_corr(self.low, _sma(self.volume, 10), 7), 6), 4)
        part2 = ts_rank(decay_linear(delta(self.vwap, 3), 10), 15)
        return part1 + part2

    def alpha045(self) -> pd.DataFrame:
        """(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))"""
        return rank(delta(self.close * 0.6 + self.open * 0.4, 1)) * rank(
            ts_corr(self.vwap, _sma(self.volume, 150), 15)
        )

    def alpha046(self) -> pd.DataFrame:
        """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)"""
        return (_sma(self.close, 3) + _sma(self.close, 6) + _sma(self.close, 12) + _sma(self.close, 24)) / (4 * self.close + 1e-10)

    def alpha047(self) -> pd.DataFrame:
        """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)"""
        hi6 = ts_max(self.high, 6)
        lo6 = ts_min(self.low, 6)
        inner = (hi6 - self.close) / (hi6 - lo6 + 1e-10) * 100
        return ema_sma(inner, n=9, m=1)

    def alpha048(self) -> pd.DataFrame:
        """(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))"""
        d1 = sign(self.close - delay(self.close, 1))
        d2 = sign(delay(self.close, 1) - delay(self.close, 2))
        d3 = sign(delay(self.close, 2) - delay(self.close, 3))
        return -1 * rank(d1 + d2 + d3) * ts_sum(self.volume, 5) / (ts_sum(self.volume, 20) + 1e-10)

    def alpha049(self) -> pd.DataFrame:
        """方向运动指标 — DI 型因子"""
        hl_sum = self.high + self.low
        hl_lag = delay(self.high, 1) + delay(self.low, 1)
        cond = hl_sum >= hl_lag
        abs_h = (self.high - delay(self.high, 1)).abs()
        abs_l = (self.low - delay(self.low, 1)).abs()
        max_hl = pd.DataFrame(
            np.maximum(abs_h.values, abs_l.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner = self._where(cond, 0.0, max_hl)
        inner_neg = self._where(~cond, 0.0, max_hl)
        s1 = ts_sum(inner, 12)
        s2 = ts_sum(inner_neg, 12)
        return s1 / (s1 + s2 + 1e-10)

    def alpha050(self) -> pd.DataFrame:
        """方向运动指标 — DI 差值因子"""
        hl_sum = self.high + self.low
        hl_lag = delay(self.high, 1) + delay(self.low, 1)
        cond_ge = hl_sum >= hl_lag
        cond_le = hl_sum <= hl_lag
        abs_h = (self.high - delay(self.high, 1)).abs()
        abs_l = (self.low - delay(self.low, 1)).abs()
        max_hl = pd.DataFrame(
            np.maximum(abs_h.values, abs_l.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner_le = self._where(cond_le, 0.0, max_hl)
        inner_ge = self._where(cond_ge, 0.0, max_hl)
        s_le = ts_sum(inner_le, 12)
        s_ge = ts_sum(inner_ge, 12)
        return s_le / (s_le + s_ge + 1e-10) - s_ge / (s_ge + s_le + 1e-10)
```

- [ ] **Step 4: 运行测试**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors001to050 -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add stockquant/indicators/alpha191/alpha191.py tests/test_alpha191.py
git commit -m "feat: implement Alpha191 factors #001-#050"
```

---

### Task 4: 实现 Alpha191 因子 #051 – #100

**Files:**
- Modify: `stockquant/indicators/alpha191/alpha191.py`
- Modify: `tests/test_alpha191.py`

- [ ] **Step 1: 追加测试**

```python
class TestAlpha191Factors051to100:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [51, 53, 54, 60, 62, 65, 66, 70, 71, 80, 85, 86, 88, 90, 99, 100])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha065_is_mean_ratio(self, engine):
        """Alpha65 = MEAN(CLOSE,6) / CLOSE"""
        result = engine.alpha065()
        expected = _sma(engine.close, 6) / (engine.close + 1e-10)
        # 比较非 NaN 部分
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result.values[mask.values], expected.values[mask.values], rtol=1e-5)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors051to100 -v`
Expected: FAIL

- [ ] **Step 3: 实现 Alpha #051 – #100**

以下是 #051 – #100 各因子实现，直接对照 PDF 公式翻译。每个方法体结构一致：读取 self.close/open/high/low/volume/vwap + 算子组合 → 返回 DataFrame。

代表性因子列表（全部需实现）：

| 编号 | 公式摘要 | 使用算子 |
|------|----------|----------|
| 51 | DI 型（同50类似，只取正向部分） | ts_sum, _where |
| 52 | SUM(MAX(0,H-DELAY(TP,1)),26)/SUM(MAX(0,DELAY(TP,1)-L),26)*100 | ts_sum, delay |
| 53 | COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100 | count |
| 54 | (-1 * RANK((STD\|C-O\| + (C-O) + CORR(C,O,10)))) | rank, _std, ts_corr |
| 55-58 | 复杂条件因子（Alpha55-58按PDF实现） | ema_sma, delay, sign |
| 59 | SUM(条件价格差, 20) | ts_sum, _where |
| 60 | SUM(CLV*VOLUME, 20) | ts_sum |
| 61-64 | RANK + DECAYLINEAR + CORR 组合 | rank, decay_linear, ts_corr |
| 65 | MEAN(CLOSE,6)/CLOSE | _sma |
| 66 | (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100 | _sma |
| 67 | SMA(MAX(C-D(C,1),0),24,1)/SMA(ABS(C-D(C,1)),24,1)*100 | ema_sma |
| 68 | SMA(TP_change*range/vol, 15, 2) | ema_sma |
| 69 | DTM/DBM 条件因子 | _where, ts_sum |
| 70 | STD(AMOUNT, 6) | _std |
| 71-80 | 价格均线偏离/SMA类/量变化类 | _sma, ema_sma, _std |
| 81 | SMA(VOLUME, 21, 2) | ema_sma |
| 82-84 | SMA + TSMAX/TSMIN 组合 | ema_sma, ts_max, ts_min |
| 85 | TSRANK(VOL/ADV20, 20) * TSRANK(-DELTA(C,7), 8) | ts_rank, delta |
| 86 | 加速度条件因子 | _where, delay |
| 87 | RANK(DECAYLINEAR) + TSRANK(DECAYLINEAR) 组合 | rank, ts_rank, decay_linear |
| 88 | (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100 | delay |
| 89 | 2*(SMA(C,13,2)-SMA(C,27,2)-SMA(前两者差,10,2)) | ema_sma |
| 90 | RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1 | rank, ts_corr |
| 91-100 | 类似结构：RANK^CORR、DECAYLINEAR、STD | 各种组合 |

每个方法按照与 Task 3 相同的模式实现：直接翻译 PDF 公式。

- [ ] **Step 4: 运行测试**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors051to100 -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add stockquant/indicators/alpha191/alpha191.py tests/test_alpha191.py
git commit -m "feat: implement Alpha191 factors #051-#100"
```

---

### Task 5: 实现 Alpha191 因子 #101 – #150

**Files:**
- Modify: `stockquant/indicators/alpha191/alpha191.py`
- Modify: `tests/test_alpha191.py`

- [ ] **Step 1: 追加测试**

```python
class TestAlpha191Factors101to150:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [101, 102, 105, 106, 107, 108, 113, 114, 117, 120, 126, 132, 134, 136, 140, 141, 142, 148])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha106_is_20day_return(self, engine):
        """Alpha106 = CLOSE - DELAY(CLOSE, 20)"""
        result = engine.alpha106()
        expected = engine.close - delay(engine.close, 20)
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_alpha126_is_typical_price(self, engine):
        """Alpha126 = (CLOSE+HIGH+LOW)/3"""
        result = engine.alpha126()
        expected = (engine.close + engine.high + engine.low) / 3
        pd.testing.assert_frame_equal(result, expected, check_names=False)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors101to150 -v`
Expected: FAIL

- [ ] **Step 3: 实现 Alpha #101 – #150**

同 Task 3/4 模式，对照 PDF 公式逐一翻译。关键因子：

| 编号 | 公式摘要 | 注意 |
|------|----------|------|
| 101 | RANK(CORR) 比较型 | 复用 Alpha101 同编号模式 |
| 102 | SMA(MAX(VOL变化,0),6,1)/SMA(ABS(VOL变化),6,1)*100 | RSI-like |
| 103 | ((20-LOWDAY(LOW,20))/20)*100 | 需要 lowday 算子 |
| 104-105 | CORR + RANK 组合 | 标准模式 |
| 106 | CLOSE-DELAY(CLOSE,20) | 简单差分 |
| 107 | 三因子开盘价偏差 | delay |
| 108 | RANK^CORR | signedpower |
| 109-112 | SMA 嵌套因子 | ema_sma |
| 113-115 | RANK * CORR 组合 | 标准模式 |
| 116 | REGBETA(CLOSE, SEQUENCE, 20) | regbeta + sequence |
| 117 | TSRANK * TSRANK * TSRANK | ts_rank |
| 118-122 | SMA / RANK / VWAP 组合 | 各种 |
| 126 | (CLOSE+HIGH+LOW)/3 | 简单 |
| 127-132 | 统计量/均值/STD | 各种 |
| 133 | HIGHDAY + LOWDAY | 新算子 |
| 134-142 | 价量偏离/CORR/RANK | 标准模式 |
| 143 | 递归 SELF 因子 — 用迭代实现 | 特殊处理 |
| 144 | SUMIF + COUNT | 新算子组合 |
| 145-148 | SMA/MEAN/RANK 组合 | 标准模式 |

- [ ] **Step 4: 运行测试**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors101to150 -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add stockquant/indicators/alpha191/alpha191.py tests/test_alpha191.py
git commit -m "feat: implement Alpha191 factors #101-#150"
```

---

### Task 6: 实现 Alpha191 因子 #151 – #191

**Files:**
- Modify: `stockquant/indicators/alpha191/alpha191.py`
- Modify: `tests/test_alpha191.py`

- [ ] **Step 1: 追加测试**

```python
class TestAlpha191Factors151to191:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [153, 155, 158, 160, 161, 163, 167, 168, 170, 171, 176, 184, 185, 188, 189, 191])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha191_formula(self, engine):
        """Alpha191 = ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)"""
        result = engine.alpha191()
        expected = ts_corr(_sma(engine.volume, 20), engine.low, 5) + (engine.high + engine.low) / 2 - engine.close
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result.values[mask.values], expected.values[mask.values], rtol=1e-5)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors151to191 -v`
Expected: FAIL

- [ ] **Step 3: 实现 Alpha #151 – #191**

关键因子：

| 编号 | 公式摘要 | 注意 |
|------|----------|------|
| 152-155 | SMA 嵌套 | ema_sma |
| 156-157 | RANK + DECAYLINEAR + PROD | rank, decay_linear, ts_product |
| 158 | (H-SMA(C,15,2))-(L-SMA(C,15,2))/C | ema_sma |
| 159 | 复杂多周期 CCI-like | ts_sum, delay |
| 160-162 | SMA 条件因子 | ema_sma, _where |
| 163 | RANK((-RET * ADV20 * VWAP * (H-C))) | rank |
| 164-166 | SMA 条件 / 偏度 | ema_sma |
| 167-169 | SUM 条件量 / SMA 差值 | ts_sum, ema_sma |
| 170-171 | 量价排名 / 价格比率 | rank |
| 172, 186 | ADX — 需要 LD/HD/TR 中间变量 | 特殊实现 |
| 173-174 | 三重 SMA 平滑 | ema_sma |
| 175-177 | CORR / MEAN / ABS | 标准模式 |
| 181-182 | 需要基准指数 | _bm_close_panel |
| 183-185 | SUMAC / CORR / RANK | sumac |
| 187-189 | SUM 条件 / SMA / MEAN | 标准模式 |
| 190 | LOG + SUMIF — 需要基准 | 复杂，可选跳过 |
| 191 | CORR(MEAN(VOL,20), LOW, 5) + (H+L)/2 - C | 简单 |

- [ ] **Step 4: 运行测试**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Factors151to191 -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add stockquant/indicators/alpha191/alpha191.py tests/test_alpha191.py
git commit -m "feat: implement Alpha191 factors #151-#191"
```

---

### Task 7: 注册 + 集成 + 全量回归测试

**Files:**
- Modify: `stockquant/indicators/__init__.py`
- Modify: `stockquant/indicators/alpha191/alpha191.py` (底部加注册)
- Modify: `tests/test_alpha191.py`

- [ ] **Step 1: 追加集成测试**

```python
# ======================================================================
# 注册与集成测试
# ======================================================================

class TestAlpha191Registry:

    def test_registered_in_indicator_registry(self):
        from stockquant.indicators import IndicatorRegistry
        assert "alpha191" in IndicatorRegistry.list()

    def test_create_via_registry(self):
        from stockquant.indicators import IndicatorRegistry
        ind = IndicatorRegistry.create("alpha191")
        assert ind.name == "Alpha191"

    def test_importable_from_indicators(self):
        from stockquant.indicators import Alpha191Indicators
        assert Alpha191Indicators is not None


class TestAlpha191ComputeAll:

    def test_compute_all_returns_dict(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        engine = Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )
        results = engine.compute_all()
        assert isinstance(results, dict)
        assert len(results) >= 180  # 至少 180 个因子（排除 SKIP 和未实现的）

    def test_compute_factors_subset(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        engine = Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )
        results = engine.compute_factors([1, 14, 65, 191])
        assert len(results) == 4
        assert all(isinstance(v, pd.DataFrame) for v in results.values())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py::TestAlpha191Registry -v`
Expected: FAIL — not registered

- [ ] **Step 3: 注册 Alpha191Indicators**

在 `stockquant/indicators/alpha191/alpha191.py` 文件末尾追加：

```python
IndicatorRegistry.register("alpha191", Alpha191Indicators)
```

- [ ] **Step 4: 更新 `indicators/__init__.py`**

```python
# stockquant/indicators/__init__.py
"""技术指标计算模块"""

from .base import BaseIndicator, IndicatorRegistry
from .trend import TrendIndicators
from .oscillator import OscillatorIndicators
from .volume import VolumeIndicators
from .volatility import VolatilityIndicators
from .alpha101 import Alpha101Indicators
from .alpha191 import Alpha191Indicators

__all__ = [
    "BaseIndicator",
    "IndicatorRegistry",
    "TrendIndicators",
    "OscillatorIndicators",
    "VolumeIndicators",
    "VolatilityIndicators",
    "Alpha101Indicators",
    "Alpha191Indicators",
]
```

- [ ] **Step 5: 运行 Alpha191 全部测试**

Run: `.venv/bin/python -m pytest tests/test_alpha191.py -v`
Expected: ALL PASSED

- [ ] **Step 6: 运行全量回归测试**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_hs300_daily.py`
Expected: 全部通过（173 原有 + ~60 新增 ≈ 233 tests）

- [ ] **Step 7: Commit**

```bash
git add stockquant/indicators/__init__.py stockquant/indicators/alpha191/alpha191.py tests/test_alpha191.py
git commit -m "feat: register Alpha191 + indicators integration + full regression pass"
```

---

## 验证

```bash
# Alpha191 专项测试
.venv/bin/python -m pytest tests/test_alpha191.py -v

# 全量回归测试
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_hs300_daily.py

# 快速验收：计算全部因子
.venv/bin/python -c "
from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n, m = 120, 10
dates = pd.bdate_range('2024-01-02', periods=n)
codes = [f'{i:06d}' for i in range(1, m+1)]
close = pd.DataFrame(10 + np.cumsum(rng.standard_normal((n, m))*0.2, axis=0), index=dates, columns=codes).clip(lower=1)
engine = Alpha191Engine(
    open_=close*0.99, high=close*1.02, low=close*0.98,
    close=close, volume=pd.DataFrame(rng.integers(1e5, 1e6, (n,m)), index=dates, columns=codes, dtype=float),
)
results = engine.compute_all()
print(f'成功计算 {len(results)} 个 Alpha191 因子')
for k, v in sorted(results.items())[:5]:
    print(f'  Alpha{k:03d}: shape={v.shape}, 有效率={v.notna().mean().mean():.1%}')
"
```
