"""Alpha191 专用算子 — 补充 Alpha101 算子库中不包含的运算。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema_sma(
    x: pd.DataFrame | pd.Series, n: int, m: int,
) -> pd.DataFrame | pd.Series:
    """指数加权移动平均: Y[t] = (x[t]*m + Y[t-1]*(n-m)) / n。

    对应 Alpha191 公式中的 SMA(X, N, M) 运算符。
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
    """滚动窗口 OLS 回归斜率 beta。"""
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
