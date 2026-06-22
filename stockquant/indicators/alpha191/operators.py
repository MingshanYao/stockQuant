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


def regresi(
    y: pd.DataFrame | pd.Series,
    x: pd.DataFrame,
    n: int,
) -> pd.DataFrame | pd.Series:
    """滚动窗口多元 OLS 回归残差。

    对每个时间窗口 [t-n+1, t]，用 y 对 x 的每一列做 OLS，
    返回窗口最后一期的残差 ε_t = y_t - ŷ_t。

    对应 Alpha191 公式中的 REGRESI(y, x_1, x_2, ..., n) 运算符。

    Parameters
    ----------
    y : DataFrame or Series
        因变量（如每只股票的日收益率面板）。
    x : DataFrame
        自变量矩阵，每列为一个因子（如 MKT/SMB/HML）。
        index 需与 y 的 index 对齐，值对所有股票相同。
    n : int
        滚动窗口长度。

    Returns
    -------
    DataFrame or Series
        残差序列，形状与 y 相同。
    """
    x_values = x.values

    if isinstance(y, pd.DataFrame):
        result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
        for col in y.columns:
            yc = y[col].values
            vals = np.full(len(yc), np.nan)
            for i in range(max(n - 1, 0), len(yc)):
                y_win = yc[i - n + 1: i + 1]
                x_win = x_values[i - n + 1: i + 1]
                valid = np.isfinite(y_win) & np.all(np.isfinite(x_win), axis=1)
                if valid.sum() < max(3, x.shape[1] + 1):
                    continue
                try:
                    coef, _, _, _ = np.linalg.lstsq(
                        np.column_stack([np.ones(valid.sum()), x_win[valid]]),
                        y_win[valid],
                        rcond=None,
                    )
                    pred = coef[0] + x_win[-1] @ coef[1:]
                    vals[i] = float(y_win[-1] - pred)
                except np.linalg.LinAlgError:
                    vals[i] = np.nan
            result[col] = vals
        return result
    else:
        y_arr = y.values
        vals = np.full(len(y_arr), np.nan)
        for i in range(max(n - 1, 0), len(y_arr)):
            y_win = y_arr[i - n + 1: i + 1]
            x_win = x_values[i - n + 1: i + 1]
            valid = np.isfinite(y_win) & np.all(np.isfinite(x_win), axis=1)
            if valid.sum() < max(3, x.shape[1] + 1):
                continue
            try:
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(valid.sum()), x_win[valid]]),
                    y_win[valid],
                    rcond=None,
                )
                pred = coef[0] + x_win[-1] @ coef[1:]
                vals[i] = float(y_arr[i] - pred)
            except np.linalg.LinAlgError:
                vals[i] = np.nan
        return pd.Series(vals, index=y.index)
