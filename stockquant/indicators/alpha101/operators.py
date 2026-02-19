"""
Alpha101 核心运算符 — 时间序列 / 截面 / 辅助运算。

基于 Kakushadze (2016) "101 Formulaic Alphas" 论文定义的运算符。
所有函数同时支持 ``pd.Series``（单股票）和 ``pd.DataFrame``（面板数据:
行 = 日期，列 = 股票代码）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ======================================================================
# 截面运算符 (Cross-Sectional Operators)
# ======================================================================


def rank(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """截面百分位排名。

    对每个日期（行），将所有股票（列）排名并归一化到 [0, 1]。
    若输入为 Series 则对整个 Series 排名。
    """
    if isinstance(x, pd.DataFrame):
        return x.rank(axis=1, pct=True)
    return x.rank(pct=True)


def scale(x: pd.DataFrame | pd.Series, a: float = 1.0) -> pd.DataFrame | pd.Series:
    """截面归一化: scale(x) = a * x / sum(|x|)。

    使截面绝对值之和等于 ``a``。
    """
    if isinstance(x, pd.DataFrame):
        denom = x.abs().sum(axis=1).replace(0, np.nan)
        return x.div(denom, axis=0) * a
    s = x.abs().sum()
    return x / s * a if s != 0 else x * 0


def ind_neutralize(
    x: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    """行业中性化 — 对每个日期、每个行业去均值。

    Parameters
    ----------
    x : DataFrame
        行=日期, 列=股票代码。
    industry : DataFrame
        同结构, 值=行业代码（字符串或整数）。
    """
    result = x.copy()
    common_dates = x.index.intersection(industry.index)
    for date in common_dates:
        groups = industry.loc[date]
        vals = x.loc[date]
        for grp in groups.dropna().unique():
            mask = groups == grp
            result.loc[date, mask] = vals[mask] - vals[mask].mean()
    return result


# ======================================================================
# 时间序列运算符 (Time-Series Operators)
# ======================================================================


def delay(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """时间序列延迟: delay(x, d) = x[t − d]。"""
    return x.shift(int(d))


def delta(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """差分: delta(x, d) = x[t] − x[t − d]。"""
    return x - delay(x, d)


def ts_min(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动最小值。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).min()


def ts_max(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动最大值。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).max()


def ts_argmin(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动窗口内最小值的位置 (0 = 窗口最早, d−1 = 最新)。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmin, raw=True)


def ts_argmax(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动窗口内最大值的位置。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).apply(np.argmax, raw=True)


def ts_rank(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动百分位排名: 当前值在过去 d 个值中的百分位排名 [0, 1]。"""
    d = int(d)

    def _rank_pct(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        return float(s.rank(pct=True).iloc[-1])

    return x.rolling(d, min_periods=max(1, d // 2)).apply(_rank_pct, raw=True)


def ts_sum(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动求和。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).sum()


def ts_product(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动乘积。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).apply(np.prod, raw=True)


def ts_stddev(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """滚动标准差。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).std()


def sma(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """简单移动平均。"""
    d = int(d)
    return x.rolling(d, min_periods=max(1, d // 2)).mean()


def ts_corr(
    x: pd.DataFrame | pd.Series,
    y: pd.DataFrame | pd.Series,
    d: int,
) -> pd.DataFrame | pd.Series:
    """滚动相关系数（向量化实现，支持 DataFrame）。

    使用公式: corr = cov(X,Y) / (std(X) * std(Y))
    完全向量化，避免逐列迭代。
    """
    d = int(d)
    mp = max(1, d // 2)
    mean_x = x.rolling(d, min_periods=mp).mean()
    mean_y = y.rolling(d, min_periods=mp).mean()
    mean_xy = (x * y).rolling(d, min_periods=mp).mean()
    std_x = x.rolling(d, min_periods=mp).std()
    std_y = y.rolling(d, min_periods=mp).std()
    cov_xy = mean_xy - mean_x * mean_y
    denom = std_x * std_y
    result = cov_xy / denom.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def ts_cov(
    x: pd.DataFrame | pd.Series,
    y: pd.DataFrame | pd.Series,
    d: int,
) -> pd.DataFrame | pd.Series:
    """滚动协方差（向量化实现）。"""
    d = int(d)
    mp = max(1, d // 2)
    mean_x = x.rolling(d, min_periods=mp).mean()
    mean_y = y.rolling(d, min_periods=mp).mean()
    mean_xy = (x * y).rolling(d, min_periods=mp).mean()
    return mean_xy - mean_x * mean_y


def decay_linear(x: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """线性衰减加权移动平均。

    权重 w = [1, 2, ..., d] / sum(1..d)，越近期权重越大。
    """
    d = int(d)
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()

    def _wma(arr: np.ndarray) -> float:
        n = len(arr)
        if n < d:
            w = np.arange(1, n + 1, dtype=float)
            w /= w.sum()
            return float(np.dot(arr, w))
        return float(np.dot(arr, weights))

    return x.rolling(d, min_periods=max(1, d // 2)).apply(_wma, raw=True)


# ======================================================================
# 辅助运算符
# ======================================================================


def sign(x):
    """符号函数: −1 / 0 / +1。"""
    return np.sign(x)


def signedpower(x, a: float):
    """有符号幂: sign(x) × |x|^a。"""
    return np.sign(x) * (np.abs(x) ** a)


def log(x):
    """自然对数 (安全版本，对非正数返回 NaN)。"""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return np.log(x.clip(lower=1e-10))
    return np.log(max(x, 1e-10))


def adv(volume: pd.DataFrame | pd.Series, d: int) -> pd.DataFrame | pd.Series:
    """平均日成交量: adv{d} = 过去 d 天成交量均值。"""
    return sma(volume, d)
