"""
WorldQuant Alpha#101 因子指标包。

本包是 ``stockquant.indicators`` 体系的一部分，提供基于
Kakushadze (2016) 论文的全部 101 个 Alpha 因子公式。

主要入口
--------
- :class:`Alpha101Indicators`  —  继承 ``BaseIndicator``，标准 ``compute(df)`` 接口
- :class:`Alpha101Engine`  —  面板计算引擎（多股票截面计算）

快速开始::

    # 单股票模式 (BaseIndicator 接口)
    from stockquant.indicators import Alpha101Indicators

    ind = Alpha101Indicators()
    df = ind.compute(df)                         # 全部因子
    df = ind.compute(df, alphas=[1, 6, 101])     # 指定因子

    # 多股票面板模式
    engine = Alpha101Indicators.panel(
        open_=open_df, high=high_df, low=low_df,
        close=close_df, volume=volume_df,
    )
    factor_1 = engine.alpha001()
    all_factors = engine.compute_all()
"""

from stockquant.indicators.alpha101.alpha101 import (
    Alpha101Engine,
    Alpha101Indicators,
    INDUSTRY_ALPHAS,
)
from stockquant.indicators.alpha101.operators import (
    adv,
    decay_linear,
    delay,
    delta,
    ind_neutralize,
    log,
    rank,
    scale,
    sign,
    signedpower,
    sma,
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

__all__ = [
    # 核心类
    "Alpha101Indicators",
    "Alpha101Engine",
    "INDUSTRY_ALPHAS",
    # 算子
    "rank",
    "scale",
    "ind_neutralize",
    "delay",
    "delta",
    "ts_min",
    "ts_max",
    "ts_argmin",
    "ts_argmax",
    "ts_rank",
    "ts_sum",
    "ts_product",
    "ts_stddev",
    "sma",
    "ts_corr",
    "ts_cov",
    "decay_linear",
    "sign",
    "signedpower",
    "log",
    "adv",
]
