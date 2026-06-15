"""
市场状态检测器 — 基于基准指数的趋势+波动率进行市场状态分类。
"""
from __future__ import annotations

import enum
import numpy as np
import pandas as pd

from stockquant.utils.logger import get_logger

logger = get_logger("risk.regime")


class Regime(enum.Enum):
    BULL = "牛市"
    NORMAL = "正常"
    BEAR = "熊市"
    CRISIS = "危机"


class MarketRegimeDetector:
    """基于基准指数的市场状态检测。

    使用 MA20/MA60 均线交叉判断趋势方向,用 20日/252日 波动率比值判断波动状态。
    """

    def __init__(
        self,
        vol_ratio_threshold: float = 1.5,
        ma_short: int = 20,
        ma_long: int = 60,
        vol_short_window: int = 20,
        vol_long_window: int = 252,
    ) -> None:
        self.vol_ratio_threshold = vol_ratio_threshold
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.vol_short_window = vol_short_window
        self.vol_long_window = vol_long_window

        self._scale_map: dict[Regime, float] = {
            Regime.BULL: 1.0,
            Regime.NORMAL: 0.8,
            Regime.BEAR: 0.5,
            Regime.CRISIS: 0.2,
        }

    def detect(self, prices: pd.Series) -> Regime:
        """从价格序列检测当前市场状态。"""
        if len(prices) < max(self.ma_long, self.vol_long_window):
            return Regime.NORMAL

        returns = prices.pct_change().dropna()

        ma_short_val = prices.rolling(self.ma_short).mean().iloc[-1]
        ma_long_val = prices.rolling(self.ma_long).mean().iloc[-1]
        trend_up = ma_short_val > ma_long_val

        short_vol = returns.rolling(self.vol_short_window).std().iloc[-1]
        long_vol = returns.rolling(self.vol_long_window).std().iloc[-1]
        vol_ratio = short_vol / long_vol if long_vol and long_vol > 0 else 1.0
        high_vol = vol_ratio > self.vol_ratio_threshold

        recent_5d_return = (
            prices.iloc[-1] / prices.iloc[-min(5, len(prices))] - 1
            if len(prices) >= 5
            else 0.0
        )

        if recent_5d_return < -0.10 and high_vol:
            return Regime.CRISIS
        if not trend_up and high_vol:
            return Regime.BEAR
        if not trend_up:
            return Regime.BEAR
        if trend_up and not high_vol:
            return Regime.BULL

        return Regime.NORMAL

    def get_position_scale(self, regime: Regime) -> float:
        """获取该市场状态对应的仓位系数。"""
        return self._scale_map.get(regime, 0.8)

    def compute_scale(self, prices: pd.Series) -> float:
        """一站式：从价格序列直接算出仓位系数。"""
        regime = self.detect(prices)
        scale = self.get_position_scale(regime)
        logger.debug(f"市场状态: {regime.value}, 仓位系数: {scale:.0%}")
        return scale
