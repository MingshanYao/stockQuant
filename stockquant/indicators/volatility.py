"""
波动率类指标 — ATR / 历史波动率 / 真实波幅 等。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from stockquant.indicators.base import BaseIndicator


class VolatilityIndicators(BaseIndicator):
    """波动率类指标集合。"""

    @property
    def name(self) -> str:
        return "Volatility"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df = self.atr(df)
        df = self.historical_volatility(df)
        df = self.true_range(df)
        return df

    # ------------------------------------------------------------------

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """平均真实波幅 (ATR)。"""
        df[f"atr{length}"] = ta.atr(df["high"], df["low"], df["close"], length=length)
        return df

    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.DataFrame:
        """真实波幅 (True Range)。"""
        tr = ta.true_range(df["high"], df["low"], df["close"])
        if tr is not None:
            df["true_range"] = tr
        return df

    @staticmethod
    def historical_volatility(
        df: pd.DataFrame,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """历史波动率。"""
        log_returns = np.log(df["close"] / df["close"].shift(1))
        hv = log_returns.rolling(window=window).std()
        if annualize:
            hv = hv * np.sqrt(252)
        df[f"hv{window}"] = hv
        return df
