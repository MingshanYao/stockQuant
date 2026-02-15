"""
趋势类指标 — MA / EMA / MACD / BOLL 等。
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from stockquant.indicators.base import BaseIndicator


class TrendIndicators(BaseIndicator):
    """趋势类指标集合。"""

    @property
    def name(self) -> str:
        return "Trend"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df = self.ma(df)
        df = self.ema(df)
        df = self.macd(df)
        df = self.boll(df)
        return df

    # ------------------------------------------------------------------

    @staticmethod
    def ma(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """简单移动平均线 (SMA)。"""
        periods = periods or [5, 10, 20, 60]
        for p in periods:
            df[f"ma{p}"] = ta.sma(df["close"], length=p)
        return df

    @staticmethod
    def ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """指数移动平均线 (EMA)。"""
        periods = periods or [12, 26]
        for p in periods:
            df[f"ema{p}"] = ta.ema(df["close"], length=p)
        return df

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD 指标。"""
        macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        return df

    @staticmethod
    def boll(
        df: pd.DataFrame,
        length: int = 20,
        std: float = 2.0,
    ) -> pd.DataFrame:
        """布林带 (Bollinger Bands)。"""
        bbands = ta.bbands(df["close"], length=length, std=std)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)
        return df
