"""
震荡类指标 — RSI / KDJ / CCI / Williams %R 等。
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from stockquant.indicators.base import BaseIndicator


class OscillatorIndicators(BaseIndicator):
    """震荡类指标集合。"""

    @property
    def name(self) -> str:
        return "Oscillator"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df = self.rsi(df)
        df = self.kdj(df)
        df = self.cci(df)
        df = self.willr(df)
        return df

    # ------------------------------------------------------------------

    @staticmethod
    def rsi(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """相对强弱指标 (RSI)。"""
        periods = periods or [6, 12, 24]
        for p in periods:
            df[f"rsi{p}"] = ta.rsi(df["close"], length=p)
        return df

    @staticmethod
    def kdj(
        df: pd.DataFrame,
        k_period: int = 9,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> pd.DataFrame:
        """KDJ 随机指标。"""
        stoch = ta.stoch(
            df["high"], df["low"], df["close"],
            k=k_period, d=d_period, smooth_k=smooth_k,
        )
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)
            # 计算 J 值 = 3K - 2D
            k_col = [c for c in stoch.columns if "K" in c.upper()]
            d_col = [c for c in stoch.columns if "D" in c.upper()]
            if k_col and d_col:
                df["KDJ_J"] = 3 * df[k_col[0]] - 2 * df[d_col[0]]
        return df

    @staticmethod
    def cci(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """顺势指标 (CCI)。"""
        df[f"cci{length}"] = ta.cci(df["high"], df["low"], df["close"], length=length)
        return df

    @staticmethod
    def willr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """威廉指标 (Williams %R)。"""
        df[f"willr{length}"] = ta.willr(df["high"], df["low"], df["close"], length=length)
        return df
