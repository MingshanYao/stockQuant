"""
量价类指标 — OBV / VWAP / AD / MFI 等。
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from stockquant.indicators.base import BaseIndicator


class VolumeIndicators(BaseIndicator):
    """量价类指标集合。"""

    @property
    def name(self) -> str:
        return "Volume"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df = self.obv(df)
        df = self.vwap(df)
        df = self.ad(df)
        df = self.mfi(df)
        return df

    # ------------------------------------------------------------------

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.DataFrame:
        """能量潮 (OBV)。"""
        df["obv"] = ta.obv(df["close"], df["volume"])
        return df

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.DataFrame:
        """成交量加权平均价 (VWAP)。"""
        try:
            vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap is not None:
                df["vwap"] = vwap
        except Exception:
            pass
        return df

    @staticmethod
    def ad(df: pd.DataFrame) -> pd.DataFrame:
        """累积/分配线 (AD)。"""
        df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
        return df

    @staticmethod
    def mfi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """资金流量指标 (MFI)。"""
        df[f"mfi{length}"] = ta.mfi(
            df["high"], df["low"], df["close"], df["volume"], length=length
        )
        return df
