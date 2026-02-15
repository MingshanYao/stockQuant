"""
标的过滤器 — ST / 次新 / 停牌 / 黑名单过滤。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("risk.filter")


class StockFilter:
    """标的过滤器。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self.exclude_st: bool = cfg.get("stock_pool.exclude_st", True)
        self.exclude_new: bool = cfg.get("stock_pool.exclude_new", True)
        self.exclude_suspended: bool = cfg.get("stock_pool.exclude_suspended", True)
        self.blacklist: list[str] = cfg.get("stock_pool.blacklist", [])
        self._new_days: int = 60  # 上市不满 60 天视为次新

    def filter(
        self,
        stock_list: pd.DataFrame,
        trade_date: dt.date | None = None,
    ) -> pd.DataFrame:
        """过滤不符合条件的标的。

        Parameters
        ----------
        stock_list : DataFrame
            至少包含 ``code``, ``name`` 列，可选 ``list_date``。
        trade_date : date, optional
            当前交易日（用于判断次新股）。
        """
        df = stock_list.copy()
        before = len(df)

        # ST 过滤
        if self.exclude_st and "name" in df.columns:
            df = df[~df["name"].str.contains("ST", case=False, na=False)]

        # 次新股过滤
        if self.exclude_new and "list_date" in df.columns and trade_date:
            cutoff = trade_date - dt.timedelta(days=self._new_days)
            df["list_date"] = pd.to_datetime(df["list_date"])
            df = df[df["list_date"].dt.date <= cutoff]

        # 黑名单
        if self.blacklist:
            df = df[~df["code"].isin(self.blacklist)]

        after = len(df)
        if after < before:
            logger.info(f"标的过滤: {before} → {after} (移除 {before - after} 只)")

        return df.reset_index(drop=True)
