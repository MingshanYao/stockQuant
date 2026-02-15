"""
数据源抽象基类与工厂。

所有数据源适配器都继承 ``BaseDataSource`` 并实现统一接口，
通过 ``DataSourceFactory`` 按配置名称获取对应实例。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import datetime as dt


class BaseDataSource(ABC):
    """数据源抽象基类。"""

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取全部 A 股股票列表。

        Returns
        -------
        DataFrame
            至少包含 ``code`` 和 ``name`` 两列。
        """

    @abstractmethod
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线行情数据。

        Parameters
        ----------
        code : str
            股票代码（纯 6 位数字）。
        start_date / end_date : str | date
            起止日期。
        adjust : str
            复权方式: ``qfq`` / ``hfq`` / ``none``。

        Returns
        -------
        DataFrame
            标准列: date, open, high, low, close, volume, amount, turnover
        """

    @abstractmethod
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        """获取指数日线行情。"""

    @abstractmethod
    def get_finance_data(self, code: str) -> pd.DataFrame:
        """获取基本财务数据。"""

    @abstractmethod
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        """获取区间内交易日历。"""


class DataSourceFactory:
    """数据源工厂 — 根据名称返回对应数据源实例。"""

    _registry: dict[str, type[BaseDataSource]] = {}

    @classmethod
    def register(cls, name: str, source_cls: type[BaseDataSource]) -> None:
        cls._registry[name.lower()] = source_cls

    @classmethod
    def create(cls, name: str) -> BaseDataSource:
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(
                f"未注册的数据源: {name}，可用: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()
