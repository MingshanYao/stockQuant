"""数据采集与管理模块。

导入此包时自动注册 AkShare / BaoStock 数据源到 :class:`DataSourceFactory`。
"""

from .data_source import (
    BaseDataSource,
    DataSourceFactory,
    DAILY_BAR_COLS,
    INDEX_DAILY_COLS,
    standardize_daily,
    standardize_index,
)
from .database import Database
from .data_cleaner import DataCleaner

# 触发数据源注册（副作用 import）
from . import source_akshare  # noqa: F401
from . import source_baostock  # noqa: F401

from .data_manager import DataManager
from .universe import Pool, StockUniverse, BacktestDataset

__all__ = [
    "BaseDataSource",
    "DataSourceFactory",
    "DAILY_BAR_COLS",
    "INDEX_DAILY_COLS",
    "standardize_daily",
    "standardize_index",
    "Database",
    "DataCleaner",
    "DataManager",
    "Pool",
    "StockUniverse",
    "BacktestDataset",
]
