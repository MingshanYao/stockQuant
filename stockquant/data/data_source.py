"""
数据源抽象基类与工厂。

所有数据源适配器都继承 ``BaseDataSource`` 并实现统一接口，
通过 ``DataSourceFactory`` 按配置名称获取对应实例。

本模块同时提供 schema 标准化工具 ``standardize_daily`` / ``standardize_index``，
各数据源适配器应在返回前统一经此函数收口，保证下游表结构一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from stockquant.utils.helpers import RateLimiter

if TYPE_CHECKING:
    import datetime as dt


# ======================================================================
# 标准列定义（与 Database 表 schema 对齐）
# ======================================================================

DAILY_BAR_COLS: tuple[str, ...] = (
    "code", "date", "open", "high", "low", "close",
    "volume", "amount", "turnover", "pct_change", "change",
)

INDEX_DAILY_COLS: tuple[str, ...] = (
    "code", "date", "open", "high", "low", "close", "volume", "amount",
)

# 中文 → 标准列名映射（AkShare 默认返回中文列）
_CN_DAILY_RENAME = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
}

# BaoStock 等英文返回值的映射（仅列出非标准的）
_EN_DAILY_RENAME = {
    "turn": "turnover",
    "pctChg": "pct_change",
}


def standardize_daily(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """统一日线 DataFrame schema，输出列与 daily_bars 表对齐。

    支持输入为中文列名（AkShare）或英文列名（BaoStock / 新浪）。
    多余列被丢弃，缺失列保留为空。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={**_CN_DAILY_RENAME, **_EN_DAILY_RENAME})
    df["code"] = code
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.reindex(columns=list(DAILY_BAR_COLS))


def standardize_index(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """统一指数日线 DataFrame schema，输出列与 index_daily 表对齐。

    多余列被丢弃，缺失列保留为空。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={**_CN_DAILY_RENAME, **_EN_DAILY_RENAME})
    df["code"] = code
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.reindex(columns=list(INDEX_DAILY_COLS))


# ======================================================================
# 抽象基类
# ======================================================================

class BaseDataSource(ABC):
    """数据源抽象基类。

    子类可在 ``__init__`` 中设置 ``_rate_limiter`` 以启用请求限速。
    """

    _rate_limiter: RateLimiter | None = None

    def _throttle(self) -> float:
        """若配置了限速器，阻塞直到允许下一次请求。

        Returns
        -------
        float
            实际等待的秒数（0 表示无需等待）。
        """
        if self._rate_limiter is not None:
            return self._rate_limiter.acquire()
        return 0.0

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
        adjust: str = "hfq",
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
            标准列见 :data:`DAILY_BAR_COLS`。
        """

    @abstractmethod
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        """获取指数日线行情。

        Returns
        -------
        DataFrame
            标准列见 :data:`INDEX_DAILY_COLS`。
        """

    @abstractmethod
    def get_index_constituents(self, index_code: str) -> list[str]:
        """获取指数成分股代码列表（纯 6 位数字、去重）。"""

    @abstractmethod
    def get_finance_data(self, code: str) -> pd.DataFrame:
        """获取基本财务数据。"""

    @abstractmethod
    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息。

        Returns
        -------
        DataFrame
            标准列: code, name, industry, sector, market,
            list_date, total_shares, float_shares, total_cap, float_cap
        """

    @abstractmethod
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        """获取区间内交易日历。"""


# ======================================================================
# 工厂
# ======================================================================

class DataSourceFactory:
    """数据源工厂 — 根据名称返回对应数据源实例（按名称缓存）。"""

    _registry: dict[str, type[BaseDataSource]] = {}
    _instances: dict[str, BaseDataSource] = {}

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
        if name not in cls._instances:
            cls._instances[name] = cls._registry[name]()
        return cls._instances[name]

    @classmethod
    def reset_instances(cls) -> None:
        """清空实例缓存（测试用）。"""
        cls._instances.clear()
