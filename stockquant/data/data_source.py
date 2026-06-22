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
    "code", "date", "open", "high", "low", "close", "pre_close",
    "volume", "amount", "vwap", "turnover", "pct_change", "adj_factor",
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
    "昨收": "pre_close",
    "前收盘": "pre_close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_change",
    "涨跌额": "change",       # only used to derive pre_close if missing
}

# BaoStock 等英文返回值的映射（仅列出非标准的）
_EN_DAILY_RENAME = {
    "preclose": "pre_close",
    "turn": "turnover",
    "pctChg": "pct_change",
}


def standardize_daily(
    df: pd.DataFrame,
    code: str,
    *,
    volume_unit: str = "shares",
    adj_factor: pd.Series | None = None,
) -> pd.DataFrame:
    """统一日线 DataFrame schema，输出列与 daily_bars 表对齐。

    支持输入为中文列名（AkShare）或英文列名（BaoStock / 新浪）。

    Parameters
    ----------
    df : DataFrame
        原始日线数据。
    code : str
        股票代码（纯 6 位数字）。
    volume_unit : str
        原始成交量单位: ``"shares"`` (股) 或 ``"lots"`` (手，100股/手)。
        输出统一为 shares (股)。
    adj_factor : Series, optional
        复权因子序列（与 df 行对齐）。传入时直接使用，未传入时尝试从
        ``close / close_raw`` 推导（需同时存在两列）。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={**_CN_DAILY_RENAME, **_EN_DAILY_RENAME})
    df["code"] = code
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # ---- 1. volume 单位归一化: 手 → 股 ----
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        if volume_unit == "lots":
            df["volume"] = df["volume"] * 100

    # ---- 2. pre_close 推导 ----
    if "pre_close" not in df.columns or df["pre_close"].isna().all():
        # 从 pct_change 反推
        if "pct_change" in df.columns and "close" in df.columns:
            df["pre_close"] = df["close"] / (1 + df["pct_change"] / 100)
        # 从 change 反推
        elif "change" in df.columns and "close" in df.columns:
            df["pre_close"] = df["close"] - df["change"]
        # 从 close 前移
        elif "close" in df.columns:
            df["pre_close"] = df["close"].shift(1)

    # ---- 3. adj_factor 处理 ----
    if adj_factor is not None:
        df["adj_factor"] = adj_factor
    elif "adj_factor" not in df.columns or df["adj_factor"].isna().all():
        # 从 close_raw（列名映射后仍然存在的话）和 close 推导
        if "close_raw" in df.columns:
            raw = pd.to_numeric(df["close_raw"], errors="coerce")
            cls = pd.to_numeric(df["close"], errors="coerce")
            with pd.option_context("mode.use_inf_as_na", True):
                df["adj_factor"] = cls / raw
        else:
            df["adj_factor"] = 1.0

    # ---- 4. vwap 计算 ----
    if "vwap" not in df.columns or df["vwap"].isna().all():
        if "amount" in df.columns and "volume" in df.columns:
            df["vwap"] = df["amount"] / (df["volume"].replace(0, pd.NA) + 1e-10)

    return df.reindex(columns=list(DAILY_BAR_COLS))


# 需要复权调整的价格列
_PRICE_COLS = ["open", "high", "low", "close", "pre_close", "vwap"]


def apply_price_adjustment(
    df: pd.DataFrame, method: str = "hfq",
) -> pd.DataFrame:
    """对日线 DataFrame 就地应用复权价格调整。

    Parameters
    ----------
    df : DataFrame
        含原始价格列和 ``adj_factor`` 列的日线数据。
    method : str
        ``"hfq"`` — 后复权：price × adj_factor。
        ``"qfq"`` — 前复权：price × adj_factor / adj_factor.max()。
        ``"none"`` — 不复权，原样返回。

    Returns
    -------
    DataFrame
        调整后的 DataFrame（就地修改）。
    """
    if method == "none" or "adj_factor" not in df.columns:
        return df

    factor = df["adj_factor"].copy()

    if method == "hfq":
        pass  # factor as-is
    elif method == "qfq":
        # 前复权 = 后复权价格 / 最近一期后复权因子
        latest = factor.max()
        if latest > 0:
            factor = factor / latest
    else:
        return df

    for col in _PRICE_COLS:
        if col in df.columns:
            df[col] = df[col] * factor

    return df


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
