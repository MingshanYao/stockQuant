"""
股票标的池 (Stock Universe) — 类型安全的可投资股票集合定义与加载。

使用 ``Pool`` 枚举代替字符串，杜绝拼写错误；
Builder 链式 API 支持多标的池并集、按板块 / 个股排除，
最终通过 ``DataManager`` 从本地数据库加载日线数据构建回测数据集。

典型用法
--------
>>> from stockquant.data.universe import Pool, StockUniverse
>>>
>>> # 沪深300 ∪ 中证500，排除科创板和创业板
>>> dataset = (
...     StockUniverse()
...     .scope(Pool.CSI300, Pool.CSI500)
...     .exclude(Pool.STAR, Pool.CHINEXT)
...     .load("2020-01-01", "2025-12-31")
... )
>>>
>>> # 自定义代码列表
>>> dataset = (
...     StockUniverse()
...     .scope(["600519", "000858", "601318"])
...     .load("2020-01-01", "2025-12-31")
... )
>>>
>>> # 全部 A 股，排除科创板 + 北交所 + 个别股票
>>> codes = (
...     StockUniverse()
...     .scope(Pool.ALL_A)
...     .exclude(Pool.STAR, Pool.BSE, "600519")
...     .codes()
... )
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

import akshare as ak
import pandas as pd

from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

if TYPE_CHECKING:
    from stockquant.data.data_manager import DataManager

logger = get_logger("data.universe")


# ======================================================================
# Pool 枚举 — 类型安全的标的池定义
# ======================================================================

class Pool(Enum):
    """预定义标的池类型。

    可同时用于 :meth:`StockUniverse.scope` （定义投资范围）
    和 :meth:`StockUniverse.exclude` （排除标的）。

    Attributes
    ----------
    display_name : str
        中文显示名称。
    index_code : str | None
        指数成分股类型对应的指数代码，板块 / 全市场类型为 ``None``。
    code_prefixes : tuple[str, ...] | None
        板块类型对应的代码前缀，指数 / 全市场类型为 ``None``。

    Examples
    --------
    >>> Pool.CSI300.display_name   # '沪深300'
    >>> Pool.CSI300.index_code     # '000300'
    >>> Pool.STAR.code_prefixes    # ('688',)
    >>> Pool.CSI300.is_index       # True
    >>> Pool.STAR.is_board         # True
    """

    # ---- 指数成分股 ----
    SSE50   = ("上证50",   "000016", None)
    CSI300  = ("沪深300",  "000300", None)
    CSI500  = ("中证500",  "000905", None)
    CSI1000 = ("中证1000", "000852", None)
    CSI2000 = ("中证2000", "932000", None)

    # ---- 板块 ----
    STAR    = ("科创板",   None, ("688",))
    CHINEXT = ("创业板",   None, ("300", "301"))
    BSE     = ("北交所",   None, ("43", "83", "87"))
    SH_MAIN = ("沪市主板", None, ("600", "601", "603", "605"))
    SZ_MAIN = ("深市主板", None, ("000", "001", "002", "003"))

    # ---- 全市场 ----
    ALL_A   = ("全部A股",  None, None)

    def __init__(
        self,
        display_name: str,
        index_code: str | None,
        code_prefixes: tuple[str, ...] | None,
    ) -> None:
        self.display_name = display_name
        self.index_code = index_code
        self.code_prefixes = code_prefixes

    @property
    def is_index(self) -> bool:
        """是否为指数成分股类型。"""
        return self.index_code is not None

    @property
    def is_board(self) -> bool:
        """是否为板块类型（可用代码前缀快速过滤）。"""
        return self.code_prefixes is not None

    @property
    def is_all(self) -> bool:
        """是否为全市场。"""
        return self == Pool.ALL_A

    def __str__(self) -> str:
        return self.display_name


# ======================================================================
# BacktestDataset — 回测数据集
# ======================================================================

@dataclasses.dataclass
class BacktestDataset:
    """回测数据集，由 :meth:`StockUniverse.load` 返回。

    Attributes
    ----------
    stock_data : dict[str, DataFrame]
        代码 → 日线 DataFrame（仅包含有数据的股票）。
    codes : list[str]
        实际有数据的股票代码列表。
    benchmark : DataFrame
        基准指数日线数据。
    benchmark_code : str
        基准指数代码。
    start_date : str
        起始日期。
    end_date : str
        截止日期。
    missing_codes : list[str]
        本地无数据的股票代码（可提示用户先运行 updater）。
    """

    stock_data: dict[str, pd.DataFrame]
    codes: list[str]
    benchmark: pd.DataFrame
    benchmark_code: str
    start_date: str
    end_date: str
    missing_codes: list[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        n_total = len(self.codes) + len(self.missing_codes)
        n_loaded = len(self.codes)
        bars = sum(len(df) for df in self.stock_data.values())
        return (
            f"BacktestDataset("
            f"{n_loaded}/{n_total} stocks loaded, "
            f"{bars:,} bars, "
            f"benchmark={self.benchmark_code}, "
            f"{self.start_date} ~ {self.end_date})"
        )

    def summary(self) -> str:
        """返回友好的数据集摘要字符串。"""
        lines = [
            f"📊 回测数据集摘要",
            f"   标的数量: {len(self.codes)} 只"
            f"（共请求 {len(self.codes) + len(self.missing_codes)} 只）",
            f"   日期范围: {self.start_date} ~ {self.end_date}",
            f"   基准指数: {self.benchmark_code}"
            f"（{len(self.benchmark)} 条）",
        ]
        if self.stock_data:
            bars = sum(len(df) for df in self.stock_data.values())
            lines.append(f"   日线总条数: {bars:,}")
        if self.missing_codes:
            lines.append(f"   ⚠️ 缺失数据: {self.missing_codes}")
        return "\n".join(lines)


# ======================================================================
# StockUniverse — Builder 核心
# ======================================================================

class StockUniverse:
    """股票标的池 Builder — 类型安全的范围定义 → 排除 → 加载数据。

    Parameters
    ----------
    dm : DataManager, optional
        数据管理器实例。省略时在首次需要数据库时延迟创建。

    Examples
    --------
    >>> from stockquant.data.universe import Pool, StockUniverse
    >>>
    >>> # 沪深300 ∪ 中证500，排除科创板和创业板
    >>> dataset = (
    ...     StockUniverse()
    ...     .scope(Pool.CSI300, Pool.CSI500)
    ...     .exclude(Pool.STAR, Pool.CHINEXT)
    ...     .load("2020-01-01", "2025-12-31")
    ... )
    >>>
    >>> # 自定义代码
    >>> dataset = (
    ...     StockUniverse()
    ...     .scope(["600519", "000858", "601318"])
    ...     .load("2020-01-01", "2025-12-31")
    ... )
    >>>
    >>> # 全部主板 + 排除个股
    >>> dataset = (
    ...     StockUniverse()
    ...     .scope(Pool.ALL_A)
    ...     .exclude(Pool.STAR, Pool.CHINEXT, Pool.BSE, "600519")
    ...     .load("2020-01-01", "2025-12-31")
    ... )
    """

    def __init__(self, dm: Optional[DataManager] = None) -> None:
        self.__dm = dm
        self._scope_codes: list[str] = []
        self._scope_labels: list[str] = []
        self._exclude_pools: list[Pool] = []
        self._exclude_codes: set[str] = set()

    # ---------- lazy DataManager ----------

    @property
    def _dm(self) -> DataManager:
        if self.__dm is None:
            from stockquant.data.data_manager import DataManager
            self.__dm = DataManager()
        return self.__dm

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def scope(self, *args: Pool | Sequence[str]) -> StockUniverse:
        """设置标的池范围（多个参数取并集，多次调用累加）。

        Parameters
        ----------
        *args : Pool | list[str]
            ``Pool`` 枚举 → 预定义标的池。
            ``list[str]`` → 自定义股票代码列表。
            传入多个参数或多次调用均取并集。

        Returns
        -------
        self
            支持链式调用。

        Examples
        --------
        >>> u.scope(Pool.CSI300)                          # 沪深300
        >>> u.scope(Pool.CSI300, Pool.CSI500)             # 沪深300 ∪ 中证500
        >>> u.scope(["600519", "000858"])                  # 自定义代码
        >>> u.scope(Pool.CSI300, ["600519"])               # 混合
        >>> u.scope(Pool.CSI300).scope(Pool.CSI500)        # 链式并集
        """
        for arg in args:
            if isinstance(arg, Pool):
                resolved = self._resolve_pool(arg)
                self._scope_codes.extend(resolved)
                self._scope_labels.append(arg.display_name)
            elif isinstance(arg, (list, tuple)):
                codes = [normalize_stock_code(c) for c in arg]
                self._scope_codes.extend(codes)
                self._scope_labels.append(f"自定义({len(codes)}只)")
            else:
                raise TypeError(
                    f"scope() 参数类型错误: {type(arg).__name__}。"
                    f"请传入 Pool 枚举或代码列表 list[str]。"
                )
        # 保序去重
        self._scope_codes = list(dict.fromkeys(self._scope_codes))
        return self

    def exclude(self, *items: Pool | str) -> StockUniverse:
        """排除标的池或个股（多次调用累加）。

        Parameters
        ----------
        *items : Pool | str
            ``Pool`` → 排除整个标的池（板块按前缀过滤，指数按成分股过滤）。
            ``str``  → 排除单个股票代码。

        Returns
        -------
        self
            支持链式调用。

        Examples
        --------
        >>> u.exclude(Pool.STAR)                           # 排除科创板
        >>> u.exclude(Pool.STAR, Pool.CHINEXT)             # 排除多个板块
        >>> u.exclude(Pool.STAR, "600519", "000001")       # 排除板块 + 个股
        """
        for item in items:
            if isinstance(item, Pool):
                if item not in self._exclude_pools:
                    self._exclude_pools.append(item)
            elif isinstance(item, str):
                self._exclude_codes.add(normalize_stock_code(item))
            else:
                raise TypeError(
                    f"exclude() 参数类型错误: {type(item).__name__}。"
                    f"请传入 Pool 枚举或股票代码 str。"
                )
        return self

    def codes(self) -> list[str]:
        """获取过滤后的股票代码列表（不加载日线数据）。

        Returns
        -------
        list[str]
            过滤后的代码列表（6 位纯数字）。
        """
        result = list(self._scope_codes)

        # 按 Pool 排除
        for pool in self._exclude_pools:
            result = self._apply_pool_exclude(result, pool)

        # 按个股代码排除
        if self._exclude_codes:
            result = [c for c in result if c not in self._exclude_codes]

        return result

    def load(
        self,
        start_date: str,
        end_date: str,
        benchmark: Pool | str = Pool.CSI300,
    ) -> BacktestDataset:
        """从本地数据库加载日线数据，构建回测数据集。

        Parameters
        ----------
        start_date : str
            起始日期，如 ``"2020-01-01"``。
        end_date : str
            截止日期，如 ``"2025-12-31"``。
        benchmark : Pool | str
            基准指数。传入 ``Pool`` 枚举（需为指数类型）或指数代码字符串，
            默认 ``Pool.CSI300``。

        Returns
        -------
        BacktestDataset
        """
        final_codes = self.codes()

        scope_label = " ∪ ".join(self._scope_labels) if self._scope_labels else "未定义"
        exclude_parts = (
            [p.display_name for p in self._exclude_pools]
            + sorted(self._exclude_codes)
        )
        exclude_label = ", ".join(exclude_parts)

        logger.info(
            f"加载标的池: {scope_label}"
            + (f"（排除: {exclude_label}）" if exclude_label else "")
            + f" → {len(final_codes)} 只"
        )

        # --- 解析基准 ---
        benchmark_code = self._resolve_benchmark(benchmark)

        # --- 加载个股日线 ---
        stock_data: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        total = len(final_codes)

        for i, code in enumerate(final_codes, 1):
            df = self._dm.fetch_daily(code, start_date=start_date, end_date=end_date)
            if df.empty:
                missing.append(code)
                logger.warning(f"[{i}/{total}] ⚠️  {code} 无数据")
            else:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                stock_data[code] = df
                date_min = df["date"].min().date()
                date_max = df["date"].max().date()
                logger.debug(
                    f"[{i}/{total}] ✅ {code}: {len(df)} bars"
                    f" [{date_min} ~ {date_max}]"
                )

        loaded_codes = list(stock_data.keys())
        logger.info(f"日线加载完成: {len(loaded_codes)}/{total} 只")

        # --- 加载基准指数 ---
        benchmark_df = self._load_benchmark(benchmark_code, start_date, end_date)

        return BacktestDataset(
            stock_data=stock_data,
            codes=loaded_codes,
            benchmark=benchmark_df,
            benchmark_code=benchmark_code,
            start_date=start_date,
            end_date=end_date,
            missing_codes=missing,
        )

    # ------------------------------------------------------------------
    # 内部 — Pool 解析
    # ------------------------------------------------------------------

    def _resolve_pool(self, pool: Pool) -> list[str]:
        """将 Pool 枚举解析为股票代码列表。"""
        if pool.is_all:
            return self._get_all_a_codes()
        if pool.is_index:
            return self._get_index_constituents(pool.index_code)
        if pool.is_board:
            return self._get_codes_by_prefix(pool.code_prefixes)
        return []

    def _apply_pool_exclude(self, codes: list[str], pool: Pool) -> list[str]:
        """从代码列表中排除 Pool 对应的股票。"""
        before = len(codes)
        if pool.is_board:
            # 板块类型：前缀快速过滤
            prefixes = pool.code_prefixes
            codes = [c for c in codes if not c.startswith(prefixes)]
        elif pool.is_index:
            # 指数类型：获取成分股后排除
            constituents = set(self._get_index_constituents(pool.index_code))
            codes = [c for c in codes if c not in constituents]
        elif pool.is_all:
            codes = []

        removed = before - len(codes)
        if removed:
            logger.info(f"排除 {pool.display_name}: 移除 {removed} 只")
        return codes

    @staticmethod
    def _resolve_benchmark(benchmark: Pool | str) -> str:
        """解析基准参数为指数代码。"""
        if isinstance(benchmark, Pool):
            if not benchmark.is_index:
                raise ValueError(
                    f"基准必须是指数类型的 Pool（如 Pool.CSI300），"
                    f"'{benchmark.display_name}' 不是指数类型。"
                )
            return benchmark.index_code
        return normalize_stock_code(benchmark)

    # ------------------------------------------------------------------
    # 内部 — 数据获取
    # ------------------------------------------------------------------

    def _get_all_a_codes(self) -> list[str]:
        """获取全部 A 股代码，优先本地 stock_info 表。"""
        try:
            df = self._dm.get_stock_list()
            if not df.empty:
                codes = df["code"].astype(str).str.zfill(6).tolist()
                logger.info(f"全部 A 股: {len(codes)} 只（本地）")
                return codes
        except Exception:
            pass

        # fallback: AkShare
        try:
            df = ak.stock_info_a_code_name()
            df.columns = ["code", "name"]
            codes = df["code"].astype(str).str.zfill(6).tolist()
            logger.info(f"全部 A 股: {len(codes)} 只（远程）")
            return codes
        except Exception as e:
            logger.error(f"获取全部 A 股失败: {e}")
            return []

    @staticmethod
    def _get_index_constituents(index_code: str) -> list[str]:
        """获取指数成分股代码列表（AkShare）。"""
        logger.info(f"获取指数 {index_code} 成分股")

        # 首选 index_stock_cons
        try:
            df = ak.index_stock_cons(symbol=index_code)
            if df is not None and not df.empty:
                col = "品种代码" if "品种代码" in df.columns else df.columns[0]
                codes = df[col].astype(str).str.zfill(6).drop_duplicates().tolist()
                logger.info(f"指数 {index_code} 成分股: {len(codes)} 只")
                return codes
        except Exception as e:
            logger.debug(f"index_stock_cons({index_code}) 失败: {e}")

        # 备用 index_stock_cons_csindex
        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
            if df is not None and not df.empty:
                col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
                codes = df[col].astype(str).str.zfill(6).drop_duplicates().tolist()
                logger.info(f"指数 {index_code} 成分股: {len(codes)} 只（csindex）")
                return codes
        except Exception as e:
            logger.error(f"获取指数 {index_code} 成分股失败: {e}")

        return []

    def _get_codes_by_prefix(self, prefixes: tuple[str, ...]) -> list[str]:
        """从全部 A 股中按代码前缀筛选。"""
        all_codes = self._get_all_a_codes()
        return [c for c in all_codes if c.startswith(prefixes)]

    # ------------------------------------------------------------------
    # 内部 — 基准加载
    # ------------------------------------------------------------------

    def _load_benchmark(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载基准指数日线，优先本地数据库。"""
        # 本地查询
        try:
            df = self._dm.db.query(
                "SELECT * FROM index_daily "
                "WHERE code = ? AND date >= ? AND date <= ? "
                "ORDER BY date",
                [code, start_date, end_date],
            )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                logger.info(f"基准 {code}: {len(df)} 条（本地）")
                return df
        except Exception:
            pass

        # 远程拉取
        try:
            df = self._dm.fetch_index_daily(code, start_date, end_date)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                logger.info(f"基准 {code}: {len(df)} 条（远程）")
                return df
        except Exception as e:
            logger.error(f"加载基准 {code} 失败: {e}")

        return pd.DataFrame()
