"""
数据更新器 — 提供按指数成分股 / 板块 / 全市场批量更新日线数据的能力。

典型用法::

    from stockquant.data.updater import DataUpdater

    updater = DataUpdater()

    # 更新沪深300成分股日线（排除科创板 & 北交所 & 创业板）
    updater.update_index_daily("000300", include_star=False, include_bse=False, include_gem=False)

    # 更新中证1000成分股日线
    updater.update_index_daily("000852")

    # 更新全部A股（含科创板 & 北交所）
    updater.update_all_daily()

    # 仅更新科创板
    updater.update_star_daily()

    # 仅更新北交所
    updater.update_bse_daily()
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Sequence

import akshare as ak
import pandas as pd

from stockquant.data.data_source import DataSourceFactory, BaseDataSource
from stockquant.data.database import Database
from stockquant.data.data_cleaner import DataCleaner
from stockquant.utils.config import Config
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("data.updater")

# ======================================================================
# 常用指数代码
# ======================================================================
INDEX_HS300 = "000300"   # 沪深300
INDEX_CSI500 = "000905"  # 中证500
INDEX_CSI1000 = "000852"  # 中证1000
INDEX_CSI2000 = "932000"  # 中证2000

# ======================================================================
# 板块代码前缀
# ======================================================================
_STAR_PREFIXES = ("688",)       # 科创板
_BSE_PREFIXES = ("4", "8")      # 北交所 (43xxxx / 83xxxx / 87xxxx 等)
_MAIN_SH_PREFIXES = ("60",)     # 沪市主板
_MAIN_SZ_PREFIXES = ("00",)     # 深市主板
_GEM_PREFIXES = ("30",)         # 创业板


# ======================================================================
# 工具函数
# ======================================================================

def _is_star(code: str) -> bool:
    """判断是否为科创板股票。"""
    return code.startswith(_STAR_PREFIXES)


def _is_bse(code: str) -> bool:
    """判断是否为北交所股票。"""
    return code.startswith(_BSE_PREFIXES) and not code.startswith(_STAR_PREFIXES)


def _is_gem(code: str) -> bool:
    """判断是否为创业板股票。"""
    return code.startswith(_GEM_PREFIXES)


def filter_codes(
    codes: list[str],
    *,
    include_star: bool = True,
    include_bse: bool = True,
    include_gem: bool = True,
) -> list[str]:
    """根据板块过滤股票代码列表。

    Parameters
    ----------
    codes : list[str]
        原始代码列表（6 位纯数字）。
    include_star : bool
        是否包含科创板 (688xxx)，默认 True。
    include_bse : bool
        是否包含北交所 (4xxxxx / 8xxxxx)，默认 True。
    include_gem : bool
        是否包含创业板 (30xxxx)，默认 True。

    Returns
    -------
    list[str]
        过滤后的代码列表。
    """
    result = []
    for c in codes:
        if not include_star and _is_star(c):
            continue
        if not include_bse and _is_bse(c):
            continue
        if not include_gem and _is_gem(c):
            continue
        result.append(c)
    return result


def get_index_constituents(index_code: str) -> list[str]:
    """获取指数成分股代码列表。

    Parameters
    ----------
    index_code : str
        指数代码，如 "000300"。

    Returns
    -------
    list[str]
        成分股代码列表（6 位纯数字，已去重）。
    """
    index_code = normalize_stock_code(index_code)
    logger.info(f"获取指数 {index_code} 成分股列表")
    try:
        df = ak.index_stock_cons(symbol=index_code)
        codes = df["品种代码"].astype(str).str.zfill(6).drop_duplicates().tolist()
        logger.info(f"指数 {index_code} 成分股: {len(codes)} 只")
        return codes
    except Exception as e:
        logger.error(f"获取指数 {index_code} 成分股失败: {e}")
        return []


def get_all_a_codes() -> list[str]:
    """获取全部 A 股代码列表。"""
    logger.info("获取全部 A 股代码列表")
    try:
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        codes = df["code"].astype(str).str.zfill(6).tolist()
        logger.info(f"全部 A 股: {len(codes)} 只")
        return codes
    except Exception as e:
        logger.error(f"获取 A 股列表失败: {e}")
        return []


# ======================================================================
# DataUpdater
# ======================================================================

class DataUpdater:
    """数据更新器 — 批量拉取 & 入库日线数据。

    Parameters
    ----------
    config : Config, optional
        配置实例，默认使用全局单例。
    db : Database, optional
        数据库实例，默认根据配置自动创建。
    """

    def __init__(
        self,
        config: Config | None = None,
        db: Database | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.db = db or Database()
        self.db.init_tables()
        self.cleaner = DataCleaner()

        # 加载数据源
        import stockquant.data.source_akshare  # noqa: F401
        primary = self.cfg.get("data_source.primary", "akshare")
        self._source: BaseDataSource = DataSourceFactory.create(primary)

    # ------------------------------------------------------------------
    # 公开接口：按指数更新
    # ------------------------------------------------------------------

    def update_index_daily(
        self,
        index_code: str = INDEX_HS300,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
        *,
        include_star: bool = True,
        include_bse: bool = True,
        include_gem: bool = True,
    ) -> dict[str, int]:
        """更新某个指数成分股的日线数据。

        Parameters
        ----------
        index_code : str
            指数代码，默认沪深300。
        start_date / end_date : str | date, optional
            起止日期，默认使用配置或增量更新。
        adjust : str, optional
            复权方式。
        include_star : bool
            是否包含科创板，默认 True。
        include_bse : bool
            是否包含北交所，默认 True。
        include_gem : bool
            是否包含创业板，默认 True。

        Returns
        -------
        dict[str, int]
            {code: 写入行数} 的映射。
        """
        codes = get_index_constituents(index_code)
        codes = filter_codes(codes, include_star=include_star, include_bse=include_bse, include_gem=include_gem)
        logger.info(
            f"更新指数 {index_code} 成分股日线: {len(codes)} 只 "
            f"(include_star={include_star}, include_bse={include_bse}, include_gem={include_gem})"
        )
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_hs300_daily(self, **kwargs) -> dict[str, int]:
        """更新沪深300成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_HS300, **kwargs)

    def update_csi500_daily(self, **kwargs) -> dict[str, int]:
        """更新中证500成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_CSI500, **kwargs)

    def update_csi1000_daily(self, **kwargs) -> dict[str, int]:
        """更新中证1000成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_CSI1000, **kwargs)

    def update_csi2000_daily(self, **kwargs) -> dict[str, int]:
        """更新中证2000成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_CSI2000, **kwargs)

    # ------------------------------------------------------------------
    # 公开接口：全市场更新
    # ------------------------------------------------------------------

    def update_all_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
        *,
        include_star: bool = True,
        include_bse: bool = True,
        include_gem: bool = True,
    ) -> dict[str, int]:
        """更新全部 A 股日线数据。

        Parameters
        ----------
        include_star : bool
            是否包含科创板，默认 True。
        include_bse : bool
            是否包含北交所，默认 True。
        include_gem : bool
            是否包含创业板，默认 True。
        """
        codes = get_all_a_codes()
        codes = filter_codes(codes, include_star=include_star, include_bse=include_bse, include_gem=include_gem)
        logger.info(
            f"更新全部 A 股日线: {len(codes)} 只 "
            f"(include_star={include_star}, include_bse={include_bse}, include_gem={include_gem})"
        )
        return self._batch_update(codes, start_date, end_date, adjust)

    # ------------------------------------------------------------------
    # 公开接口：按板块更新
    # ------------------------------------------------------------------

    def update_star_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新科创板 (688xxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [c for c in codes if _is_star(c)]
        logger.info(f"更新科创板日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_bse_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新北交所 (4xxxxx / 8xxxxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [c for c in codes if _is_bse(c)]
        logger.info(f"更新北交所日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_main_board_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新沪深主板 (60xxxx / 00xxxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [
            c for c in codes
            if c.startswith(_MAIN_SH_PREFIXES) or c.startswith(_MAIN_SZ_PREFIXES)
        ]
        logger.info(f"更新沪深主板日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_gem_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新创业板 (30xxxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [c for c in codes if c.startswith(_GEM_PREFIXES)]
        logger.info(f"更新创业板日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    # ------------------------------------------------------------------
    # 公开接口：自定义代码列表
    # ------------------------------------------------------------------

    def update_codes_daily(
        self,
        codes: Sequence[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
        *,
        include_star: bool = True,
        include_bse: bool = True,
        include_gem: bool = True,
    ) -> dict[str, int]:
        """更新指定代码列表的日线数据。"""
        codes = [normalize_stock_code(c) for c in codes]
        codes = filter_codes(codes, include_star=include_star, include_bse=include_bse, include_gem=include_gem)
        logger.info(f"更新自定义列表日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    # ------------------------------------------------------------------
    # 内部：批量更新核心逻辑
    # ------------------------------------------------------------------

    def _batch_update(
        self,
        codes: list[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """批量拉取 & 入库日线数据（支持增量更新）。

        Returns
        -------
        dict[str, int]
            {code: 实际写入行数}，失败的股票不包含在内。
        """
        adjust = adjust or self.cfg.get("data_fetch.adjust", "qfq")
        end_date = ensure_date(end_date) or dt.date.today()

        total = len(codes)
        results: dict[str, int] = {}
        failed: list[str] = []
        t_start = time.time()

        for i, code in enumerate(codes, 1):
            code = normalize_stock_code(code)
            try:
                # 增量更新：从数据库中已有的最新日期的下一天开始拉取
                sd = self._resolve_start_date(code, start_date)
                if sd > end_date:
                    # 数据已是最新
                    results[code] = 0
                    continue

                df = self._source.get_daily_bars(code, sd, end_date, adjust)
                if df.empty:
                    results[code] = 0
                    continue

                df = self.cleaner.clean_pipeline(df)
                rows = self.db.save_dataframe(df, "daily_bars")
                results[code] = rows

            except Exception as e:
                failed.append(code)
                logger.warning(f"[{i}/{total}] {code} 更新失败: {e}")

            # 进度日志
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t_start
                speed = i / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[{i}/{total}] 已完成 | 成功 {len(results)} | "
                    f"失败 {len(failed)} | 速率 {speed:.1f} 只/s"
                )

        elapsed = time.time() - t_start
        logger.info(
            f"批量更新完成: 共 {total} 只, 成功 {len(results)}, "
            f"失败 {len(failed)}, 耗时 {elapsed:.1f}s"
        )
        if failed:
            logger.warning(f"失败代码 (前20): {failed[:20]}")

        return results

    def _resolve_start_date(
        self,
        code: str,
        user_start: str | dt.date | None,
    ) -> dt.date:
        """决定某只股票的拉取起始日期。

        优先级:
            1. 增量更新 — 数据库中最新日期 + 1 天
            2. 用户指定的 start_date
            3. 配置文件中的 data_fetch.start_date
        """
        # 尝试增量
        if user_start is None:
            latest = self.db.get_latest_date("daily_bars", code)
            if latest:
                latest_date = ensure_date(latest)
                if latest_date:
                    return latest_date + dt.timedelta(days=1)

        # 回退到用户指定 / 配置默认
        fallback = user_start or self.cfg.get("data_fetch.start_date", "2020-01-01")
        return ensure_date(fallback) or dt.date(2020, 1, 1)
