"""
数据管理器 — 只读门面。

职责：
- 从本地 DuckDB 查询日线 / 指数 / 股票信息；
- 当本地数据缺失时按需远程拉取，并将增量写回本地（保证下次查询命中）；
- 委托数据源做股票池 / 成分股 / 交易日历等元数据查询。

写库由 :class:`DataUpdater` 负责，本类不再做全量 / 覆盖写。
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import pandas as pd

from stockquant.data.data_source import BaseDataSource, DataSourceFactory
from stockquant.data.data_cleaner import DataCleaner
from stockquant.data.database import Database
from stockquant.utils.config import Config
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("data.manager")


class DataManager:
    """高层数据门面：本地查询 → 缺则远程拉取 → 写回本地。

    Parameters
    ----------
    config : Config, optional
        配置实例，默认使用全局单例。
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.db = Database()
        self.db.init_tables()
        self.cleaner = DataCleaner()

        primary = self.cfg.get("data_source.primary", "akshare")
        self._source: BaseDataSource = DataSourceFactory.create(primary)
        logger.info(f"DataManager 初始化，主数据源: {primary}")

    # ------------------------------------------------------------------
    # 股票列表 / 基本信息（只读）
    # ------------------------------------------------------------------

    def get_stock_list(self) -> pd.DataFrame:
        """获取本地股票列表（code, name）。

        本地表为空时返回空 DataFrame —— 不再隐式触发写库，
        请显式调用 ``DataUpdater.update_stock_info()`` 初始化数据。
        """
        if not self.db.table_exists("stock_info"):
            logger.warning("stock_info 表不存在，请先运行 DataUpdater.update_stock_info()")
            return pd.DataFrame(columns=["code", "name"])
        df = self.db.query("SELECT code, name FROM stock_info")
        if df.empty:
            logger.warning("stock_info 表为空，请先运行 DataUpdater.update_stock_info()")
        return df

    def get_stock_info(self, codes: Sequence[str] | None = None) -> pd.DataFrame:
        """获取股票基本信息（行业 / 市值等）。

        Parameters
        ----------
        codes : list[str], optional
            股票代码列表。None 返回全部。

        Returns
        -------
        DataFrame
            包含 code, name, industry, sector, market, total_cap, float_cap 等列。
        """
        if not self.db.table_exists("stock_info"):
            logger.warning("stock_info 表不存在，请先运行 DataUpdater.update_stock_info()")
            return pd.DataFrame()

        if codes:
            placeholders = ", ".join("?" * len(codes))
            return self.db.query(
                f"SELECT * FROM stock_info WHERE code IN ({placeholders})",
                [normalize_stock_code(c) for c in codes],
            )
        return self.db.query("SELECT * FROM stock_info")

    def get_all_a_codes(self) -> list[str]:
        """获取全部 A 股代码（优先本地 stock_info，缺失则走数据源）。"""
        try:
            df = self.get_stock_list()
            if not df.empty:
                return df["code"].astype(str).str.zfill(6).tolist()
        except Exception as e:
            logger.debug(f"本地股票列表查询失败: {e}")

        df = self._source.get_stock_list()
        if df.empty:
            return []
        return df["code"].astype(str).str.zfill(6).tolist()

    def get_index_constituents(self, index_code: str) -> list[str]:
        """获取指数成分股代码列表（透传到数据源）。"""
        return self._source.get_index_constituents(index_code)

    # ------------------------------------------------------------------
    # 日线（本地优先 + 远程增量 write-back）
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        code: str,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> pd.DataFrame:
        """获取日线数据：本地优先，缺失增量远程拉取并写回。"""
        code = normalize_stock_code(code)
        start_date = ensure_date(start_date or self.cfg.get("data_fetch.start_date", "2020-01-01"))
        end_date = ensure_date(end_date) or dt.date.today()
        adjust = adjust or self.cfg.get("data_fetch.adjust", "hfq")

        local_df = self._load_local_daily(code, start_date, end_date)
        if not local_df.empty:
            latest = local_df["date"].max()
            latest_date = latest.date() if hasattr(latest, "date") else latest
            if latest_date >= end_date:
                return local_df

            next_day = latest_date + dt.timedelta(days=1)
            new_df = self._fetch_and_persist_daily(code, next_day, end_date, adjust)
            if not new_df.empty:
                return pd.concat([local_df, new_df], ignore_index=True)
            return local_df

        return self._fetch_and_persist_daily(code, start_date, end_date, adjust)

    def batch_fetch_daily(
        self,
        codes: Sequence[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """串行批量获取多只股票的日线数据。

        大批量场景请使用 :class:`DataUpdater` 的并发更新接口。
        """
        result: dict[str, pd.DataFrame] = {}
        total = len(codes)
        for i, code in enumerate(codes, 1):
            logger.info(f"[{i}/{total}] 拉取 {code}")
            try:
                result[normalize_stock_code(code)] = self.fetch_daily(
                    code, start_date, end_date, adjust,
                )
            except Exception as e:
                logger.error(f"拉取 {code} 失败: {e}")
        return result

    # ------------------------------------------------------------------
    # 指数日线（同样本地优先 + 写回）
    # ------------------------------------------------------------------

    def fetch_index_daily(
        self,
        code: str = "000300",
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        start_date = ensure_date(start_date or self.cfg.get("data_fetch.start_date"))
        end_date = ensure_date(end_date) or dt.date.today()

        df = self._source.get_index_daily(code, start_date, end_date)
        if not df.empty and self.db.table_exists("index_daily"):
            self.db.insert_or_ignore(df, "index_daily")
        return df

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def get_trade_dates(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> list[str]:
        start_date = start_date or self.cfg.get("data_fetch.start_date")
        end_date = end_date or str(dt.date.today())
        return self._source.get_trade_dates(start_date, end_date)

    # ------------------------------------------------------------------
    # 直接查询
    # ------------------------------------------------------------------

    def query_daily(
        self,
        code: str,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> pd.DataFrame:
        """从本地数据库查询日线数据。"""
        code = normalize_stock_code(code)
        sql = "SELECT * FROM daily_bars WHERE code = ?"
        params: list[object] = [code]
        if start_date:
            sql += " AND date >= ?"
            params.append(str(ensure_date(start_date)))
        if end_date:
            sql += " AND date <= ?"
            params.append(str(ensure_date(end_date)))
        sql += " ORDER BY date"
        return self.db.query(sql, params)

    # ------------------------------------------------------------------
    # 财务数据查询
    # ------------------------------------------------------------------

    def get_financials(
        self,
        code: str | None = None,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """从本地数据库查询季频财务数据。

        Parameters
        ----------
        code : str, optional
            股票代码。省略时查询全部。
        start_date / end_date : str | date, optional
            report_date 范围过滤。
        fields : list[str], optional
            要返回的列。省略时返回全部列。

        Returns
        -------
        DataFrame
        """
        if not self.db.table_exists("financials"):
            logger.warning("financials 表不存在，请先运行 DataUpdater.update_financials()")
            return pd.DataFrame()

        cols = ", ".join(fields) if fields else "*"
        sql = f"SELECT {cols} FROM financials WHERE 1=1"
        params: list[object] = []

        if code:
            sql += " AND code = ?"
            params.append(normalize_stock_code(code))
        if start_date:
            sql += " AND report_date >= ?"
            params.append(str(ensure_date(start_date)))
        if end_date:
            sql += " AND report_date <= ?"
            params.append(str(ensure_date(end_date)))

        sql += " ORDER BY code, report_date"
        return self.db.query(sql, params)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_local_daily(self, code: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        try:
            if not self.db.table_exists("daily_bars"):
                return pd.DataFrame()
            return self.query_daily(code, start_date, end_date)
        except Exception:
            return pd.DataFrame()

    def _fetch_and_persist_daily(
        self,
        code: str,
        start_date: dt.date,
        end_date: dt.date,
        adjust: str,
    ) -> pd.DataFrame:
        """远程拉取 → 清洗 → 写回 daily_bars。"""
        df = self._source.get_daily_bars(code, start_date, end_date, adjust)
        if df.empty:
            return df
        df = self.cleaner.clean_pipeline(df)
        if self.db.table_exists("daily_bars"):
            self.db.insert_or_ignore(df, "daily_bars")
        return df
