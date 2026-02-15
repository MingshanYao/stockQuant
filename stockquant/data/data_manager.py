"""
数据管理器 — 统一的数据获取 / 存储 / 更新入口。
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import pandas as pd

from stockquant.data.data_source import DataSourceFactory, BaseDataSource
from stockquant.data.database import Database
from stockquant.data.data_cleaner import DataCleaner
from stockquant.utils.config import Config
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("data.manager")


class DataManager:
    """高层数据管理器：采集 → 清洗 → 存储 → 查询。

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

        # 加载数据源适配器（导入时自动注册）
        import stockquant.data.source_akshare  # noqa: F401
        import stockquant.data.source_baostock  # noqa: F401

        primary = self.cfg.get("data_source.primary", "akshare")
        self._source: BaseDataSource = DataSourceFactory.create(primary)
        logger.info(f"数据管理器初始化完成，主数据源: {primary}")

    # ------------------------------------------------------------------
    # 股票列表
    # ------------------------------------------------------------------

    def update_stock_list(self) -> pd.DataFrame:
        """更新 A 股股票列表到本地数据库。"""
        df = self._source.get_stock_list()
        if not df.empty:
            self.db.save_dataframe(df, "stock_info", if_exists="replace")
            logger.info(f"已更新股票列表: {len(df)} 只")
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """获取本地股票列表。"""
        if self.db.table_exists("stock_info"):
            df = self.db.query("SELECT code, name FROM stock_info")
            if not df.empty:
                return df
        return self.update_stock_list()

    # ------------------------------------------------------------------
    # 日线数据
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        code: str,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> pd.DataFrame:
        """获取日线数据（优先本地，缺失则远程拉取）。

        Parameters
        ----------
        code : str
            股票代码。
        start_date / end_date : str | date, optional
            起止日期，默认使用配置值。
        adjust : str, optional
            复权方式，默认使用配置值。
        """
        code = normalize_stock_code(code)
        start_date = ensure_date(start_date or self.cfg.get("data_fetch.start_date", "2020-01-01"))
        end_date = ensure_date(end_date) or dt.date.today()
        adjust = adjust or self.cfg.get("data_fetch.adjust", "qfq")

        # 尝试从本地读取
        local_df = self._load_local_daily(code, start_date, end_date)
        if not local_df.empty:
            # 检查是否需要增量更新
            latest = local_df["date"].max().date() if hasattr(local_df["date"].max(), "date") else local_df["date"].max()
            if latest >= end_date:
                return local_df
            # 增量拉取
            next_day = latest + dt.timedelta(days=1)
            new_df = self._fetch_remote_daily(code, next_day, end_date, adjust)
            if not new_df.empty:
                return pd.concat([local_df, new_df], ignore_index=True)
            return local_df

        # 全量拉取
        return self._fetch_remote_daily(code, start_date, end_date, adjust)

    def batch_fetch_daily(
        self,
        codes: Sequence[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """批量获取多只股票的日线数据。"""
        result = {}
        total = len(codes)
        for i, code in enumerate(codes, 1):
            logger.info(f"[{i}/{total}] 拉取 {code}")
            try:
                df = self.fetch_daily(code, start_date, end_date, adjust)
                result[normalize_stock_code(code)] = df
            except Exception as e:
                logger.error(f"拉取 {code} 失败: {e}")
        return result

    # ------------------------------------------------------------------
    # 指数数据
    # ------------------------------------------------------------------

    def fetch_index_daily(
        self,
        code: str = "000300",
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> pd.DataFrame:
        """获取指数日线数据。"""
        code = normalize_stock_code(code)
        start_date = ensure_date(start_date or self.cfg.get("data_fetch.start_date"))
        end_date = ensure_date(end_date) or dt.date.today()

        df = self._source.get_index_daily(code, start_date, end_date)
        if not df.empty:
            self.db.save_dataframe(df, "index_daily")
        return df

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def get_trade_dates(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> list[str]:
        """获取交易日历。"""
        start_date = start_date or self.cfg.get("data_fetch.start_date")
        end_date = end_date or str(dt.date.today())
        return self._source.get_trade_dates(start_date, end_date)

    # ------------------------------------------------------------------
    # 数据查询
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
        params = [code]
        if start_date:
            sql += " AND date >= ?"
            params.append(str(ensure_date(start_date)))
        if end_date:
            sql += " AND date <= ?"
            params.append(str(ensure_date(end_date)))
        sql += " ORDER BY date"
        return self.db.query(sql, params)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_local_daily(self, code: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        try:
            return self.query_daily(code, start_date, end_date)
        except Exception:
            return pd.DataFrame()

    def _fetch_remote_daily(
        self,
        code: str,
        start_date: dt.date,
        end_date: dt.date,
        adjust: str,
    ) -> pd.DataFrame:
        df = self._source.get_daily_bars(code, start_date, end_date, adjust)
        if df.empty:
            return df
        df = self.cleaner.clean_pipeline(df)
        self.db.save_dataframe(df, "daily_bars")
        return df
