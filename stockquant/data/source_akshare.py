"""
AkShare 数据源适配器。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import akshare as ak
import pandas as pd

from stockquant.data.data_source import BaseDataSource, DataSourceFactory
from stockquant.utils.helpers import normalize_stock_code, get_market_prefix, ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("data.akshare")


class AkShareDataSource(BaseDataSource):
    """基于 AkShare 的数据源实现。"""

    # ------------------------------------------------------------------
    # 股票列表
    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        logger.info("通过 AkShare 获取 A 股股票列表")
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 日线行情
    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        start_str = str(ensure_date(start_date)).replace("-", "")
        end_str = str(ensure_date(end_date)).replace("-", "")

        logger.debug(f"AkShare 日线: {code} [{start_str} ~ {end_str}] adjust={adjust}")

        adjust_map = {"qfq": "qfq", "hfq": "hfq", "none": ""}
        ak_adjust = adjust_map.get(adjust, "qfq")

        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=ak_adjust,
            )
        except Exception as e:
            logger.error(f"获取 {code} 日线数据失败: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        df = self._standardize_daily(df, code)
        return df

    # ------------------------------------------------------------------
    # 指数日线
    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        start_str = str(ensure_date(start_date)).replace("-", "")
        end_str = str(ensure_date(end_date)).replace("-", "")

        logger.debug(f"AkShare 指数日线: {code}")
        try:
            df = ak.stock_zh_index_daily(symbol=f"{get_market_prefix(code)}{code}")
            df = df[(df["date"] >= start_str) & (df["date"] <= end_str)]
        except Exception as e:
            logger.error(f"获取指数 {code} 日线失败: {e}")
            return pd.DataFrame()

        return self._standardize_index(df, code)

    # ------------------------------------------------------------------
    # 基本财务
    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        code = normalize_stock_code(code)
        logger.debug(f"AkShare 财务数据: {code}")
        try:
            df = ak.stock_financial_abstract_ths(symbol=code)
        except Exception as e:
            logger.warning(f"获取 {code} 财务数据失败: {e}")
            return pd.DataFrame()
        return df

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        try:
            df = ak.tool_trade_date_hist_sina()
            dates = pd.to_datetime(df["trade_date"])
            sd = pd.Timestamp(ensure_date(start_date))
            ed = pd.Timestamp(ensure_date(end_date))
            filtered = dates[(dates >= sd) & (dates <= ed)]
            return [d.strftime("%Y-%m-%d") for d in filtered]
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

    # ------------------------------------------------------------------
    # 内部：列名标准化
    # ------------------------------------------------------------------
    @staticmethod
    def _standardize_daily(df: pd.DataFrame, code: str) -> pd.DataFrame:
        col_map = {
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
        df = df.rename(columns=col_map)
        df["code"] = code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        # 只保留数据库 daily_bars 表中定义的列，丢弃多余列（如 股票代码、振幅）
        standard_cols = [
            "code", "date", "open", "high", "low", "close",
            "volume", "amount", "turnover", "pct_change", "change",
        ]
        df = df[[c for c in standard_cols if c in df.columns]]
        return df

    @staticmethod
    def _standardize_index(df: pd.DataFrame, code: str) -> pd.DataFrame:
        df = df.copy()
        df["code"] = code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df


# 注册到工厂
DataSourceFactory.register("akshare", AkShareDataSource)
