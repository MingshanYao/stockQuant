"""
BaoStock 数据源适配器。
"""

from __future__ import annotations

import datetime as dt

import baostock as bs
import pandas as pd

from stockquant.data.data_source import (
    BaseDataSource,
    DataSourceFactory,
    standardize_daily,
    standardize_index,
)
from stockquant.utils.helpers import normalize_stock_code, get_market_prefix, ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("data.baostock")


class BaoStockDataSource(BaseDataSource):
    """基于 BaoStock 的数据源实现。

    BaoStock 客户端为进程内全局单例，登录后保持连接复用；
    析构时自动 logout。
    """

    def __init__(self) -> None:
        self._logged_in = False

    def _ensure_login(self) -> None:
        if self._logged_in:
            return
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"BaoStock 登录失败: {lg.error_msg}")
            return
        self._logged_in = True

    def __del__(self) -> None:
        if getattr(self, "_logged_in", False):
            try:
                bs.logout()
            except Exception:
                pass

    @staticmethod
    def _bs_code(code: str) -> str:
        """6 位代码转 BaoStock 格式（sh.600000 / sz.000001）。"""
        code = normalize_stock_code(code)
        return f"{get_market_prefix(code)}.{code}"

    def _rs_to_df(self, rs) -> pd.DataFrame:
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            logger.warning(f"BaoStock 查询异常 [{rs.error_code}]: {rs.error_msg}")
            self._logged_in = False
        return pd.DataFrame(rows, columns=rs.fields)

    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        self._ensure_login()
        df = self._rs_to_df(bs.query_stock_basic())
        df = df[df["type"] == "1"]  # 1=股票
        df = df.rename(columns={"code_name": "name"})
        df["code"] = df["code"].str.split(".").str[1]
        return df[["code", "name"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        bs_code = self._bs_code(code)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        adjust_map = {"qfq": "2", "hfq": "1", "none": "3"}

        self._ensure_login()
        df = self._rs_to_df(bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=sd,
            end_date=ed,
            frequency="d",
            adjustflag=adjust_map.get(adjust, "1"),
        ))

        if df.empty:
            return df

        for col in ("open", "high", "low", "close", "volume", "amount", "turn", "pctChg"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return standardize_daily(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        bs_code = self._bs_code(code)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        self._ensure_login()
        df = self._rs_to_df(bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=sd,
            end_date=ed,
            frequency="d",
        ))

        if df.empty:
            return df

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return standardize_index(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_constituents(self, index_code: str) -> list[str]:
        """BaoStock 不提供通用指数成分股接口。"""
        logger.warning("BaoStock 数据源不支持 get_index_constituents，请使用 AkShare")
        raise NotImplementedError(
            "BaoStock 暂不支持 get_index_constituents，请切换 AkShare 数据源。"
        )

    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        bs_code = self._bs_code(code)

        self._ensure_login()
        return self._rs_to_df(bs.query_profit_data(code=bs_code, year=2024, quarter=4))

    # ------------------------------------------------------------------
    def get_stock_info(self) -> pd.DataFrame:
        """BaoStock 暂不支持批量获取股票基本信息。"""
        logger.warning("BaoStock 数据源不支持 get_stock_info，请使用 AkShare")
        raise NotImplementedError(
            "BaoStock 暂不支持 get_stock_info，请切换 AkShare 数据源。"
        )

    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        self._ensure_login()
        df = self._rs_to_df(bs.query_trade_dates(start_date=sd, end_date=ed))
        return df[df["is_trading_day"] == "1"]["calendar_date"].tolist()


# 注册到工厂
DataSourceFactory.register("baostock", BaoStockDataSource)
