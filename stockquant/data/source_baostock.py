"""
BaoStock 数据源适配器。
"""

from __future__ import annotations

import datetime as dt

import baostock as bs
import pandas as pd

from stockquant.data.data_source import BaseDataSource, DataSourceFactory
from stockquant.utils.helpers import normalize_stock_code, get_market_prefix, ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("data.baostock")


class BaoStockDataSource(BaseDataSource):
    """基于 BaoStock 的数据源实现。"""

    def __init__(self) -> None:
        self._logged_in = False

    def _login(self) -> None:
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                logger.error(f"BaoStock 登录失败: {lg.error_msg}")
            else:
                self._logged_in = True

    def _logout(self) -> None:
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        self._login()
        try:
            rs = bs.query_stock_basic()
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            df = df[df["type"] == "1"]  # 1=股票
            df = df.rename(columns={"code_name": "name"})
            df["code"] = df["code"].str.split(".").str[1]
            return df[["code", "name"]].reset_index(drop=True)
        finally:
            self._logout()

    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        bs_code = f"{prefix}.{code}"
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        adjust_map = {"qfq": "2", "hfq": "1", "none": "3"}

        self._login()
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=sd,
                end_date=ed,
                frequency="d",
                adjustflag=adjust_map.get(adjust, "1"),
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            self._logout()

        if df.empty:
            return df

        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={"turn": "turnover", "pctChg": "pct_change"})
        df["code"] = code
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        prefix = get_market_prefix(code)
        bs_code = f"{prefix}.{code}"
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        self._login()
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=sd,
                end_date=ed,
                frequency="d",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            self._logout()

        df["code"] = code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        code = normalize_stock_code(code)
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        bs_code = f"{prefix}.{code}"

        self._login()
        try:
            rs = bs.query_profit_data(code=bs_code, year=2024, quarter=4)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            self._logout()

        return df

    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        self._login()
        try:
            rs = bs.query_trade_dates(start_date=sd, end_date=ed)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            self._logout()

        trade_days = df[df["is_trading_day"] == "1"]["calendar_date"].tolist()
        return trade_days


# 注册到工厂
DataSourceFactory.register("baostock", BaoStockDataSource)
