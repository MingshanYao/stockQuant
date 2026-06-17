"""
Tushare 数据源适配器。

Tushare 免费版 API 频率限制：
- stock_basic: 1 次/小时
- daily_basic: 500 次/天
- trade_cal: 不限
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd
import requests

from stockquant.data.data_source import (
    BaseDataSource,
    DataSourceFactory,
    standardize_daily,
    standardize_index,
)
from stockquant.utils.helpers import normalize_stock_code, ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("data.tushare")

TUSHARE_URL = "http://api.waditu.com"

# stock_info 表标准列
STOCK_INFO_COLS = [
    "code", "name", "industry", "sector", "market",
    "list_date", "total_shares", "float_shares", "total_cap", "float_cap",
]


class TushareDataSource(BaseDataSource):
    """基于 Tushare 的数据源实现。"""

    def __init__(self, token: str = "") -> None:
        self._token = token or os.environ.get("TUSHARE_TOKEN", "")
        self._session = requests.Session()

    def _call(self, api_name: str, params: dict[str, Any] | None = None,
              fields: str = "") -> list[list[Any]]:
        """调用 Tushare HTTP API，返回 items 列表。"""
        payload: dict[str, Any] = {
            "api_name": api_name,
            "token": self._token,
            "params": params or {},
        }
        if fields:
            payload["fields"] = fields

        resp = self._session.post(TUSHARE_URL, json=payload, timeout=30)
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Tushare {api_name} 错误: {data.get('msg')}")

        items = data.get("data", {}).get("items") or []
        flds = data["data"]["fields"]
        return items, flds

    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        items, fields = self._call("stock_basic",
                                   params={"exchange": "", "list_status": "L"},
                                   fields="ts_code,symbol,name")
        rows = []
        for it in items:
            d = dict(zip(fields, it))
            rows.append({"code": d["symbol"], "name": d["name"]})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        # Tushare 日线接口: daily (后复权)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        items, fields = self._call("daily",
                                   params={"ts_code": self._ts_code(code),
                                           "start_date": sd.replace("-", ""),
                                           "end_date": ed.replace("-", "")},
                                   fields="trade_date,open,high,low,close,"
                                          "vol,amount,pct_chg,change")
        if not items:
            return pd.DataFrame()

        rows = []
        for it in items:
            d = dict(zip(fields, it))
            rows.append({
                "date": d["trade_date"],
                "open": float(d.get("open") or 0),
                "high": float(d.get("high") or 0),
                "low": float(d.get("low") or 0),
                "close": float(d.get("close") or 0),
                "volume": float(d.get("vol") or 0),
                "amount": float(d.get("amount") or 0),
                "pct_change": float(d.get("pct_chg") or 0),
                "change": float(d.get("change") or 0),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return standardize_daily(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        ts_map = {"000001": "000001.SH", "399001": "399001.SZ",
                   "000300": "000300.SH", "000905": "000905.SH"}
        ts_code = ts_map.get(code, f"{code}.SH")

        items, fields = self._call("index_daily",
                                   params={"ts_code": ts_code,
                                           "start_date": sd.replace("-", ""),
                                           "end_date": ed.replace("-", "")},
                                   fields="trade_date,open,high,low,close,vol,amount")
        if not items:
            return pd.DataFrame()

        rows = []
        for it in items:
            d = dict(zip(fields, it))
            rows.append({
                "date": d["trade_date"],
                "open": float(d.get("open") or 0),
                "high": float(d.get("high") or 0),
                "low": float(d.get("low") or 0),
                "close": float(d.get("close") or 0),
                "volume": float(d.get("vol") or 0),
                "amount": float(d.get("amount") or 0),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return standardize_index(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_constituents(self, index_code: str) -> list[str]:
        ts_map = {"000905": "000905.SH", "000300": "000300.SH"}
        ts_code = ts_map.get(index_code, f"{index_code}.SH")

        items, fields = self._call("index_weight",
                                   params={"index_code": ts_code,
                                           "trade_date": "20151231"},
                                   fields="con_code")
        codes = list({d["con_code"].split(".")[0]
                      for d in (dict(zip(fields, it)) for it in items)})
        return sorted(codes)

    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        logger.warning("Tushare 数据源暂不支持 get_finance_data")
        raise NotImplementedError("Tushare 暂不支持 get_finance_data，请切换 AkShare。")

    # ------------------------------------------------------------------
    def get_stock_info(self) -> pd.DataFrame:
        """拉取全市场股票基本信息（行业分类 + 市值）。

        分两步：
        1. stock_basic — 获取 code/name/industry/area/market/list_date
        2. daily_basic — 获取 total_mv/circ_mv（取最近一个交易日）
        """
        # Step 1: 股票基本信息
        logger.info("拉取 stock_basic（行业分类）...")
        items, fields = self._call("stock_basic",
                                   params={"exchange": "", "list_status": "L"},
                                   fields="ts_code,symbol,name,area,industry,"
                                          "market,list_date")
        info = {}
        for it in items:
            d = dict(zip(fields, it))
            info[d["symbol"]] = {
                "code": d["symbol"],
                "name": d["name"],
                "industry": d.get("industry") or "",
                "sector": d.get("area") or "",
                "market": d.get("market") or "",
                "list_date": d.get("list_date") or "",
            }
        logger.info(f"  获取 {len(info)} 只股票基本信息")

        # Step 2: 市值数据 — 取最近一个交易日的 circ_mv
        # daily_basic 免费版不支持 trade_date='' 取最新，用一个近期日期
        logger.info("拉取 daily_basic（市值数据）...")
        try:
            items_mv, fields_mv = self._call("daily_basic",
                                             params={"trade_date": "20250102"},
                                             fields="ts_code,total_mv,circ_mv")
            for it in items_mv:
                d = dict(zip(fields_mv, it))
                sym = d["ts_code"].split(".")[0]
                if sym in info:
                    info[sym]["total_cap"] = float(d.get("total_mv") or 0) * 1e4  # 万→元
                    info[sym]["float_cap"] = float(d.get("circ_mv") or 0) * 1e4
            logger.info(f"  市值覆盖: {sum(1 for v in info.values() if 'total_cap' in v)}/{len(info)}")
        except Exception as e:
            logger.warning(f"  市值拉取失败: {e}")

        # 构建 DataFrame
        rows = []
        for v in info.values():
            rows.append({
                "code": v["code"],
                "name": v["name"],
                "industry": v.get("industry", ""),
                "sector": v.get("sector", ""),
                "market": v.get("market", ""),
                "list_date": v.get("list_date", ""),
                "total_shares": None,
                "float_shares": None,
                "total_cap": v.get("total_cap"),
                "float_cap": v.get("float_cap"),
            })

        return pd.DataFrame(rows, columns=STOCK_INFO_COLS)

    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        items, fields = self._call("trade_cal",
                                   params={"exchange": "SSE",
                                           "start_date": sd.replace("-", ""),
                                           "end_date": ed.replace("-", ""),
                                           "is_open": "1"},
                                   fields="cal_date")
        return [dict(zip(fields, it))["cal_date"] for it in items]

    # ------------------------------------------------------------------
    @staticmethod
    def _ts_code(code: str) -> str:
        code = normalize_stock_code(code)
        suffix = "SH" if code.startswith(("6", "9")) else "SZ"
        return f"{code}.{suffix}"


# 注册到工厂
DataSourceFactory.register("tushare", TushareDataSource)
