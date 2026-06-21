"""
iTick 数据源适配器。

iTick 提供 REST API 查询股票/指数/期货/基金/外汇/加密货币的实时行情和历史K线。
不支持行业分类和财务数据（市值等），适合用做行情数据源。

API 文档: https://itick.org/docs/python
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd
import requests
import urllib3

from stockquant.data.data_source import (
    BaseDataSource,
    DataSourceFactory,
    standardize_daily,
    standardize_index,
)
from stockquant.utils.helpers import (
    normalize_stock_code, ensure_date,
    RateLimiter, call_with_retries,
)
from stockquant.utils.logger import get_logger

logger = get_logger("data.itick")

ITICK_BASE_URL = "https://api.itick.org"

# iTick K线周期 → 自然日数量映射
PERIOD_MAP = {
    "1d": 1,
    "1w": 2,
    "1m": 3,
}


class ITickDataSource(BaseDataSource):
    """基于 iTick 的数据源实现。

    iTick 的 ``get_stock_info`` 是单只股票查询，不支持批量获取行业/市值。
    因此 ``get_stock_info()`` 仅返回代码和名称（通过 symbol/list），
    行业和市值字段留空。
    """

    def __init__(self, token: str = "") -> None:
        self._token = token or os.environ.get("ITICK_TOKEN", "")
        self._session = requests.Session()
        self._session.verify = False
        urllib3.disable_warnings()
        # iTick: 5 req/s, burst=10 支持短时突发
        self._rate_limiter = RateLimiter(rate=5.0, burst=10)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        def _call() -> Any:
            self._throttle()
            url = f"{ITICK_BASE_URL}{path}"
            headers = {"accept": "application/json", "token": self._token}
            resp = self._session.get(url, params=params or {}, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(
                    f"iTick {path} 错误 [{data.get('code')}]: {data.get('msg')}"
                )
            return data.get("data")

        def _is_rate_limit(exc: Exception) -> bool:
            """检测 HTTP 429 限流错误。"""
            if hasattr(exc, "response") and getattr(exc.response, "status_code", 0) == 429:
                return True
            msg = str(exc).lower()
            return "429" in msg or "rate limit" in msg or "too many" in msg

        return call_with_retries(
            _call, attempts=3, delay=2.0, backoff=2.0,
            is_rate_limit=_is_rate_limit, rate_limit_wait=15.0,
            label=f"iTick {path}",
        )

    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        data = self._get("/symbol/list")
        rows = []
        # symbol/list 返回格式: 需要实际确认，暂按通用格式解析
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    code = item.get("code") or item.get("f_code") or ""
                    name = item.get("name") or item.get("f_name") or ""
                    rows.append({"code": str(code), "name": str(name)})
                elif isinstance(item, str):
                    rows.append({"code": item, "name": ""})
        elif isinstance(data, dict):
            items = data.get("items") or data.get("list") or []
            for item in items:
                if isinstance(item, dict):
                    rows.append({
                        "code": str(item.get("code", "")),
                        "name": str(item.get("name", "")),
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        region = "SH" if code.startswith(("6", "9")) else "SZ"
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)

        # iTick K线: period=1 (日线), limit 取日期跨度
        days = max((ed - sd).days + 1, 1)
        data = self._get("/stock/kline", {
            "region": region,
            "code": code,
            "period": 1,
            "limit": min(days, 5000),
            "end": str(ed),
        })

        if not data:
            return pd.DataFrame()

        rows = _parse_kline_data(data)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(sd)) & (df["date"] <= pd.Timestamp(ed))]
        if df.empty:
            return df
        return standardize_daily(df, code)

    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)
        days = max((ed - sd).days + 1, 1)

        region_map = {
            "000001": "SH", "000300": "SH", "000905": "SH",
            "399001": "SZ", "399005": "SZ", "399006": "SZ",
        }
        region = region_map.get(code, "SH")

        data = self._get("/indices/kline", {
            "region": region,
            "code": code,
            "period": 1,
            "limit": min(days, 5000),
            "end": str(ed),
        })

        if not data:
            return pd.DataFrame()

        rows = _parse_kline_data(data)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(sd)) & (df["date"] <= pd.Timestamp(ed))]
        if df.empty:
            return df
        return standardize_index(df, code)

    # ------------------------------------------------------------------
    def get_index_constituents(self, index_code: str) -> list[str]:
        logger.warning("iTick 数据源不支持 get_index_constituents")
        raise NotImplementedError(
            "iTick 暂不支持 get_index_constituents，请切换 AkShare 或 Tushare。"
        )

    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        logger.warning("iTick 数据源不支持 get_finance_data")
        raise NotImplementedError(
            "iTick 暂不支持 get_finance_data，请切换 AkShare 或 Tushare。"
        )

    # ------------------------------------------------------------------
    def get_stock_info(self) -> pd.DataFrame:
        """iTick 不提供行业/市值等基本面数据，仅返回代码和名称占位。

        行业和市值字段全部留空，待 Tushare 或 Sina 等其他数据源补充。
        """
        logger.warning(
            "iTick 不提供行业分类和市值数据，stock_info 仅填充 code/name，"
            "行业/市值请用 Tushare 或 Sina 数据源补充。"
        )
        stock_list = self.get_stock_list()
        if stock_list.empty:
            return pd.DataFrame()

        stock_list["industry"] = ""
        stock_list["sector"] = ""
        stock_list["market"] = ""
        stock_list["list_date"] = None
        stock_list["total_shares"] = None
        stock_list["float_shares"] = None
        stock_list["total_cap"] = None
        stock_list["float_cap"] = None
        return stock_list

    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        try:
            data = self._get("/symbol/holidays")
            holidays = set()
            if isinstance(data, list):
                holidays = {str(d) for d in data}
            elif isinstance(data, dict):
                holidays = {str(d) for d in (data.get("holidays") or [])}

            sd = ensure_date(start_date)
            ed = ensure_date(end_date)
            dates = pd.bdate_range(start=sd, end=ed)
            return [d.strftime("%Y-%m-%d") for d in dates
                    if d.strftime("%Y-%m-%d") not in holidays]
        except Exception:
            logger.warning("iTick 获取交易日历失败，使用 pandas 默认")
            sd = ensure_date(start_date)
            ed = ensure_date(end_date)
            dates = pd.bdate_range(start=sd, end=ed)
            return [d.strftime("%Y-%m-%d") for d in dates]


def _parse_kline_data(data: Any) -> list[dict]:
    """解析 iTick K线返回数据为统一行格式。"""
    rows = []
    items = data if isinstance(data, list) else data.get("items", [])
    for it in items:
        if isinstance(it, dict):
            rows.append({
                "date": it.get("date") or it.get("t") or it.get("time") or "",
                "open": float(it.get("open") or it.get("o") or 0),
                "high": float(it.get("high") or it.get("h") or 0),
                "low": float(it.get("low") or it.get("l") or 0),
                "close": float(it.get("close") or it.get("c") or 0),
                "volume": float(it.get("volume") or it.get("v") or 0),
                "amount": float(it.get("amount") or it.get("a") or 0),
            })
        elif isinstance(it, list):
            # 数组格式: [date, open, high, low, close, volume, amount]
            rows.append({
                "date": str(it[0]) if len(it) > 0 else "",
                "open": float(it[1]) if len(it) > 1 else 0,
                "high": float(it[2]) if len(it) > 2 else 0,
                "low": float(it[3]) if len(it) > 3 else 0,
                "close": float(it[4]) if len(it) > 4 else 0,
                "volume": float(it[5]) if len(it) > 5 else 0,
                "amount": float(it[6]) if len(it) > 6 else 0,
            })
    return rows


# 注册到工厂
DataSourceFactory.register("itick", ITickDataSource)
