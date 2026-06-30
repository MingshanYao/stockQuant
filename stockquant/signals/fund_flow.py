"""
个股资金流向 — 分钟级（盘中实时）+ 日级（120 日历史）。

端点:
  - push2.eastmoney.com   — 分钟级实时资金流
  - push2his.eastmoney.com — 日级历史资金流
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import UA, em_get, empty_df
from stockquant.utils.helpers import get_market_prefix, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.fund_flow")

FUND_FLOW_COLS = ("date", "main_net", "small_net", "mid_net",
                  "large_net", "super_net")
FUND_FLOW_MINUTE_COLS = ("time", "main_net", "small_net", "mid_net",
                         "large_net", "super_net")


def get_fund_flow(code: str, days: int = 120) -> pd.DataFrame:
    """获取个股资金流向日级数据。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。
    days : int
        拉取最近多少个交易日，默认 120。

    Returns
    -------
    pd.DataFrame
        列: date, main_net, small_net, mid_net, large_net, super_net。
        金额单位: **元**。请求失败时返回带列名的空 DataFrame。
    """
    code = normalize_stock_code(code)
    prefix = get_market_prefix(code)
    market_code = 1 if prefix == "sh" else 0

    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": f"{market_code}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": str(days),
    }
    headers = {
        "User-Agent": UA,
        "Referer": "https://quote.eastmoney.com/",
        "Origin": "https://quote.eastmoney.com",
    }
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        d = r.json()
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"资金流向请求失败 code={code}: {e}")
        return _empty_fund_flow()
    except Exception:
        logger.exception(f"资金流向未预期错误 code={code}")
        return _empty_fund_flow()

    klines = (d.get("data") or {}).get("klines") or []
    if not klines:
        logger.info(f"资金流向返回空数据 code={code}（可能被东财间歇风控）")
        return _empty_fund_flow()

    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 7:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0.0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0.0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0.0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0.0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0.0,
            })

    if not rows:
        return _empty_fund_flow()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _empty_fund_flow() -> pd.DataFrame:
    df = pd.DataFrame(columns=list(FUND_FLOW_COLS))
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_fund_flow_minute(code: str) -> pd.DataFrame:
    """获取个股资金流向分钟级数据（当日盘中实时）。

    端点: push2.eastmoney.com，交易时段返回当日所有分钟 K 线，
    非交易时段返回空 DataFrame。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。

    Returns
    -------
    pd.DataFrame
        列: time, main_net, small_net, mid_net, large_net, super_net。
        金额单位: **元**。请求失败或非交易时段返回带正确 dtype 的空 DataFrame。
    """
    code = normalize_stock_code(code)
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"

    url = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
    params = {
        "secid": secid,
        "klt": 1,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    headers = {
        "User-Agent": UA,
        "Referer": "https://quote.eastmoney.com/",
        "Origin": "https://quote.eastmoney.com",
    }
    try:
        r = em_get(url, params=params, headers=headers, timeout=10)
        d = r.json()
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"资金流向分钟级请求失败 code={code}: {e}")
        return _empty_fund_flow_minute()
    except Exception:
        logger.exception(f"资金流向分钟级未预期错误 code={code}")
        return _empty_fund_flow_minute()

    klines = (d.get("data") or {}).get("klines") or []
    if not klines:
        return _empty_fund_flow_minute()

    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append({
                "time": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0.0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0.0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0.0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0.0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0.0,
            })

    if not rows:
        return _empty_fund_flow_minute()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


def _empty_fund_flow_minute() -> pd.DataFrame:
    return empty_df(FUND_FLOW_MINUTE_COLS, ("time",))
