"""
龙虎榜数据 — 个股上榜记录 + 买卖席位 + 全市场榜单。

端点: datacenter-web.eastmoney.com
  - RPT_DAILYBILLBOARD_DETAILSNEW   — 上榜记录
  - RPT_BILLBOARD_DAILYDETAILSBUY   — 买入席位
  - RPT_BILLBOARD_DAILYDETAILSSELL  — 卖出席位
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_datacenter, empty_df
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.dragon_tiger")

RECORD_COLS = ("date", "reason", "net_buy_wan", "turnover_pct")
SEAT_COLS = ("name", "buy_amt_wan", "sell_amt_wan", "net_wan")
DAILY_COLS = (
    "code", "name", "reason", "close", "change_pct",
    "net_buy_wan", "buy_wan", "sell_wan", "turnover_pct",
)


def get_dragon_tiger_board(
    code: str,
    date: str | dt.date | dt.datetime | None = None,
    look_back: int = 30,
) -> dict:
    """获取个股龙虎榜上榜记录和买卖席位。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"002475"`` / ``"sz002475"`` / ``"002475.SZ"``。
    date : str | date | datetime | None
        查询截止日期，默认今天。
    look_back : int
        往回看多少天，默认 30。

    Returns
    -------
    dict
        ``{"records": DataFrame, "seats": {"buy": list, "sell": list},
        "institution": {"buy_wan": float, "sell_wan": float, "net_wan": float}}``
        无数据时 records 为空 DataFrame，seats 为空列表。
    """
    code = normalize_stock_code(code)
    trade_date = ensure_date(date) or dt.date.today()
    start = trade_date - dt.timedelta(days=look_back)
    start_str = start.strftime("%Y-%m-%d")
    end_str = trade_date.strftime("%Y-%m-%d")

    # 1. 上榜记录
    try:
        record_data = em_datacenter(
            "RPT_DAILYBILLBOARD_DETAILSNEW",
            filter_str=(
                f"(TRADE_DATE>='{start_str}')"
                f"(TRADE_DATE<='{end_str}')"
                f'(SECURITY_CODE="{code}")'
            ),
            page_size=50,
            sort_columns="TRADE_DATE",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"龙虎榜请求失败 code={code}: {e}")
        return _empty_board_result()
    except Exception:
        logger.exception(f"龙虎榜未预期错误 code={code}")
        return _empty_board_result()

    records = []
    for row in record_data:
        records.append({
            "date": str(row.get("TRADE_DATE", ""))[:10],
            "reason": row.get("EXPLANATION", ""),
            "net_buy_wan": round((row.get("BILLBOARD_NET_AMT") or 0) / 10000, 1),
            "turnover_pct": round(float(row.get("TURNOVERRATE") or 0), 2),
        })

    # 2. 最近上榜日的买卖席位
    seats: dict = {"buy": [], "sell": []}
    institution = {"buy_wan": 0.0, "sell_wan": 0.0, "net_wan": 0.0}

    if records:
        latest_date = records[0]["date"]
        buy_data, sell_data = _fetch_seat_details(code, latest_date)

        for row in buy_data[:5]:
            seats["buy"].append({
                "name": row.get("OPERATEDEPT_NAME", ""),
                "buy_amt_wan": round((row.get("BUY") or 0) / 10000, 1),
                "sell_amt_wan": round((row.get("SELL") or 0) / 10000, 1),
                "net_wan": round((row.get("NET") or 0) / 10000, 1),
            })
        for row in sell_data[:5]:
            seats["sell"].append({
                "name": row.get("OPERATEDEPT_NAME", ""),
                "buy_amt_wan": round((row.get("BUY") or 0) / 10000, 1),
                "sell_amt_wan": round((row.get("SELL") or 0) / 10000, 1),
                "net_wan": round((row.get("NET") or 0) / 10000, 1),
            })

        # 3. 机构买卖统计（OPERATEDEPT_CODE="0" 为机构专用席位）
        for row in buy_data:
            if str(row.get("OPERATEDEPT_CODE", "")) == "0":
                institution["buy_wan"] += (row.get("BUY") or 0) / 10000
        for row in sell_data:
            if str(row.get("OPERATEDEPT_CODE", "")) == "0":
                institution["sell_wan"] += (row.get("SELL") or 0) / 10000
        institution["buy_wan"] = round(institution["buy_wan"], 1)
        institution["sell_wan"] = round(institution["sell_wan"], 1)
        institution["net_wan"] = round(
            institution["buy_wan"] - institution["sell_wan"], 1
        )

    df = pd.DataFrame(records, columns=list(RECORD_COLS)) if records else \
        empty_df(RECORD_COLS, ("date",))
    return {"records": df, "seats": seats, "institution": institution}


def get_daily_dragon_tiger(
    date: str | dt.date | dt.datetime | None = None,
    min_net_buy: float | None = None,
) -> pd.DataFrame:
    """获取全市场龙虎榜 — 当日所有上榜股票。

    Parameters
    ----------
    date : str | date | datetime | None
        查询日期，默认今天。
    min_net_buy : float | None
        净买入下限（万元），None 不过滤。

    Returns
    -------
    pd.DataFrame
        列: code, name, reason, close, change_pct, net_buy_wan,
        buy_wan, sell_wan, turnover_pct。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y-%m-%d")

    try:
        data = em_datacenter(
            "RPT_DAILYBILLBOARD_DETAILSNEW",
            filter_str=f"(TRADE_DATE>='{date_str}')(TRADE_DATE<='{date_str}')",
            page_size=500,
            sort_columns="BILLBOARD_NET_AMT",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"全市场龙虎榜请求失败: {e}")
        return pd.DataFrame(columns=list(DAILY_COLS))
    except Exception:
        logger.exception("全市场龙虎榜未预期错误")
        return pd.DataFrame(columns=list(DAILY_COLS))

    if not data:
        return pd.DataFrame(columns=list(DAILY_COLS))

    rows = []
    for row in data:
        net_buy = (row.get("BILLBOARD_NET_AMT") or 0) / 10000
        if min_net_buy is not None and net_buy < min_net_buy:
            continue
        rows.append({
            "code": row.get("SECURITY_CODE", ""),
            "name": row.get("SECURITY_NAME_ABBR", ""),
            "reason": row.get("EXPLANATION", ""),
            "close": row.get("CLOSE_PRICE") or 0,
            "change_pct": round(float(row.get("CHANGE_RATE") or 0), 2),
            "net_buy_wan": round(net_buy, 1),
            "buy_wan": round((row.get("BILLBOARD_BUY_AMT") or 0) / 10000, 1),
            "sell_wan": round((row.get("BILLBOARD_SELL_AMT") or 0) / 10000, 1),
            "turnover_pct": round(float(row.get("TURNOVERRATE") or 0), 2),
        })

    if not rows:
        return pd.DataFrame(columns=list(DAILY_COLS))
    return pd.DataFrame(rows)


def _fetch_seat_details(code: str, trade_date: str) -> tuple[list, list]:
    """获取某日买卖席位明细。"""
    buy_data: list = []
    sell_data: list = []
    try:
        buy_data = em_datacenter(
            "RPT_BILLBOARD_DAILYDETAILSBUY",
            filter_str=f"(TRADE_DATE='{trade_date}')(SECURITY_CODE=\"{code}\")",
            page_size=10,
            sort_columns="BUY",
            sort_types="-1",
        )
    except Exception as e:
        logger.warning(f"买入席位查询失败: {e}")

    try:
        sell_data = em_datacenter(
            "RPT_BILLBOARD_DAILYDETAILSSELL",
            filter_str=f"(TRADE_DATE='{trade_date}')(SECURITY_CODE=\"{code}\")",
            page_size=10,
            sort_columns="SELL",
            sort_types="-1",
        )
    except Exception as e:
        logger.warning(f"卖出席位查询失败: {e}")

    return buy_data, sell_data


def _empty_board_result() -> dict:
    return {
        "records": empty_df(RECORD_COLS, ("date",)),
        "seats": {"buy": [], "sell": []},
        "institution": {"buy_wan": 0.0, "sell_wan": 0.0, "net_wan": 0.0},
    }
