"""
打板层 — 涨停池 / 炸板池 / 跌停池 / 昨涨停池 + 涨停原因 + 打板情绪。

端点:
  - push2ex.eastmoney.com   — 东财涨停板行情中心 (四池, 走 em_get 限流)
  - data.10jqka.com.cn      — 同花顺涨停揭秘 (涨停原因题材增强)
"""

from __future__ import annotations

import datetime as dt
import random
import time

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_get, UA
from stockquant.utils.helpers import ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("signals.limit_up")

ZTB_UT = "7eea3edcaed734bea9cbfc24409ed989"

ZT_COLS = (
    "code", "name", "price", "pct", "amount", "float_cap",
    "turnover", "limit_days", "first_seal", "last_seal",
    "seal_fund", "break_times", "industry", "zt_stat",
)
ZB_COLS = (
    "code", "name", "price", "limit_price", "pct", "turnover",
    "first_seal", "break_times", "amplitude", "speed",
    "industry", "zt_stat",
)
DT_COLS = (
    "code", "name", "price", "pct", "turnover", "pe",
    "seal_fund", "last_seal", "board_amount", "dt_days",
    "open_times", "industry",
)
YZT_COLS = (
    "code", "name", "price", "pct", "turnover", "amplitude",
    "speed", "y_first_seal", "y_limit_days", "industry", "zt_stat",
)
REASON_COLS = (
    "code", "name", "price", "pct", "reason", "board_type",
    "seal_rate", "break_times", "seal_amount", "high_days",
    "first_time", "is_again",
)

_THS_SESSION = requests.Session()
_THS_SESSION.headers.update({
    "User-Agent": UA,
})


def _fmt_zt_time(t: int | str) -> str:
    """涨停板时间整数 → HH:MM:SS（92500 → 09:25:00）。"""
    s = str(int(t)).zfill(6)
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"


def _em_zt_api(endpoint: str, sort: str, date_str: str) -> list[dict]:
    """东财涨停板行情中心通用请求（push2ex，走 em_get 限流）。"""
    url = f"https://push2ex.eastmoney.com/{endpoint}"
    params = {
        "ut": ZTB_UT, "dpt": "wz.ztzt", "Pageindex": 0,
        "pagesize": 10000, "sort": sort, "date": date_str,
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=10)
        return (r.json().get("data") or {}).get("pool") or []
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"涨停板池 {endpoint} 请求失败: {e}")
        return []
    except Exception:
        logger.exception(f"涨停板池 {endpoint} 未预期错误")
        return []


def get_limit_up_pool(
    date: str | dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """涨停池 — 当日涨停股票 + 连板数 + 封板资金 + 炸板次数。

    Parameters
    ----------
    date : str | date | datetime | None
        日期，格式 ``"YYYYMMDD"`` 或 ``"YYYY-MM-DD"``，默认今天。

    Returns
    -------
    pd.DataFrame
        列: code, name, price, pct, amount, float_cap, turnover,
        limit_days (连板数), first_seal, last_seal, seal_fund (封板资金/元),
        break_times (炸板次数), industry, zt_stat (N天M板)。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y%m%d")

    pool = _em_zt_api("getTopicZTPool", "fbt:asc", date_str)
    if not pool:
        return pd.DataFrame(columns=list(ZT_COLS))

    rows = []
    for p in pool:
        rows.append({
            "code": p["c"], "name": p["n"],
            "price": p["p"] / 1000,
            "pct": round(p["zdp"], 2),
            "amount": p["amount"], "float_cap": p["ltsz"],
            "turnover": round(p["hs"], 2), "limit_days": p["lbc"],
            "first_seal": _fmt_zt_time(p["fbt"]),
            "last_seal": _fmt_zt_time(p["lbt"]),
            "seal_fund": p["fund"], "break_times": p["zbc"],
            "industry": p.get("hybk", ""),
            "zt_stat": (
                f'{(p.get("zttj") or {}).get("days", "?")}天'
                f'{(p.get("zttj") or {}).get("ct", "?")}板'
            ),
        })
    return pd.DataFrame(rows)


def get_broken_board_pool(
    date: str | dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """炸板池 — 涨停后开板的股票。

    Parameters
    ----------
    date : str | date | datetime | None
        日期，默认今天。

    Returns
    -------
    pd.DataFrame
        列: code, name, price, limit_price, pct, turnover, first_seal,
        break_times, amplitude, speed, industry, zt_stat。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y%m%d")

    pool = _em_zt_api("getTopicZBPool", "fbt:asc", date_str)
    if not pool:
        return pd.DataFrame(columns=list(ZB_COLS))

    rows = []
    for p in pool:
        rows.append({
            "code": p["c"], "name": p["n"],
            "price": p["p"] / 1000,
            "limit_price": p["ztp"] / 1000,
            "pct": round(p["zdp"], 2),
            "turnover": round(p["hs"], 2),
            "first_seal": _fmt_zt_time(p["fbt"]),
            "break_times": p["zbc"],
            "amplitude": round(p["zf"], 2),
            "speed": round(p["zs"], 2),
            "industry": p.get("hybk", ""),
            "zt_stat": (
                f'{(p.get("zttj") or {}).get("days", "?")}天'
                f'{(p.get("zttj") or {}).get("ct", "?")}板'
            ),
        })
    return pd.DataFrame(rows)


def get_limit_down_pool(
    date: str | dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """跌停池 — 当日跌停股票。

    Parameters
    ----------
    date : str | date | datetime | None
        日期，默认今天。

    Returns
    -------
    pd.DataFrame
        列: code, name, price, pct, turnover, pe, seal_fund,
        last_seal, board_amount, dt_days (连续跌停), open_times, industry。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y%m%d")

    pool = _em_zt_api("getTopicDTPool", "fund:asc", date_str)
    if not pool:
        return pd.DataFrame(columns=list(DT_COLS))

    rows = []
    for p in pool:
        rows.append({
            "code": p["c"], "name": p["n"],
            "price": p["p"] / 1000,
            "pct": round(p["zdp"], 2),
            "turnover": round(p["hs"], 2), "pe": p.get("pe"),
            "seal_fund": p["fund"],
            "last_seal": _fmt_zt_time(p["lbt"]),
            "board_amount": p.get("fba"),
            "dt_days": p.get("days"),
            "open_times": p.get("oc"),
            "industry": p.get("hybk", ""),
        })
    return pd.DataFrame(rows)


def get_yesterday_limit_up_pool(
    date: str | dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """昨日涨停池 — 昨涨停股今日表现，可用于计算晋级率和赚钱效应。

    Parameters
    ----------
    date : str | date | datetime | None
        日期，默认今天。

    Returns
    -------
    pd.DataFrame
        列: code, name, price, pct (今日涨幅), turnover, amplitude,
        speed, y_first_seal (昨封板时间), y_limit_days (昨连板),
        industry, zt_stat。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y%m%d")

    pool = _em_zt_api("getYesterdayZTPool", "zs:desc", date_str)
    if not pool:
        return pd.DataFrame(columns=list(YZT_COLS))

    rows = []
    for p in pool:
        rows.append({
            "code": p["c"], "name": p["n"],
            "price": p["p"] / 1000,
            "pct": round(p["zdp"], 2),
            "turnover": round(p["hs"], 2),
            "amplitude": round(p["zf"], 2),
            "speed": round(p["zs"], 2),
            "y_first_seal": _fmt_zt_time(p["yfbt"]),
            "y_limit_days": p["ylbc"],
            "industry": p.get("hybk", ""),
            "zt_stat": (
                f'{(p.get("zttj") or {}).get("days", "?")}天'
                f'{(p.get("zttj") or {}).get("ct", "?")}板'
            ),
        })
    return pd.DataFrame(rows)


def get_limit_up_reasons(
    date: str | dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """同花顺涨停揭秘 — 涨停原因题材 + 封板成功率 + 板型。

    Parameters
    ----------
    date : str | date | datetime | None
        日期，格式 ``"YYYYMMDD"``，默认今天。

    Returns
    -------
    pd.DataFrame
        列: code, name, price, pct, reason (涨停原因题材),
        board_type (换手板/一字板/T字板), seal_rate (封板成功率),
        break_times (炸板次数), seal_amount (封单额/元),
        high_days (几天几板), first_time (首次涨停时间),
        is_again (是否回封)。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y%m%d")

    url = "https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool"
    params = {
        "page": 1, "limit": 200,
        "field": "199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004",
        "filter": "HS,GEM2STAR",
        "order_field": "330324", "order_type": "0",
        "date": date_str,
    }

    last_exc = None
    for attempt in range(3):
        try:
            r = _THS_SESSION.get(url, params=params, timeout=10)
            info = (r.json().get("data") or {}).get("info", [])
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.3))
        except (ValueError, KeyError) as e:
            logger.warning(f"涨停揭秘 JSON 解析失败: {e}")
            return pd.DataFrame(columns=list(REASON_COLS))
    else:
        logger.warning(f"涨停揭秘请求失败: {last_exc}")
        return pd.DataFrame(columns=list(REASON_COLS))

    rows = []
    for it in info:
        ft = it.get("first_limit_up_time")
        rows.append({
            "code": it.get("code"), "name": it.get("name"),
            "price": it.get("latest"), "pct": it.get("change_rate"),
            "reason": it.get("reason_type", ""),
            "board_type": it.get("limit_up_type", ""),
            "seal_rate": it.get("limit_up_suc_rate"),
            "break_times": it.get("open_num") or 0,
            "seal_amount": it.get("order_amount"),
            "high_days": it.get("high_days", ""),
            "first_time": (
                dt.datetime.fromtimestamp(int(ft)).strftime("%H:%M:%S")
                if ft else ""
            ),
            "is_again": it.get("is_again_limit"),
        })
    return pd.DataFrame(rows)


def get_limit_up_sentiment(
    date: str | dt.date | dt.datetime | None = None,
) -> dict:
    """打板情绪温度计 — 连板梯队 + 炸板率 + 涨跌停对比 + 晋级率。

    一次调用拉取涨停/炸板/跌停/昨涨停四池，速算关键情绪指标。

    Parameters
    ----------
    date : str | date | datetime | None
        日期，默认今天。

    Returns
    -------
    dict
        ``{date, zt_count, zb_count, dt_count, break_rate, max_height,
        ladder, promotion_rate}``
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y%m%d")

    zt = get_limit_up_pool(date_str)
    zb = get_broken_board_pool(date_str)
    dt_pool = get_limit_down_pool(date_str)
    yzt = get_yesterday_limit_up_pool(date_str)

    zt_n = len(zt)
    zb_n = len(zb)
    dt_n = len(dt_pool)

    # 连板梯队: {板数: 家数}
    ladder: dict[int, int] = {}
    if not zt.empty and "limit_days" in zt.columns:
        for days in zt["limit_days"]:
            d = int(days)
            ladder[d] = ladder.get(d, 0) + 1

    # 炸板率
    total_attempt = zt_n + zb_n
    break_rate = round(zb_n / total_attempt * 100, 1) if total_attempt else 0.0

    # 最高连板
    max_height = max(ladder.keys(), default=0)

    # 晋级率: 昨涨停今日继续涨停 / 昨涨停总数
    yzt_total = len(yzt)
    if yzt_total and not yzt.empty and "pct" in yzt.columns:
        promoted = (yzt["pct"] >= 9.8).sum()
    else:
        promoted = 0
    promotion_rate = round(promoted / yzt_total * 100, 1) if yzt_total else 0.0

    return {
        "date": date_str,
        "zt_count": zt_n,
        "zb_count": zb_n,
        "dt_count": dt_n,
        "break_rate": break_rate,
        "max_height": max_height,
        "ladder": dict(sorted(ladder.items())),
        "promotion_rate": promotion_rate,
    }
