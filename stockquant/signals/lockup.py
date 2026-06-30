"""
限售解禁日历 — 历史解禁 + 未来待解禁。

端点: datacenter-web.eastmoney.com → RPT_LIFT_STAGE
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_datacenter, empty_df
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.lockup")

LOCKUP_COLS = ("date", "type", "shares", "ratio")


def get_lockup_expiry(
    code: str,
    date: str | dt.date | dt.datetime | None = None,
    forward_days: int = 90,
) -> dict:
    """获取限售解禁日历 — 历史解禁 + 未来待解禁。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"002475"`` / ``"sz002475"`` / ``"002475.SZ"``。
    date : str | date | datetime | None
        基准日期，默认今天。
    forward_days : int
        向前查询未来多少天，默认 90。

    Returns
    -------
    dict
        ``{"history": DataFrame, "upcoming": DataFrame}``
        无解禁记录时返回空 DataFrame。
    """
    code = normalize_stock_code(code)
    trade_date = ensure_date(date) or dt.date.today()

    # 1. 历史解禁
    history = empty_df(LOCKUP_COLS, ("date",))
    try:
        hist_data = em_datacenter(
            "RPT_LIFT_STAGE",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=15,
            sort_columns="FREE_DATE",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"历史解禁请求失败 code={code}: {e}")
        hist_data = []
    except Exception:
        logger.exception(f"历史解禁未预期错误 code={code}")
        hist_data = []

    if hist_data:
        rows = []
        for row in hist_data:
            rows.append({
                "date": str(row.get("FREE_DATE", ""))[:10],
                "type": row.get("LIMITED_STOCK_TYPE", ""),
                "shares": row.get("FREE_SHARES_NUM", 0),
                "ratio": row.get("FREE_RATIO", 0),
            })
        history = pd.DataFrame(rows)
        history["date"] = pd.to_datetime(history["date"])

    # 2. 未来待解禁
    upcoming = empty_df(LOCKUP_COLS, ("date",))
    end_date = trade_date + dt.timedelta(days=forward_days)
    date_str = trade_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        up_data = em_datacenter(
            "RPT_LIFT_STAGE",
            filter_str=(
                f'(SECURITY_CODE="{code}")'
                f"(FREE_DATE>='{date_str}')"
                f"(FREE_DATE<='{end_str}')"
            ),
            page_size=20,
            sort_columns="FREE_DATE",
            sort_types="1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"待解禁请求失败 code={code}: {e}")
        up_data = []
    except Exception:
        logger.exception(f"待解禁未预期错误 code={code}")
        up_data = []

    if up_data:
        rows = []
        for row in up_data:
            rows.append({
                "date": str(row.get("FREE_DATE", ""))[:10],
                "type": row.get("LIMITED_STOCK_TYPE", ""),
                "shares": row.get("FREE_SHARES_NUM", 0),
                "ratio": row.get("FREE_RATIO", 0),
            })
        upcoming = pd.DataFrame(rows)
        upcoming["date"] = pd.to_datetime(upcoming["date"])

    return {"history": history, "upcoming": upcoming}
