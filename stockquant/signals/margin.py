"""
融资融券日级明细。

端点: datacenter-web.eastmoney.com -> RPTA_WEB_RZRQ_GGMX
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_datacenter
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.margin")

MARGIN_COLS = ("date", "rzye", "rzmre", "rzche",
               "rqye", "rqmcl", "rqchl", "rzrqye")


def get_margin_trading(code: str, page_size: int = 30) -> pd.DataFrame:
    """获取个股融资融券日级明细。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。
    page_size : int
        返回最近多少条记录，默认 30。

    Returns
    -------
    pd.DataFrame
        列:
        - date       — 日期
        - rzye       — 融资余额 (元)
        - rzmre      — 融资买入额 (元)
        - rzche      — 融资偿还额 (元)
        - rqye       — 融券余额 (元)
        - rqmcl      — 融券卖出量 (股)
        - rqchl      — 融券偿还量 (股)
        - rzrqye     — 融资融券余额合计 (元)
    """
    code = normalize_stock_code(code)

    try:
        data = em_datacenter(
            "RPTA_WEB_RZRQ_GGMX",
            filter_str=f'(SCODE="{code}")',
            page_size=page_size,
            sort_columns="DATE",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"融资融券请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(MARGIN_COLS))
    except Exception:
        logger.exception(f"融资融券未预期错误 code={code}")
        return pd.DataFrame(columns=list(MARGIN_COLS))

    if not data:
        return pd.DataFrame(columns=list(MARGIN_COLS))

    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("DATE", ""))[:10],
            "rzye": row.get("RZYE") or 0,
            "rzmre": row.get("RZMRE") or 0,
            "rzche": row.get("RZCHE") or 0,
            "rqye": row.get("RQYE") or 0,
            "rqmcl": row.get("RQMCL") or 0,
            "rqchl": row.get("RQCHL") or 0,
            "rzrqye": row.get("RZRQYE") or 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
