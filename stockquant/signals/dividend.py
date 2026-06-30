"""
分红送转历史。

端点: datacenter-web.eastmoney.com → RPT_SHAREBONUS_DET
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_datacenter
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.dividend")

DIVIDEND_COLS = (
    "date", "bonus_rmb", "transfer_ratio",
    "bonus_ratio", "plan",
)


def get_dividend_history(code: str, page_size: int = 20) -> pd.DataFrame:
    """获取个股分红送转历史。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。
    page_size : int
        返回最近多少期，默认 20。

    Returns
    -------
    pd.DataFrame
        列:
        - date            — 除权除息日
        - bonus_rmb       — 每股派息（税前，元）
        - transfer_ratio  — 每 10 股转增（股）
        - bonus_ratio     — 每 10 股送股（股）
        - plan            — 方案进度（实施/预案等）
    """
    code = normalize_stock_code(code)

    try:
        data = em_datacenter(
            "RPT_SHAREBONUS_DET",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=page_size,
            sort_columns="EX_DIVIDEND_DATE",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"分红送转请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(DIVIDEND_COLS))
    except Exception:
        logger.exception(f"分红送转未预期错误 code={code}")
        return pd.DataFrame(columns=list(DIVIDEND_COLS))

    if not data:
        return pd.DataFrame(columns=list(DIVIDEND_COLS))

    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("EX_DIVIDEND_DATE", ""))[:10],
            "bonus_rmb": row.get("PRETAX_BONUS_RMB", 0),
            "transfer_ratio": row.get("TRANSFER_RATIO", 0),
            "bonus_ratio": row.get("BONUS_RATIO", 0),
            "plan": row.get("ASSIGN_PROGRESS", ""),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
