"""
股东户数变化（筹码集中度）。

端点: datacenter-web.eastmoney.com → RPT_HOLDERNUMLATEST
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_datacenter
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.holders")

HOLDER_COLS = ("date", "holder_num", "change_num",
               "change_ratio", "avg_shares")


def get_holder_changes(code: str, periods: int = 10) -> pd.DataFrame:
    """获取个股股东户数变化（季频，筹码集中度信号）。

    股东户数持续减少 → 筹码集中 → 主力吸筹信号。
    股东户数持续增加 → 筹码分散 → 主力出货信号。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。
    periods : int
        返回最近多少期（季度），默认 10。

    Returns
    -------
    pd.DataFrame
        列:
        - date          — 截止日期
        - holder_num    — 股东户数
        - change_num    — 较上期变化（户, 负值=减少=集中）
        - change_ratio  — 较上期变化比例 (%)
        - avg_shares    — 户均持股数（股）
    """
    code = normalize_stock_code(code)

    try:
        data = em_datacenter(
            "RPT_HOLDERNUMLATEST",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=periods,
            sort_columns="END_DATE",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"股东户数请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(HOLDER_COLS))
    except Exception:
        logger.exception(f"股东户数未预期错误 code={code}")
        return pd.DataFrame(columns=list(HOLDER_COLS))

    if not data:
        return pd.DataFrame(columns=list(HOLDER_COLS))

    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("END_DATE", ""))[:10],
            "holder_num": row.get("HOLDER_NUM") or 0,
            "change_num": row.get("HOLDER_NUM_CHANGE") or 0,
            "change_ratio": row.get("HOLDER_NUM_RATIO") or 0.0,
            "avg_shares": row.get("AVG_FREE_SHARES") or 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
