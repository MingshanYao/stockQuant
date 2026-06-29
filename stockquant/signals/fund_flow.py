"""
个股资金流向（日级）。

端点: push2his.eastmoney.com
"""

from __future__ import annotations

import pandas as pd

from stockquant.signals._eastmoney import em_get, UA
from stockquant.utils.logger import get_logger

logger = get_logger("signals.fund_flow")

FUND_FLOW_COLS = ["date", "main_net", "small_net", "mid_net",
                  "large_net", "super_net"]


def get_fund_flow(code: str, days: int = 120) -> pd.DataFrame:
    """获取个股资金流向日级数据。

    Parameters
    ----------
    code : str
        6 位股票代码，如 ``"600519"``。
    days : int
        拉取最近多少个交易日，默认 120。

    Returns
    -------
    pd.DataFrame
        列: date, main_net, small_net, mid_net, large_net, super_net。
        金额单位: **元**。空结果时返回带列名的空 DataFrame。
    """
    market_code = 1 if code.startswith("6") else 0
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
    except Exception as e:
        logger.warning(f"资金流向请求失败 code={code}: {e}")
        return pd.DataFrame(columns=FUND_FLOW_COLS)

    klines = (d.get("data") or {}).get("klines") or []
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
        return pd.DataFrame(columns=FUND_FLOW_COLS)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
