"""
行业板块排名 — 全市场行业涨跌/涨跌家数/领涨股。

端点: push2.eastmoney.com/api/qt/clist/get (东财)
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_get, UA
from stockquant.utils.logger import get_logger

logger = get_logger("signals.industry")

INDUSTRY_COLS = (
    "rank", "name", "change_pct", "code",
    "up_count", "down_count", "leader", "leader_change",
)


def get_industry_ranking(top_n: int = 20) -> pd.DataFrame:
    """获取全市场行业板块涨跌幅排名。

    一次调用覆盖 ~100 个东财行业板块，适用于行业轮动分析。

    Parameters
    ----------
    top_n : int
        返回前 N 个和后 N 个行业，默认 20。

    Returns
    -------
    pd.DataFrame
        列: rank, name, change_pct, code (BK码), up_count (上涨家数),
        down_count (下跌家数), leader (领涨股), leader_change (领涨股涨幅)。
        按涨跌幅降序排列。
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "fltt": "2", "invt": "2",
        "fs": "m:90+t:2",
        "fields": "f2,f3,f4,f12,f13,f14,f104,f105,f128,f136,f140,f141,f207",
    }
    headers = {"User-Agent": UA}

    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        d = r.json()
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"行业排名请求失败: {e}")
        return pd.DataFrame(columns=list(INDUSTRY_COLS))
    except Exception:
        logger.exception("行业排名未预期错误")
        return pd.DataFrame(columns=list(INDUSTRY_COLS))

    items = d.get("data", {}).get("diff", [])
    if not items:
        return pd.DataFrame(columns=list(INDUSTRY_COLS))

    rows = []
    for i, item in enumerate(items):
        rows.append({
            "rank": i + 1,
            "name": item.get("f14", ""),
            "change_pct": item.get("f3", 0),
            "code": item.get("f12", ""),
            "up_count": item.get("f104", 0),
            "down_count": item.get("f105", 0),
            "leader": item.get("f140", ""),
            "leader_change": item.get("f136", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("change_pct", ascending=False)
        df["rank"] = range(1, len(df) + 1)
    return df
