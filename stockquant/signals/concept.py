"""
个股板块/概念归属。

端点: push2.eastmoney.com/api/qt/slist/get (东财, spt=3)
一次请求拿全个股所属行业 + 概念 + 地域板块，含 BK 码 + 涨跌幅 + 龙头股。
"""

from __future__ import annotations

import requests

from stockquant.signals._eastmoney import em_get, UA
from stockquant.utils.helpers import get_market_prefix, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.concept")

BOARD_COLS = ("name", "code", "change_pct", "lead_stock")


def get_concept_blocks(code: str) -> dict:
    """获取个股所属板块/概念归属（东财 slist，行业+概念+地域混合）。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。

    Returns
    -------
    dict
        ``{"total": int, "boards": list[dict], "tags": list[str]}``
        boards 每项: name (板块名), code (BK码), change_pct (涨跌幅),
        lead_stock (龙头股)。tags 是板块名的便捷列表。
    """
    code = normalize_stock_code(code)
    prefix = get_market_prefix(code)
    market_code = 1 if prefix == "sh" else 0

    params = {
        "fltt": "2", "invt": "2",
        "secid": f"{market_code}.{code}",
        "spt": "3", "pi": "0", "pz": "200", "po": "1",
        "fields": "f12,f14,f3,f128",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}

    try:
        r = em_get(
            "https://push2.eastmoney.com/api/qt/slist/get",
            params=params,
            headers=headers,
            timeout=15,
        )
        d = r.json()
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"板块归属请求失败 code={code}: {e}")
        return {"total": 0, "boards": [], "tags": []}
    except Exception:
        logger.exception(f"板块归属未预期错误 code={code}")
        return {"total": 0, "boards": [], "tags": []}

    diff = (d.get("data") or {}).get("diff") or {}
    items = diff.values() if isinstance(diff, dict) else diff

    boards = []
    for it in items:
        boards.append({
            "name": it.get("f14", ""),
            "code": it.get("f12", ""),
            "change_pct": it.get("f3", ""),
            "lead_stock": it.get("f128", ""),
        })

    return {
        "total": len(boards),
        "boards": boards,
        "tags": [b["name"] for b in boards],
    }
