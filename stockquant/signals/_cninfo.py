"""
巨潮公告 helper — orgId 动态映射 + 公告查询。

提供:
    _cninfo_orgid(code) → str      (查巨潮 orgId，模块级缓存)
    _cninfo_ts_to_date(ts) → str   (巨潮毫秒时间戳 → 日期字符串)
"""

from __future__ import annotations

from datetime import datetime

import requests

from stockquant.signals._eastmoney import UA
from stockquant.utils.logger import get_logger

logger = get_logger("signals._cninfo")

_ORGID_MAP: dict[str, str] = {}


def _cninfo_ts_to_date(ts) -> str:
    """巨潮 announcementTime 返回 Unix 毫秒整数 → 日期字符串。"""
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        except (OSError, ValueError):
            return str(ts)[:10] if ts else ""
    return str(ts)[:10] if ts else ""


def _cninfo_orgid(code: str) -> str:
    """查巨潮股票真实 orgId — 动态查官方映射表，模块级缓存。

    巨潮 orgId 并非统一 ``gssx0{code}`` 格式（如 601318→9900002221、
    601398→jjxt0000019），硬编码会导致大量股票返回 totalAnnouncement=0。
    优先动态查官方映射表，查不到再回退硬编码。
    """
    global _ORGID_MAP

    if not _ORGID_MAP:
        try:
            r = requests.get(
                "http://www.cninfo.com.cn/new/data/szse_stock.json",
                headers={"User-Agent": UA},
                timeout=15,
            )
            _ORGID_MAP = {
                s["code"]: s["orgId"]
                for s in r.json().get("stockList", [])
            }
        except Exception as e:
            logger.warning(f"巨潮 orgId 映射表拉取失败，回退硬编码: {e}")

    org = _ORGID_MAP.get(code)
    if org:
        return org

    # fallback：老格式（仅部分老股票适用）
    if code.startswith("6"):
        return f"gssh0{code}"
    elif code.startswith("8") or code.startswith("4"):
        return f"gsbj0{code}"
    return f"gssz0{code}"
