"""
公共工具函数。
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence


def ensure_date(date_str: str | dt.date | dt.datetime | None) -> dt.date | None:
    """将日期字符串或 datetime 对象统一转为 date 对象。"""
    if date_str is None:
        return None
    if isinstance(date_str, dt.datetime):
        return date_str.date()
    if isinstance(date_str, dt.date):
        return date_str
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def normalize_stock_code(code: str) -> str:
    """将股票代码统一为纯数字 6 位格式。

    支持输入格式:
        - "600000"
        - "sh600000" / "sz000001"
        - "600000.SH" / "000001.SZ"
    """
    code = code.strip().upper()
    # 去除前缀
    for prefix in ("SH", "SZ", "BJ"):
        if code.startswith(prefix):
            code = code[len(prefix):]
            break
    # 去除后缀
    for suffix in (".SH", ".SZ", ".BJ"):
        if code.endswith(suffix):
            code = code[: -len(suffix)]
            break
    return code.zfill(6)


def get_market_prefix(code: str) -> str:
    """根据股票代码返回市场前缀 (sh / sz / bj)。"""
    code = normalize_stock_code(code)
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith(("0", "2", "3")):
        return "sz"
    elif code.startswith(("4", "8")):
        return "bj"
    return "sh"


def split_list(lst: Sequence, chunk_size: int) -> list[list]:
    """将列表按指定大小分块。"""
    return [list(lst[i: i + chunk_size]) for i in range(0, len(lst), chunk_size)]
