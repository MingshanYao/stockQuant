"""
公共工具函数。
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Callable, Sequence, TypeVar

from stockquant.utils.logger import get_logger

logger = get_logger("utils.helpers")

T = TypeVar("T")


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


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    *,
    on_error: Callable[[Exception, int], None] | None = None,
    label: str = "调用",
) -> T:
    """带指数退避的重试。

    Parameters
    ----------
    fn : Callable[[], T]
        无参可调用对象。
    attempts : int
        最大尝试次数（含首次），默认 3。
    delay : float
        首次失败后的初始等待秒数。
    backoff : float
        每次失败后等待时间的倍数因子。
    on_error : Callable[[Exception, int], None], optional
        每次失败的钩子，参数为 ``(exc, attempt_number)``；默认仅 WARNING 日志。
    label : str
        日志中显示的任务标签。

    Returns
    -------
    T
        ``fn()`` 的返回值。

    Raises
    ------
    Exception
        最后一次尝试抛出的异常。
    """
    last_exc: Exception | None = None
    cur_delay = delay
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if on_error is not None:
                on_error(e, i)
            else:
                logger.warning(f"[{label}] 第 {i}/{attempts} 次失败: {e}")
            if i < attempts:
                time.sleep(cur_delay)
                cur_delay *= backoff
    assert last_exc is not None
    raise last_exc
