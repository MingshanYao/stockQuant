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


# ======================================================================
# Token Bucket 限速器
# ======================================================================

class RateLimiter:
    """Token Bucket 限速器，线程安全。

    允许短时突发（桶内有攒的令牌时），长期平均速率严格受限。

    Parameters
    ----------
    rate : float
        令牌补充速率（个/秒）。
    burst : int
        桶容量（最大积攒令牌数），默认等于 rate 的整数值（最小 1）。

    Examples
    --------
    >>> limiter = RateLimiter(rate=1.0, burst=60)  # 60 次/分钟
    >>> for i in range(100):
    ...     limiter.acquire()  # 前 60 个瞬间通过，之后每秒放行 1 个
    ...     make_api_call()
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        self._rate = float(rate)
        self._burst = burst if burst is not None else max(int(rate), 1)
        self._tokens = float(self._burst)  # 初始满桶
        self._last_refill = time.monotonic()
        self._lock = __import__("threading").Lock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def burst(self) -> int:
        return self._burst

    def acquire(self) -> float:
        """获取一个令牌，阻塞直到可用。

        Returns
        -------
        float
            实际等待的秒数（0 表示立即可用）。
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # 需要等待的时间
            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
            self._last_refill += wait

        time.sleep(wait)
        return wait


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    *,
    on_error: Callable[[Exception, int], None] | None = None,
    is_rate_limit: Callable[[Exception], bool] | None = None,
    rate_limit_wait: float = 65.0,
    label: str = "调用",
) -> T:
    """带指数退避的重试，支持限流错误特殊处理。

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
    is_rate_limit : Callable[[Exception], bool], optional
        判断异常是否为限流错误的钩子。限流错误不按指数退避，
        而是等待 ``rate_limit_wait`` 秒后重试（默认 65s，超过 1 分钟窗口）。
    rate_limit_wait : float
        限流错误的等待秒数，默认 65。
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
        except NotImplementedError:
            raise
        except Exception as e:
            last_exc = e
            if on_error is not None:
                on_error(e, i)
            else:
                logger.warning(f"[{label}] 第 {i}/{attempts} 次失败: {e}")

            if i >= attempts:
                break

            if is_rate_limit is not None and is_rate_limit(e):
                logger.info(
                    f"[{label}] 检测到限流，等待 {rate_limit_wait:.0f}s 后重试"
                    f" ({i + 1}/{attempts})"
                )
                time.sleep(rate_limit_wait)
                cur_delay = delay  # 重置退避，限流等待已足够
            else:
                time.sleep(cur_delay)
                cur_delay *= backoff
    assert last_exc is not None
    raise last_exc
