"""
东财 HTTP 请求共享基础设施。

提供:
    em_get()          — 带串行限流 + 重试的 GET，所有 eastmoney.com 端点专用
    em_datacenter()   — datacenter-web 通用查询模板（龙虎榜/融资融券/股东户数等共用）

限流规则: 两次请求最小间隔 1.0 秒 + 随机抖动(0.1~0.5秒)，串行不并发。
所有东财端点必须走 em_get/em_datacenter，禁止裸 requests.get 直接打东财。
"""

from __future__ import annotations

import random
import time

import requests
from requests.adapters import HTTPAdapter

from stockquant.utils.logger import get_logger

logger = get_logger("signals._eastmoney")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# ── Session + Retry Adapter ──────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": UA})

try:
    from urllib3.util.retry import Retry

    _adapter = HTTPAdapter(max_retries=Retry(
        total=3,
        connect=2,       # DNS / TCP 握手失败重试
        read=2,          # RemoteDisconnected / ProtocolError 重试
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    ))
    _SESSION.mount("https://", _adapter)
    _SESSION.mount("http://", _adapter)
except Exception:
    pass  # urllib3 版本不兼容时降级，不影响主流程

# ── 限流 ──────────────────────────────────────────────────────────────────
EM_MIN_INTERVAL = 1.0       # 公开可调，批量场景可调大到 1.5~2s
_em_last_call = [0.0]       # 可变容器，跨模块共享

# ── 重试 ──────────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0        # 基础退避秒数，实际: 1s → 2s → 4s


def em_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 15,
    **kwargs,
) -> requests.Response:
    """东财统一 GET 请求 — 串行限流 + 会话复用 + 自动重试。

    所有 eastmoney.com 的 HTTP 请求必须走此函数，避免高频被封 IP。
    被封表现: 403 / 429 / 连接超时 / RemoteDisconnected。

    Raises
    ------
    requests.ConnectionError, requests.Timeout
        3 次重试全部失败时抛出最后一次异常。
    """
    # 串行限流
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _SESSION.get(url, params=params, headers=headers,
                                timeout=timeout, **kwargs)
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                backoff = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"东财请求失败 (第{attempt + 1}次): {e}，"
                    f"{backoff:.1f}s 后重试..."
                )
                time.sleep(backoff + random.uniform(0, 0.3))
        finally:
            _em_last_call[0] = time.time()

    assert last_exc is not None
    raise last_exc


def em_datacenter(
    report_name: str,
    *,
    filter_str: str = "",
    page_size: int = 50,
    sort_columns: str = "",
    sort_types: str = "-1",
) -> list[dict]:
    """东财 datacenter-web 通用查询。

    龙虎榜、融资融券、限售解禁、大宗交易、股东户数、分红送转等共用此端点。

    Parameters
    ----------
    report_name : str
        RPT 报表名称，如 ``"RPTA_WEB_RZRQ_GGMX"``。
    filter_str : str
        过滤条件，如 ``'(SCODE="600519")'``。
    page_size : int
        每页条数，上限 500。
    sort_columns : str
        排序字段。
    sort_types : str
        排序方向: ``"-1"`` 降序 / ``"1"`` 升序。

    Returns
    -------
    list[dict]
        数据行列表，无数据或出错时返回空列表。
    """
    params = {
        "reportName": report_name,
        "columns": "ALL",
        "filter": filter_str,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "source": "WEB",
        "client": "WEB",
    }
    try:
        r = em_get(
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            params=params,
            timeout=15,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"datacenter 请求失败 ({report_name}): {e}")
        return []

    if r.status_code != 200:
        logger.warning(
            f"datacenter 返回 HTTP {r.status_code} ({report_name}): "
            f"{r.text[:200]}"
        )
        return []

    try:
        d = r.json()
    except ValueError:
        logger.warning(
            f"datacenter JSON 解析失败 ({report_name}): {r.text[:200]}"
        )
        return []

    result = d.get("result")
    if result is None:
        return []
    return result.get("data") or []
