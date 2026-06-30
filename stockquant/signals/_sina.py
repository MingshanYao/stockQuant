"""
新浪财经数据 helper — GBK 编码 + 逗号分隔 + Referer header。

提供:
    sina_get(url) → requests.Response  (GBK decoded)
    sina_opt_list(param) → list[str]   (去 var hq_str_XXX="..." 壳)
"""

from __future__ import annotations

import requests

from stockquant.utils.logger import get_logger

logger = get_logger("signals._sina")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

SINA_REFERER = "https://stock.finance.sina.com.cn/"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": UA,
    "Referer": SINA_REFERER,
})


def sina_get(url: str, params: dict | None = None, timeout: int = 10) -> requests.Response:
    """新浪 HTTP GET — GBK 解码 + 自动带 Referer。

    Parameters
    ----------
    url : str
        请求 URL。
    params : dict | None
        URL 参数。
    timeout : int
        超时秒数。

    Returns
    -------
    requests.Response
        已 GBK 解码的 Response。
    """
    r = _SESSION.get(url, params=params, timeout=timeout)
    r.encoding = "gbk"
    return r


def sina_opt_list(param: str) -> list[str]:
    """
    新浪 hq.sinajs.cn 取值 — GBK，逗号分隔，去 ``var hq_str_XXX="..."`` 壳。

    Parameters
    ----------
    param : str
        新浪行情参数，如 ``"CON_OP_5100502506"``。

    Returns
    -------
    list[str]
        逗号分隔后的字段列表，请求失败返回空列表。
    """
    try:
        r = sina_get(
            f"https://hq.sinajs.cn/list={param}",
            timeout=10,
        )
        text = r.text
        if '"' in text:
            return text.split('"')[1].split(",")
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"新浪行情请求失败 param={param}: {e}")
    except Exception:
        logger.exception(f"新浪行情未预期错误 param={param}")
    return []
