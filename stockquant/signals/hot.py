"""
同花顺当日强势股 + 题材归因。

端点: zx.10jqka.com.cn (同花顺, 零鉴权, 不走东财限流)

返回字段: code, name, reason (题材归因标签), date, market。
价格/涨幅等行情数据需配合腾讯财经或 mootdx 补充。
"""

from __future__ import annotations

import datetime as dt
import random
import time

import pandas as pd
import requests

from stockquant.utils.helpers import ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("signals.hot")

HOT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Chrome/117.0.0.0 Safari/537.36"
    ),
}

_SESSION = requests.Session()
_SESSION.headers.update(HOT_HEADERS)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0

HOT_COLS = ("code", "name", "reason", "date", "market")


def get_hot_stocks(
    date: str | dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """获取同花顺当日强势股 + 题材归因。

    核心价值：同花顺编辑部人工运营的题材标签，如
    "算力租赁+Token工厂+AI政务"，适合题材归因和热点追踪。

    Note: 此端点仅返回代码/名称/题材/日期/市场，不含行情数据。
    如需价格/涨幅，可配合腾讯财经 ``tencent_quote()`` 补充。

    Parameters
    ----------
    date : str | date | datetime | None
        查询日期，格式 ``"YYYY-MM-DD"``，默认今天。

    Returns
    -------
    pd.DataFrame
        列: code, name, reason (题材归因), date, market (市场代码)。
        非交易日或请求失败时返回空 DataFrame。
    """
    trade_date = ensure_date(date) or dt.date.today()
    date_str = trade_date.strftime("%Y-%m-%d")

    url = (
        f"http://zx.10jqka.com.cn/event/api/getharden/"
        f"date/{date_str}/orderby/date/orderway/desc/charset/GBK/"
    )

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = _SESSION.get(url, timeout=10)
            data = r.json()
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                backoff = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"同花顺热点请求失败 (第{attempt + 1}次): {e}，"
                    f"{backoff:.1f}s 后重试..."
                )
                time.sleep(backoff + random.uniform(0, 0.3))
        except ValueError as e:
            logger.warning(f"同花顺热点 JSON 解析失败: {e}")
            return pd.DataFrame(columns=list(HOT_COLS))
    else:
        logger.warning(f"同花顺热点请求失败 (已重试{_MAX_RETRIES}次): {last_exc}")
        return pd.DataFrame(columns=list(HOT_COLS))

    if data.get("errocode", 0) != 0:
        logger.warning(f"同花顺热点 API 错误: {data.get('errormsg', '')}")
        return pd.DataFrame(columns=list(HOT_COLS))

    rows = data.get("data") or []
    if not rows:
        return pd.DataFrame(columns=list(HOT_COLS))

    df = pd.DataFrame(rows)
    keep_cols = [c for c in HOT_COLS if c in df.columns]
    return df[keep_cols]
