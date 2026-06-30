"""
新闻层 — 东财个股新闻 + 东财全球资讯（7x24 快讯）。

端点:
  - search-api-web.eastmoney.com  — 个股新闻 JSONP (em_get 限流)
  - np-weblist.eastmoney.com      — 全球资讯快讯 (em_get 限流)
"""

from __future__ import annotations

import json
import re
import uuid

import pandas as pd

from stockquant.signals._eastmoney import em_get, empty_df
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.news")

STOCK_NEWS_COLS = ("title", "content", "time", "source", "url")
GLOBAL_NEWS_COLS = ("title", "summary", "time")


def get_stock_news(code: str, page_size: int = 20) -> pd.DataFrame:
    """获取东财个股相关新闻。

    Parameters
    ----------
    code : str
        6 位股票代码。
    page_size : int
        返回条数，默认 20。

    Returns
    -------
    pd.DataFrame
        列: title, content, time, source, url。
    """
    code = normalize_stock_code(code)
    cb = "jQuery_news"

    inner = json.dumps({
        "uid": "",
        "keyword": code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": page_size,
                "preTag": "",
                "postTag": "",
            },
        },
    }, separators=(',', ':'))

    params = {"cb": cb, "param": inner}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
        "Referer": "https://so.eastmoney.com/",
    }

    try:
        r = em_get(
            "https://search-api-web.eastmoney.com/search/jsonp",
            params=params, headers=headers, timeout=15,
        )
    except Exception:
        logger.exception(f"个股新闻请求失败 code={code}")
        return empty_df(STOCK_NEWS_COLS, ("time",))

    try:
        text = r.text
        json_str = text[text.index("(") + 1 : text.rindex(")")]
        d = json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"个股新闻 JSONP 解析失败 code={code}: {e}")
        return empty_df(STOCK_NEWS_COLS, ("time",))

    articles = d.get("result", {}).get("cmsArticleWebOld", []) or []
    if isinstance(articles, dict):
        articles = articles.get("list") or []

    rows = []
    for a in articles:
        rows.append({
            "title": re.sub(r'<[^>]+>', '', a.get("title", "")),
            "content": re.sub(r'<[^>]+>', '', a.get("content", ""))[:200],
            "time": a.get("date", ""),
            "source": a.get("mediaName", ""),
            "url": a.get("url", ""),
        })

    if not rows:
        return empty_df(STOCK_NEWS_COLS, ("time",))

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def get_global_news(page_size: int = 50) -> pd.DataFrame:
    """获取东财全球财经资讯（7x24 滚动快讯）。

    Parameters
    ----------
    page_size : int
        返回条数，默认 50。

    Returns
    -------
    pd.DataFrame
        列: title, summary, time。
    """
    url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
    params = {
        "client": "web",
        "biz": "web_724",
        "fastColumn": "102",
        "sortEnd": "",
        "pageSize": str(page_size),
        "req_trace": str(uuid.uuid4()),
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
        "Referer": "https://kuaixun.eastmoney.com/",
    }

    try:
        r = em_get(url, params=params, headers=headers, timeout=10)
        d = r.json()
    except Exception:
        logger.exception("全球资讯请求失败")
        return empty_df(GLOBAL_NEWS_COLS, ("time",))

    items = d.get("data", {}).get("fastNewsList") or []
    rows = []
    for item in items:
        rows.append({
            "title": item.get("title", ""),
            "summary": (item.get("summary") or "")[:200],
            "time": item.get("showTime", ""),
        })

    if not rows:
        return empty_df(GLOBAL_NEWS_COLS, ("time",))

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df
