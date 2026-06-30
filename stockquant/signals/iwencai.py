"""
iwencai NL语义搜索研报（需要 API Key）。

端点: openapi.iwencai.com (SkillHub 2.0, 需 X-Claw Headers)

环境变量:
    IWENCAI_API_KEY  — API Key (从 https://www.iwencai.com/skillhub 获取)
    IWENCAI_BASE_URL — 默认 "https://openapi.iwencai.com"
"""

from __future__ import annotations

import os
import secrets

import pandas as pd
import requests

from stockquant.signals._eastmoney import empty_df
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.iwencai")

IWENCAI_BASE = os.environ.get("IWENCAI_BASE_URL", "https://openapi.iwencai.com")
IWENCAI_KEY = os.environ.get("IWENCAI_API_KEY", "")

SEARCH_COLS = ("uid", "title", "publish_date", "score", "summary", "source")


def _claw_headers(call_type: str = "normal") -> dict:
    """SkillHub 2.0 必须的 X-Claw 鉴权头。"""
    return {
        "X-Claw-Call-Type": call_type,
        "X-Claw-Skill-Id": "report-search",
        "X-Claw-Skill-Version": "2.0.0",
        "X-Claw-Plugin-Id": "none",
        "X-Claw-Plugin-Version": "none",
        "X-Claw-Trace-Id": secrets.token_hex(32),
    }


def _check_key() -> bool:
    if not IWENCAI_KEY:
        logger.warning("IWENCAI_API_KEY 未设置，iwencai 功能不可用。"
                       "请从 https://www.iwencai.com/skillhub 获取 Key 后 "
                       "export IWENCAI_API_KEY=your_key")
        return False
    return True


def iwencai_search(query: str, channel: str = "report",
                   size: int = 50) -> pd.DataFrame:
    """iwencai 语义搜索研报/公告/新闻。

    Parameters
    ----------
    query : str
        搜索查询，如 ``"人形机器人 行星滚柱丝杠 2026"``。
    channel : str
        频道: ``"report"`` (研报) / ``"announcement"`` (公告) / ``"news"`` (新闻)。
    size : int
        返回条数，默认 50。

    Returns
    -------
    pd.DataFrame
        列: uid, title, publish_date, score, summary, source。
    """
    if not _check_key():
        return empty_df(SEARCH_COLS, ("publish_date",))

    headers = {
        "Authorization": f"Bearer {IWENCAI_KEY}",
        "Content-Type": "application/json",
        **_claw_headers(),
    }
    payload = {
        "channels": [channel],
        "app_id": "AIME_SKILL",
        "query": query,
        "size": size,
    }

    try:
        r = requests.post(
            f"{IWENCAI_BASE}/v1/comprehensive/search",
            json=payload, headers=headers, timeout=30,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"iwencai 搜索请求失败: {e}")
        return empty_df(SEARCH_COLS, ("publish_date",))
    except Exception:
        logger.exception("iwencai 搜索未预期错误")
        return empty_df(SEARCH_COLS, ("publish_date",))

    if r.status_code != 200:
        logger.warning(f"iwencai HTTP {r.status_code}: {r.text[:200]}")
        return empty_df(SEARCH_COLS, ("publish_date",))

    try:
        data = r.json()
    except ValueError as e:
        logger.warning(f"iwencai JSON 解析失败: {e}")
        return empty_df(SEARCH_COLS, ("publish_date",))

    if data.get("status_code", 0) != 0:
        logger.warning(f"iwencai API 错误: {data.get('status_msg', '')}")
        return empty_df(SEARCH_COLS, ("publish_date",))

    articles = data.get("data") or []
    if not articles:
        return empty_df(SEARCH_COLS, ("publish_date",))

    rows = []
    for a in articles:
        extra = a.get("extra") or {}
        rows.append({
            "uid": a.get("uid", ""),
            "title": a.get("title", ""),
            "publish_date": a.get("publish_date", ""),
            "score": a.get("score", ""),
            "summary": (a.get("content") or "")[:300],
            "source": extra.get("source", ""),
        })

    df = pd.DataFrame(rows)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    return df


def iwencai_query(query: str, page: int = 1, limit: int = 50) -> pd.DataFrame:
    """iwencai NL 数据查询（结构化字段，类 SQL）。

    例: ``"贵州茅台 ROE"`` → 结构化 DataFrame。

    Parameters
    ----------
    query : str
        NL 查询语句。
    page : int
        页码。
    limit : int
        每页条数。

    Returns
    -------
    pd.DataFrame
    """
    if not _check_key():
        return pd.DataFrame()

    headers = {
        "Authorization": f"Bearer {IWENCAI_KEY}",
        "Content-Type": "application/json",
        **_claw_headers(),
    }
    payload = {
        "query": query,
        "page": str(page),
        "limit": str(limit),
        "is_cache": "1",
        "expand_index": "true",
    }

    try:
        r = requests.post(
            f"{IWENCAI_BASE}/v1/query2data",
            json=payload, headers=headers, timeout=30,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"iwencai query 请求失败: {e}")
        return pd.DataFrame()
    except Exception:
        logger.exception("iwencai query 未预期错误")
        return pd.DataFrame()

    if r.status_code != 200:
        logger.warning(f"iwencai HTTP {r.status_code}: {r.text[:200]}")
        return pd.DataFrame()

    try:
        data = r.json()
    except ValueError:
        return pd.DataFrame()

    if data.get("status_code", 0) != 0:
        logger.warning(f"iwencai error: {data.get('status_msg', '')}")
        return pd.DataFrame()

    rows = data.get("datas") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def dedup_articles(articles: list[dict]) -> list[dict]:
    """对 iwencai 搜索结果去重 — 同一 uid 仅保留 score 最高的条目。

    Parameters
    ----------
    articles : list[dict]
        ``iwencai_search`` 返回的原始数据（DataFrame.to_dict("records")）。

    Returns
    -------
    list[dict]
        按 publish_date 降序排列的去重后列表。
    """
    best: dict[str, dict] = {}
    for a in articles:
        uid = a.get("uid", "") or f"{a.get('title', '')}|{a.get('publish_date', '')}"
        score = float(a.get("score", 0))
        if uid not in best or score > float(best[uid].get("score", 0)):
            best[uid] = a
    return sorted(best.values(), key=lambda x: str(x.get("publish_date", "")), reverse=True)
