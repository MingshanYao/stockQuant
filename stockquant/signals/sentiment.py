"""
舆情互动层 — 互动易问答 + 同花顺热榜 + 东财人气榜 + 概念命中。

端点:
  - irm.cninfo.com.cn           — 巨潮互动易（投资者提问 + 公司回复）
  - dq.10jqka.com.cn            — 同花顺热榜（人气值 + 概念标签）
  - emappdata.eastmoney.com     — 东财人气榜 + 概念命中 (em_get 限流)
  - push2.eastmoney.com         — 东财 ulist 批量补名称/价格 (em_get 限流)
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from stockquant.signals._eastmoney import UA, em_get
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.sentiment")

EM_HOT_BODY = {
    "appId": "appId01",
    "globalId": "786e4c21-70dc-435a-93bb-38",
}

IRM_COLS = ("code", "company", "question", "answer", "answerer", "ask_time")
THS_HOT_COLS = ("rank", "code", "name", "heat", "pct",
                "rank_chg", "concepts", "tag")
EM_HOT_COLS = ("rank", "code", "name", "price", "pct", "rank_chg")
EM_CONCEPT_COLS = ("concept", "bk", "hit")

_IRM_SESSION = requests.Session()
_IRM_SESSION.headers.update({"User-Agent": UA})


def get_irm_qa(
    code: str,
    page_size: int = 30,
    page_num: int = 1,
) -> pd.DataFrame:
    """互动易问答（巨潮 — 投资者提问 + 公司回复）。

    核心价值：能查到公司对特定传闻/利好/事件的官方回应，别处拿不到。

    Parameters
    ----------
    code : str
        6 位股票代码。
    page_size : int
        每页条数。
    page_num : int
        页码。

    Returns
    -------
    pd.DataFrame
        列: code, company, question, answer, answerer, ask_time。
        ``answer`` 为 None 表示尚未回复。
    """
    code = normalize_stock_code(code)

    try:
        r1 = _IRM_SESSION.post(
            "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo",
            data={"keyWord": code}, timeout=10,
        )
        d1 = r1.json().get("data") or []
        if not d1:
            return pd.DataFrame(columns=list(IRM_COLS))
        org_id = d1[0].get("secid")
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"互动易第一步请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(IRM_COLS))
    except Exception:
        logger.exception(f"互动易第一步未预期错误 code={code}")
        return pd.DataFrame(columns=list(IRM_COLS))

    # 第二步参数在 query string 里（POST body 空），否则 HTTP 400
    try:
        r2 = _IRM_SESSION.post(
            "https://irm.cninfo.com.cn/newircs/company/question",
            params={
                "_t": 1, "stockcode": code, "orgId": org_id,
                "pageSize": page_size, "pageNum": page_num,
                "keyWord": "", "startDay": "", "endDay": "",
            },
            timeout=10,
        )
        rows = r2.json().get("rows") or []
    except (requests.ConnectionError, requests.Timeout, ValueError) as e:
        logger.warning(f"互动易第二步请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(IRM_COLS))
    except Exception:
        logger.exception(f"互动易第二步未预期错误 code={code}")
        return pd.DataFrame(columns=list(IRM_COLS))

    if not rows:
        return pd.DataFrame(columns=list(IRM_COLS))

    out = []
    for it in rows:
        pd_ts = it.get("pubDate")
        out.append({
            "code": it.get("stockCode"),
            "company": it.get("companyShortName"),
            "question": it.get("mainContent"),
            "answer": it.get("attachedContent"),
            "answerer": it.get("attachedAuthor"),
            "ask_time": (
                dt.datetime.fromtimestamp(pd_ts / 1000).strftime("%Y-%m-%d %H:%M")
                if pd_ts else ""
            ),
        })
    return pd.DataFrame(out)


def get_ths_hot_list(period: str = "hour") -> pd.DataFrame:
    """同花顺热榜 — 人气值 + 概念标签 + 排名变化。

    Parameters
    ----------
    period : str
        ``"hour"`` 或 ``"day"``，默认 ``"hour"``。

    Returns
    -------
    pd.DataFrame
        列: rank, code, name, heat (人气值), pct (涨跌幅),
        rank_chg (排名变化), concepts (概念标签列表), tag (热度标签)。
    """
    try:
        r = requests.get(
            "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock",
            params={"stock_type": "a", "type": period, "list_type": "normal"},
            headers={"User-Agent": UA}, timeout=10,
        )
        lst = (r.json().get("data") or {}).get("stock_list") or []
    except Exception:
        logger.exception("同花顺热榜请求失败")
        return pd.DataFrame(columns=list(THS_HOT_COLS))

    if not lst:
        return pd.DataFrame(columns=list(THS_HOT_COLS))

    rows = []
    for it in lst:
        tag = it.get("tag") or {}
        rows.append({
            "rank": it.get("order"),
            "code": it.get("code"),
            "name": it.get("name"),
            "heat": it.get("rate"),
            "pct": it.get("rise_and_fall"),
            "rank_chg": it.get("hot_rank_chg"),
            "concepts": tag.get("concept_tag") or [],
            "tag": tag.get("popularity_tag", ""),
        })
    return pd.DataFrame(rows)


def get_em_hot_rank(top: int = 50) -> pd.DataFrame:
    """东财人气榜 — 排名 + 排名变化 + 名称/价格。

    注：东财人气榜只返回带前缀代码（SZ/SH），内部会通过 ulist.np
    批量补充名称和价格。

    Parameters
    ----------
    top : int
        返回 TOP N，默认 50。

    Returns
    -------
    pd.DataFrame
        列: rank, code, name, price, pct, rank_chg。
    """
    try:
        r = requests.post(
            "https://emappdata.eastmoney.com/stockrank/getAllCurrentList",
            json={**EM_HOT_BODY, "marketType": "", "pageNo": 1, "pageSize": top},
            headers={"User-Agent": UA}, timeout=10,
        )
        data = r.json().get("data") or []
    except Exception:
        logger.exception("东财人气榜请求失败")
        return pd.DataFrame(columns=list(EM_HOT_COLS))

    if not data:
        return pd.DataFrame(columns=list(EM_HOT_COLS))

    # 批量补名称/价格
    secids = [
        ("0." if it["sc"].startswith("SZ") else "1.") + it["sc"][2:]
        for it in data
    ]
    name_map = {}
    try:
        u = em_get(
            "https://push2.eastmoney.com/api/qt/ulist.np/get",
            params={
                "ut": "f057cbcbce2a86e2866ab8877db1d059",
                "fltt": 2, "invt": 2,
                "fields": "f14,f3,f12,f2",
                "secids": ",".join(secids),
            },
            headers={
                "User-Agent": UA,
                "Referer": "https://quote.eastmoney.com/",
            },
            timeout=10,
        )
        diff = (u.json().get("data") or {}).get("diff") or []
        if isinstance(diff, dict):
            diff = list(diff.values())
        for x in diff:
            name_map[x["f12"]] = (x.get("f14"), x.get("f2"), x.get("f3"))
    except Exception as e:
        logger.warning(f"东财人气榜 名称补充失败: {e}")

    rows = []
    for it in data:
        code = it["sc"][2:]
        name, price, pct = name_map.get(code, ("", None, None))
        rows.append({
            "rank": it["rk"],
            "code": code,
            "name": name,
            "price": price,
            "pct": pct,
            "rank_chg": it.get("hisRc"),
        })
    return pd.DataFrame(rows)


def get_em_hot_concept(code: str) -> pd.DataFrame:
    """东财个股热门概念命中 — 当前被市场归到哪些概念在炒。

    Parameters
    ----------
    code : str
        6 位股票代码。

    Returns
    -------
    pd.DataFrame
        列: concept, bk (概念ID), hit (命中热度)，按热度降序。
    """
    code = normalize_stock_code(code)
    prefix = "SH" if code.startswith("6") else "SZ"

    try:
        r = requests.post(
            "https://emappdata.eastmoney.com/stockrank/getHotStockRankList",
            json={**EM_HOT_BODY, "srcSecurityCode": prefix + code},
            headers={"User-Agent": UA}, timeout=10,
        )
        data = r.json().get("data") or []
    except Exception:
        logger.exception(f"东财概念命中失败 code={code}")
        return pd.DataFrame(columns=list(EM_CONCEPT_COLS))

    rows = []
    for x in data:
        rows.append({
            "concept": x.get("conceptName"),
            "bk": x.get("conceptId"),
            "hit": x.get("hitCount"),
        })
    return pd.DataFrame(rows)
