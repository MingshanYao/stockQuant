"""
巨潮公告全文检索。

端点: cninfo.com.cn (巨潮资讯网, 零鉴权)
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._cninfo import _cninfo_orgid, _cninfo_ts_to_date
from stockquant.signals._eastmoney import UA, empty_df
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.announcement")

ANNOUNCE_COLS = ("title", "type", "date", "url")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": UA})


def get_announcements(code: str, page_size: int = 30) -> pd.DataFrame:
    """获取巨潮公告全文检索列表。

    Parameters
    ----------
    code : str
        6 位股票代码。
    page_size : int
        返回条数，默认 30。

    Returns
    -------
    pd.DataFrame
        列: title, type, date, url。
    """
    code = normalize_stock_code(code)
    org_id = _cninfo_orgid(code)

    payload = {
        "stock": f"{code},{org_id}",
        "tabName": "fulltext",
        "pageSize": str(page_size),
        "pageNum": "1",
        "column": "",
        "category": "",
        "plate": "",
        "seDate": "",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    headers = {
        "User-Agent": UA,
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.cninfo.com.cn/new/disclosure",
        "Origin": "https://www.cninfo.com.cn",
    }

    try:
        r = _SESSION.post(
            "https://www.cninfo.com.cn/new/hisAnnouncement/query",
            data=payload, headers=headers, timeout=15,
        )
        d = r.json()
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"巨潮公告请求失败 code={code}: {e}")
        return empty_df(ANNOUNCE_COLS, ("date",))
    except (ValueError, KeyError) as e:
        logger.warning(f"巨潮公告 JSON 解析失败 code={code}: {e}")
        return empty_df(ANNOUNCE_COLS, ("date",))
    except Exception:
        logger.exception(f"巨潮公告未预期错误 code={code}")
        return empty_df(ANNOUNCE_COLS, ("date",))

    rows = []
    for item in d.get("announcements", []) or []:
        rows.append({
            "title": item.get("announcementTitle", ""),
            "type": item.get("announcementTypeName", ""),
            "date": _cninfo_ts_to_date(item.get("announcementTime")),
            "url": (
                "https://www.cninfo.com.cn/new/disclosure/detail"
                f"?annoId={item.get('announcementId', '')}"
            ),
        })

    if not rows:
        return empty_df(ANNOUNCE_COLS, ("date",))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
