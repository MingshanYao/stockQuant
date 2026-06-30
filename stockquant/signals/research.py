"""
研报层 — 东财个股/行业研报 + PDF下载 + 同花顺一致预期EPS。

端点:
  - reportapi.eastmoney.com  — 东财研报列表 (em_get 限流)
  - pdf.dfcfw.com             — 东财研报 PDF 下载 (em_get 限流)
  - basic.10jqka.com.cn       — 同花顺一致预期EPS (GBK HTML)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from stockquant.signals._eastmoney import UA, em_get, empty_df
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.research")

REPORT_API = "https://reportapi.eastmoney.com/report/list"
PDF_TPL = "https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"

REPORT_COLS = (
    "title", "publish_date", "org_name", "info_code",
    "eps_cur", "eps_next", "eps_next2", "rating",
    "industry", "attach_pages",
)
IND_REPORT_COLS = (
    "title", "publish_date", "org_name", "info_code",
    "industry_name", "industry_code", "rating",
    "report_type", "attach_pages", "attach_size",
)
EPS_COLS = ("year", "institution_count", "eps_min", "eps_mean", "eps_max")

_THS_SESSION = requests.Session()
_THS_SESSION.headers.update({"User-Agent": UA})


def get_research_reports(
    code: str,
    max_pages: int = 5,
    begin: str = "2024-01-01",
) -> pd.DataFrame:
    """获取个股研报列表（东财 reportapi）。

    Parameters
    ----------
    code : str
        6 位股票代码。
    max_pages : int
        最多翻页数，每页 100 条。
    begin : str
        起始日期 ``"YYYY-MM-DD"``。

    Returns
    -------
    pd.DataFrame
        列: title, publish_date, org_name, info_code, eps_cur,
        eps_next, eps_next2, rating, industry, attach_pages。
    """
    code = normalize_stock_code(code)
    all_records = []
    end = "2030-01-01"

    for page in range(1, max_pages + 1):
        params = {
            "industryCode": "*", "pageSize": "100", "industry": "*",
            "rating": "*", "ratingChange": "*",
            "beginTime": begin, "endTime": end,
            "pageNo": str(page), "fields": "", "qType": "0",
            "orgCode": "", "code": code, "rcode": "",
            "p": str(page), "pageNum": str(page), "pageNumber": str(page),
        }
        try:
            r = em_get(REPORT_API, params=params,
                       headers={"Referer": "https://data.eastmoney.com/"}, timeout=30)
            d = r.json()
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            logger.warning(f"个股研报请求失败 code={code}: {e}")
            break
        except Exception:
            logger.exception(f"个股研报未预期错误 code={code}")
            break

        rows = d.get("data") or []
        if not rows:
            break
        all_records.extend(rows)
        total_pages = d.get("TotalPage", 1) or 1
        if page >= total_pages:
            break

    if not all_records:
        return empty_df(REPORT_COLS, ("publish_date",))

    processed = []
    for rec in all_records:
        processed.append({
            "title": rec.get("title", ""),
            "publish_date": (rec.get("publishDate") or "")[:10],
            "org_name": rec.get("orgSName", ""),
            "info_code": rec.get("infoCode", ""),
            "eps_cur": rec.get("predictThisYearEps"),
            "eps_next": rec.get("predictNextYearEps"),
            "eps_next2": rec.get("predictNextTwoYearEps"),
            "rating": rec.get("emRatingName", ""),
            "industry": rec.get("indvInduName", ""),
            "attach_pages": rec.get("attachPages"),
        })

    df = pd.DataFrame(processed)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    return df


def get_industry_reports(
    industry_code: str = "*",
    max_pages: int = 5,
    begin: str = "2024-01-01",
) -> pd.DataFrame:
    """获取行业研报列表（东财 reportapi, qType=1）。

    Parameters
    ----------
    industry_code : str
        东财行业码，``"*"`` 拉全行业。传具体码如 ``"1238"`` (IT服务Ⅱ) 精确过滤。
    max_pages : int
        最多翻页数。
    begin : str
        起始日期 ``"YYYY-MM-DD"``。

    Returns
    -------
    pd.DataFrame
        列: title, publish_date, org_name, info_code, industry_name,
        industry_code, rating, report_type, attach_pages, attach_size。
    """
    all_records = []
    end = "2030-01-01"

    for page in range(1, max_pages + 1):
        params = {
            "industryCode": industry_code, "pageSize": "100", "industry": "*",
            "rating": "*", "ratingChange": "*",
            "beginTime": begin, "endTime": end,
            "pageNo": str(page), "fields": "", "qType": "1",
        }
        try:
            r = em_get(REPORT_API, params=params,
                       headers={"Referer": "https://data.eastmoney.com/"}, timeout=30)
            d = r.json()
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            logger.warning(f"行业研报请求失败 industry={industry_code}: {e}")
            break
        except Exception:
            logger.exception(f"行业研报未预期错误 industry={industry_code}")
            break

        rows = d.get("data") or []
        if not rows:
            break
        all_records.extend(rows)
        total_pages = d.get("TotalPage", 1) or 1
        if page >= total_pages:
            break

    if not all_records:
        return empty_df(IND_REPORT_COLS, ("publish_date",))

    processed = []
    for rec in all_records:
        processed.append({
            "title": rec.get("title", ""),
            "publish_date": (rec.get("publishDate") or "")[:10],
            "org_name": rec.get("orgSName", ""),
            "info_code": rec.get("infoCode", ""),
            "industry_name": rec.get("industryName", ""),
            "industry_code": rec.get("industryCode", ""),
            "rating": rec.get("emRatingName", ""),
            "report_type": rec.get("reportType", ""),
            "attach_pages": rec.get("attachPages"),
            "attach_size": rec.get("attachSize"),
        })

    df = pd.DataFrame(processed)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    return df


def download_report_pdf(record: dict, target_dir: str = "./reports") -> str | None:
    """下载单份研报 PDF。

    Parameters
    ----------
    record : dict
        研报记录（来自 get_research_reports / get_industry_reports），
        需含 ``info_code`` 键。
    target_dir : str
        保存目录，默认 ``"./reports"``。

    Returns
    -------
    str | None
        保存路径，下载失败返回 None。
    """
    info_code = record.get("info_code", "")
    if not info_code:
        return None

    date = (record.get("publish_date") or "")[:10]
    org = re.sub(r'[\\/:*?"<>|]', "_", str(record.get("org_name", "未知")))[:40]
    title = re.sub(r'[\\/:*?"<>|]', "_", str(record.get("title", "")))[:80]
    fname = f"{date}_{org}_{title}.pdf"
    target = Path(target_dir) / fname

    if target.exists():
        return str(target)

    url = PDF_TPL.format(info_code=info_code)
    try:
        r = em_get(url, headers={"Referer": "https://data.eastmoney.com/"}, timeout=60)
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"PDF 下载失败 {info_code}: {e}")
        return None
    except Exception:
        logger.exception(f"PDF 下载未预期错误 {info_code}")
        return None

    if r.status_code == 200 and len(r.content) >= 1024:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        return str(target)
    return None


def get_consensus_eps(code: str) -> pd.DataFrame:
    """同花顺机构一致预期 EPS（直连 basic.10jqka.com.cn, 解析 HTML 表格）。

    注：小盘/次新/ST 股可能无机构覆盖，返回空 DataFrame。

    Parameters
    ----------
    code : str
        6 位股票代码。

    Returns
    -------
    pd.DataFrame
        列: year, institution_count, eps_min, eps_mean, eps_max。
        ``eps_mean`` 即机构一致预期 EPS。
    """
    code = normalize_stock_code(code)
    url = f"https://basic.10jqka.com.cn/new/{code}/worth.html"

    try:
        r = _THS_SESSION.get(url, timeout=15)
        r.encoding = "gbk"
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"一致预期EPS请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(EPS_COLS))
    except Exception:
        logger.exception(f"一致预期EPS未预期错误 code={code}")
        return pd.DataFrame(columns=list(EPS_COLS))

    from io import StringIO

    try:
        dfs = pd.read_html(StringIO(r.text))
    except (ValueError, OSError, ImportError):
        logger.warning(f"一致预期EPS HTML 无表格 code={code}")
        return pd.DataFrame(columns=list(EPS_COLS))

    # 找含"每股收益"或"均值"的表格
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if any("每股收益" in c or "均值" in c for c in cols):
            return _normalize_eps_df(df)

    # fallback: 返回第一个表
    if dfs:
        return _normalize_eps_df(dfs[0])
    return pd.DataFrame(columns=list(EPS_COLS))


def _normalize_eps_df(df: pd.DataFrame) -> pd.DataFrame:
    """将同花顺 HTML 表格归一化为标准列。"""
    if df.empty:
        return pd.DataFrame(columns=list(EPS_COLS))

    col_map = {}
    for c in df.columns:
        cs = str(c)
        if "年度" in cs or "year" in cs.lower():
            col_map[c] = "year"
        elif "机构" in cs or "家数" in cs:
            col_map[c] = "institution_count"
        elif "最小" in cs:
            col_map[c] = "eps_min"
        elif "均值" in cs or "平均" in cs:
            col_map[c] = "eps_mean"
        elif "最大" in cs:
            col_map[c] = "eps_max"

    if not col_map:
        return pd.DataFrame(columns=list(EPS_COLS))

    df = df.rename(columns=col_map)
    keep = [c for c in EPS_COLS if c in df.columns]
    return df[keep]
