"""
基础数据层 — 季报快照 + F10 公司资料 + 新浪财报三表 + 东财个股信息。

端点:
  - mootdx (TCP 7709)         — 季报快照(37字段) + F10(9类文本)
  - push2.eastmoney.com       — 个股基本面 (em_get 限流)
  - quotes.sina.cn            — 资产负债表/利润表/现金流量表 (GBK, 零鉴权)
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import UA, em_get
from stockquant.utils.helpers import get_market_prefix, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.finance")

FINANCE_SNAPSHOT_COLS = (
    "code", "name", "industry", "total_shares", "float_shares",
    "mcap", "float_mcap", "list_date", "price",
)


def get_stock_info(code: str) -> dict:
    """东财个股基本面信息（push2 API）。

    Parameters
    ----------
    code : str
        6 位股票代码。

    Returns
    -------
    dict
        ``{code, name, industry, total_shares, float_shares,
        mcap, float_mcap, list_date, price}``。
    """
    code = normalize_stock_code(code)
    prefix = get_market_prefix(code)
    market_code = 1 if prefix == "sh" else 0

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "fltt": "2", "invt": "2",
        "fields": "f57,f58,f84,f85,f127,f116,f117,f189,f43",
        "secid": f"{market_code}.{code}",
    }
    headers = {"User-Agent": UA}

    try:
        r = em_get(url, params=params, headers=headers, timeout=10)
        d = r.json().get("data", {})
    except Exception as e:
        logger.warning(f"个股信息请求失败 code={code}: {e}")
        return {}

    return {
        "code": d.get("f57", ""),
        "name": d.get("f58", ""),
        "industry": d.get("f127", ""),
        "total_shares": d.get("f84", 0),
        "float_shares": d.get("f85", 0),
        "mcap": d.get("f116", 0),
        "float_mcap": d.get("f117", 0),
        "list_date": str(d.get("f189", "")),
        "price": d.get("f43", 0),
    }


def get_finance_snapshot(code: str) -> dict:
    """mootdx 季报财务快照（37 字段，EPS/ROE/净利等）。

    需要 mootdx 依赖。未安装时返回空 dict。

    Parameters
    ----------
    code : str
        6 位股票代码。

    Returns
    -------
    dict
        37 个季报字段。
    """
    try:
        from stockquant.signals._mootdx import tdx_client
        client = tdx_client()
        return client.finance(symbol=code)
    except ImportError as e:
        logger.warning(f"mootdx 不可用: {e}")
        return {}
    except Exception as e:
        logger.warning(f"财务快照获取失败 code={code}: {e}")
        return {}


def get_f10_profile(code: str, category: str = "公司概况") -> str | None:
    """mootdx F10 公司文本资料（9 大类）。

    需要 mootdx 依赖。未安装时返回 None。

    Parameters
    ----------
    code : str
        6 位股票代码。
    category : str
        类别: 最新提示/公司概况/财务分析/股东研究/股本结构/
        资本运作/业内点评/行业分析/公司大事。

    Returns
    -------
    str | None
    """
    try:
        from stockquant.signals._mootdx import tdx_client
        client = tdx_client()
        return client.F10(symbol=code, name=category)
    except ImportError as e:
        logger.warning(f"mootdx 不可用: {e}")
        return None
    except Exception as e:
        logger.warning(f"F10 获取失败 code={code}: {e}")
        return None


def get_sina_financials(code: str, report_type: str = "lrb",
                        num: int = 8) -> pd.DataFrame:
    """新浪财报三表（资产负债表/利润表/现金流量表）。

    Parameters
    ----------
    code : str
        6 位股票代码。
    report_type : str
        ``"fzb"`` (资产负债表) / ``"lrb"`` (利润表) / ``"llb"`` (现金流量表)。
    num : int
        取最近 N 期，默认 8。

    Returns
    -------
    pd.DataFrame
        每行一个报告期，列为科目名称（含 _同比 后缀列）。
    """
    code = normalize_stock_code(code)
    prefix = "sh" if code.startswith("6") else "sz"
    paper_code = f"{prefix}{code}"

    url = (
        "https://quotes.sina.cn/cn/api/openapi.php/"
        "CompanyFinanceService.getFinanceReport2022"
    )
    params = {
        "paperCode": paper_code,
        "source": report_type,
        "type": "0",
        "page": "1",
        "num": str(num),
    }

    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": UA}, timeout=15)
        report_list = (
            r.json().get("result", {}).get("data", {}).get("report_list", {}) or {}
        )
    except Exception as e:
        logger.warning(f"新浪财报请求失败 code={code}: {e}")
        return pd.DataFrame()

    rows = []
    for period in sorted(report_list.keys(), reverse=True)[:num]:
        obj = report_list[period]
        rec: dict = {
            "report_date": f"{period[:4]}-{period[4:6]}-{period[6:8]}",
        }
        for it in obj.get("data", []) or []:
            title = it.get("item_title", "")
            if not title or it.get("item_value") is None:
                continue
            rec[title] = it.get("item_value")
            tongbi = it.get("item_tongbi")
            if tongbi not in (None, ""):
                rec[title + "_同比"] = tongbi
        rows.append(rec)
    return pd.DataFrame(rows)
