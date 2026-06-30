"""
估值分析辅助函数 — PE/PEG/消化年限 + 完整估值流程。

基于腾讯实时行情 + 同花顺一致预期 EPS 计算。
"""

from __future__ import annotations

import math

import pandas as pd

from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.valuation")


def forward_pe(price: float, eps_forecast: float) -> float:
    """前向 PE = 当前股价 / 未来年度一致预期 EPS。

    若 eps_forecast <= 0 返回 inf。
    """
    if eps_forecast <= 0:
        return float("inf")
    return price / eps_forecast


def pe_digestion(current_pe: float, cagr: float, target_pe: float = 30) -> float:
    """当前 PE 消化到目标 PE 需要的年数。

    默认 target_pe=30x（A 股成长股合理估值锚点）。
    若 PE 已低于目标值返回 0，cagr <= 0 返回 inf。
    """
    if current_pe <= target_pe:
        return 0.0
    if cagr <= 0:
        return float("inf")
    return math.log(current_pe / target_pe) / math.log(1 + cagr)


def calc_peg(pe: float, cagr: float) -> float:
    """PEG = 前向 PE / (CAGR × 100)。

    PEG < 1   → 便宜
    PEG 1–1.5 → 合理
    PEG > 1.5 → 贵

    cagr <= 0 返回 inf。
    """
    if cagr <= 0:
        return float("inf")
    return pe / (cagr * 100)


def full_valuation(code: str) -> dict:
    """单票完整估值分析 — 腾讯行情 + 一致预期 EPS → PE/PEG/消化年限。

    用法::

        result = full_valuation("688017")
        # {"name": "...", "price": ..., "pe_ttm": ..., "pe_fwd": ...,
        #  "cagr_pct": ..., "peg": ..., "digest_years": ..., "analyst_count": ...}

    Parameters
    ----------
    code : str
        6 位股票代码。

    Returns
    -------
    dict
        无机构覆盖或无 EPS 数据时部分字段为 None。
    """
    code = normalize_stock_code(code)

    # 1. 腾讯实时行情
    from stockquant.signals.quote import get_tencent_quotes

    quotes = get_tencent_quotes([code])
    q = quotes.get(code)
    if not q:
        logger.warning(f"full_valuation 腾讯行情缺失 code={code}")
        return _empty_valuation()

    price = q["price"]
    pe_ttm = q["pe_ttm"]
    pb = q["pb"]
    mcap = q["mcap_yi"]
    name = q["name"]

    # 2. 一致预期 EPS
    from stockquant.signals.research import get_consensus_eps

    df = get_consensus_eps(code)
    eps_cur = None
    eps_next = None
    analyst_count = 0

    if not df.empty and len(df.columns) >= 3:
        def _pick(row, *name_keys):
            for c in df.columns:
                cs = str(c)
                if any(k in cs for k in name_keys):
                    return row.get(c)
            return None

        try:
            r0 = df.iloc[0]
            v = _pick(r0, "均值", "eps_mean")
            eps_cur = float(v) if pd.notna(v) else None
            cnt = _pick(r0, "机构", "institution")
            analyst_count = int(cnt) if pd.notna(cnt) else 0
            if len(df) >= 2:
                vn = _pick(df.iloc[1], "均值", "eps_mean")
                eps_next = float(vn) if pd.notna(vn) else None
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"full_valuation EPS 解析失败 code={code}: {e}")

    # 3. 估值指标
    pe_fwd = forward_pe(price, eps_cur) if eps_cur else float("inf")
    cagr = (eps_next / eps_cur - 1) if (eps_cur and eps_next and eps_cur > 0) else 0.0
    peg = calc_peg(pe_fwd, cagr) if pe_fwd != float("inf") else float("inf")
    digest = pe_digestion(pe_fwd, cagr) if pe_fwd != float("inf") else float("inf")

    return {
        "name": name,
        "price": price,
        "mcap_yi": mcap,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "eps_cur": eps_cur,
        "eps_next": eps_next,
        "pe_fwd": round(pe_fwd, 1) if eps_cur else None,
        "cagr_pct": round(cagr * 100, 0) if cagr else None,
        "peg": round(peg, 2) if peg != float("inf") else None,
        "digest_years": round(digest, 1),
        "analyst_count": analyst_count,
    }


def _empty_valuation() -> dict:
    return {
        "name": None,
        "price": None, "mcap_yi": None, "pe_ttm": None, "pb": None,
        "eps_cur": None, "eps_next": None, "pe_fwd": None,
        "cagr_pct": None, "peg": None, "digest_years": None,
        "analyst_count": 0,
    }
