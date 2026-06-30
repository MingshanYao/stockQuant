"""
ETF期权层 — 合约清单 + T型报价 + 希腊字母 + 隐含波动率。

端点: hq.sinajs.cn / stock.finance.sina.com.cn (新浪, GBK, 需 Referer)

注：50ETF/300ETF/科创50ETF/500ETF 期权。希腊字母/IV 由交易所预先算好，
无需本地 BSM 计算。
"""

from __future__ import annotations

import pandas as pd

from stockquant.signals._sina import sina_get, sina_opt_list
from stockquant.utils.logger import get_logger

logger = get_logger("signals.options")

_UNDERLYING_MAP = {
    "510050": "50ETF",
    "510300": "300ETF",
    "588000": "科创50ETF",
    "510500": "500ETF",
}

TQUOTE_COLS = (
    "bid_vol", "bid", "last", "ask", "ask_vol", "open_interest",
    "pct", "strike", "prev_close", "open", "limit_up", "limit_down",
    "name", "amplitude", "high", "low", "volume", "amount",
)
GREEKS_COLS = (
    "name", "volume", "delta", "gamma", "theta", "vega",
    "iv", "high", "low", "trade_code", "strike", "last", "theory",
)


def _opt_f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return x


def get_option_codes(underlying: str = "510050", call: bool = True) -> dict[str, list[str]]:
    """获取 ETF 期权合约清单。

    Parameters
    ----------
    underlying : str
        标的 ETF 代码，支持 ``"510050"`` / ``"510300"`` / ``"588000"`` / ``"510500"``。
    call : bool
        True=认购期权, False=认沽期权。

    Returns
    -------
    dict[str, list[str]]
        ``{月份YYMM: [合约代码, ...]}``，第一个 key 即近月合约。
    """
    cate = _UNDERLYING_MAP.get(underlying, "50ETF")
    url = (
        "https://stock.finance.sina.com.cn/futures/api/openapi.php/"
        "StockOptionService.getStockName"
    )
    params = {"exchange": "null", "cate": cate}

    try:
        r = sina_get(url, params=params, timeout=10)
        months = r.json()["result"]["data"]["contractMonth"]
    except Exception:
        logger.exception(f"期权月份获取失败 underlying={underlying}")
        return {}

    if not months or len(months) < 2:
        return {}

    months = [m.replace("-", "")[2:] for m in months[1:]]  # YYMM
    flag = "OP_UP_" if call else "OP_DOWN_"
    out: dict[str, list[str]] = {}
    for m in months:
        codes = [
            c.replace("CON_OP_", "")
            for c in sina_opt_list(f"{flag}{underlying}{m}")
            if c.startswith("CON_OP_")
        ]
        if codes:
            out[m] = codes
    return out


def get_option_tquote(code: str) -> dict:
    """获取期权 T 型报价。

    Parameters
    ----------
    code : str
        合约代码（来自 get_option_codes）。

    Returns
    -------
    dict
        bid_vol, bid, last, ask, ask_vol, open_interest, pct,
        strike, prev_close, open, limit_up, limit_down, name,
        amplitude, high, low, volume, amount。
    """
    v = sina_opt_list(f"CON_OP_{code}")
    if len(v) < 43:
        return {}

    return {
        "bid_vol": _opt_f(v[0]), "bid": _opt_f(v[1]),
        "last": _opt_f(v[2]), "ask": _opt_f(v[3]),
        "ask_vol": _opt_f(v[4]), "open_interest": _opt_f(v[5]),
        "pct": _opt_f(v[6]), "strike": _opt_f(v[7]),
        "prev_close": _opt_f(v[8]), "open": _opt_f(v[9]),
        "limit_up": _opt_f(v[10]), "limit_down": _opt_f(v[11]),
        "name": v[37], "amplitude": _opt_f(v[38]),
        "high": _opt_f(v[39]), "low": _opt_f(v[40]),
        "volume": _opt_f(v[41]), "amount": _opt_f(v[42]),
    }


def get_option_greeks(code: str) -> dict:
    """获取期权希腊字母 + 隐含波动率 (IV)。

    Parameters
    ----------
    code : str
        合约代码（来自 get_option_codes）。

    Returns
    -------
    dict
        delta, gamma, theta, vega, iv (隐含波动率, 小数, 如 0.1735=17.35%),
        name, volume, high, low, trade_code, strike, last, theory。
    """
    raw = sina_opt_list(f"CON_SO_{code}")
    if len(raw) < 16:
        return {}

    v = [raw[0]] + raw[4:]  # raw[1:4] 是 3 个空串
    return {
        "name": v[0],
        "volume": _opt_f(v[1]),
        "delta": _opt_f(v[2]),
        "gamma": _opt_f(v[3]),
        "theta": _opt_f(v[4]),
        "vega": _opt_f(v[5]),
        "iv": _opt_f(v[6]),
        "high": _opt_f(v[7]),
        "low": _opt_f(v[8]),
        "trade_code": v[9],
        "strike": _opt_f(v[10]),
        "last": _opt_f(v[11]),
        "theory": _opt_f(v[12]),
    }
