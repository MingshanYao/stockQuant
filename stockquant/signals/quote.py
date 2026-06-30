"""
行情层 — 实时行情（补充 BaoStock / TickFlow 已有覆盖）。

端点:
  - mootdx (TCP 7709)      — K线 + 五档盘口 + 逐笔成交 (需 mootdx 依赖)
  - 腾讯财经 API (HTTP)    — PE/PB/市值/换手率/涨跌停/指数/ETF (GBK, 不封IP)
  - 百度股市通 (HTTP)      — K线带MA5/10/20 (零鉴权)
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.utils.logger import get_logger

logger = get_logger("signals.quote")

QUOTE_COLS = (
    "name", "price", "last_close", "open", "high", "low",
    "change_amt", "change_pct", "amount_wan", "turnover_pct",
    "pe_ttm", "pb", "mcap_yi", "float_mcap_yi", "amplitude_pct",
    "limit_up", "limit_down", "vol_ratio", "pe_static",
)


def get_tencent_quotes(codes: list[str]) -> dict[str, dict]:
    """批量拉取腾讯财经实时行情（PE/PB/市值/换手率/涨跌停/量比等）。

    不封 IP，不限频。也支持指数（000001=上证/000300=沪深300/399006=创业板指）
    和 ETF（510050/510300 等）。

    Parameters
    ----------
    codes : list[str]
        6 位代码列表，如 ``["688017", "600519"]``。

    Returns
    -------
    dict[str, dict]
        ``{code: {name, price, pe_ttm, pb, mcap_yi, ...}}``。
    """
    prefixed = []
    for c in codes:
        if c.startswith(("6", "9")):
            prefixed.append(f"sh{c}")
        elif c.startswith("8"):
            prefixed.append(f"bj{c}")
        else:
            prefixed.append(f"sz{c}")

    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.content.decode("gbk")
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"腾讯行情请求失败: {e}")
        return {}
    except Exception:
        logger.exception("腾讯行情未预期错误")
        return {}

    result = {}
    for line in data.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue
        code = key[2:]
        result[code] = {
            "name": vals[1],
            "price": float(vals[3]) if vals[3] else 0,
            "last_close": float(vals[4]) if vals[4] else 0,
            "open": float(vals[5]) if vals[5] else 0,
            "high": float(vals[33]) if vals[33] else 0,
            "low": float(vals[34]) if vals[34] else 0,
            "change_amt": float(vals[31]) if vals[31] else 0,
            "change_pct": float(vals[32]) if vals[32] else 0,
            "amount_wan": float(vals[37]) if vals[37] else 0,
            "turnover_pct": float(vals[38]) if vals[38] else 0,
            "pe_ttm": float(vals[39]) if vals[39] else 0,
            "pb": float(vals[46]) if vals[46] else 0,
            "mcap_yi": float(vals[44]) if vals[44] else 0,
            "float_mcap_yi": float(vals[45]) if vals[45] else 0,
            "amplitude_pct": float(vals[43]) if vals[43] else 0,
            "limit_up": float(vals[47]) if vals[47] else 0,
            "limit_down": float(vals[48]) if vals[48] else 0,
            "vol_ratio": float(vals[49]) if vals[49] else 0,
            "pe_static": float(vals[52]) if vals[52] else 0,
        }
    return result


def get_bars(
    code: str,
    frequency: int = 9,
    offset: int = 100,
) -> pd.DataFrame:
    """获取 K 线数据（mootdx TCP，可选依赖）。

    频率值表 (mootdx 0.11.7):
      0=5分钟  1=15分钟  2=30分钟  3=60分钟  4=日线
      5=周线  6=月线  8=1分钟  9=日线(默认)  10=季线  11=年线

    Parameters
    ----------
    code : str
        6 位股票代码。
    frequency : int
        K线频率，默认 9（日线）。
    offset : int
        返回根数，默认 100。

    Returns
    -------
    pd.DataFrame
        列: open, close, high, low, vol, amount, datetime。
        mootdx 未安装时返回空 DataFrame。
    """
    try:
        from stockquant.signals._mootdx import tdx_client
        client = tdx_client()
        result = client.bars(symbol=code, frequency=frequency, offset=offset)
    except ImportError as e:
        logger.warning(f"mootdx 不可用: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"K线获取失败 code={code}: {e}")
        return pd.DataFrame()

    if not result:
        return pd.DataFrame()
    return pd.DataFrame(result)


def get_level2_orderbook(code: str, n: int = 5) -> pd.DataFrame:
    """获取五档盘口（mootdx TCP，可选依赖）。

    Parameters
    ----------
    code : str
        6 位股票代码。
    n : int
        档位数，默认 5（五档）。

    Returns
    -------
    pd.DataFrame
        列: price, bid_vol, ask_vol。mootdx 未安装时返回空。
    """
    try:
        from stockquant.signals._mootdx import tdx_client
        client = tdx_client()
        quotes = client.quotes(symbol=[code])
    except ImportError as e:
        logger.warning(f"mootdx 不可用: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"盘口获取失败 code={code}: {e}")
        return pd.DataFrame()

    if not quotes:
        return pd.DataFrame()

    rows = []
    q = quotes[0] if isinstance(quotes, list) else quotes
    for i in range(1, n + 1):
        rows.append({
            "price": q.get(f"bid{i}"),
            "bid_vol": q.get(f"bid_vol{i}"),
            "ask_vol": q.get(f"ask_vol{i}"),
        })
    return pd.DataFrame(rows)


def get_tick_transactions(code: str, date: str | None = None) -> pd.DataFrame:
    """获取逐笔成交（mootdx TCP，非交易时间返回空）。

    Parameters
    ----------
    code : str
        6 位股票代码。
    date : str | None
        日期 ``"YYYYMMDD"``，默认最新交易日。

    Returns
    -------
    pd.DataFrame
        列: time, price, vol, num, buyorsell (0买/1卖/2中性)。
    """
    try:
        from stockquant.signals._mootdx import tdx_client
        client = tdx_client()
        trades = client.transaction(symbol=code, date=date)
    except ImportError as e:
        logger.warning(f"mootdx 不可用: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"逐笔成交获取失败 code={code}: {e}")
        return pd.DataFrame()

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


def get_baidu_kline_ma(code: str, start_time: str = "") -> dict:
    """百度股市通 K 线 — 返回自带 MA5/MA10/MA20 均价。

    Parameters
    ----------
    code : str
        6 位股票代码。
    start_time : str
        起始时间，留空取全部。

    Returns
    -------
    dict
        ``{"keys": [...], "rows": [...]}``。
    """
    url = "https://finance.pae.baidu.com/selfselect/getstockquotation"
    params = {
        "all": "1", "isIndex": "false", "isBk": "false",
        "isBlock": "false", "isFutures": "false", "isStock": "true",
        "newFormat": "1", "group": "quotation_kline_ab",
        "finClientType": "pc", "code": code,
        "start_time": start_time, "ktype": "1",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        d = r.json()
    except Exception as e:
        logger.warning(f"百度K线请求失败 code={code}: {e}")
        return {"keys": [], "rows": []}

    result = d.get("Result", {})
    md = result.get("newMarketData", {})
    return {
        "keys": md.get("keys", []),
        "rows": (md.get("marketData") or "").split(";"),
    }
