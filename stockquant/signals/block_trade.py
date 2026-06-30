"""
大宗交易记录。

端点: datacenter-web.eastmoney.com → RPT_DATA_BLOCKTRADE
"""

from __future__ import annotations

import pandas as pd
import requests

from stockquant.signals._eastmoney import em_datacenter
from stockquant.utils.helpers import normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("signals.block_trade")

BLOCK_TRADE_COLS = (
    "date", "price", "close", "premium_pct",
    "vol", "amount", "buyer", "seller",
)


def get_block_trade(code: str, page_size: int = 20) -> pd.DataFrame:
    """获取个股大宗交易记录。

    溢价率 > 0 → 买方愿意支付高于市价的价格，通常为利好信号。
    折价率过大 → 可能为大股东减持套现。

    Parameters
    ----------
    code : str
        6 位股票代码，支持 ``"600519"`` / ``"sh600519"`` / ``"600519.SH"``。
    page_size : int
        返回最近多少条记录，默认 20。

    Returns
    -------
    pd.DataFrame
        列: date, price (成交价), close (收盘价), premium_pct (溢价率%),
        vol (成交量), amount (成交额), buyer (买方营业部), seller (卖方营业部)。
    """
    code = normalize_stock_code(code)

    try:
        data = em_datacenter(
            "RPT_DATA_BLOCKTRADE",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=page_size,
            sort_columns="TRADE_DATE",
            sort_types="-1",
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"大宗交易请求失败 code={code}: {e}")
        return pd.DataFrame(columns=list(BLOCK_TRADE_COLS))
    except Exception:
        logger.exception(f"大宗交易未预期错误 code={code}")
        return pd.DataFrame(columns=list(BLOCK_TRADE_COLS))

    if not data:
        return pd.DataFrame(columns=list(BLOCK_TRADE_COLS))

    rows = []
    for row in data:
        close = row.get("CLOSE_PRICE") or 0
        deal_price = row.get("DEAL_PRICE") or 0
        premium = ((deal_price / close - 1) * 100) if close else 0.0
        rows.append({
            "date": str(row.get("TRADE_DATE", ""))[:10],
            "price": deal_price,
            "close": close,
            "premium_pct": round(premium, 2),
            "vol": row.get("DEAL_VOLUME", 0),
            "amount": row.get("DEAL_AMT", 0),
            "buyer": row.get("BUYER_NAME", ""),
            "seller": row.get("SELLER_NAME", ""),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
