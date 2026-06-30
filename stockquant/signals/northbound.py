"""
北向资金流向（沪深股通）。

端点: data.hexin.cn/market/hsgtApi/ (同花顺, 零鉴权, 不走东财限流)
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import requests

from stockquant.utils.logger import get_logger

logger = get_logger("signals.northbound")

HSGT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Chrome/117.0.0.0 Safari/537.36"
    ),
    "Host": "data.hexin.cn",
    "Referer": "https://data.hexin.cn/",
}

_SESSION = requests.Session()
_SESSION.headers.update(HSGT_HEADERS)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0

_CACHE_DIR = Path.home() / ".tradingagents" / "cache"


def get_northbound_realtime() -> pd.DataFrame:
    """获取当日沪深股通实时分钟级流向。

    Returns
    -------
    pd.DataFrame
        列: time, hgt_yi (沪股通累计净买入/亿元),
        sgt_yi (深股通累计净买入/亿元)。
        非交易时段返回含历史缓存的空 DataFrame。
    """
    url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = _SESSION.get(url, timeout=10)
            d = r.json()
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                backoff = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"北向资金请求失败 (第{attempt + 1}次): {e}，"
                    f"{backoff:.1f}s 后重试..."
                )
                time.sleep(backoff + random.uniform(0, 0.3))
        except ValueError as e:
            logger.warning(f"北向资金 JSON 解析失败: {e}")
            return pd.DataFrame(columns=["time", "hgt_yi", "sgt_yi"])
    else:
        logger.warning(f"北向资金请求失败 (已重试{_MAX_RETRIES}次): {last_exc}")
        return pd.DataFrame(columns=["time", "hgt_yi", "sgt_yi"])

    times = d.get("time") or []
    hgt = d.get("hgt") or []
    sgt = d.get("sgt") or []

    # 截断到最短长度，避免不齐时填 None 导致数据错位
    min_len = min(len(times), len(hgt), len(sgt))
    times = times[:min_len]
    hgt = hgt[:min_len]
    sgt = sgt[:min_len]

    df = pd.DataFrame({
        "time": times,
        "hgt_yi": hgt,
        "sgt_yi": sgt,
    })

    _save_snapshot(df)
    return df


def get_northbound_history(n: int = 20) -> pd.DataFrame:
    """读取本地缓存的北向资金日级历史数据。

    缓存文件在 ``~/.tradingagents/cache/northbound_daily.csv``，
    由 ``get_northbound_realtime()`` 每次调用时自动追加。

    Parameters
    ----------
    n : int
        返回最近 N 个交易日，默认 20。

    Returns
    -------
    pd.DataFrame
        列: date, hgt, sgt。无缓存文件时返回空 DataFrame。
    """
    path = _CACHE_DIR / "northbound_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "hgt", "sgt"])

    try:
        df = pd.read_csv(path)
        return df.tail(n)
    except Exception as e:
        logger.warning(f"北向历史读取失败: {e}")
        return pd.DataFrame(columns=["date", "hgt", "sgt"])


def _save_snapshot(df: pd.DataFrame) -> None:
    """从分钟 DataFrame 提取收盘快照写入本地 CSV（内部函数）。"""
    valid = df.dropna(subset=["hgt_yi", "sgt_yi"])
    if valid.empty:
        return

    last = valid.iloc[-1]
    time_val = last["time"]
    date_str = str(time_val)[:10] if time_val else ""

    if not date_str:
        return

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "northbound_daily.csv"

    rows: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().strip().split("\n")[1:]:
            parts = line.split(",")
            if len(parts) == 3:
                rows[parts[0]] = line

    rows[date_str] = f"{date_str},{last['hgt_yi']},{last['sgt_yi']}"

    with open(path, "w") as f:
        f.write("date,hgt,sgt\n")
        for d in sorted(rows.keys()):
            f.write(rows[d] + "\n")
