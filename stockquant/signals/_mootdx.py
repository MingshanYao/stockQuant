"""
通达信 TCP 客户端 helper — 规避 mootdx 0.11.x BESTIP.HQ 空串 bug。

提供:
    tdx_client(market='std') → Quotes
    顺序探测可用服务器 → 回退 bestip → 回退裸 factory → RuntimeError

mootdx 为可选依赖。如果未安装，调用 tdx_client() 会抛出 ImportError。
"""

from __future__ import annotations

import socket

from stockquant.utils.logger import get_logger

logger = get_logger("signals._mootdx")

_TDX_SERVERS = [
    ("119.97.185.59", 7709), ("124.70.133.119", 7709),
    ("116.205.183.150", 7709), ("123.60.73.44", 7709),
    ("116.205.163.254", 7709), ("121.36.225.169", 7709),
    ("123.60.70.228", 7709), ("124.71.9.153", 7709),
    ("110.41.147.114", 7709), ("124.71.187.122", 7709),
]

_CLIENT = None


def _probe(ip: str, port: int, timeout: float = 2.0) -> bool:
    """TCP 握手探测，判断服务器是否可达。"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def tdx_client(market: str = "std"):
    """创建 mootdx 客户端，规避 0.11.x BESTIP.HQ 空串 bug。

    顺序兜底:
      1) 顺序探测 _TDX_SERVERS，用第一个 TCP 可达的显式 server；
      2) 全部不可达 → 回退 mootdx 自带 bestip 测速选优；
      3) 再不行 → 回退裸 factory（老用户 config 已有可用 BESTIP 时成立）；
      4) 仍失败 → 抛 RuntimeError。

    Parameters
    ----------
    market : str
        市场，默认 ``"std"``（标准市场）。

    Returns
    -------
    mootdx.quotes.Quotes
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    try:
        from mootdx.quotes import Quotes
    except ImportError:
        raise ImportError(
            "mootdx 未安装。请运行: pip install mootdx"
        )

    for ip, port in _TDX_SERVERS:
        if _probe(ip, port):
            _CLIENT = Quotes.factory(market=market, server=(ip, port))
            return _CLIENT

    try:
        _CLIENT = Quotes.factory(market=market, bestip=True)
        return _CLIENT
    except Exception:
        pass

    try:
        _CLIENT = Quotes.factory(market=market)
        return _CLIENT
    except Exception as e:
        raise RuntimeError(
            "所有 mootdx 服务器均不可达。海外网络通常全部超时（TCP 7709），"
            "请走国内代理或更新 _TDX_SERVERS 列表。原始错误：%s" % e
        )
