"""
stockQuant 信号数据层。

提供量化策略常用的信号/另类数据接口，覆盖资金流向、融资融券、
北向资金、股东户数、龙虎榜、限售解禁等。东财系接口统一内置限流。

第一期（资金面/筹码层）::

    from stockquant.signals import (
        get_fund_flow,              # 个股资金流向 120 日
        get_margin_trading,         # 融资融券明细
        get_northbound_realtime,    # 北向资金实时分钟流向
        get_northbound_history,     # 北向资金历史日级
        get_holder_changes,         # 股东户数变化
    )

设计文档: docs/superpowers/specs/2026-06-29-signals-layer-design.md
"""

from stockquant.signals.fund_flow import get_fund_flow
from stockquant.signals.holders import get_holder_changes
from stockquant.signals.margin import get_margin_trading
from stockquant.signals.northbound import get_northbound_history, get_northbound_realtime

__all__ = [
    "get_fund_flow",
    "get_holder_changes",
    "get_margin_trading",
    "get_northbound_history",
    "get_northbound_realtime",
]
