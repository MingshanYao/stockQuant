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

第二期（事件型）::

    from stockquant.signals import (
        get_dragon_tiger_board,     # 个股龙虎榜 + 买卖席位
        get_daily_dragon_tiger,     # 全市场龙虎榜
        get_lockup_expiry,          # 限售解禁日历
        get_block_trade,            # 大宗交易
        get_dividend_history,       # 分红送转历史
    )

第三期（截面/文本型）::

    from stockquant.signals import (
        get_hot_stocks,             # 同花顺强势股 + 题材归因
        get_concept_blocks,         # 个股板块/概念归属
        get_industry_ranking,       # 行业板块排名
    )

设计文档: docs/superpowers/specs/2026-06-29-signals-layer-design.md
"""

from stockquant.signals.block_trade import get_block_trade
from stockquant.signals.concept import get_concept_blocks
from stockquant.signals.dividend import get_dividend_history
from stockquant.signals.dragon_tiger import (
    get_daily_dragon_tiger,
    get_dragon_tiger_board,
)
from stockquant.signals.fund_flow import get_fund_flow
from stockquant.signals.holders import get_holder_changes
from stockquant.signals.hot import get_hot_stocks
from stockquant.signals.industry import get_industry_ranking
from stockquant.signals.limit_up import (
    get_broken_board_pool,
    get_limit_down_pool,
    get_limit_up_pool,
    get_limit_up_reasons,
    get_limit_up_sentiment,
    get_yesterday_limit_up_pool,
)
from stockquant.signals.lockup import get_lockup_expiry
from stockquant.signals.margin import get_margin_trading
from stockquant.signals.northbound import get_northbound_history, get_northbound_realtime

__all__ = [
    "get_block_trade",
    "get_broken_board_pool",
    "get_concept_blocks",
    "get_daily_dragon_tiger",
    "get_dividend_history",
    "get_dragon_tiger_board",
    "get_fund_flow",
    "get_holder_changes",
    "get_hot_stocks",
    "get_industry_ranking",
    "get_limit_down_pool",
    "get_limit_up_pool",
    "get_limit_up_reasons",
    "get_limit_up_sentiment",
    "get_lockup_expiry",
    "get_margin_trading",
    "get_northbound_history",
    "get_northbound_realtime",
    "get_yesterday_limit_up_pool",
]
