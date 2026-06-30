"""
stockQuant 信号数据层。

提供量化策略常用的信号/另类数据接口，覆盖资金流向、融资融券、
北向资金、股东户数、龙虎榜、限售解禁等。东财系接口统一内置限流。

第一期（资金面/筹码层）::

    from stockquant.signals import (
        get_fund_flow,              # 个股资金流向 120 日
        get_fund_flow_minute,       # 个股资金流向 分钟级
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
        get_limit_up_pool,          # 涨停池 + 连板梯队
        get_limit_up_sentiment,     # 打板情绪温度计
        get_research_reports,       # 东财个股研报
        get_consensus_eps,          # 一致预期EPS
        get_stock_news,             # 东财个股新闻
        get_global_news,            # 7x24全球快讯
        get_irm_qa,                 # 互动易问答
        get_ths_hot_list,           # 同花顺热榜
    )

第四期（行情/基础/公告/期权）::

    from stockquant.signals import (
        get_tencent_quotes,         # 腾讯实时行情 PE/PB/市值
        get_bars,                   # mootdx K线 (多频率)
        get_level2_orderbook,       # mootdx 五档盘口
        get_tick_transactions,      # mootdx 逐笔成交
        get_baidu_kline_ma,         # 百度K线带MA均线
        get_stock_info,             # 东财个股基本面
        get_sina_financials,        # 新浪财报三表
        get_announcements,          # 巨潮公告全文检索
        get_option_codes,           # ETF期权合约清单
        get_option_tquote,          # 期权T型报价
        get_option_greeks,          # 期权希腊字母+IV
        iwencai_search,             # NL语义搜索研报
        dedup_articles,             # iwencai 搜索结果去重
        full_valuation,             # 单票完整估值分析
    )

设计文档: docs/superpowers/specs/2026-06-29-signals-layer-design.md
"""

from stockquant.signals.announcement import get_announcements
from stockquant.signals.block_trade import get_block_trade
from stockquant.signals.concept import get_concept_blocks
from stockquant.signals.dividend import get_dividend_history
from stockquant.signals.dragon_tiger import (
    get_daily_dragon_tiger,
    get_dragon_tiger_board,
)
from stockquant.signals.finance import (
    get_f10_profile,
    get_finance_snapshot,
    get_sina_financials,
    get_stock_info,
)
from stockquant.signals.fund_flow import get_fund_flow, get_fund_flow_minute
from stockquant.signals.holders import get_holder_changes
from stockquant.signals.hot import get_hot_stocks
from stockquant.signals.industry import get_industry_ranking
from stockquant.signals.iwencai import dedup_articles, iwencai_query, iwencai_search
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
from stockquant.signals.news import get_global_news, get_stock_news
from stockquant.signals.northbound import get_northbound_history, get_northbound_realtime
from stockquant.signals.options import (
    get_option_codes,
    get_option_greeks,
    get_option_tquote,
)
from stockquant.signals.quote import (
    get_baidu_kline_ma,
    get_bars,
    get_level2_orderbook,
    get_tencent_quotes,
    get_tick_transactions,
)
from stockquant.signals.research import (
    download_report_pdf,
    get_consensus_eps,
    get_industry_reports,
    get_research_reports,
)
from stockquant.signals.valuation import (
    calc_peg,
    forward_pe,
    full_valuation,
    pe_digestion,
)
from stockquant.signals.sentiment import (
    get_em_hot_concept,
    get_em_hot_rank,
    get_irm_qa,
    get_ths_hot_list,
)

__all__ = [
    "calc_peg",
    "dedup_articles",
    "download_report_pdf",
    "forward_pe",
    "full_valuation",
    "get_announcements",
    "get_baidu_kline_ma",
    "get_bars",
    "get_block_trade",
    "get_broken_board_pool",
    "get_concept_blocks",
    "get_consensus_eps",
    "get_daily_dragon_tiger",
    "get_dividend_history",
    "get_dragon_tiger_board",
    "get_em_hot_concept",
    "get_em_hot_rank",
    "get_f10_profile",
    "get_finance_snapshot",
    "get_fund_flow",
    "get_fund_flow_minute",
    "get_global_news",
    "get_holder_changes",
    "get_hot_stocks",
    "get_industry_ranking",
    "get_industry_reports",
    "get_irm_qa",
    "get_level2_orderbook",
    "get_limit_down_pool",
    "get_limit_up_pool",
    "get_limit_up_reasons",
    "get_limit_up_sentiment",
    "get_lockup_expiry",
    "get_margin_trading",
    "get_northbound_history",
    "get_northbound_realtime",
    "get_option_codes",
    "get_option_greeks",
    "get_option_tquote",
    "get_research_reports",
    "get_sina_financials",
    "get_stock_info",
    "get_stock_news",
    "get_tencent_quotes",
    "get_ths_hot_list",
    "get_tick_transactions",
    "get_yesterday_limit_up_pool",
    "iwencai_query",
    "iwencai_search",
    "pe_digestion",
]
