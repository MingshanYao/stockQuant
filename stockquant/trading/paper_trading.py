"""
模拟交易引擎 (Paper Trading)。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from stockquant.backtest.broker import Broker
from stockquant.backtest.context import Context
from stockquant.strategy.base_strategy import BaseStrategy, Order
from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("trading.paper")


class PaperTradingEngine:
    """模拟交易引擎。

    提供与回测引擎类似的接口，但使用实时/延时行情驱动。
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        initial_capital = self.cfg.get("backtest.initial_capital", 1_000_000.0)
        self.context = Context(initial_capital=initial_capital)
        self.broker = Broker(self.context, self.cfg)
        self.strategy: BaseStrategy | None = None
        self._running: bool = False

    def set_strategy(self, strategy: BaseStrategy) -> None:
        self.strategy = strategy
        self.strategy.set_context(self.context)

    def start(self) -> None:
        """启动模拟交易。"""
        if not self.strategy:
            raise RuntimeError("未设置策略")
        self._running = True
        self.strategy.initialize()
        logger.info("模拟交易引擎已启动")

    def stop(self) -> None:
        """停止模拟交易。"""
        self._running = False
        logger.info("模拟交易引擎已停止")

    def on_bar(self, bar: dict[str, pd.DataFrame]) -> list[Order]:
        """接收新行情并执行策略逻辑。"""
        if not self._running or not self.strategy:
            return []

        self.context.current_date = dt.date.today()
        self.broker.on_new_day()

        # 更新价格
        for code, df in bar.items():
            if not df.empty:
                self.context.update_price(code, df["close"].iloc[-1])

        # 执行策略
        self.strategy._orders.clear()
        self.strategy.handle_bar(bar)

        # 撮合
        filled = []
        for order in self.strategy.get_pending_orders():
            result = self.broker.process_order(order)
            if result.status == "filled":
                filled.append(result)

        self.context.record_equity()
        return filled

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """当前投资组合摘要。"""
        return {
            "日期": str(self.context.current_date),
            "总市值": f"{self.context.portfolio_value:,.0f}",
            "现金": f"{self.context.cash:,.0f}",
            "持仓数": len([p for p in self.context.positions.values() if p.quantity > 0]),
        }
