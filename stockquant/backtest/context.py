"""
回测上下文 — 维护回测过程中的账户与市场状态。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from stockquant.strategy.base_strategy import Position


@dataclass
class Portfolio:
    """投资组合状态。"""

    initial_capital: float = 1_000_000.0
    cash: float = 1_000_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    total_value: float = 1_000_000.0
    daily_returns: list[float] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)


class Context:
    """回测上下文，由引擎创建并注入策略。"""

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
            total_value=initial_capital,
        )
        self.current_date: dt.date | None = None
        self.current_data: dict[str, pd.DataFrame] = {}
        self._price_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 属性快捷方式
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        return self.portfolio.total_value

    @property
    def cash(self) -> float:
        return self.portfolio.cash

    @cash.setter
    def cash(self, value: float) -> None:
        self.portfolio.cash = value

    @property
    def positions(self) -> dict[str, Position]:
        return self.portfolio.positions

    # ------------------------------------------------------------------
    # 持仓操作
    # ------------------------------------------------------------------

    def get_position(self, code: str) -> Position:
        """获取持仓（若不存在则返回空持仓）。"""
        if code not in self.portfolio.positions:
            self.portfolio.positions[code] = Position(code=code)
        return self.portfolio.positions[code]

    def get_position_value(self, code: str) -> float:
        """获取某只股票持仓市值。"""
        pos = self.get_position(code)
        return pos.quantity * pos.current_price

    def get_current_price(self, code: str) -> float:
        """获取当前价格。"""
        return self._price_cache.get(code, 0.0)

    def update_price(self, code: str, price: float) -> None:
        """更新价格缓存。"""
        self._price_cache[code] = price
        if code in self.portfolio.positions:
            pos = self.portfolio.positions[code]
            pos.current_price = price
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    def update_portfolio_value(self) -> None:
        """重新计算组合总市值。"""
        position_value = sum(
            pos.quantity * pos.current_price
            for pos in self.portfolio.positions.values()
        )
        self.portfolio.total_value = self.portfolio.cash + position_value

    def record_equity(self) -> None:
        """记录每日权益曲线。"""
        self.update_portfolio_value()
        self.portfolio.equity_curve.append({
            "date": self.current_date,
            "total_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "position_value": self.portfolio.total_value - self.portfolio.cash,
        })

        # 计算日收益率
        if len(self.portfolio.equity_curve) >= 2:
            prev = self.portfolio.equity_curve[-2]["total_value"]
            curr = self.portfolio.total_value
            daily_ret = (curr - prev) / prev if prev > 0 else 0.0
        else:
            daily_ret = 0.0
        self.portfolio.daily_returns.append(daily_ret)
