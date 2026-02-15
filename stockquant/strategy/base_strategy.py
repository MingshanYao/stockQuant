"""
策略基类 — 所有策略继承此类实现 ``initialize`` / ``handle_bar``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from stockquant.backtest.context import Context


@dataclass
class Order:
    """订单数据结构。"""

    code: str
    direction: str          # "buy" / "sell"
    quantity: int           # 股数
    price: float | None     # 限价，None 表示市价
    order_type: str = "market"  # "market" / "limit"
    status: str = "pending"     # "pending" / "filled" / "rejected" / "cancelled"
    filled_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    timestamp: str = ""
    reason: str = ""


@dataclass
class Position:
    """持仓数据结构。"""

    code: str
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    buy_date: str = ""       # 买入日期，用于 T+1 判断
    frozen: int = 0          # 冻结数量（T+1 当日买入部分）


class BaseStrategy(ABC):
    """策略基类。

    用户通过继承本类并实现以下方法来开发策略:
        - ``initialize()``: 策略初始化，设置参数/订阅数据。
        - ``handle_bar(bar)``: 每个 bar 触发的交易逻辑。
        - ``on_order(order)`` (可选): 订单状态回调。
        - ``on_trade(order)`` (可选): 成交回调。
        - ``before_trading()`` (可选): 盘前回调。
        - ``after_trading()`` (可选): 收盘后回调。
    """

    def __init__(self) -> None:
        self.context: Context | None = None
        self._orders: list[Order] = []
        self._params: dict[str, Any] = {}

    def set_context(self, context: Context) -> None:
        """由回测引擎注入上下文。"""
        self.context = context

    def set_params(self, **kwargs) -> None:
        """设置策略参数。"""
        self._params.update(kwargs)

    def get_param(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    # ------------------------------------------------------------------
    # 必须实现的方法
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """策略初始化（仅在回测开始时调用一次）。"""

    @abstractmethod
    def handle_bar(self, bar: dict[str, pd.DataFrame]) -> None:
        """每个 bar（交易日）触发的核心逻辑。

        Parameters
        ----------
        bar : dict[str, DataFrame]
            当前 bar 数据，键为股票代码。
        """

    # ------------------------------------------------------------------
    # 可选回调
    # ------------------------------------------------------------------

    def before_trading(self) -> None:
        """每日盘前回调。"""

    def after_trading(self) -> None:
        """每日收盘后回调。"""

    def on_order(self, order: Order) -> None:
        """订单状态更新回调。"""

    def on_trade(self, order: Order) -> None:
        """成交回调。"""

    # ------------------------------------------------------------------
    # 下单接口（由策略调用）
    # ------------------------------------------------------------------

    def buy(
        self,
        code: str,
        quantity: int,
        price: float | None = None,
        reason: str = "",
    ) -> Order:
        """买入下单。"""
        order = Order(
            code=code,
            direction="buy",
            quantity=quantity,
            price=price,
            order_type="limit" if price else "market",
            reason=reason,
        )
        self._orders.append(order)
        return order

    def sell(
        self,
        code: str,
        quantity: int,
        price: float | None = None,
        reason: str = "",
    ) -> Order:
        """卖出下单。"""
        order = Order(
            code=code,
            direction="sell",
            quantity=quantity,
            price=price,
            order_type="limit" if price else "market",
            reason=reason,
        )
        self._orders.append(order)
        return order

    def order_target_percent(
        self,
        code: str,
        target_pct: float,
        price: float | None = None,
        reason: str = "",
    ) -> Order | None:
        """按目标仓位比例下单。"""
        if self.context is None:
            return None

        total_value = self.context.portfolio_value
        target_value = total_value * target_pct
        current_value = self.context.get_position_value(code)
        diff_value = target_value - current_value

        if abs(diff_value) < 100:  # 忽略极小差异
            return None

        current_price = price or self.context.get_current_price(code)
        if current_price <= 0:
            return None

        quantity = int(abs(diff_value) / current_price / 100) * 100
        if quantity <= 0:
            return None

        if diff_value > 0:
            return self.buy(code, quantity, price, reason)
        else:
            return self.sell(code, quantity, price, reason)

    def cancel_all(self) -> None:
        """撤销所有未成交订单。"""
        for order in self._orders:
            if order.status == "pending":
                order.status = "cancelled"

    def get_pending_orders(self) -> list[Order]:
        """获取未成交订单列表。"""
        return [o for o in self._orders if o.status == "pending"]
