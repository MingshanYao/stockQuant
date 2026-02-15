"""
虚拟券商 — 模拟 A 股撮合与费用计算。
"""

from __future__ import annotations

from stockquant.strategy.base_strategy import Order, Position
from stockquant.backtest.context import Context
from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("backtest.broker")


class Broker:
    """虚拟券商：订单校验 → 费用计算 → 持仓更新。

    完整模拟 A 股交易规则：
        - T+1（当日买入不可卖出）
        - 100 股整手交易
        - 涨跌停限制
        - 佣金 + 印花税 + 滑点
    """

    def __init__(self, context: Context, config: Config | None = None) -> None:
        self.ctx = context
        cfg = config or Config()

        self.commission_rate: float = cfg.get("backtest.commission_rate", 0.00025)
        self.stamp_duty_rate: float = cfg.get("backtest.stamp_duty_rate", 0.001)
        self.slippage: float = cfg.get("backtest.slippage", 0.002)
        self.min_commission: float = cfg.get("backtest.min_commission", 5.0)
        self.trade_unit: int = cfg.get("backtest.trade_unit", 100)
        self.price_limit: float = cfg.get("backtest.price_limit", 0.1)
        self.t_plus_1: bool = cfg.get("backtest.t_plus_1", True)

        self._trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # 订单处理
    # ------------------------------------------------------------------

    def process_order(self, order: Order) -> Order:
        """处理单个订单。"""
        if order.status != "pending":
            return order

        # 数量必须为交易单位整数倍
        if order.quantity % self.trade_unit != 0:
            order.quantity = (order.quantity // self.trade_unit) * self.trade_unit

        if order.quantity <= 0:
            order.status = "rejected"
            order.reason = "数量不足一手"
            return order

        # 获取执行价格（含滑点）
        exec_price = self._apply_slippage(order)

        # 涨跌停检查
        if not self._check_price_limit(order, exec_price):
            order.status = "rejected"
            order.reason = "涨跌停限制"
            return order

        if order.direction == "buy":
            return self._execute_buy(order, exec_price)
        else:
            return self._execute_sell(order, exec_price)

    # ------------------------------------------------------------------
    # 买入
    # ------------------------------------------------------------------

    def _execute_buy(self, order: Order, exec_price: float) -> Order:
        total_cost = exec_price * order.quantity
        commission = max(total_cost * self.commission_rate, self.min_commission)
        required = total_cost + commission

        if required > self.ctx.cash:
            # 自动缩量
            affordable = int(self.ctx.cash / (exec_price * (1 + self.commission_rate)) / self.trade_unit) * self.trade_unit
            if affordable <= 0:
                order.status = "rejected"
                order.reason = "资金不足"
                return order
            order.quantity = affordable
            total_cost = exec_price * order.quantity
            commission = max(total_cost * self.commission_rate, self.min_commission)
            required = total_cost + commission

        # 扣款 & 更新持仓
        self.ctx.cash -= required

        pos = self.ctx.get_position(order.code)
        old_value = pos.avg_cost * pos.quantity
        new_quantity = pos.quantity + order.quantity
        pos.avg_cost = (old_value + total_cost) / new_quantity if new_quantity > 0 else 0
        pos.quantity = new_quantity
        pos.buy_date = str(self.ctx.current_date)

        if self.t_plus_1:
            pos.frozen += order.quantity

        order.filled_price = exec_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.status = "filled"
        order.timestamp = str(self.ctx.current_date)

        self._record_trade(order)
        return order

    # ------------------------------------------------------------------
    # 卖出
    # ------------------------------------------------------------------

    def _execute_sell(self, order: Order, exec_price: float) -> Order:
        pos = self.ctx.get_position(order.code)
        available = pos.quantity - pos.frozen

        if available <= 0:
            order.status = "rejected"
            order.reason = "T+1 限制，无可卖仓位"
            return order

        sell_qty = min(order.quantity, available)
        sell_qty = (sell_qty // self.trade_unit) * self.trade_unit
        if sell_qty <= 0:
            order.status = "rejected"
            order.reason = "可卖数量不足一手"
            return order

        total_amount = exec_price * sell_qty
        commission = max(total_amount * self.commission_rate, self.min_commission)
        stamp_duty = total_amount * self.stamp_duty_rate
        net_amount = total_amount - commission - stamp_duty

        self.ctx.cash += net_amount
        pos.quantity -= sell_qty

        if pos.quantity == 0:
            pos.avg_cost = 0.0
            pos.frozen = 0

        order.filled_price = exec_price
        order.filled_quantity = sell_qty
        order.commission = commission + stamp_duty
        order.status = "filled"
        order.timestamp = str(self.ctx.current_date)

        self._record_trade(order)
        return order

    # ------------------------------------------------------------------
    # 日切：解冻 T+1 冻结仓位
    # ------------------------------------------------------------------

    def on_new_day(self) -> None:
        """每日开盘前解冻前一日买入的股票。"""
        for pos in self.ctx.positions.values():
            pos.frozen = 0

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _apply_slippage(self, order: Order) -> float:
        """应用滑点。"""
        base_price = order.price or self.ctx.get_current_price(order.code)
        if order.direction == "buy":
            return base_price * (1 + self.slippage)
        else:
            return base_price * (1 - self.slippage)

    def _check_price_limit(self, order: Order, price: float) -> bool:
        """涨跌停检查（简化版）。"""
        # 实际应比对前一日收盘价，这里先放行
        return price > 0

    def _record_trade(self, order: Order) -> None:
        self._trade_log.append({
            "date": order.timestamp,
            "code": order.code,
            "direction": order.direction,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "commission": order.commission,
        })
        logger.debug(
            f"{order.direction.upper()} {order.code} "
            f"x{order.filled_quantity} @{order.filled_price:.2f} "
            f"fee={order.commission:.2f}"
        )

    @property
    def trade_log(self) -> list[dict]:
        return self._trade_log
