"""
止盈止损管理器 — 固定 / 移动 / ATR 三种模式。
"""

from __future__ import annotations

from dataclasses import dataclass

from stockquant.strategy.base_strategy import Position
from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("risk.stoploss")


@dataclass
class StopResult:
    """止盈止损检查结果。"""

    triggered: bool = False
    action: str = ""          # "stop_loss" / "take_profit"
    code: str = ""
    reason: str = ""


class StopLossManager:
    """止盈止损管理器。

    支持模式:
        - ``fixed``:    固定比例止损/止盈
        - ``trailing``: 移动止损
        - ``atr``:      基于 ATR 的动态止损
    """

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()

        # 止损
        self.sl_enabled: bool = cfg.get("risk.stop_loss.enabled", True)
        self.sl_method: str = cfg.get("risk.stop_loss.method", "fixed")
        self.sl_fixed_pct: float = cfg.get("risk.stop_loss.fixed_pct", 0.08)
        self.sl_trailing_pct: float = cfg.get("risk.stop_loss.trailing_pct", 0.10)
        self.sl_atr_mult: float = cfg.get("risk.stop_loss.atr_multiplier", 2.0)

        # 止盈
        self.tp_enabled: bool = cfg.get("risk.take_profit.enabled", True)
        self.tp_method: str = cfg.get("risk.take_profit.method", "fixed")
        self.tp_fixed_pct: float = cfg.get("risk.take_profit.fixed_pct", 0.20)

        # 移动止损：记录持仓期最高价
        self._highest_price: dict[str, float] = {}

    def check(self, position: Position, current_price: float, atr: float | None = None) -> StopResult:
        """检查是否触发止盈止损。"""
        if position.quantity <= 0 or position.avg_cost <= 0:
            return StopResult()

        pnl_pct = (current_price - position.avg_cost) / position.avg_cost

        # 止损检查
        if self.sl_enabled:
            result = self._check_stop_loss(position, current_price, pnl_pct, atr)
            if result.triggered:
                return result

        # 止盈检查
        if self.tp_enabled:
            result = self._check_take_profit(position, pnl_pct)
            if result.triggered:
                return result

        # 更新最高价（用于移动止损）
        code = position.code
        self._highest_price[code] = max(
            self._highest_price.get(code, current_price), current_price
        )

        return StopResult()

    def _check_stop_loss(
        self,
        position: Position,
        current_price: float,
        pnl_pct: float,
        atr: float | None,
    ) -> StopResult:
        code = position.code

        if self.sl_method == "fixed":
            if pnl_pct <= -self.sl_fixed_pct:
                return StopResult(
                    triggered=True, action="stop_loss", code=code,
                    reason=f"固定止损: 亏损 {pnl_pct:.2%}",
                )

        elif self.sl_method == "trailing":
            highest = self._highest_price.get(code, current_price)
            drawdown = (highest - current_price) / highest if highest > 0 else 0
            if drawdown >= self.sl_trailing_pct:
                return StopResult(
                    triggered=True, action="stop_loss", code=code,
                    reason=f"移动止损: 从最高价回撤 {drawdown:.2%}",
                )

        elif self.sl_method == "atr" and atr is not None:
            stop_price = position.avg_cost - atr * self.sl_atr_mult
            if current_price <= stop_price:
                return StopResult(
                    triggered=True, action="stop_loss", code=code,
                    reason=f"ATR止损: 价格 {current_price:.2f} < 止损线 {stop_price:.2f}",
                )

        return StopResult()

    def _check_take_profit(self, position: Position, pnl_pct: float) -> StopResult:
        if self.tp_method == "fixed" and pnl_pct >= self.tp_fixed_pct:
            return StopResult(
                triggered=True, action="take_profit", code=position.code,
                reason=f"固定止盈: 盈利 {pnl_pct:.2%}",
            )
        return StopResult()

    def reset(self, code: str) -> None:
        """清除某只股票的跟踪数据。"""
        self._highest_price.pop(code, None)
