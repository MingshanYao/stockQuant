"""
仓位管理器 — 控制单只与总仓位上限。
"""

from __future__ import annotations

from stockquant.backtest.context import Context
from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("risk.position")


class PositionManager:
    """仓位控制。

    规则:
        - 单只股票持仓不超过总资产的 ``max_position_pct``。
        - 总持仓不超过总资产的 ``max_total_position_pct``。
    """

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self.max_position_pct: float = cfg.get("risk.max_position_pct", 0.25)
        self.max_total_position_pct: float = cfg.get("risk.max_total_position_pct", 0.95)

    def check_buy(self, context: Context, code: str, amount: float) -> float:
        """校验买入金额，返回允许的最大买入金额。

        Parameters
        ----------
        context : Context
            回测上下文。
        code : str
            股票代码。
        amount : float
            请求买入金额。

        Returns
        -------
        float
            允许的最大买入金额（可能缩减）。
        """
        total_value = context.portfolio_value
        current_position_value = context.get_position_value(code)

        # 单只上限
        max_single = total_value * self.max_position_pct - current_position_value
        max_single = max(max_single, 0)

        # 总仓位上限
        total_position_value = sum(
            pos.quantity * pos.current_price
            for pos in context.positions.values()
        )
        max_total = total_value * self.max_total_position_pct - total_position_value
        max_total = max(max_total, 0)

        allowed = min(amount, max_single, max_total, context.cash)

        if allowed < amount:
            logger.info(
                f"仓位控制: {code} 请求 {amount:.0f}, 允许 {allowed:.0f}"
            )

        return allowed

    def get_available_buy_amount(self, context: Context, code: str) -> float:
        """获取某只股票当前可买入的最大金额。"""
        return self.check_buy(context, code, float("inf"))
