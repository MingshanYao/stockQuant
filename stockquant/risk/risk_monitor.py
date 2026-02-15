"""
风险监控器 — 最大回撤熔断。
"""

from __future__ import annotations

from stockquant.backtest.context import Context
from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("risk.monitor")


class RiskMonitor:
    """账户级风险监控。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self.max_drawdown_limit: float = cfg.get("risk.max_drawdown_limit", 0.20)
        self._peak_value: float = 0.0
        self._is_frozen: bool = False

    def update(self, context: Context) -> bool:
        """每日更新风险状态。

        Returns
        -------
        bool
            True 表示触发熔断，应停止交易。
        """
        value = context.portfolio_value
        self._peak_value = max(self._peak_value, value)

        drawdown = (self._peak_value - value) / self._peak_value if self._peak_value > 0 else 0

        if drawdown >= self.max_drawdown_limit and not self._is_frozen:
            self._is_frozen = True
            logger.warning(
                f"⚠️ 触发最大回撤熔断! 当前回撤: {drawdown:.2%}, 阈值: {self.max_drawdown_limit:.2%}"
            )
            return True

        return self._is_frozen

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def current_drawdown(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._peak_value) / self._peak_value

    def reset(self) -> None:
        self._peak_value = 0.0
        self._is_frozen = False
