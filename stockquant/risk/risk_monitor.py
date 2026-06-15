"""
风险监控器 — 分级回撤预警与动态仓位压缩。
"""
from __future__ import annotations

import enum

from stockquant.utils.logger import get_logger

logger = get_logger("risk.monitor")


class AlertLevel(enum.Enum):
    """回撤预警级别。"""
    GREEN = ("正常", 1.0)       # DD < green_threshold
    YELLOW = ("一级预警", 0.8)   # green_threshold ≤ DD < yellow_threshold
    ORANGE = ("二级预警", 0.6)   # yellow_threshold ≤ DD < orange_threshold
    RED = ("危机预警", 0.3)      # DD ≥ orange_threshold

    def __init__(self, label: str, position_scale: float):
        self.label = label
        self.position_scale = position_scale


class RiskMonitor:
    """账户级分级回撤监控。

    逐日更新峰值和回撤，返回当前预警级别。
    与旧版的区别：
    - 旧版：二元冻结（触发后永久停止交易）
    - 新版：四级预警（可恢复），每级对应不同的仓位压缩系数
    """

    def __init__(
        self,
        green_threshold: float = 0.04,
        yellow_threshold: float = 0.07,
        orange_threshold: float = 0.10,
    ) -> None:
        self.green_threshold = green_threshold
        self.yellow_threshold = yellow_threshold
        self.orange_threshold = orange_threshold
        self._peak_value: float = 0.0
        self._last_value: float = 0.0
        self._current_level: AlertLevel = AlertLevel.GREEN

    def update(self, context) -> AlertLevel:
        """从 Context 更新（兼容旧接口）。

        Returns
        -------
        AlertLevel
        """
        return self.update_value(context.portfolio_value)

    def update_value(self, portfolio_value: float) -> AlertLevel:
        """从账户总值更新并返回预警级别。

        Returns
        -------
        AlertLevel
        """
        self._last_value = portfolio_value
        self._peak_value = max(self._peak_value, portfolio_value)

        if self._peak_value <= 0:
            self._current_level = AlertLevel.GREEN
            return self._current_level

        dd = (self._peak_value - portfolio_value) / self._peak_value

        if dd >= self.orange_threshold:
            self._current_level = AlertLevel.RED
        elif dd >= self.yellow_threshold:
            self._current_level = AlertLevel.ORANGE
        elif dd >= self.green_threshold:
            self._current_level = AlertLevel.YELLOW
        else:
            self._current_level = AlertLevel.GREEN

        if self._current_level != AlertLevel.GREEN:
            logger.warning(
                f"回撤预警: {self._current_level.label} "
                f"(当前DD={dd:.2%}, 峰值={self._peak_value:,.0f})"
            )

        return self._current_level

    @property
    def current_level(self) -> AlertLevel:
        return self._current_level

    @property
    def current_drawdown(self) -> float:
        """最近一次 update 时的回撤比例。"""
        if self._peak_value <= 0 or self._last_value <= 0:
            return 0.0
        return (self._peak_value - self._last_value) / self._peak_value

    @property
    def position_scale(self) -> float:
        """当前预警级别对应的仓位压缩系数。"""
        return self._current_level.position_scale

    @property
    def is_frozen(self) -> bool:
        """兼容旧接口：RED 级别视为熔断。"""
        return self._current_level == AlertLevel.RED

    def reset(self) -> None:
        self._peak_value = 0.0
        self._last_value = 0.0
        self._current_level = AlertLevel.GREEN
