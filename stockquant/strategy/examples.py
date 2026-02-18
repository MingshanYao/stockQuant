"""
示例策略 — 双均线交叉策略（含风险监控 + 止损 + 止盈）。
"""

from __future__ import annotations

import pandas as pd

from stockquant.strategy.base_strategy import BaseStrategy
from stockquant.risk.risk_monitor import RiskMonitor


class DualMAStrategy(BaseStrategy):
    """双均线交叉策略示例。

    信号逻辑
    --------
    - 短均线上穿长均线（金叉）→ 买入目标仓位
    - 短均线下穿长均线（死叉）→ 清仓卖出

    风险控制（优先于均线信号）
    --------------------------
    - 持仓浮亏超过 ``stop_loss_pct``  → 止损强制平仓
    - 持仓浮盈超过 ``take_profit_pct`` → 止盈强制平仓
    - 账户最大回撤超过 ``max_drawdown_limit`` → 账户级熔断，停止全部交易

    可调参数（通过 set_params 设置）
    --------------------------------
    short_window       : int   = 5      短均线窗口
    long_window        : int   = 20     长均线窗口
    position_pct       : float = 0.20   每只股票目标仓位比例
    stop_loss_pct      : float = -0.07  止损线，如 -0.07 表示 -7%
    take_profit_pct    : float = 0.20   止盈线，如  0.20 表示 +20%
    max_drawdown_limit : float = 0.20   账户最大回撤熔断阈值
    """

    def initialize(self) -> None:
        self.set_params(
            short_window=self.get_param("short_window", 5),
            long_window=self.get_param("long_window", 20),
            position_pct=self.get_param("position_pct", 0.2),
            stop_loss_pct=self.get_param("stop_loss_pct", -0.07),
            take_profit_pct=self.get_param("take_profit_pct", 0.20),
        )
        # 账户级风险监控（最大回撤熔断）
        self._risk_monitor = RiskMonitor()
        self._risk_monitor.max_drawdown_limit = self.get_param("max_drawdown_limit", 0.20)

    def handle_bar(self, bar: dict[str, pd.DataFrame]) -> None:
        # ---- 账户级熔断：触发后全部停止交易 ----
        if self.context and self._risk_monitor.update(self.context):
            return

        short_win = self.get_param("short_window")
        long_win = self.get_param("long_window")
        position_pct = self.get_param("position_pct")
        stop_loss_pct = self.get_param("stop_loss_pct")
        take_profit_pct = self.get_param("take_profit_pct")

        for code, df in bar.items():
            if len(df) < long_win:
                continue

            close = df["close"]
            short_ma = close.rolling(short_win).mean()
            long_ma = close.rolling(long_win).mean()
            current_price = close.iloc[-1]

            pos = self.context.get_position(code) if self.context else None

            # ---- 持仓中：先检查止损 / 止盈（优先于均线信号）----
            if pos and pos.quantity > 0 and pos.avg_cost > 0:
                pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost

                if pnl_pct <= stop_loss_pct:
                    self.sell(code, pos.quantity, reason=f"止损卖出 ({pnl_pct:.2%})")
                    continue

                if pnl_pct >= take_profit_pct:
                    self.sell(code, pos.quantity, reason=f"止盈卖出 ({pnl_pct:.2%})")
                    continue

            # ---- 死叉卖出 ----
            if short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                if pos and pos.quantity > 0:
                    self.sell(code, pos.quantity, reason="死叉卖出")

            # ---- 金叉买入 ----
            elif short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                if pos and pos.quantity == 0:
                    self.order_target_percent(code, position_pct, reason="金叉买入")
