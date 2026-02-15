"""
示例策略 — 双均线交叉策略。
"""

from __future__ import annotations

import pandas as pd

from stockquant.strategy.base_strategy import BaseStrategy


class DualMAStrategy(BaseStrategy):
    """双均线交叉策略示例。

    - 短均线上穿长均线 → 买入
    - 短均线下穿长均线 → 卖出
    """

    def initialize(self) -> None:
        self.set_params(
            short_window=self.get_param("short_window", 5),
            long_window=self.get_param("long_window", 20),
            position_pct=self.get_param("position_pct", 0.2),
        )

    def handle_bar(self, bar: dict[str, pd.DataFrame]) -> None:
        short_win = self.get_param("short_window")
        long_win = self.get_param("long_window")
        position_pct = self.get_param("position_pct")

        for code, df in bar.items():
            if len(df) < long_win:
                continue

            short_ma = df["close"].rolling(short_win).mean()
            long_ma = df["close"].rolling(long_win).mean()

            # 金叉
            if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                if self.context and self.context.get_position(code).quantity == 0:
                    self.order_target_percent(code, position_pct, reason="金叉买入")

            # 死叉
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                pos = self.context.get_position(code) if self.context else None
                if pos and pos.quantity > 0:
                    self.sell(code, pos.quantity, reason="死叉卖出")
