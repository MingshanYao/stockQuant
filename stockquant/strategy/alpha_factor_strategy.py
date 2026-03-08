"""
通用截面因子选股策略 — 基于任意 Alpha 因子面板的等权轮动策略。

适用于 WorldQuant Alpha#101 或任何形如 ``(日期 × 股票代码)`` 面板的因子数据，
每隔 ``rebalance_freq`` 个交易日根据截面排名选 Top-N 只股票等权持仓。

典型用法
--------
>>> from stockquant.strategy import AlphaFactorStrategy
>>>
>>> strategy = AlphaFactorStrategy()
>>> strategy.set_params(
...     alpha_panel    = alpha001_panel,   # DataFrame: index=日期, columns=股票代码
...     max_positions  = 10,
...     rebalance_freq = 5,                # 每 5 个交易日调仓
... )
"""

from __future__ import annotations

import pandas as pd

from stockquant.strategy.base_strategy import BaseStrategy
from stockquant.utils.logger import get_logger

logger = get_logger("strategy.alpha_factor")


class AlphaFactorStrategy(BaseStrategy):
    """通用截面因子选股策略。

    每隔 ``rebalance_freq`` 个交易日触发调仓：
    1. 获取当日（或最近可用日）的截面因子值；
    2. 在当日有行情的股票中，按因子值排名选出 Top-N；
    3. 卖出不在名单的持仓（遵守 T+1 冻结限制）；
    4. 对名单内股票按目标仓位 ``1 / max_positions`` 等权调仓。

    参数（通过 :meth:`set_params` 设置）
    -------------------------------------
    alpha_panel    : DataFrame  **必填**  ``index=日期, columns=股票代码``
    max_positions  : int = 10   最多同时持仓只数
    rebalance_freq : int = 5    调仓频率（交易日数）
    ascending      : bool = False
        因子排序方向。``False`` 表示值越大越好（适用于大多数 Alpha 因子），
        ``True`` 表示值越小越好。
    label          : str = "AlphaFactor"
        策略标签，用于日志和交易备注。
    """

    def initialize(self) -> None:
        self._alpha_panel: pd.DataFrame = self.get_param("alpha_panel")
        self._max_pos: int = self.get_param("max_positions", 10)
        self._freq: int = self.get_param("rebalance_freq", 5)
        self._ascending: bool = self.get_param("ascending", False)
        self._label: str = self.get_param("label", "AlphaFactor")
        self._day_count: int = 0
        self._target_pct: float = 1.0 / max(self._max_pos, 1)

        # 统一为 DatetimeIndex，以便与 context.current_date 对齐
        self._alpha_panel.index = pd.to_datetime(self._alpha_panel.index)

        logger.info(
            f"[{self._label}] 初始化完成 — "
            f"max_pos={self._max_pos}, freq={self._freq}, "
            f"target_pct={self._target_pct:.1%}, ascending={self._ascending}"
        )

    def handle_bar(self, bar: dict) -> None:
        self._day_count += 1

        # ── 非调仓日跳过 ──────────────────────────────────────────
        if self._day_count % self._freq != 0:
            return

        if self.context is None:
            return

        # ── 1. 获取当日（或最近可用日）截面因子值 ─────────────────
        current_ts = pd.Timestamp(self.context.current_date)
        valid_idx = self._alpha_panel.index[self._alpha_panel.index <= current_ts]
        if valid_idx.empty:
            return
        today_alpha = self._alpha_panel.loc[valid_idx[-1]].dropna()

        # 只保留当日 bar 中有行情的股票
        available = [c for c in today_alpha.index if c in bar and not bar[c].empty]
        if not available:
            return
        today_alpha = today_alpha[available]

        # ── 2. 截面排名取 Top N ────────────────────────────────────
        if self._ascending:
            # 值越小越好（如反转因子）
            top_stocks = set(today_alpha.nsmallest(self._max_pos).index.tolist())
        else:
            # 值越大越好（适用于大多数 alpha 因子）
            top_stocks = set(today_alpha.nlargest(self._max_pos).index.tolist())

        # ── 3. 卖出不在目标名单的现有持仓 ─────────────────────────
        for code, pos in list(self.context.positions.items()):
            if pos.quantity <= 0:
                continue
            if code not in top_stocks:
                # T+1: 扣除冻结（当日买入不可卖）
                avail_qty = pos.quantity - pos.frozen
                if avail_qty > 0:
                    self.sell(code, avail_qty, reason=f"{self._label}调仓-剔除")

        # ── 4. 等权建仓 / 调整目标名单中的股票 ────────────────────
        for code in top_stocks:
            self.order_target_percent(
                code,
                self._target_pct,
                reason=f"{self._label}调仓-买入",
            )
