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
>>>
>>> # 可选启用风险管理
>>> strategy.set_params(
...     enable_risk_mgmt = True,
...     stop_loss_pct    = 0.08,
...     take_profit_pct  = 0.20,
...     max_drawdown_limit = 0.20,
... )
"""

from __future__ import annotations

import pandas as pd

from stockquant.risk.risk_monitor import RiskMonitor
from stockquant.risk.stop_loss import StopLossManager
from stockquant.strategy.base_strategy import BaseStrategy, StrategyRegistry
from stockquant.utils.logger import get_logger

logger = get_logger("strategy.alpha_factor")


class AlphaFactorStrategy(BaseStrategy):
    """通用截面因子选股策略。

    每隔 ``rebalance_freq`` 个交易日触发调仓：
    1. 获取当日（或最近可用日）的截面因子值；
    2. 在当日有行情的股票中，按因子值排名选出 Top-N；
    3. 卖出不在名单的持仓（遵守 T+1 冻结限制）；
    4. 对名单内股票按目标仓位 ``1 / max_positions`` 等权调仓。

    可选风险管理（通过 ``enable_risk_mgmt`` 开启）：
    - 固定比例止损/止盈（每 bar 独立检查）
    - 账户级最大回撤熔断

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
    enable_risk_mgmt: bool = False
        是否启用风险管理（止损/止盈 + 回撤熔断）。
    stop_loss_pct  : float = 0.08
        固定止损比例（亏损超过此值触发卖出）。
    take_profit_pct: float = 0.20
        固定止盈比例（盈利超过此值触发卖出）。
    max_drawdown_limit: float = 0.20
        账户最大回撤熔断阈值。
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

        # ── 风险管理 ──
        self._risk_enabled: bool = self.get_param("enable_risk_mgmt", False)

        if self._risk_enabled:
            sl_pct: float = self.get_param("stop_loss_pct", 0.08)
            tp_pct: float = self.get_param("take_profit_pct", 0.20)
            dd_limit: float = self.get_param("max_drawdown_limit", 0.20)

            self._stop_loss = StopLossManager()
            self._stop_loss.sl_fixed_pct = sl_pct
            self._stop_loss.tp_fixed_pct = tp_pct

            self._risk_monitor = RiskMonitor()
            self._risk_monitor.max_drawdown_limit = dd_limit

            logger.info(
                f"[{self._label}] 风险管理已启用 — "
                f"止损={sl_pct:.0%}, 止盈={tp_pct:.0%}, "
                f"回撤熔断={dd_limit:.0%}"
            )
        else:
            self._stop_loss = None
            self._risk_monitor = None

        logger.info(
            f"[{self._label}] 初始化完成 — "
            f"max_pos={self._max_pos}, freq={self._freq}, "
            f"target_pct={self._target_pct:.1%}, ascending={self._ascending}, "
            f"risk_mgmt={self._risk_enabled}"
        )

    def handle_bar(self, bar: dict) -> None:
        # ── 风险管理（每 bar 运行，优先于调仓逻辑）────
        if self._risk_enabled and self.context is not None:
            # 1. 账户级回撤熔断
            if self._risk_monitor.update(self.context):
                return

            # 2. 持仓止损/止盈检查
            for code, pos in list(self.context.positions.items()):
                if pos.quantity <= 0:
                    continue
                current_price = self.context.get_current_price(code)
                if current_price <= 0:
                    continue
                result = self._stop_loss.check(pos, current_price)
                if result.triggered:
                    avail_qty = pos.quantity - pos.frozen
                    if avail_qty > 0:
                        self.sell(code, avail_qty, reason=result.reason)

        # ── 调仓逻辑（仅调仓日执行）───────────────────
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


StrategyRegistry.register("alpha_factor", AlphaFactorStrategy)
