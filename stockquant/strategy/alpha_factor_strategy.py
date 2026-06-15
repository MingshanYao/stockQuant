"""
通用截面因子选股策略 — 集成完整回撤控制框架。

风险控制层次（感知→决策→执行）：
  1. 市场状态识别（牛/熊/震荡/危机）→ 仓位上限
  2. 分级回撤预警（绿/黄/橙/红）→ 仓位压缩
  3. 个股止损/止盈（固定+移动止损）→ 单仓退出
"""
from __future__ import annotations

import pandas as pd

from stockquant.risk.regime_detector import MarketRegimeDetector
from stockquant.risk.risk_monitor import AlertLevel, RiskMonitor
from stockquant.risk.stop_loss import StopLossManager
from stockquant.strategy.base_strategy import BaseStrategy, StrategyRegistry
from stockquant.utils.logger import get_logger

logger = get_logger("strategy.alpha_factor")


class AlphaFactorStrategy(BaseStrategy):
    """通用截面因子选股策略 — 集成完整回撤控制。

    每隔 ``rebalance_freq`` 个交易日触发调仓，调仓时按因子截面排名选 Top-N。
    每 bar 运行风险检查（优先于调仓逻辑）：

    **感知层：**
    - 市场状态识别：基于基准指数 (中证500/沪深300) 的 MA 交叉 + 波动率比值
    - 回撤预警：四级分级预警 (绿/黄/橙/红)

    **决策层：**
    - 总仓位 = 基准仓位 × 市场状态系数 × 回撤预警系数
    - 个股止损：移动止损 + 止盈

    参数
    ----
    alpha_panel    : DataFrame  **必填**
    max_positions  : int = 50
    rebalance_freq : int = 5
    ascending      : bool = False
    label          : str = "AlphaFactor"
    enable_risk_mgmt : bool = False
        启用完整回撤控制框架。
    stop_loss_pct  : float = 0.08
        固定止损比例（用于 StopLossManager 兼容）。
    take_profit_pct : float = 0.20
        固定止盈比例。
    trailing_stop_pct : float = 0.05
        移动止损回撤比例（从持仓期最高价计算）。
    max_drawdown_green  : float = 0.04  绿→黄 阈值
    max_drawdown_yellow : float = 0.07  黄→橙 阈值
    max_drawdown_orange : float = 0.10  橙→红 阈值
    """

    def initialize(self) -> None:
        self._alpha_panel: pd.DataFrame = self.get_param("alpha_panel")
        self._max_pos: int = self.get_param("max_positions", 50)
        self._freq: int = self.get_param("rebalance_freq", 5)
        self._ascending: bool = self.get_param("ascending", False)
        self._label: str = self.get_param("label", "AlphaFactor")
        self._day_count: int = 0
        self._base_target_pct: float = 1.0 / max(self._max_pos, 1)

        self._alpha_panel.index = pd.to_datetime(self._alpha_panel.index)

        # ── 风险管理 ──
        self._risk_enabled: bool = self.get_param("enable_risk_mgmt", False)

        if self._risk_enabled:
            sl_pct: float = self.get_param("stop_loss_pct", 0.08)
            tp_pct: float = self.get_param("take_profit_pct", 0.20)
            ts_pct: float = self.get_param("trailing_stop_pct", 0.05)

            # 感知层: 市场状态检测
            self._regime_detector = MarketRegimeDetector()

            # 感知层: 分级回撤预警
            self._risk_monitor = RiskMonitor(
                green_threshold=self.get_param("max_drawdown_green", 0.04),
                yellow_threshold=self.get_param("max_drawdown_yellow", 0.07),
                orange_threshold=self.get_param("max_drawdown_orange", 0.10),
            )

            # 决策层: 个股止损/止盈 — 使用移动止损模式（会跟踪持仓期最高价，
            # 从最高点回撤 > trailing_stop_pct 时触发，天然比固定止损更紧）
            self._stop_loss = StopLossManager()
            self._stop_loss.sl_enabled = True
            self._stop_loss.sl_method = "trailing"
            self._stop_loss.sl_trailing_pct = ts_pct
            self._stop_loss.sl_fixed_pct = sl_pct  # 保留作为兜底
            self._stop_loss.tp_enabled = True
            self._stop_loss.tp_fixed_pct = tp_pct

            logger.info(
                f"[{self._label}] 回撤控制框架已启用 — "
                f"移动止损={ts_pct:.0%}, 止盈={tp_pct:.0%}, "
                f"预警阈值: 绿<{self._risk_monitor.green_threshold:.0%}"
                f"<黄<{self._risk_monitor.yellow_threshold:.0%}"
                f"<橙<{self._risk_monitor.orange_threshold:.0%}<红"
            )
        else:
            self._regime_detector = None
            self._risk_monitor = None
            self._stop_loss = None

        logger.info(
            f"[{self._label}] 初始化完成 — "
            f"max_pos={self._max_pos}, freq={self._freq}, "
            f"base_target_pct={self._base_target_pct:.1%}, "
            f"ascending={self._ascending}, risk_mgmt={self._risk_enabled}"
        )

    def _compute_position_scale(self) -> float:
        """计算当前应使用的仓位系数 (0.0~1.0)。

        两层压缩：
        1. 市场状态 → 仓位上限系数
        2. 回撤预警 → 仓位压缩系数
        取两者乘积作为最终系数。
        """
        regime_scale = 1.0
        dd_scale = 1.0

        # 第一层：市场状态
        if self._regime_detector is not None and self.context is not None:
            bm_prices = self.context.benchmark_prices
            if bm_prices is not None and len(bm_prices) >= 60:
                regime_scale = self._regime_detector.compute_scale(bm_prices)

        # 第二层：回撤预警
        if self._risk_monitor is not None:
            dd_scale = self._risk_monitor.position_scale

        return regime_scale * dd_scale

    def handle_bar(self, bar: dict) -> None:
        # ── 感知层: 更新回撤预警（每 bar）──────────
        alert_level = AlertLevel.GREEN
        if self._risk_enabled and self.context is not None:
            alert_level = self._risk_monitor.update(self.context)

        # ── 决策层: 个股止损/止盈（每 bar）─────────
        if self._risk_enabled and self.context is not None:
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
                        self._stop_loss.reset(code)

        # ── 调仓逻辑（仅调仓日执行）───────────────────
        self._day_count += 1
        if self._day_count % self._freq != 0:
            return
        if self.context is None:
            return

        # ── 决策层: 计算动态仓位系数 ─────────────────
        position_scale = self._compute_position_scale() if self._risk_enabled else 1.0
        if position_scale <= 0:
            return

        effective_target_pct = self._base_target_pct * position_scale

        # ── 1. 获取当日截面因子值 ─────────────────
        current_ts = pd.Timestamp(self.context.current_date)
        valid_idx = self._alpha_panel.index[self._alpha_panel.index <= current_ts]
        if valid_idx.empty:
            return
        today_alpha = self._alpha_panel.loc[valid_idx[-1]].dropna()

        available = [c for c in today_alpha.index if c in bar and not bar[c].empty]
        if not available:
            return
        today_alpha = today_alpha[available]

        # ── 2. 截面排名取 Top N ────────────────────
        if self._ascending:
            top_stocks = set(today_alpha.nsmallest(self._max_pos).index.tolist())
        else:
            top_stocks = set(today_alpha.nlargest(self._max_pos).index.tolist())

        # ── 3. 卖出不在名单的持仓 ─────────────────
        for code, pos in list(self.context.positions.items()):
            if pos.quantity <= 0:
                continue
            if code not in top_stocks:
                avail_qty = pos.quantity - pos.frozen
                if avail_qty > 0:
                    self.sell(code, avail_qty, reason=f"{self._label}调仓-剔除")

        # ── 4. 等权建仓（使用动态仓位）─────────────
        for code in top_stocks:
            self.order_target_percent(
                code,
                effective_target_pct,
                reason=f"{self._label}调仓-买入"
                + (f"({alert_level.label}, scale={position_scale:.0%})" if self._risk_enabled else ""),
            )


StrategyRegistry.register("alpha_factor", AlphaFactorStrategy)
