"""
通用截面因子选股策略 — 集成完整回撤控制框架 + 风格中性组合构建。

风险控制层次（感知→决策→执行）：
  1. 市场状态识别（牛/熊/震荡/危机）→ 仓位上限
  2. 分级回撤预警（绿/黄/橙/红）→ 仓位压缩
  3. 个股止损/止盈（固定+移动止损）→ 单仓退出

风格中性化（可选，需 cvxpy）：
  使用线性规划优化器构建行业/市值中性组合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockquant.risk.regime_detector import MarketRegimeDetector
from stockquant.risk.risk_monitor import AlertLevel, RiskMonitor
from stockquant.risk.stop_loss import StopLossManager
from stockquant.strategy.base_strategy import BaseStrategy, StrategyRegistry
from stockquant.utils.logger import get_logger

logger = get_logger("strategy.alpha_factor")

# cvxpy 可选导入
HAS_CVXPY = False
try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    pass


class AlphaFactorStrategy(BaseStrategy):
    """通用截面因子选股策略 — 集成完整回撤控制 + 风格中性组合。

    每隔 ``rebalance_freq`` 个交易日触发调仓。
    风格中性化模式下，使用 cvxpy 优化器构建行业/市值中性的投资组合。

    参数
    ----
    alpha_panel    : DataFrame  **必填**
    max_positions  : int = 50
    rebalance_freq : int = 5
    ascending      : bool = False
    label          : str = "AlphaFactor"
    enable_risk_mgmt : bool = False
    enable_style_neutral : bool = False
        使用 cvxpy 构建行业/市值中性组合（需要传入 industry_map 和 market_cap）。
    industry_map : Series, optional
        股票代码 → 行业名称的映射。
    market_cap : Series, optional
        股票代码 → 当前市值的映射。
    industry_exposure_limit : float = 0.05
        行业权重偏差上限（相对比例）。
    size_exposure_limit : float = 0.1
        市值 Z-score 偏差上限（标准差单位）。
    """

    uses_lightweight_bar = True

    def initialize(self) -> None:
        self._alpha_panel: pd.DataFrame = self.get_param("alpha_panel")
        self._max_pos: int = self.get_param("max_positions", 50)
        self._freq: int = self.get_param("rebalance_freq", 5)
        self._ascending: bool = self.get_param("ascending", False)
        self._label: str = self.get_param("label", "AlphaFactor")
        self._day_count: int = 0
        self._base_target_pct: float = 1.0 / max(self._max_pos, 1)

        self._alpha_panel.index = pd.to_datetime(self._alpha_panel.index)

        # ── 风格中性化 ──
        self._style_neutral: bool = self.get_param("enable_style_neutral", False)
        self._industry_map: pd.Series | None = self.get_param("industry_map", None)
        self._market_cap: pd.Series | None = self.get_param("market_cap", None)
        self._industry_limit: float = self.get_param("industry_exposure_limit", 0.05)
        self._size_limit: float = self.get_param("size_exposure_limit", 0.1)

        if self._style_neutral and not HAS_CVXPY:
            logger.warning(
                f"[{self._label}] enable_style_neutral=True 但 cvxpy 未安装，"
                "回退到简单 Top-N 选股。安装: pip install cvxpy"
            )
            self._style_neutral = False

        if self._style_neutral:
            logger.info(
                f"[{self._label}] 风格中性化已启用 — "
                f"行业偏差≤{self._industry_limit:.0%}, "
                f"市值偏差≤{self._size_limit:.1f}σ"
            )

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

            # 决策层: 个股止损/止盈
            self._stop_loss = StopLossManager()
            self._stop_loss.sl_enabled = True
            self._stop_loss.sl_method = "trailing"
            self._stop_loss.sl_trailing_pct = ts_pct
            self._stop_loss.sl_fixed_pct = sl_pct
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
            f"ascending={self._ascending}, risk_mgmt={self._risk_enabled}, "
            f"style_neutral={self._style_neutral}"
        )

    # ------------------------------------------------------------------
    # 风格中性化组合构建
    # ------------------------------------------------------------------

    def _build_skip_neutral(self, today_alpha: pd.Series) -> list[str]:
        """简单取 Top-N，无行业/市值约束（回退方案）。"""
        if self._ascending:
            return today_alpha.nsmallest(self._max_pos).index.tolist()
        return today_alpha.nlargest(self._max_pos).index.tolist()

    def _build_optimized_portfolio(self, today_alpha: pd.Series) -> list[str]:
        """使用 cvxpy 构建行业/市值中性的优化组合。

        约束：
          - 行业权重偏差 ≤ industry_exposure_limit
          - 市值 Z-score 偏差 ≤ size_exposure_limit
          - 满仓投资
          - 个股上限 = 1/max_positions
          - 个股下限 = 0

        目标：maximize Σ w_i × factor_score_i
        """
        if not HAS_CVXPY:
            return self._build_skip_neutral(today_alpha)

        scores = today_alpha.dropna()
        codes = scores.index.tolist()
        n = len(codes)

        if n == 0:
            return []

        if self._ascending:
            scores = -scores  # 翻转方向，统一最大化

        max_w = 1.0 / max(self._max_pos, 1)

        # 行业约束
        industry_constraints = []
        if self._industry_map is not None:
            ind = self._industry_map.reindex(codes).dropna()
            if len(ind) > 0:
                industries = ind.unique()
                for industry in industries:
                    mask = (ind == industry).values.astype(float)
                    target_weight = mask.sum() / len(ind)  # 截面行业占比
                    lower = max(0, target_weight - self._industry_limit)
                    upper = min(1.0, target_weight + self._industry_limit)
                    industry_constraints.append((mask, lower, upper))

        # 市值约束
        size_constraint = None
        if self._market_cap is not None:
            mc = self._market_cap.reindex(codes).dropna()
            if len(mc) > 0:
                log_mc = np.log(mc.replace(0, np.nan)).dropna()
                mc_z = (log_mc - log_mc.mean()) / log_mc.std()
                size_constraint = mc_z.values

        # 构建 cvxpy 问题
        w = cp.Variable(n, nonneg=True)
        score_vec = scores.loc[codes].values.astype(float)
        objective = cp.Maximize(score_vec @ w)

        constraints = [cp.sum(w) <= 1.0, w <= max_w]

        # 行业约束
        for mask, lower, upper in industry_constraints:
            industry_weight = mask @ w
            constraints.append(industry_weight >= lower)
            constraints.append(industry_weight <= upper)

        # 市值约束
        if size_constraint is not None:
            size_exposure = size_constraint @ w
            constraints.append(size_exposure >= -self._size_limit)
            constraints.append(size_exposure <= self._size_limit)

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(warm_start=True, verbose=False)
        except Exception as e:
            logger.warning(f"[{self._label}] cvxpy 求解失败: {e}，回退到简单 Top-N")
            return self._build_skip_neutral(today_alpha)

        if w.value is None:
            logger.warning(f"[{self._label}] cvxpy 未找到可行解，回退到简单 Top-N")
            return self._build_skip_neutral(today_alpha)

        selected = [codes[i] for i in range(n) if w.value[i] > 1e-6]
        return selected if selected else self._build_skip_neutral(today_alpha)

    # ------------------------------------------------------------------
    # 仓位系数
    # ------------------------------------------------------------------

    def _compute_position_scale(self) -> float:
        """计算当前应使用的仓位系数 (0.0~1.0)。"""
        regime_scale = 1.0
        dd_scale = 1.0

        if self._regime_detector is not None and self.context is not None:
            bm_prices = self.context.benchmark_prices
            if bm_prices is not None and len(bm_prices) >= 60:
                regime_scale = self._regime_detector.compute_scale(bm_prices)

        if self._risk_monitor is not None:
            dd_scale = self._risk_monitor.position_scale

        return regime_scale * dd_scale

    # ------------------------------------------------------------------
    # 主逻辑
    # ------------------------------------------------------------------

    def handle_bar(self, bar) -> None:
        # ── 感知层: 更新回撤预警 ──
        alert_level = AlertLevel.GREEN
        if self._risk_enabled and self.context is not None:
            alert_level = self._risk_monitor.update(self.context)

        # ── 决策层: 个股止损/止盈 ──
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

        # ── 调仓逻辑 ──
        self._day_count += 1
        if self._day_count % self._freq != 0:
            return
        if self.context is None:
            return

        position_scale = self._compute_position_scale() if self._risk_enabled else 1.0
        if position_scale <= 0:
            return

        effective_target_pct = self._base_target_pct * position_scale

        # ── 1. 获取当日截面因子值 ──
        current_ts = pd.Timestamp(self.context.current_date)
        valid_idx = self._alpha_panel.index[self._alpha_panel.index <= current_ts]
        if valid_idx.empty:
            return
        today_alpha = self._alpha_panel.loc[valid_idx[-1]].dropna()

        available = [c for c in today_alpha.index if c in bar]
        if not available:
            return
        today_alpha = today_alpha[available]

        # ── 2. 选中个股（风格中性 or 简单 Top-N）──
        if self._style_neutral:
            top_stocks = set(self._build_optimized_portfolio(today_alpha))
        else:
            if self._ascending:
                top_stocks = set(today_alpha.nsmallest(self._max_pos).index.tolist())
            else:
                top_stocks = set(today_alpha.nlargest(self._max_pos).index.tolist())

        # ── 3. 卖出不在名单的持仓 ──
        for code, pos in list(self.context.positions.items()):
            if pos.quantity <= 0:
                continue
            if code not in top_stocks:
                avail_qty = pos.quantity - pos.frozen
                if avail_qty > 0:
                    self.sell(code, avail_qty, reason=f"{self._label}调仓-剔除")

        # ── 4. 等权建仓 ──
        for code in top_stocks:
            self.order_target_percent(
                code,
                effective_target_pct,
                reason=f"{self._label}调仓-买入"
                + (f"({alert_level.label}, scale={position_scale:.0%})" if self._risk_enabled else ""),
            )


StrategyRegistry.register("alpha_factor", AlphaFactorStrategy)
