"""
绩效分析器 — 收益 / 风险 / 风险调整 / 交易统计。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockquant.utils.logger import get_logger

logger = get_logger("analysis.performance")

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03  # 年化无风险利率


class PerformanceAnalyzer:
    """回测绩效分析器。"""

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        daily_returns: list[float] | None = None,
        benchmark_returns: list[float] | None = None,
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> None:
        self.equity = equity_curve
        self.trades = trade_log
        self.returns = np.array(daily_returns or [])
        self.benchmark_returns = np.array(benchmark_returns or [])
        self.rf = risk_free_rate

    # ==================================================================
    # 收益指标
    # ==================================================================

    def total_return(self) -> float:
        """总收益率。"""
        if self.equity.empty:
            return 0.0
        return (
            self.equity["total_value"].iloc[-1] / self.equity["total_value"].iloc[0] - 1
        )

    def annualized_return(self) -> float:
        """年化收益率。"""
        n_days = len(self.returns)
        if n_days <= 0:
            return 0.0
        total = self.total_return()
        years = n_days / TRADING_DAYS_PER_YEAR
        return (1 + total) ** (1 / years) - 1 if years > 0 else 0.0

    def alpha(self) -> float:
        """Alpha（相对基准的超额收益）。"""
        if len(self.benchmark_returns) == 0:
            return self.annualized_return() - self.rf
        benchmark_annual = self._annualize_returns(self.benchmark_returns)
        return self.annualized_return() - benchmark_annual

    def beta(self) -> float:
        """Beta（相对基准的系统性风险敞口）。"""
        if len(self.benchmark_returns) == 0 or len(self.returns) == 0:
            return 1.0
        n = min(len(self.returns), len(self.benchmark_returns))
        cov = np.cov(self.returns[:n], self.benchmark_returns[:n])
        if cov[1, 1] == 0:
            return 1.0
        return cov[0, 1] / cov[1, 1]

    # ==================================================================
    # 风险指标
    # ==================================================================

    def max_drawdown(self) -> float:
        """最大回撤。"""
        if self.equity.empty:
            return 0.0
        values = self.equity["total_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return float(np.max(drawdown))

    def max_drawdown_duration(self) -> int:
        """最大回撤持续天数。"""
        if self.equity.empty:
            return 0
        values = self.equity["total_value"].values
        peak = np.maximum.accumulate(values)
        in_drawdown = values < peak
        max_dur = 0
        current = 0
        for is_dd in in_drawdown:
            if is_dd:
                current += 1
                max_dur = max(max_dur, current)
            else:
                current = 0
        return max_dur

    def volatility(self) -> float:
        """年化波动率。"""
        if len(self.returns) == 0:
            return 0.0
        return float(np.std(self.returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    def var(self, confidence: float = 0.95) -> float:
        """历史 VaR（Value at Risk）。"""
        if len(self.returns) == 0:
            return 0.0
        return float(np.percentile(self.returns, (1 - confidence) * 100))

    # ==================================================================
    # 风险调整收益
    # ==================================================================

    def sharpe_ratio(self) -> float:
        """夏普比率。"""
        vol = self.volatility()
        if vol == 0:
            return 0.0
        return (self.annualized_return() - self.rf) / vol

    def sortino_ratio(self) -> float:
        """索提诺比率（仅考虑下行波动）。"""
        if len(self.returns) == 0:
            return 0.0
        downside = self.returns[self.returns < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if downside_std == 0:
            return 0.0
        return (self.annualized_return() - self.rf) / downside_std

    def calmar_ratio(self) -> float:
        """卡玛比率（年化收益 / 最大回撤）。"""
        mdd = self.max_drawdown()
        if mdd == 0:
            return 0.0
        return self.annualized_return() / mdd

    # ==================================================================
    # 交易统计
    # ==================================================================

    def trade_statistics(self) -> dict[str, Any]:
        """交易统计摘要。"""
        if self.trades.empty:
            return {"总交易次数": 0}

        buys = self.trades[self.trades["direction"] == "buy"]
        sells = self.trades[self.trades["direction"] == "sell"]

        # 简化盈亏计算
        stats: dict[str, Any] = {
            "总交易次数": len(self.trades),
            "买入次数": len(buys),
            "卖出次数": len(sells),
            "总佣金": float(self.trades["commission"].sum()),
        }

        return stats

    # ==================================================================
    # 综合报告
    # ==================================================================

    def full_report(self) -> dict[str, Any]:
        """生成完整绩效报告。"""
        report = {
            # 收益
            "总收益率": f"{self.total_return():.2%}",
            "年化收益率": f"{self.annualized_return():.2%}",
            "Alpha": f"{self.alpha():.2%}",
            "Beta": f"{self.beta():.2f}",
            # 风险
            "最大回撤": f"{self.max_drawdown():.2%}",
            "最大回撤天数": self.max_drawdown_duration(),
            "年化波动率": f"{self.volatility():.2%}",
            "VaR(95%)": f"{self.var():.4f}",
            # 风险调整
            "夏普比率": f"{self.sharpe_ratio():.2f}",
            "索提诺比率": f"{self.sortino_ratio():.2f}",
            "卡玛比率": f"{self.calmar_ratio():.2f}",
        }
        report.update(self.trade_statistics())

        logger.info("绩效报告:")
        for k, v in report.items():
            logger.info(f"  {k}: {v}")

        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _annualize_returns(returns: np.ndarray) -> float:
        cumulative = (1 + returns).prod() - 1
        years = len(returns) / TRADING_DAYS_PER_YEAR
        return (1 + cumulative) ** (1 / years) - 1 if years > 0 else 0.0
