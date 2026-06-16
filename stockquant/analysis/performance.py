"""
绩效分析器 — 收益 / 风险 / 风险调整 / 交易统计 / Alpha 收益。
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
    # 交易统计（增强版：包含胜率、盈亏比、Alpha盈亏等）
    # ==================================================================

    def _pair_trades(self) -> pd.DataFrame:
        """将交易日志中的买卖配对，返回每笔完整交易的盈亏明细。

        Returns
        -------
        pd.DataFrame
            每行一笔完整交易，包含：code, buy_date, sell_date, buy_price, sell_price,
            shares, pnl, pnl_pct, holding_days, commission_total
        """
        if self.trades.empty:
            return pd.DataFrame()

        rows = []
        # 按股票代码和时间排序，配对买卖
        for code, group in self.trades.sort_values("date").groupby("code"):
            buys = group[group["direction"] == "buy"]
            sells = group[group["direction"] == "sell"]
            for _, buy in buys.iterrows():
                # 对于每笔买入，找对应的卖出
                # 如果卖出行数少于买入，说明部分未平仓
                if len(sells) == 0:
                    continue
                sell = sells.iloc[0]
                sells = sells.iloc[1:]

                buy_value = buy["price"] * buy.get("shares", buy.get("quantity", 0))
                sell_value = sell["price"] * sell.get("shares", sell.get("quantity", 0))
                shares = min(
                    buy.get("shares", buy.get("quantity", 0)),
                    sell.get("shares", sell.get("quantity", 0)),
                )
                buy_comm = buy.get("commission", 0.0)
                sell_comm = sell.get("commission", 0.0)
                total_comm = buy_comm + sell_comm

                pnl = sell_value - buy_value - total_comm
                pnl_pct = pnl / buy_value if buy_value > 0 else 0.0

                buy_date = pd.Timestamp(buy["date"])
                sell_date = pd.Timestamp(sell["date"])
                holding_days = (sell_date - buy_date).days

                rows.append({
                    "code": code,
                    "buy_date": buy_date,
                    "sell_date": sell_date,
                    "buy_price": buy["price"],
                    "sell_price": sell["price"],
                    "shares": shares,
                    "buy_value": buy_value,
                    "sell_value": sell_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                    "commission_total": total_comm,
                })

        result = pd.DataFrame(rows)
        if len(result) == 0:
            return result
        result["pnl"] = result["pnl"].astype(float)
        result["pnl_pct"] = result["pnl_pct"].astype(float)
        return result

    def _rolling_beta(self, window: int = 60) -> np.ndarray:
        """计算策略 vs 基准的滚动 Beta。

        Returns
        -------
        np.ndarray
            长度与 self.returns 相同，每个位置的滚动 Beta 值。
            前 window-1 个值为初始期 Beta（用全部数据计算）。
        """
        if len(self.returns) == 0 or len(self.benchmark_returns) == 0:
            return np.ones_like(self.returns)

        n = min(len(self.returns), len(self.benchmark_returns))
        strat = self.returns[:n]
        bench = self.benchmark_returns[:n]

        betas = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - window + 1)
            seg_strat = strat[start : i + 1]
            seg_bench = bench[start : i + 1]
            if len(seg_strat) < 5:
                continue
            try:
                cov = np.cov(seg_strat, seg_bench)
                if cov[1, 1] > 1e-12:
                    betas[i] = cov[0, 1] / cov[1, 1]
            except Exception:
                pass

        # 前 window-1 个 NaN 用全期 Beta 填充
        full_beta = self.beta()
        betas = np.where(np.isfinite(betas), betas, full_beta)
        return betas

    def paired_trades(self) -> pd.DataFrame:
        """返回配对后的交易明细（对外接口）。"""
        return self._pair_trades()

    def excess_returns(self) -> pd.Series:
        """超额收益率序列（策略 - 基准，逐日）。

        存在基准数据时返回策略日收益率减去基准日收益率；
        缺少基准数据时返回全零序列。
        """
        if len(self.benchmark_returns) == 0:
            return pd.Series(np.zeros_like(self.returns), name="excess")
        n = min(len(self.returns), len(self.benchmark_returns))
        return pd.Series(
            self.returns[:n] - self.benchmark_returns[:n],
            name="excess",
        )

    def alpha_returns(self, window: int = 60) -> pd.Series:
        """纯 Alpha 收益率序列（策略 - 滚动Beta × 基准，逐日）。

        Parameters
        ----------
        window : int
            滚动 Beta 计算窗口（交易日数），默认 60。

        Returns
        -------
        pd.Series
            每日 Alpha 收益率。
        """
        if len(self.benchmark_returns) == 0:
            return pd.Series(np.zeros_like(self.returns), name="alpha")

        betas = self._rolling_beta(window=window)
        n = min(len(self.returns), len(betas), len(self.benchmark_returns))
        return pd.Series(
            self.returns[:n] - betas[:n] * self.benchmark_returns[:n],
            name="alpha",
        )

    def cumulative_excess(self) -> pd.Series:
        """累计超额收益曲线（从 1.0 开始）。"""
        excess = self.excess_returns()
        return (1 + excess).cumprod()

    def cumulative_alpha(self, window: int = 60) -> pd.Series:
        """累计纯 Alpha 收益曲线（从 1.0 开始）。

        Parameters
        ----------
        window : int
            滚动 Beta 计算窗口，默认 60。
        """
        alpha = self.alpha_returns(window=window)
        return (1 + alpha).cumprod()

    def per_trade_alpha_pnl(self) -> pd.DataFrame:
        """计算每笔交易的 Alpha 盈亏。

        对每笔配对交易：
        1. 计算持仓期间基准的累计收益率
        2. 使用该股票的滚动 Beta
        3. Alpha PnL = 实际 PnL - Beta × (投入资金 × 基准累计收益)

        Returns
        -------
        pd.DataFrame
            配对交易明细 + alpha_pnl, alpha_pnl_pct 列。
        """
        paired = self._pair_trades()
        if paired.empty or len(self.benchmark_returns) == 0:
            paired["alpha_pnl"] = paired["pnl"]
            paired["alpha_pnl_pct"] = paired["pnl_pct"]
            return paired

        # 从 equity curve 中提取日期索引
        if self.equity.empty or "date" not in self.equity.columns:
            paired["alpha_pnl"] = paired["pnl"]
            paired["alpha_pnl_pct"] = paired["pnl_pct"]
            return paired

        equity_dates = pd.to_datetime(self.equity["date"]).sort_values()
        # 基准收益率的时间索引
        n = min(len(equity_dates), len(self.benchmark_returns))
        bench_series = pd.Series(
            self.benchmark_returns[:n] + 1,  # 转为日收益率因子
            index=equity_dates[:n],
        ).cumprod()
        # 转为以 1.0 为起点的基准净值
        bench_nav = bench_series / bench_series.iloc[0]

        # 策略滚动 Beta 序列
        betas = self._rolling_beta(window=60)

        alpha_pnls = []
        for _, trade in paired.iterrows():
            buy_ts = trade["buy_date"]
            sell_ts = trade["sell_date"]

            # 持仓期间基准累计收益
            mask = (bench_nav.index >= buy_ts) & (bench_nav.index <= sell_ts)
            period_bench = bench_nav[mask]
            if len(period_bench) < 2:
                bench_return = 0.0
            else:
                bench_return = period_bench.iloc[-1] / period_bench.iloc[0] - 1

            # 用持仓期内的平均 Beta
            idx_start = equity_dates.get_loc(buy_ts, method="ffill") if buy_ts in equity_dates else 0
            idx_end = equity_dates.get_loc(sell_ts, method="ffill") if sell_ts in equity_dates else len(betas)
            trade_beta = np.nanmean(betas[max(0, idx_start) : min(len(betas), idx_end + 1)])
            if not np.isfinite(trade_beta):
                trade_beta = 1.0

            # Alpha PnL = 实际 PnL - Beta × 资金成本
            capital = trade["buy_value"]
            alpha = trade["pnl"] - trade_beta * capital * bench_return
            alpha_pct = alpha / capital if capital > 0 else 0.0
            alpha_pnls.append({"alpha_pnl": alpha, "alpha_pnl_pct": alpha_pct})

        if alpha_pnls:
            alpha_df = pd.DataFrame(alpha_pnls, index=paired.index)
            paired = pd.concat([paired, alpha_df], axis=1)
        else:
            paired["alpha_pnl"] = paired["pnl"]
            paired["alpha_pnl_pct"] = paired["pnl_pct"]

        return paired

    def trade_statistics(self) -> dict[str, Any]:
        """交易统计摘要（含胜率/盈亏比/Profit Factor 等）。"""
        if self.trades.empty:
            return {"总交易次数": 0}

        buys = self.trades[self.trades["direction"] == "buy"]
        sells = self.trades[self.trades["direction"] == "sell"]

        stats: dict[str, Any] = {
            "总交易次数": len(self.trades),
            "买入次数": len(buys),
            "卖出次数": len(sells),
            "总佣金": float(self.trades["commission"].sum()),
        }

        # 配对交易计算胜率
        paired = self._pair_trades()
        if paired.empty:
            return stats

        n_trades = len(paired)
        winning = paired[paired["pnl"] > 0]
        losing = paired[paired["pnl"] < 0]
        flat = paired[paired["pnl"] == 0]

        n_win = len(winning)
        n_lose = len(losing)

        win_rate = n_win / n_trades if n_trades > 0 else 0.0
        avg_win = winning["pnl"].mean() if n_win > 0 else 0.0
        avg_loss = losing["pnl"].mean() if n_lose > 0 else 0.0
        total_win = winning["pnl"].sum() if n_win > 0 else 0.0
        total_loss = abs(losing["pnl"].sum()) if n_lose > 0 else 0.0

        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # 最大连胜/连败
        pnl_signs = paired["pnl"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).values
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        current_sign = 0
        for s in pnl_signs:
            if s == 0:
                current_streak = 0
                current_sign = 0
            elif s == current_sign:
                current_streak += 1
            else:
                current_streak = 1
                current_sign = s
            if current_sign == 1:
                max_win_streak = max(max_win_streak, current_streak)
            elif current_sign == -1:
                max_loss_streak = max(max_loss_streak, current_streak)

        # 平均持仓天数（按盈亏分组）
        avg_hold_win = float(winning["holding_days"].mean()) if n_win > 0 else 0.0
        avg_hold_loss = float(losing["holding_days"].mean()) if n_lose > 0 else 0.0

        stats.update({
            "配对交易次数": n_trades,
            "盈利交易": n_win,
            "亏损交易": n_lose,
            "持平交易": len(flat),
            "胜率": win_rate,
            "平均盈利": avg_win,
            "平均亏损": avg_loss,
            "盈亏比": win_loss_ratio,
            "Profit Factor": profit_factor,
            "最大连胜": max_win_streak,
            "最大连败": max_loss_streak,
            "平均持仓天数(盈利)": avg_hold_win,
            "平均持仓天数(亏损)": avg_hold_loss,
        })

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
