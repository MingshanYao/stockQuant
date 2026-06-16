"""
可视化引擎 — 收益曲线 / 回撤 / K线 / 热力图。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stockquant.utils.logger import get_logger

logger = get_logger("visualization")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class PlotEngine:
    """统一可视化接口，支持 Matplotlib 和 Plotly 后端。"""

    def __init__(self, backend: str = "matplotlib") -> None:
        self.backend = backend

    # ------------------------------------------------------------------
    # 1. 收益曲线
    # ------------------------------------------------------------------

    def plot_equity_curve(
        self,
        equity: pd.DataFrame,
        benchmark: pd.DataFrame | None = None,
        title: str = "策略收益曲线",
        save_path: str | None = None,
    ) -> None:
        """绘制策略净值曲线。"""
        if self.backend == "plotly" and HAS_PLOTLY:
            self._plotly_equity(equity, benchmark, title, save_path)
        elif HAS_MPL:
            self._mpl_equity(equity, benchmark, title, save_path)

    def _mpl_equity(self, equity, benchmark, title, save_path):
        fig, ax = plt.subplots(figsize=(14, 6))
        nav = equity["total_value"] / equity["total_value"].iloc[0]
        ax.plot(equity["date"], nav, label="策略", linewidth=1.5)
        if benchmark is not None and "close" in benchmark.columns:
            bm_nav = benchmark["close"] / benchmark["close"].iloc[0]
            ax.plot(benchmark["date"], bm_nav, label="基准", linewidth=1, alpha=0.7)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    def _plotly_equity(self, equity, benchmark, title, save_path):
        fig = go.Figure()
        nav = equity["total_value"] / equity["total_value"].iloc[0]
        fig.add_trace(go.Scatter(x=equity["date"], y=nav, name="策略"))
        if benchmark is not None and "close" in benchmark.columns:
            bm_nav = benchmark["close"] / benchmark["close"].iloc[0]
            fig.add_trace(go.Scatter(x=benchmark["date"], y=bm_nav, name="基准", opacity=0.7))
        fig.update_layout(title=title, xaxis_title="日期", yaxis_title="净值")
        if save_path:
            fig.write_html(save_path)
        fig.show()

    # ------------------------------------------------------------------
    # 2. 回撤曲线
    # ------------------------------------------------------------------

    def plot_drawdown(
        self,
        equity: pd.DataFrame,
        title: str = "策略回撤",
        save_path: str | None = None,
    ) -> None:
        """绘制回撤曲线。"""
        values = equity["total_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak

        if self.backend == "plotly" and HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity["date"], y=-drawdown, fill="tozeroy", name="回撤",
            ))
            fig.update_layout(title=title, yaxis_title="回撤", yaxis_tickformat=".1%")
            if save_path:
                fig.write_html(save_path)
            fig.show()
        elif HAS_MPL:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.fill_between(equity["date"], -drawdown, 0, alpha=0.5, color="red")
            ax.set_title(title)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.grid(True, alpha=0.3)
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # 3. K线 + 买卖点
    # ------------------------------------------------------------------

    def build_kline_traces(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        name: str = "",
    ) -> list:
        """构建 K 线 + 成交量的 Plotly traces 列表（不显示，用于组合子图）。

        Parameters
        ----------
        df : DataFrame
            OHLCV 数据，需包含 date/open/high/low/close 列，可选 volume。
        trades : DataFrame, optional
            买卖点，需包含 date, price, direction 列。
        name : str
            K 线名称前缀（用于图例）。

        Returns
        -------
        list[tuple[go.BaseTraceType, str]]
            每个元素为 ``(trace, target_row)``，target_row 为
            ``"candle"`` 或 ``"volume"``，方便调用方添加到对应子图行。
        """
        if not HAS_PLOTLY:
            logger.warning("K线图需要 plotly，请安装: pip install plotly")
            return []

        traces: list[tuple] = []
        x = df["date"] if "date" in df.columns else df.index

        # K 线
        traces.append((
            go.Candlestick(
                x=x, open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name=f"{name} K线" if name else "K线",
                increasing_line_color="red",
                decreasing_line_color="green",
                showlegend=bool(name),
            ),
            "candle",
        ))

        # 成交量
        if "volume" in df.columns:
            colors = [
                "red" if c >= o else "green"
                for c, o in zip(df["close"], df["open"])
            ]
            traces.append((
                go.Bar(
                    x=x, y=df["volume"],
                    name="成交量",
                    marker_color=colors, opacity=0.5,
                    showlegend=False,
                ),
                "volume",
            ))

        # 买卖点
        if trades is not None and not trades.empty:
            buys = trades[trades["direction"] == "buy"]
            sells = trades[trades["direction"] == "sell"]
            if not buys.empty:
                traces.append((
                    go.Scatter(
                        x=buys["date"], y=buys["price"],
                        mode="markers", name="买入",
                        marker=dict(symbol="triangle-up", size=10, color="magenta"),
                    ),
                    "candle",
                ))
            if not sells.empty:
                traces.append((
                    go.Scatter(
                        x=sells["date"], y=sells["price"],
                        mode="markers", name="卖出",
                        marker=dict(symbol="triangle-down", size=10, color="blue"),
                    ),
                    "candle",
                ))

        return traces

    def plot_kline(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        title: str = "K线图",
        save_path: str | None = None,
    ) -> None:
        """绘制 K 线图并标注买卖点。"""
        if not HAS_PLOTLY:
            logger.warning("K线图需要 plotly，请安装: pip install plotly")
            return

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        for trace, target in self.build_kline_traces(df, trades):
            row = 1 if target == "candle" else 2
            fig.add_trace(trace, row=row, col=1)

        fig.update_layout(title=title, xaxis_rangeslider_visible=False)
        if save_path:
            fig.write_html(save_path)
        fig.show()

    # ------------------------------------------------------------------
    # 4. 月度收益热力图
    # ------------------------------------------------------------------

    def plot_monthly_heatmap(
        self,
        daily_returns: pd.Series,
        title: str = "月度收益热力图",
        save_path: str | None = None,
    ) -> None:
        """绘制月度收益热力图。"""
        if not HAS_MPL:
            return

        monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        pivot = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }).pivot(index="year", columns="month", values="return")

        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, format="%.1%%")
        ax.set_title(title)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 5. 三面板性能图（净值 + 超额 + Alpha + 月度Alpha热力图）
    # ------------------------------------------------------------------

    def plot_performance_dashboard(
        self,
        result,
        title: str = "策略绩效看板",
        save_path: str | None = None,
    ) -> None:
        """三面板绩效图。

        面板 1（顶部）：策略净值 + 基准净值 + 累计超额收益
        面板 2（中部）：纯 Alpha 累计收益曲线（剥离滚动Beta）
        面板 3（底部）：月度 Alpha 收益热力图
        """
        if not HAS_MPL:
            logger.warning("三面板图需要 matplotlib")
            return

        analyzer = result.analyze()
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

        # ---- 面板 1：净值 + 基准 + 超额 ----
        ax = axes[0]
        equity = result.equity_curve
        dates = pd.to_datetime(equity["date"])
        nav = equity["total_value"].values / equity["total_value"].iloc[0]
        ax.plot(dates, nav, label="策略净值", linewidth=1.5, color="#2c3e50")

        if not result.benchmark.empty and "close" in result.benchmark.columns:
            bm = result.benchmark
            bm_dates = pd.to_datetime(bm["date"])
            bm_nav = bm["close"].values / bm["close"].iloc[0]
            ax.plot(bm_dates, bm_nav, label="基准净值", linewidth=1, alpha=0.7, color="#95a5a6")

        excess = analyzer.cumulative_excess()
        if len(excess) > 0:
            excess_dates = dates[: len(excess)]
            ax.plot(excess_dates, excess.values, label="累计超额",
                    linewidth=1.5, linestyle="--", color="#e67e22")

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("净值", fontsize=11)
        ax.set_title(f"{title} — 策略 vs 基准 vs 超额", fontsize=13, loc="left")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

        # ---- 面板 2：纯 Alpha 累计收益 ----
        ax = axes[1]
        cum_alpha = analyzer.cumulative_alpha(window=60)
        if len(cum_alpha) > 0:
            alpha_dates = dates[: len(cum_alpha)]
            ax.plot(alpha_dates, cum_alpha.values, label="纯Alpha（剥离Beta）",
                    linewidth=1.5, color="#27ae60")
            ax.fill_between(
                alpha_dates, 1.0, cum_alpha.values,
                where=(cum_alpha.values >= 1.0),
                color="#27ae60", alpha=0.15,
            )
            ax.fill_between(
                alpha_dates, 1.0, cum_alpha.values,
                where=(cum_alpha.values < 1.0),
                color="#e74c3c", alpha=0.15,
            )

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Alpha净值", fontsize=11)
        ax.set_title("纯 Alpha 收益曲线（滚动60日Beta剥离）", fontsize=13, loc="left")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

        # ---- 面板 3：月度 Alpha 收益热力图 ----
        ax = axes[2]
        alpha_returns = analyzer.alpha_returns(window=60)
        if len(alpha_returns) > 0:
            alpha_index = pd.to_datetime(dates[: len(alpha_returns)])
            alpha_s = pd.Series(alpha_returns.values, index=alpha_index)
            monthly = alpha_s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            pivot = pd.DataFrame({
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values,
            }).pivot(index="year", columns="month", values="return")

            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(12))
            ax.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=8)
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index, fontsize=8)

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > 0.15 else "black"
                        ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                                fontsize=7, color=color)
            plt.colorbar(im, ax=ax, format="%.1%%", fraction=0.02, pad=0.04)

        ax.set_title("月度 Alpha 收益热力图", fontsize=13, loc="left")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"绩效看板已保存: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # 6. 交易仪表盘（胜率 / 盈亏分布）
    # ------------------------------------------------------------------

    def plot_trade_dashboard(
        self,
        analyzer,
        title: str = "交易分析仪表盘",
        save_path: str | None = None,
    ) -> None:
        """交易仪表盘。

        左侧：胜率卡片（胜率、盈亏比、Profit Factor、最大连败/连胜）
        右侧：逐笔交易盈亏分布直方图
        """
        if not HAS_MPL:
            logger.warning("交易仪表盘需要 matplotlib")
            return

        stats = analyzer.trade_statistics()
        paired = analyzer.paired_trades()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ---- 左侧：胜率卡片 ----
        ax = axes[0]
        ax.axis("off")

        win_rate = stats.get("胜率", 0)
        win_rate_str = f"{win_rate:.1%}" if isinstance(win_rate, float) else str(win_rate)
        avg_win = stats.get("平均盈利", 0)
        avg_loss = stats.get("平均亏损", 0)
        aw = f"{avg_win:>10,.0f}" if isinstance(avg_win, (int, float)) else str(avg_win)
        al = f"{avg_loss:>10,.0f}" if isinstance(avg_loss, (int, float)) else str(avg_loss)

        lines = [
            ("📊 交易统计", True),
            ("", False),
            (f"配对交易:    {stats.get('配对交易次数', 0)}", False),
            (f"盈利交易:    {stats.get('盈利交易', 0)}", False),
            (f"亏损交易:    {stats.get('亏损交易', 0)}", False),
            (f"持平交易:    {stats.get('持平交易', 0)}", False),
            ("", False),
            (f"🎯 胜率:        {win_rate_str}", True),
            ("", False),
            (f"平均盈利:    {aw}", False),
            (f"平均亏损:    {al}", False),
            (f"盈亏比:       {stats.get('盈亏比', 0):>8.2f}", False),
            (f"Profit Factor: {stats.get('Profit Factor', 0):>8.2f}", False),
            ("", False),
            (f"最大连胜:    {stats.get('最大连胜', 0)}", False),
            (f"最大连败:    {stats.get('最大连败', 0)}", False),
            ("", False),
        ]

        ahw = stats.get("平均持仓天数(盈利)")
        ahl = stats.get("平均持仓天数(亏损)")
        if ahw is not None:
            lines.append((f"平均持仓(盈利): {ahw:.1f} 天", False))
        if ahl is not None:
            lines.append((f"平均持仓(亏损): {ahl:.1f} 天", False))

        y_pos = 0.95
        for text, is_hdr in lines:
            fs = 13 if is_hdr else 10
            fw = "bold" if is_hdr else "normal"
            c = "#2c3e50" if is_hdr else "#34495e"
            fm = "sans-serif" if is_hdr else "monospace"
            ax.text(0.05, y_pos, text, fontsize=fs, fontweight=fw,
                    transform=ax.transAxes, color=c, fontfamily=fm)
            y_pos -= 0.045

        ax.set_title("交易胜率分析", fontsize=14, fontweight="bold", pad=10)

        # ---- 右侧：盈亏分布直方图 ----
        ax = axes[1]
        if not paired.empty and "pnl" in paired.columns:
            pnls = paired["pnl"].values
            winning = pnls[pnls > 0]
            losing = pnls[pnls < 0]
            if len(winning) > 0:
                ax.hist(winning, bins=30, alpha=0.7, color="#27ae60",
                        label=f"盈利 ({len(winning)}笔)")
            if len(losing) > 0:
                ax.hist(losing, bins=30, alpha=0.7, color="#e74c3c",
                        label=f"亏损 ({len(losing)}笔)")
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("盈亏金额", fontsize=11)
            ax.set_ylabel("交易笔数", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "无配对交易数据", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")

        ax.set_title("逐笔交易盈亏分布", fontsize=14, fontweight="bold", pad=10)
        plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"交易仪表盘已保存: {save_path}")
        plt.show()
