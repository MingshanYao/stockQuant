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

        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="K线",
        ), row=1, col=1)

        if "volume" in df.columns:
            colors = ["red" if c >= o else "green" for c, o in zip(df["close"], df["open"])]
            fig.add_trace(go.Bar(
                x=df["date"], y=df["volume"], name="成交量",
                marker_color=colors, opacity=0.5,
            ), row=2, col=1)

        if trades is not None and not trades.empty:
            buys = trades[trades["direction"] == "buy"]
            sells = trades[trades["direction"] == "sell"]

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["date"], y=buys["price"],
                    mode="markers", name="买入",
                    marker=dict(symbol="triangle-up", size=10, color="red"),
                ), row=1, col=1)

            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["date"], y=sells["price"],
                    mode="markers", name="卖出",
                    marker=dict(symbol="triangle-down", size=10, color="green"),
                ), row=1, col=1)

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
