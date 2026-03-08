"""
Alpha 因子研究工具 — 系统性研究 Alpha 因子表现的一站式框架。

提供完整的因子研究流水线：
  数据加载（外部传入）→ 因子计算（带缓存）→ 回测执行 → 绩效分析 → 可视化

典型用法
--------
>>> from stockquant.data.universe import Pool, StockUniverse
>>> from stockquant.research import AlphaResearcher
>>>
>>> dataset = (
...     StockUniverse()
...     .scope(Pool.CSI300)
...     .exclude(Pool.STAR, Pool.CHINEXT, Pool.BSE)
...     .load("2022-01-01", "2024-12-31", benchmark=Pool.CSI300)
... )
>>> researcher = AlphaResearcher(dataset, max_positions=10, rebalance_freq=5)
>>>
>>> # 单因子完整分析
>>> result = researcher.run_backtest(alpha_id=1)
>>> researcher.full_analysis(result)
>>>
>>> # 多因子批量对比
>>> results = researcher.run_multiple([1, 2, 3, 6, 12, 101])
>>> researcher.compare_factors(results)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from stockquant.backtest.engine import BacktestEngine
from stockquant.analysis.performance import PerformanceAnalyzer
from stockquant.strategy.alpha_factor_strategy import AlphaFactorStrategy
from stockquant.utils.logger import get_logger

if TYPE_CHECKING:
    from stockquant.data.universe import BacktestDataset
    from stockquant.indicators.alpha101.alpha101 import Alpha101Engine

logger = get_logger("research.alpha")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    plt.rcParams.setdefault("font.sans-serif", ["SimHei", "DejaVu Sans"])
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ======================================================================
# AlphaBacktestResult — 单因子回测结果容器
# ======================================================================

@dataclasses.dataclass
class AlphaBacktestResult:
    """单个 Alpha 因子的回测结果容器。

    Attributes
    ----------
    alpha_id : int | None
        Alpha 编号（自定义因子时为 None）。
    label : str
        显示标签，如 "Alpha001"。
    backtest_result : BacktestResult
        ``BacktestEngine.run()`` 返回的原始结果对象。
    alpha_panel : DataFrame
        该因子的面板数据（行=日期，列=股票代码）。
    benchmark_df : DataFrame
        基准指数日线数据。
    """

    alpha_id: int | None
    label: str
    backtest_result: Any          # BacktestEngine.BacktestResult
    alpha_panel: pd.DataFrame
    benchmark_df: pd.DataFrame

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def equity_curve(self) -> pd.DataFrame:
        """权益曲线 DataFrame (date, total_value, cash, ...)。"""
        return self.backtest_result.equity_curve

    @property
    def trade_log(self) -> pd.DataFrame:
        """成交记录 DataFrame。"""
        return self.backtest_result.trade_log

    @property
    def total_return(self) -> float:
        """总收益率（小数）。"""
        return self.backtest_result.total_return

    def nav(self) -> pd.Series:
        """净值序列（从 1.0 开始），index 为 DatetimeIndex。"""
        eq = self.equity_curve.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.sort_values("date").set_index("date")
        series = eq["total_value"] / eq["total_value"].iloc[0]
        series.name = self.label
        return series

    def get_analyzer(self) -> PerformanceAnalyzer:
        """返回配置好基准收益的绩效分析器实例。"""
        bm_returns = (
            self.benchmark_df["close"].pct_change().fillna(0).tolist()
            if not self.benchmark_df.empty
            else []
        )
        return PerformanceAnalyzer(
            equity_curve=self.backtest_result.equity_curve,
            trade_log=self.backtest_result.trade_log,
            daily_returns=self.backtest_result.daily_returns,
            benchmark_returns=bm_returns,
        )

    def full_report(self) -> dict[str, Any]:
        """生成完整绩效报告字典。"""
        return self.get_analyzer().full_report()


# ======================================================================
# AlphaResearcher — 因子研究核心工具
# ======================================================================

class AlphaResearcher:
    """Alpha 因子系统性研究工具。

    一站式完成因子计算（带缓存）→ 回测执行 → 绩效分析 → 可视化的全流程。
    内置因子面板缓存，多次研究同一因子无需重复计算。

    Parameters
    ----------
    dataset : BacktestDataset
        由 ``StockUniverse.load()`` 返回的回测数据集。
    initial_capital : float
        初始资金，默认 1,000,000。
    max_positions : int
        默认最多持仓只数，默认 10。
    rebalance_freq : int
        默认调仓频率（交易日数），默认 5。

    Examples
    --------
    >>> researcher = AlphaResearcher(dataset, max_positions=10, rebalance_freq=5)
    >>>
    >>> # 单因子回测 + 完整分析
    >>> result = researcher.run_backtest(alpha_id=1)
    >>> researcher.full_analysis(result)
    >>>
    >>> # 批量回测多个因子
    >>> results = researcher.run_multiple([1, 2, 3, 6, 12])
    >>> compare_df = researcher.compare_factors(results)
    >>> display(compare_df)
    """

    def __init__(
        self,
        dataset: "BacktestDataset",
        initial_capital: float = 1_000_000.0,
        max_positions: int = 10,
        rebalance_freq: int = 5,
    ) -> None:
        self.dataset = dataset
        self.benchmark_df = dataset.benchmark
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_freq = rebalance_freq

        self._alpha_engine: "Alpha101Engine | None" = None
        self._alpha_cache: dict[int, pd.DataFrame] = {}

        logger.info(
            f"AlphaResearcher 初始化: {len(dataset.codes)} 只股票, "
            f"max_pos={max_positions}, freq={rebalance_freq}"
        )

    # ==================================================================
    # 因子计算
    # ==================================================================

    @property
    def alpha_engine(self) -> "Alpha101Engine":
        """Alpha101 面板引擎（延迟创建，首次访问时构建）。"""
        if self._alpha_engine is None:
            from stockquant.indicators.alpha101.alpha101 import Alpha101Indicators
            logger.info("构建 Alpha101 面板引擎（首次访问）...")
            self._alpha_engine = Alpha101Indicators.from_dataset(self.dataset)
        return self._alpha_engine

    def get_alpha_panel(self, alpha_id: int) -> pd.DataFrame:
        """获取（或计算并缓存）指定 Alpha 因子的面板数据。

        Parameters
        ----------
        alpha_id : int
            Alpha 编号，如 ``1`` 代表 alpha001，取值范围 1~101。

        Returns
        -------
        DataFrame
            行=日期（DatetimeIndex），列=股票代码，值=因子值。
        """
        if alpha_id not in self._alpha_cache:
            logger.info(f"计算 Alpha{alpha_id:03d} 面板...")
            panel = self.alpha_engine.compute_factor(alpha_id)
            panel.index = pd.to_datetime(panel.index)
            self._alpha_cache[alpha_id] = panel
            logger.info(f"Alpha{alpha_id:03d} 计算完成，维度: {panel.shape}")
        return self._alpha_cache[alpha_id]

    def precompute_alphas(self, alpha_ids: Sequence[int]) -> None:
        """预先批量计算多个因子并缓存，避免后续重复构建引擎。

        Parameters
        ----------
        alpha_ids : list[int]
            要预计算的 Alpha 编号列表。
        """
        total = len(alpha_ids)
        for i, alpha_id in enumerate(alpha_ids, 1):
            if alpha_id not in self._alpha_cache:
                print(f"  [{i}/{total}] 计算 Alpha{alpha_id:03d}...", end="\r")
                self.get_alpha_panel(alpha_id)
        print(f"✅ 已预计算 {total} 个因子面板（含缓存）          ")

    # ==================================================================
    # 回测执行
    # ==================================================================

    def run_backtest(
        self,
        alpha_id: int | None = None,
        alpha_panel: pd.DataFrame | None = None,
        max_positions: int | None = None,
        rebalance_freq: int | None = None,
        ascending: bool = False,
        start_date: str | None = None,
        end_date: str | None = None,
        label: str | None = None,
    ) -> AlphaBacktestResult:
        """运行单个 Alpha 因子回测。

        Parameters
        ----------
        alpha_id : int, optional
            Alpha 编号（与 ``alpha_panel`` 二选一）。
        alpha_panel : DataFrame, optional
            自定义因子面板（与 ``alpha_id`` 二选一）。
        max_positions : int, optional
            覆盖默认最多持仓数。
        rebalance_freq : int, optional
            覆盖默认调仓频率。
        ascending : bool
            因子排序方向。``False``=值越大越好（默认），``True``=值越小越好。
        start_date / end_date : str, optional
            回测日期范围，默认使用 dataset 的起止日期。
        label : str, optional
            结果标签，默认 ``"Alpha{id:03d}"``。

        Returns
        -------
        AlphaBacktestResult
        """
        if alpha_panel is None:
            if alpha_id is None:
                raise ValueError("必须提供 alpha_id 或 alpha_panel 之一")
            alpha_panel = self.get_alpha_panel(alpha_id)

        max_pos = max_positions if max_positions is not None else self.max_positions
        freq = rebalance_freq if rebalance_freq is not None else self.rebalance_freq
        _label = label or (f"Alpha{alpha_id:03d}" if alpha_id is not None else "Custom")

        # 构建策略
        strategy = AlphaFactorStrategy()
        strategy.set_params(
            alpha_panel=alpha_panel,
            max_positions=max_pos,
            rebalance_freq=freq,
            ascending=ascending,
            label=_label,
        )

        # 构建回测引擎并设置初始资金
        bt_engine = BacktestEngine()
        bt_engine.context.portfolio.initial_capital = self.initial_capital
        bt_engine.context.portfolio.cash = self.initial_capital
        bt_engine.context.portfolio.total_value = self.initial_capital
        bt_engine.set_strategy(strategy)
        bt_engine.set_data(self.dataset.stock_data)

        sd = start_date or self.dataset.start_date
        ed = end_date or self.dataset.end_date
        bt_engine.set_date_range(start_date=sd, end_date=ed)

        logger.info(f"运行回测: {_label}  [{sd} ~ {ed}]")
        bt_result = bt_engine.run()

        return AlphaBacktestResult(
            alpha_id=alpha_id,
            label=_label,
            backtest_result=bt_result,
            alpha_panel=alpha_panel,
            benchmark_df=self.benchmark_df,
        )

    def run_multiple(
        self,
        alpha_ids: Sequence[int],
        **kwargs: Any,
    ) -> dict[int, AlphaBacktestResult]:
        """批量运行多个 Alpha 因子回测。

        Parameters
        ----------
        alpha_ids : list[int]
            要回测的 Alpha 编号列表，如 ``[1, 2, 3, 6, 12]``。
        **kwargs
            传递给 :meth:`run_backtest` 的其他参数（可统一设置调仓频率等）。

        Returns
        -------
        dict[int, AlphaBacktestResult]
            {alpha_id: AlphaBacktestResult}
        """
        results: dict[int, AlphaBacktestResult] = {}
        total = len(alpha_ids)
        for i, alpha_id in enumerate(alpha_ids, 1):
            print(f"[{i}/{total}] 回测 Alpha{alpha_id:03d}...", end="")
            try:
                r = self.run_backtest(alpha_id=alpha_id, **kwargs)
                results[alpha_id] = r
                print(f"  总收益: {r.total_return:+.2%}")
            except Exception as e:
                logger.error(f"Alpha{alpha_id:03d} 回测失败: {e}")
                print(f"  ⚠️ 失败: {e}")
        print(f"\n✅ 完成 {len(results)}/{total} 个因子回测")
        return results

    # ==================================================================
    # 绩效分析
    # ==================================================================

    def performance_summary(self, result: AlphaBacktestResult) -> pd.DataFrame:
        """生成完整绩效报告 DataFrame。

        Returns
        -------
        DataFrame
            两列：["指标", "值"]，包含收益/风险/风险调整/交易统计。
        """
        report = result.full_report()
        return pd.DataFrame(list(report.items()), columns=["指标", "值"])

    def metrics_table(
        self,
        results: dict[int | str, AlphaBacktestResult],
    ) -> pd.DataFrame:
        """生成多因子绩效对比表（数值型，便于排序/比较）。

        Parameters
        ----------
        results : dict
            ``{alpha_id 或自定义 key: AlphaBacktestResult}``

        Returns
        -------
        DataFrame
            行=因子标签，列=绩效指标（数值型）。
        """
        rows = []
        for _, r in results.items():
            az = r.get_analyzer()
            rows.append({
                "因子":       r.label,
                "总收益率":   az.total_return(),
                "年化收益率": az.annualized_return(),
                "最大回撤":   az.max_drawdown(),
                "夏普比率":   az.sharpe_ratio(),
                "索提诺比率": az.sortino_ratio(),
                "卡玛比率":   az.calmar_ratio(),
                "年化波动率": az.volatility(),
                "Alpha":      az.alpha(),
                "Beta":       az.beta(),
                "总交易次数": r.backtest_result.total_trades,
            })
        return pd.DataFrame(rows).set_index("因子")

    # ==================================================================
    # 可视化
    # ==================================================================

    def plot_equity(
        self,
        result: AlphaBacktestResult,
        show_benchmark: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 8),
    ) -> None:
        """绘制净值曲线 + 回撤图（上下两子图）。

        Parameters
        ----------
        result : AlphaBacktestResult
        show_benchmark : bool
            是否叠加沪深300基准净值，默认 True。
        title : str, optional
            图标题，默认自动生成。
        figsize : tuple
            图尺寸，默认 (14, 8)。
        """
        _check_matplotlib()

        equity = result.equity_curve.copy()
        equity["date"] = pd.to_datetime(equity["date"])
        equity = equity.sort_values("date")
        nav = equity["total_value"] / equity["total_value"].iloc[0]
        drawdown = (nav - nav.cummax()) / nav.cummax() * 100

        fig, axes = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        _title = title or (
            f"{result.label}  |  Top-{self.max_positions} 等权  |  "
            f"{self.dataset.start_date} ~ {self.dataset.end_date}"
        )
        fig.suptitle(_title, fontsize=13, y=0.98)

        # ── 净值曲线 ──
        ax1 = axes[0]
        ax1.plot(equity["date"], nav, label=result.label, linewidth=1.6, color="#e74c3c")

        if show_benchmark and not result.benchmark_df.empty:
            bm = result.benchmark_df.copy()
            bm["date"] = pd.to_datetime(bm["date"])
            bm = bm[bm["date"].between(equity["date"].iloc[0], equity["date"].iloc[-1])]
            bm = bm.sort_values("date")
            if not bm.empty:
                bm_nav = bm["close"] / bm["close"].iloc[0]
                ax1.plot(
                    bm["date"], bm_nav,
                    label="沪深300基准",
                    linewidth=1.2, color="#3498db", alpha=0.8, linestyle="--",
                )

        ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("净值", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ── 回撤 ──
        ax2 = axes[1]
        ax2.fill_between(equity["date"], drawdown, 0, alpha=0.5, color="#e74c3c")
        ax2.set_ylabel("回撤 (%)", fontsize=10)
        ax2.set_xlabel("日期", fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(
            f"总收益率: {nav.iloc[-1]-1:+.2%} | "
            f"最大回撤: {drawdown.min():.2f}%"
        )

    def plot_monthly_heatmap(
        self,
        result: AlphaBacktestResult,
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """绘制月度收益热力图（年×月矩阵，绿涨红跌）。

        Parameters
        ----------
        result : AlphaBacktestResult
        figsize : tuple, optional
            图尺寸，默认根据年数自动计算。
        """
        _check_matplotlib()

        equity = result.equity_curve.copy()
        equity["date"] = pd.to_datetime(equity["date"])
        equity = equity.sort_values("date")

        equity_m = equity.set_index("date")["total_value"].resample("ME").last()
        monthly_ret = (equity_m.pct_change().dropna() * 100).round(2)

        pivot = pd.DataFrame({
            "year":  monthly_ret.index.year,
            "month": monthly_ret.index.month,
            "ret":   monthly_ret.values,
        }).pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        n_years = len(pivot)
        _figsize = figsize or (14, max(3, n_years * 0.9 + 1))
        fig, ax = plt.subplots(figsize=_figsize)

        vals = pivot.values[~np.isnan(pivot.values)]
        vmax = max(abs(vals).max(), 1) if len(vals) > 0 else 5
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(12))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(n_years))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title(f"{result.label} — 月度收益热力图 (%)", fontsize=12, pad=12)

        for i in range(n_years):
            for j in range(12):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:+.1f}",
                        ha="center", va="center", fontsize=8.5,
                        color="black" if abs(val) < vmax * 0.6 else "white",
                    )

        plt.colorbar(im, ax=ax, label="月度收益 (%)", shrink=0.8)
        plt.tight_layout()
        plt.show()

    def plot_top_holdings(
        self,
        result: AlphaBacktestResult,
        date: str | None = None,
        top_n: int | None = None,
        figsize: tuple[int, int] = (10, 5),
    ) -> None:
        """绘制指定日期截面排名 Top-N 持仓的因子值条形图。

        Parameters
        ----------
        result : AlphaBacktestResult
        date : str, optional
            指定日期（默认取回测结束日前最近一个有效日）。
        top_n : int, optional
            显示只数，默认使用 ``self.max_positions``。
        figsize : tuple
        """
        _check_matplotlib()

        n = top_n or self.max_positions
        panel = result.alpha_panel
        end_ts = pd.Timestamp(date or self.dataset.end_date)
        valid_idx = panel.index[panel.index <= end_ts]
        if valid_idx.empty:
            print("⚠️ 无有效日期")
            return

        last_date = valid_idx[-1]
        last_alpha = panel.loc[last_date].dropna()
        top = last_alpha.nlargest(n)

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
        bars = ax.barh(
            [str(c) for c in top.index[::-1]],
            top.values[::-1],
            color=colors[::-1],
            edgecolor="white",
        )
        ax.set_xlabel("因子值", fontsize=10)
        ax.set_title(
            f"{result.label}  最后调仓日 ({last_date.date()})  Top-{n} 持仓",
            fontsize=11,
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        for bar, val in zip(bars, top.values[::-1]):
            ax.text(
                val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9,
            )
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compare_factors(
        self,
        results: dict[int | str, AlphaBacktestResult],
        figsize: tuple[int, int] = (14, 6),
    ) -> pd.DataFrame:
        """绘制多因子净值曲线对比图，并返回格式化绩效对比表。

        Parameters
        ----------
        results : dict
            ``{alpha_id 或自定义 key: AlphaBacktestResult}``
        figsize : tuple

        Returns
        -------
        DataFrame
            格式化绩效对比表，按夏普比率降序排列。
        """
        _check_matplotlib()

        if not results:
            print("无结果可比较")
            return pd.DataFrame()

        # ── 净值曲线对比图 ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.cm.tab10
        colors = [cmap(i / max(len(results), 1)) for i in range(len(results))]

        for (_, r), color in zip(results.items(), colors):
            nav = r.nav()
            ax.plot(nav.index, nav.values, label=r.label, linewidth=1.4, color=color)

        # 基准
        first = next(iter(results.values()))
        if not first.benchmark_df.empty:
            bm = first.benchmark_df.copy()
            bm["date"] = pd.to_datetime(bm["date"])
            bm = bm.sort_values("date")
            bm_nav = bm["close"] / bm["close"].iloc[0]
            ax.plot(
                bm["date"], bm_nav.values,
                label="沪深300基准",
                linewidth=1.6, color="black", alpha=0.45, linestyle="--",
            )

        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylabel("净值", fontsize=11)
        ax.set_xlabel("日期", fontsize=10)
        ax.set_title("多因子净值曲线对比", fontsize=12)
        ax.legend(fontsize=9, ncol=min(6, len(results) + 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── 绩效对比表 ────────────────────────────────────────────
        df = self.metrics_table(results)
        df_sorted = df.sort_values("夏普比率", ascending=False)

        fmt = pd.DataFrame(index=df_sorted.index)
        fmt["总收益率"]   = df_sorted["总收益率"].map(lambda x: f"{x:+.2%}")
        fmt["年化收益率"] = df_sorted["年化收益率"].map(lambda x: f"{x:+.2%}")
        fmt["最大回撤"]   = df_sorted["最大回撤"].map(lambda x: f"{x:.2%}")
        fmt["夏普比率"]   = df_sorted["夏普比率"].map(lambda x: f"{x:.2f}")
        fmt["索提诺比率"] = df_sorted["索提诺比率"].map(lambda x: f"{x:.2f}")
        fmt["卡玛比率"]   = df_sorted["卡玛比率"].map(lambda x: f"{x:.2f}")
        fmt["年化波动率"] = df_sorted["年化波动率"].map(lambda x: f"{x:.2%}")
        fmt["Alpha"]      = df_sorted["Alpha"].map(lambda x: f"{x:+.2%}")
        fmt["Beta"]       = df_sorted["Beta"].map(lambda x: f"{x:.2f}")
        fmt["总交易次数"] = df_sorted["总交易次数"]
        return fmt

    def full_analysis(
        self,
        result: AlphaBacktestResult,
    ) -> pd.DataFrame:
        """对单个因子运行完整分析：净值图 + 月度热力图 + 持仓图 + 绩效表。

        Parameters
        ----------
        result : AlphaBacktestResult

        Returns
        -------
        DataFrame
            绩效报告 DataFrame（["指标", "值"]）。
        """
        print(f"\n{'═'*52}")
        print(f"  📊 {result.label} — 完整分析报告")
        print(f"{'═'*52}")

        self.plot_equity(result)
        self.plot_monthly_heatmap(result)
        self.plot_top_holdings(result)

        report_df = self.performance_summary(result)
        print(f"\n{'─'*38}  绩效指标  {'─'*38}")
        for _, row in report_df.iterrows():
            print(f"  {row['指标']:<16}: {row['值']}")
        print("─" * 87)
        return report_df


# ======================================================================
# 内部工具
# ======================================================================

def _check_matplotlib() -> None:
    if not HAS_MPL:
        raise ImportError(
            "绘图需要 matplotlib，请运行: pip install matplotlib"
        )
