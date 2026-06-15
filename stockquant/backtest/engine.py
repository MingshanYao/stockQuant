"""
回测引擎 — 事件驱动架构，驱动策略 + 券商 + 风控协作。
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence, TYPE_CHECKING

import pandas as pd

from stockquant.backtest.broker import Broker
from stockquant.backtest.context import Context
from stockquant.backtest.event import EventBus, Event, EventType
from stockquant.strategy.base_strategy import BaseStrategy
from stockquant.utils.config import Config
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

if TYPE_CHECKING:
    from stockquant.analysis.performance import PerformanceAnalyzer
    from stockquant.data.universe import BacktestDataset, Pool, StockUniverse

logger = get_logger("backtest.engine")


class BacktestEngine:
    """事件驱动回测引擎。

    用法示例::

        engine = BacktestEngine()
        engine.set_strategy(MyStrategy())
        engine.set_data({"600000": df_600000, "000001": df_000001})
        result = engine.run()
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        initial_capital = self.cfg.get("backtest.initial_capital", 1_000_000.0)

        self.context = Context(initial_capital=initial_capital)
        self.broker = Broker(self.context, self.cfg)
        self.event_bus = EventBus()
        self.strategy: BaseStrategy | None = None

        self._data: dict[str, pd.DataFrame] = {}
        self._trade_dates: list[dt.date] = []
        self._benchmark: pd.DataFrame = pd.DataFrame()
        self._benchmark_code: str = ""
        self._close_panel: pd.DataFrame = pd.DataFrame()
        self._code_date_idx: dict[str, dict[dt.date, int]] = {}
        self._bm_close: "pd.Series | None" = None
        self._bm_index: "pd.DatetimeIndex | None" = None

    # ------------------------------------------------------------------
    # 配置接口
    # ------------------------------------------------------------------

    def set_strategy(self, strategy: BaseStrategy) -> None:
        """设置交易策略。"""
        self.strategy = strategy
        self.strategy.set_context(self.context)

    def set_data(self, data: dict[str, pd.DataFrame]) -> None:
        """设置回测用行情数据。

        Parameters
        ----------
        data : dict[str, DataFrame]
            键为股票代码，值为包含 OHLCV 的 DataFrame（需含 date 列）。
        """
        self._data = {}
        for code, df in data.items():
            code = normalize_stock_code(code)
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
            self._data[code] = df

        # 合并所有交易日期
        all_dates = set()
        for df in self._data.values():
            if "date" in df.columns:
                all_dates.update(df["date"].dt.date)
        self._trade_dates = sorted(all_dates)

        # 预计算收盘价面板 (dates × codes) 和日期→位置索引
        close_data: dict[str, pd.Series] = {}
        self._code_date_idx = {}
        for code, df in self._data.items():
            if "date" in df.columns and "close" in df.columns:
                close_data[code] = df.set_index("date")["close"]
            if "date" in df.columns:
                dates = df["date"]
                self._code_date_idx[code] = {
                    d.date(): i for i, d in enumerate(dates)
                }
        self._close_panel = pd.DataFrame(close_data).sort_index() if close_data else pd.DataFrame()

        logger.info(f"已加载 {len(self._data)} 只标的，{len(self._trade_dates)} 个交易日")

    def set_date_range(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> None:
        """过滤回测日期范围。"""
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)
        if sd:
            self._trade_dates = [d for d in self._trade_dates if d >= sd]
        if ed:
            self._trade_dates = [d for d in self._trade_dates if d <= ed]

    def load_universe(
        self,
        universe: StockUniverse | BacktestDataset,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        benchmark: Pool | str | None = None,
    ) -> None:
        """从 StockUniverse 或 BacktestDataset 一步加载回测数据。

        Parameters
        ----------
        universe : StockUniverse | BacktestDataset
            ``StockUniverse`` → 自动调用 ``.load()`` 拉取数据。
            ``BacktestDataset`` → 直接使用已加载的数据集。
        start_date / end_date : str | date, optional
            日期范围。传入 ``StockUniverse`` 时为必填；
            传入 ``BacktestDataset`` 时可用于进一步过滤。
        benchmark : Pool | str, optional
            基准指数（传入 ``StockUniverse`` 时生效），默认沪深300。
        """
        from stockquant.data.universe import BacktestDataset as _BD, StockUniverse as _SU, Pool as _Pool

        if isinstance(universe, _SU):
            if start_date is None or end_date is None:
                raise ValueError("传入 StockUniverse 时 start_date 和 end_date 为必填")
            bm = benchmark if benchmark is not None else _Pool.CSI300
            dataset = universe.load(str(start_date), str(end_date), bm)
        elif isinstance(universe, _BD):
            dataset = universe
        else:
            raise TypeError(
                f"universe 参数类型错误: {type(universe).__name__}，"
                f"请传入 StockUniverse 或 BacktestDataset。"
            )

        self.set_data(dataset.stock_data)
        self._benchmark = dataset.benchmark
        self._benchmark_code = dataset.benchmark_code

        if start_date or end_date:
            self.set_date_range(start_date, end_date)

        logger.info(
            f"已加载标的池: {len(dataset.codes)} 只, "
            f"基准: {dataset.benchmark_code}, "
            f"{dataset.start_date} ~ {dataset.end_date}"
        )

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """执行回测。"""
        if not self.strategy:
            raise RuntimeError("未设置策略，请先调用 set_strategy()")
        if not self._data:
            raise RuntimeError("未设置数据，请先调用 set_data()")

        logger.info("====== 回测开始 ======")
        self.strategy.initialize()

        # 预提取基准收盘价序列，用于快速查找
        if not self._benchmark.empty and "close" in self._benchmark.columns:
            bm = self._benchmark.sort_values("date").set_index("date")
            self._bm_close = bm["close"]
            self._bm_index = bm["close"].index
        else:
            self._bm_close = None
            self._bm_index = None

        _use_lightweight = getattr(self.strategy, "uses_lightweight_bar", False)

        for i, trade_date in enumerate(self._trade_dates):
            self.context.current_date = trade_date

            # 1. 日切：解冻 T+1
            self.broker.on_new_day()

            # 2. 盘前回调
            self.strategy.before_trading()

            if _use_lightweight:
                # ── 轻量 bar 模式：跳过 _build_bar，直接使用 close_panel ──
                self._setup_trade_date(trade_date)
                ts = pd.Timestamp(trade_date)
                available: set[str] = set()
                prices: dict[str, float] = {}
                if ts in self._close_panel.index:
                    row = self._close_panel.loc[ts].dropna()
                    available = set(row.index)
                    prices = row.to_dict()
                from stockquant.backtest.bar import BarSnapshot
                bar_snapshot = BarSnapshot(date=trade_date, codes=available, close=prices)
                self.strategy._orders.clear()
                self.strategy.handle_bar(bar_snapshot)
            else:
                # 传统模式：构建完整 bar（零拷贝 iloc 切片）
                self._setup_trade_date(trade_date)
                bar = self._build_bar(trade_date)
                if not bar:
                    continue
                self.strategy._orders.clear()
                self.strategy.handle_bar(bar)

            # 6. 撮合订单
            for order in self.strategy.get_pending_orders():
                filled = self.broker.process_order(order)
                self.strategy.on_order(filled)
                if filled.status == "filled":
                    self.strategy.on_trade(filled)

            # 7. 记录权益
            self.context.record_equity()

            # 8. 收盘后回调
            self.strategy.after_trading()

        logger.info("====== 回测结束 ======")
        return self._build_result()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _build_bar(self, trade_date: dt.date) -> dict[str, pd.DataFrame]:
        """使用预计算索引的零拷贝 bar 构建。"""
        bar: dict[str, pd.DataFrame] = {}
        for code, df in self._data.items():
            idx_map = self._code_date_idx.get(code)
            if idx_map is None:
                continue
            idx = idx_map.get(trade_date)
            if idx is not None:
                bar[code] = df.iloc[: idx + 1]
        return bar

    def _setup_trade_date(self, trade_date: dt.date) -> None:
        """设置当日上下文状态：基准价格切片 + 价格缓存更新。"""
        ts = pd.Timestamp(trade_date)

        # 基准价格（零拷贝 iloc 切片替代布尔掩码）
        if self._bm_index is not None and self._bm_close is not None:
            pos = self._bm_index.searchsorted(ts, side="right")
            self.context.benchmark_prices = self._bm_close.iloc[:pos]
        else:
            self.context.benchmark_prices = None

        # 价格缓存更新（从 close_panel 一次查找替代遍历 bar）
        if ts in self._close_panel.index:
            row = self._close_panel.loc[ts].dropna()
            for code, price in row.items():
                self.context.update_price(code, price)

    def _build_result(self) -> BacktestResult:
        """汇总回测结果。"""
        equity = pd.DataFrame(self.context.portfolio.equity_curve)
        trades = pd.DataFrame(self.broker.trade_log)

        return BacktestResult(
            equity_curve=equity,
            trade_log=trades,
            daily_returns=self.context.portfolio.daily_returns,
            initial_capital=self.context.portfolio.initial_capital,
            final_value=self.context.portfolio.total_value,
            benchmark=self._benchmark,
            benchmark_code=self._benchmark_code,
        )


class BacktestResult:
    """回测结果容器。"""

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        daily_returns: list[float],
        initial_capital: float,
        final_value: float,
        benchmark: pd.DataFrame | None = None,
        benchmark_code: str = "",
    ) -> None:
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.daily_returns = daily_returns
        self.initial_capital = initial_capital
        self.final_value = final_value
        self.benchmark = benchmark if benchmark is not None else pd.DataFrame()
        self.benchmark_code = benchmark_code

    @property
    def total_return(self) -> float:
        return (self.final_value - self.initial_capital) / self.initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.trade_log)

    def analyze(self) -> PerformanceAnalyzer:
        """返回预加载的绩效分析器（含基准收益率）。"""
        from stockquant.analysis.performance import PerformanceAnalyzer

        benchmark_returns: list[float] = []
        if not self.benchmark.empty and "close" in self.benchmark.columns:
            bm = self.benchmark.sort_values("date")["close"]
            benchmark_returns = bm.pct_change().dropna().tolist()

        return PerformanceAnalyzer(
            equity_curve=self.equity_curve,
            trade_log=self.trade_log,
            daily_returns=self.daily_returns,
            benchmark_returns=benchmark_returns,
        )

    def summary(self) -> dict:
        return {
            "初始资金": f"{self.initial_capital:,.0f}",
            "最终市值": f"{self.final_value:,.0f}",
            "总收益率": f"{self.total_return:.2%}",
            "总交易次数": self.total_trades,
        }

    def __repr__(self) -> str:
        return (
            f"BacktestResult(return={self.total_return:.2%}, "
            f"trades={self.total_trades}, "
            f"final_value={self.final_value:,.0f})"
        )
