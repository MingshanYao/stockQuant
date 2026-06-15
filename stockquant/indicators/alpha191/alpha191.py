"""
国泰君安 Alpha191 因子指标 — 基于短周期价量特征的多因子选股体系。

基于国泰君安 (2017) 《基于短周期价量特征的多因子选股体系》研究报告，
实现 191 个短周期交易型 Alpha 因子。
"""

from __future__ import annotations

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stockquant.data.universe import BacktestDataset

from stockquant.indicators.base import BaseIndicator, IndicatorRegistry
from stockquant.indicators.alpha101.operators import (
    adv,
    decay_linear,
    delay,
    delta,
    log,
    rank,
    scale,
    sign,
    signedpower,
    sma,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_max,
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
)
from stockquant.indicators.alpha191.operators import (
    count,
    ema_sma,
    filter_cond,
    highday,
    lowday,
    regbeta,
    sequence,
    sumac,
    sumif,
    wma,
)
from stockquant.utils.logger import get_logger

logger = get_logger("indicators.alpha191")

BENCHMARK_ALPHAS: set[int] = {75, 149, 150, 181, 182, 190}
SKIP_ALPHAS: set[int] = {30}
TOTAL_ALPHAS = 191


def _sma(x, d):
    return sma(x, d)


def _std(x, d):
    return ts_stddev(x, d)


class Alpha191Indicators(BaseIndicator):
    """国泰君安 Alpha191 因子指标。"""

    def __init__(self, alphas: Sequence[int] | None = None) -> None:
        self._alphas = list(alphas) if alphas else list(range(1, TOTAL_ALPHAS + 1))

    @property
    def name(self) -> str:
        return "Alpha191"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        alphas = kwargs.get("alphas", self._alphas)
        engine = self._build_engine_from_single(df)

        for alpha_id in alphas:
            if alpha_id in SKIP_ALPHAS:
                continue
            method = getattr(engine, f"alpha{alpha_id:03d}", None)
            if method is None:
                continue
            try:
                result = method()
                col_name = f"alpha{alpha_id:03d}"
                if isinstance(result, pd.DataFrame):
                    df[col_name] = result.iloc[:, 0].values
                else:
                    df[col_name] = result.values
            except Exception as e:
                logger.warning(f"Alpha191#{alpha_id} 计算失败: {e}")
                df[f"alpha{alpha_id:03d}"] = np.nan
        return df

    @classmethod
    def panel(
        cls,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame | None = None,
        amount: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        benchmark_close: pd.Series | None = None,
        benchmark_open: pd.Series | None = None,
    ) -> "Alpha191Engine":
        return Alpha191Engine(
            open_=open_, high=high, low=low, close=close,
            volume=volume, vwap=vwap, amount=amount, returns=returns,
            benchmark_close=benchmark_close,
            benchmark_open=benchmark_open,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: "BacktestDataset",
    ) -> "Alpha191Engine":
        from stockquant.indicators.alpha101.alpha101 import Alpha101Indicators
        stacked = Alpha101Indicators._stack_dataset(dataset)
        stacked = stacked.copy()
        if not isinstance(stacked.index, pd.MultiIndex):
            stacked = stacked.set_index(["date", "code"])

        def _pivot(col: str) -> pd.DataFrame | None:
            if col in stacked.columns:
                return stacked[col].unstack()
            return None

        bm_close = None
        bm_open = None
        if dataset.benchmark is not None and not dataset.benchmark.empty:
            bm = dataset.benchmark.copy()
            bm["date"] = pd.to_datetime(bm["date"])
            bm = bm.set_index("date").sort_index()
            if "close" in bm.columns:
                bm_close = bm["close"]
            if "open" in bm.columns:
                bm_open = bm["open"]

        return Alpha191Engine(
            open_=_pivot("open"),
            high=_pivot("high"),
            low=_pivot("low"),
            close=_pivot("close"),
            volume=_pivot("volume"),
            amount=_pivot("amount"),
            benchmark_close=bm_close,
            benchmark_open=bm_open,
        )

    @staticmethod
    def _build_engine_from_single(df: pd.DataFrame) -> "Alpha191Engine":
        code = df.get("code", pd.Series(["_stock"])).iloc[0] if "code" in df.columns else "_stock"

        def _col(col_name: str) -> pd.DataFrame | None:
            if col_name in df.columns:
                return df[[col_name]].rename(columns={col_name: code})
            return None

        return Alpha191Engine(
            open_=df[["open"]].rename(columns={"open": code}),
            high=df[["high"]].rename(columns={"high": code}),
            low=df[["low"]].rename(columns={"low": code}),
            close=df[["close"]].rename(columns={"close": code}),
            volume=df[["volume"]].rename(columns={"volume": code}),
            amount=_col("amount"),
        )


# 模块级全局变量用于 multiprocessing fork（避免 pickle engine）
_alpha191_engine = None


def _compute_one_alpha191(alpha_id: int):
    """模块级函数，用于 ProcessPoolExecutor fork。"""
    return alpha_id, _alpha191_engine.compute_factor(alpha_id)


class Alpha191Engine:
    """Alpha191 面板计算引擎。"""

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame | None = None,
        amount: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        benchmark_close: pd.Series | None = None,
        benchmark_open: pd.Series | None = None,
    ) -> None:
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.amount = amount if amount is not None else close * volume * 100

        if vwap is not None:
            self.vwap = vwap
        elif amount is not None:
            self.vwap = amount / (volume * 100 + 1e-10)
        else:
            self.vwap = (high + low + close) / 3

        self.returns = returns if returns is not None else close.pct_change()
        self.benchmark_close = benchmark_close
        self.benchmark_open = benchmark_open
        self._cache: dict[str, Any] = {}

    # ==================================================================
    # 内部工具
    # ==================================================================

    @staticmethod
    def _clean(result: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return result.replace([np.inf, -np.inf], np.nan)

    def _where(self, condition, x, y) -> pd.DataFrame:
        return pd.DataFrame(
            np.where(condition, x, y),
            index=self.close.index,
            columns=self.close.columns,
        )

    def _adv(self, d: int) -> pd.DataFrame:
        key = f"adv{d}"
        if key not in self._cache:
            self._cache[key] = adv(self.volume, d)
        return self._cache[key]

    def _bm_close_panel(self) -> pd.DataFrame:
        if self.benchmark_close is None:
            return self.close * 0
        return pd.DataFrame(
            np.tile(self.benchmark_close.reindex(self.close.index).values[:, None], (1, len(self.close.columns))),
            index=self.close.index,
            columns=self.close.columns,
        )

    def _bm_open_panel(self) -> pd.DataFrame:
        if self.benchmark_open is None:
            return self.open * 0
        return pd.DataFrame(
            np.tile(self.benchmark_open.reindex(self.close.index).values[:, None], (1, len(self.close.columns))),
            index=self.close.index,
            columns=self.close.columns,
        )

    # ==================================================================
    # 批量计算
    # ==================================================================

    def compute_factor(self, alpha_id: int) -> pd.DataFrame:
        method = getattr(self, f"alpha{alpha_id:03d}", None)
        if method is None:
            raise ValueError(f"Alpha191#{alpha_id} 未实现")
        try:
            return self._clean(method())
        except Exception as e:
            logger.warning(f"Alpha191#{alpha_id} 计算失败: {e}")
            return pd.DataFrame(
                index=self.close.index, columns=self.close.columns, dtype=float
            )

    def compute_factors(self, alpha_ids: Sequence[int]) -> dict[int, pd.DataFrame]:
        global _alpha191_engine
        _alpha191_engine = self
        to_compute = [i for i in alpha_ids if i not in SKIP_ALPHAS]
        results: dict[int, pd.DataFrame] = {}
        max_workers = min(max(os.cpu_count() - 2, 1), len(to_compute))

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_compute_one_alpha191, i): i for i in to_compute}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    idx, panel = future.result()
                    results[idx] = panel
                except Exception as e:
                    logger.warning(f"Alpha191#{i} 计算失败: {e}")

        logger.info(f"计算 {len(results)} 个指定 Alpha191 因子 (并行={max_workers}进程)")
        return results

    def compute_all(self) -> dict[int, pd.DataFrame]:
        global _alpha191_engine
        _alpha191_engine = self
        to_compute = [
            i for i in range(1, TOTAL_ALPHAS + 1)
            if i not in SKIP_ALPHAS and hasattr(self, f"alpha{i:03d}")
        ]
        results: dict[int, pd.DataFrame] = {}
        max_workers = min(max(os.cpu_count() - 2, 1), len(to_compute))

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_compute_one_alpha191, i): i for i in to_compute}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    idx, panel = future.result()
                    results[idx] = panel
                except Exception as e:
                    logger.warning(f"Alpha191#{i} 计算失败: {e}")

        logger.info(f"成功计算 {len(results)}/{TOTAL_ALPHAS} 个 Alpha191 因子 (并行={max_workers}进程)")
        return results

    # ==================================================================
    # Alpha #001 – #010
    # ==================================================================

    def alpha001(self) -> pd.DataFrame:
        """(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))"""
        return -1 * ts_corr(
            rank(delta(log(self.volume), 1)),
            rank((self.close - self.open) / (self.open + 1e-10)),
            6,
        )

    def alpha002(self) -> pd.DataFrame:
        """(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))"""
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10)
        return -1 * delta(inner, 1)

    def alpha003(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond_eq = self.close == d1
        cond_gt = self.close > d1
        inner_gt = self.close - pd.DataFrame(
            np.minimum(self.low.values, d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner_lt = self.close - pd.DataFrame(
            np.maximum(self.high.values, d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner = self._where(cond_eq, 0.0, self._where(cond_gt, inner_gt, inner_lt))
        return ts_sum(inner, 6)

    def alpha004(self) -> pd.DataFrame:
        sma8 = _sma(self.close, 8)
        std8 = _std(self.close, 8)
        sma2 = _sma(self.close, 2)
        vol_ratio = self.volume / (_sma(self.volume, 20) + 1e-10)
        cond1 = (sma8 + std8) < sma2
        cond2 = sma2 < (sma8 - std8)
        cond3 = vol_ratio >= 1
        return self._where(
            cond1, -1.0, self._where(cond2, 1.0, self._where(cond3, 1.0, -1.0))
        )

    def alpha005(self) -> pd.DataFrame:
        """(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))"""
        return -1 * ts_max(
            ts_corr(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3
        )

    def alpha006(self) -> pd.DataFrame:
        return rank(sign(delta(self.open * 0.85 + self.high * 0.15, 4))) * -1

    def alpha007(self) -> pd.DataFrame:
        diff = self.vwap - self.close
        return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(self.volume, 3))

    def alpha008(self) -> pd.DataFrame:
        return rank(delta((self.high + self.low) / 2 * 0.2 + self.vwap * 0.8, 4) * -1)

    def alpha009(self) -> pd.DataFrame:
        inner = (
            ((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2)
            * (self.high - self.low) / (self.volume + 1e-10)
        )
        return ema_sma(inner, n=7, m=2)

    def alpha010(self) -> pd.DataFrame:
        inner = self._where(self.returns < 0, _std(self.returns, 20), self.close)
        return rank(ts_max(inner ** 2, 5))

    # ==================================================================
    # Alpha #011 – #020
    # ==================================================================

    def alpha011(self) -> pd.DataFrame:
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10) * self.volume
        return ts_sum(inner, 6)

    def alpha012(self) -> pd.DataFrame:
        return rank(self.open - ts_sum(self.vwap, 10) / 10) * (-1 * rank((self.close - self.vwap).abs()))

    def alpha013(self) -> pd.DataFrame:
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha014(self) -> pd.DataFrame:
        return self.close - delay(self.close, 5)

    def alpha015(self) -> pd.DataFrame:
        return self.open / (delay(self.close, 1) + 1e-10) - 1

    def alpha016(self) -> pd.DataFrame:
        return -1 * ts_max(rank(ts_corr(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha017(self) -> pd.DataFrame:
        return signedpower(rank(self.vwap - ts_max(self.vwap, 15)), delta(self.close, 5))

    def alpha018(self) -> pd.DataFrame:
        return self.close / (delay(self.close, 5) + 1e-10)

    def alpha019(self) -> pd.DataFrame:
        d5 = delay(self.close, 5)
        cond_lt = self.close < d5
        cond_eq = self.close == d5
        ret_neg = (self.close - d5) / (d5 + 1e-10)
        ret_pos = (self.close - d5) / (self.close + 1e-10)
        return self._where(cond_lt, ret_neg, self._where(cond_eq, 0.0, ret_pos))

    def alpha020(self) -> pd.DataFrame:
        d6 = delay(self.close, 6)
        return (self.close - d6) / (d6 + 1e-10) * 100

    # ==================================================================
    # Alpha #021 – #030
    # ==================================================================

    def alpha021(self) -> pd.DataFrame:
        mean6 = _sma(self.close, 6)
        seq = sequence(6)
        if isinstance(mean6, pd.DataFrame):
            result = pd.DataFrame(index=mean6.index, columns=mean6.columns, dtype=float)
            for col in mean6.columns:
                vals = mean6[col].values
                out = np.full(len(vals), np.nan)
                for i in range(5, len(vals)):
                    y = vals[i - 5: i + 1]
                    if np.any(np.isnan(y)):
                        continue
                    x_dm = seq - seq.mean()
                    denom = np.dot(x_dm, x_dm)
                    if denom == 0:
                        continue
                    out[i] = float(np.dot(x_dm, y - y.mean()) / denom)
                result[col] = out
            return result
        return regbeta(mean6, pd.Series(seq), 6)

    def alpha022(self) -> pd.DataFrame:
        mean6 = _sma(self.close, 6)
        inner = (self.close - mean6) / (mean6 + 1e-10)
        return ema_sma(inner - delay(inner, 3), n=12, m=1)

    def alpha023(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond = self.close > d1
        std20 = _std(self.close, 20)
        upper = ema_sma(self._where(cond, std20, 0.0), n=20, m=1)
        lower_pos = ema_sma(self._where(cond, std20, 0.0), n=20, m=1)
        lower_neg = ema_sma(self._where(~cond, std20, 0.0), n=20, m=1)
        return upper / (lower_pos + lower_neg + 1e-10) * 100

    def alpha024(self) -> pd.DataFrame:
        return ema_sma(self.close - delay(self.close, 5), n=5, m=1)

    def alpha025(self) -> pd.DataFrame:
        return (
            -1 * rank(
                delta(self.close, 7)
                * (1 - rank(decay_linear(self.volume / (_sma(self.volume, 20) + 1e-10), 9)))
            )
            * (1 + rank(ts_sum(self.returns, 250)))
        )

    def alpha026(self) -> pd.DataFrame:
        return (ts_sum(self.close, 7) / 7 - self.close) + ts_corr(self.vwap, delay(self.close, 5), 230)

    def alpha027(self) -> pd.DataFrame:
        d3 = delay(self.close, 3)
        d6 = delay(self.close, 6)
        inner = (self.close - d3) / (d3 + 1e-10) * 100 + (self.close - d6) / (d6 + 1e-10) * 100
        return wma(inner, 12)

    def alpha028(self) -> pd.DataFrame:
        inner = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9) + 1e-10) * 100
        sma1 = ema_sma(inner, n=3, m=1)
        return 3 * sma1 - 2 * ema_sma(sma1, n=3, m=1)

    def alpha029(self) -> pd.DataFrame:
        d6 = delay(self.close, 6)
        return (self.close - d6) / (d6 + 1e-10) * self.volume

    # Alpha030 需要三因子模型，跳过

    # ==================================================================
    # Alpha #031 – #040
    # ==================================================================

    def alpha031(self) -> pd.DataFrame:
        mean12 = _sma(self.close, 12)
        return (self.close - mean12) / (mean12 + 1e-10) * 100

    def alpha032(self) -> pd.DataFrame:
        return -1 * ts_sum(rank(ts_corr(rank(self.high), rank(self.volume), 3)), 3)

    def alpha033(self) -> pd.DataFrame:
        part1 = -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)
        part2 = rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        return part1 * part2 * ts_rank(self.volume, 5)

    def alpha034(self) -> pd.DataFrame:
        return _sma(self.close, 12) / (self.close + 1e-10)

    def alpha035(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.open, 1), 15))
        part2 = rank(decay_linear(ts_corr(self.volume, self.open, 17), 7))
        return pd.DataFrame(
            np.minimum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        ) * -1

    def alpha036(self) -> pd.DataFrame:
        return rank(ts_sum(ts_corr(rank(self.volume), rank(self.vwap), 6), 2))

    def alpha037(self) -> pd.DataFrame:
        x = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -1 * rank(x - delay(x, 10))

    def alpha038(self) -> pd.DataFrame:
        cond = ts_sum(self.high, 20) / 20 < self.high
        return self._where(cond, -1 * delta(self.high, 2), 0.0)

    def alpha039(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.close, 2), 8))
        inner = self.vwap * 0.3 + self.open * 0.7
        part2 = rank(decay_linear(ts_corr(inner, ts_sum(_sma(self.volume, 180), 37), 14), 12))
        return (part1 - part2) * -1

    def alpha040(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond = self.close > d1
        upper = ts_sum(self._where(cond, self.volume, 0.0), 26)
        lower = ts_sum(self._where(~cond, self.volume, 0.0), 26)
        return upper / (lower + 1e-10) * 100

    # ==================================================================
    # Alpha #041 – #050
    # ==================================================================

    def alpha041(self) -> pd.DataFrame:
        return rank(ts_max(delta(self.vwap, 3), 5)) * -1

    def alpha042(self) -> pd.DataFrame:
        return -1 * rank(_std(self.high, 10)) * ts_corr(self.high, self.volume, 10)

    def alpha043(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond_gt = self.close > d1
        cond_lt = self.close < d1
        inner = self._where(cond_gt, self.volume, self._where(cond_lt, -self.volume, 0.0))
        return ts_sum(inner, 6)

    def alpha044(self) -> pd.DataFrame:
        part1 = ts_rank(decay_linear(ts_corr(self.low, _sma(self.volume, 10), 7), 6), 4)
        part2 = ts_rank(decay_linear(delta(self.vwap, 3), 10), 15)
        return part1 + part2

    def alpha045(self) -> pd.DataFrame:
        return rank(delta(self.close * 0.6 + self.open * 0.4, 1)) * rank(
            ts_corr(self.vwap, _sma(self.volume, 150), 15)
        )

    def alpha046(self) -> pd.DataFrame:
        return (_sma(self.close, 3) + _sma(self.close, 6) + _sma(self.close, 12) + _sma(self.close, 24)) / (4 * self.close + 1e-10)

    def alpha047(self) -> pd.DataFrame:
        hi6 = ts_max(self.high, 6)
        lo6 = ts_min(self.low, 6)
        inner = (hi6 - self.close) / (hi6 - lo6 + 1e-10) * 100
        return ema_sma(inner, n=9, m=1)

    def alpha048(self) -> pd.DataFrame:
        d1 = sign(self.close - delay(self.close, 1))
        d2 = sign(delay(self.close, 1) - delay(self.close, 2))
        d3 = sign(delay(self.close, 2) - delay(self.close, 3))
        return -1 * rank(d1 + d2 + d3) * ts_sum(self.volume, 5) / (ts_sum(self.volume, 20) + 1e-10)

    def alpha049(self) -> pd.DataFrame:
        hl_sum = self.high + self.low
        hl_lag = delay(self.high, 1) + delay(self.low, 1)
        cond = hl_sum >= hl_lag
        abs_h = (self.high - delay(self.high, 1)).abs()
        abs_l = (self.low - delay(self.low, 1)).abs()
        max_hl = pd.DataFrame(
            np.maximum(abs_h.values, abs_l.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner = self._where(cond, 0.0, max_hl)
        inner_neg = self._where(~cond, 0.0, max_hl)
        s1 = ts_sum(inner, 12)
        s2 = ts_sum(inner_neg, 12)
        return s1 / (s1 + s2 + 1e-10)

    def alpha050(self) -> pd.DataFrame:
        hl_sum = self.high + self.low
        hl_lag = delay(self.high, 1) + delay(self.low, 1)
        cond_ge = hl_sum >= hl_lag
        cond_le = hl_sum <= hl_lag
        abs_h = (self.high - delay(self.high, 1)).abs()
        abs_l = (self.low - delay(self.low, 1)).abs()
        max_hl = pd.DataFrame(
            np.maximum(abs_h.values, abs_l.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner_le = self._where(cond_le, 0.0, max_hl)
        inner_ge = self._where(cond_ge, 0.0, max_hl)
        s_le = ts_sum(inner_le, 12)
        s_ge = ts_sum(inner_ge, 12)
        return s_le / (s_le + s_ge + 1e-10) - s_ge / (s_ge + s_le + 1e-10)

    # ==================================================================
    # Alpha #051 – #060
    # ==================================================================

    def alpha051(self) -> pd.DataFrame:
        hl_sum = self.high + self.low
        hl_lag = delay(self.high, 1) + delay(self.low, 1)
        cond_le = hl_sum <= hl_lag
        abs_h = (self.high - delay(self.high, 1)).abs()
        abs_l = (self.low - delay(self.low, 1)).abs()
        max_hl = pd.DataFrame(
            np.maximum(abs_h.values, abs_l.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner_le = self._where(cond_le, 0.0, max_hl)
        inner_ge = self._where(~cond_le, 0.0, max_hl)
        s_le = ts_sum(inner_le, 12)
        s_ge = ts_sum(inner_ge, 12)
        return s_le / (s_le + s_ge + 1e-10)

    def alpha052(self) -> pd.DataFrame:
        tp = (self.high + self.low + self.close) / 3
        tp_lag = delay(tp, 1)
        upper = ts_sum(pd.DataFrame(np.maximum(0, self.high.values - tp_lag.values), index=self.close.index, columns=self.close.columns), 26)
        lower = ts_sum(pd.DataFrame(np.maximum(0, tp_lag.values - self.low.values), index=self.close.index, columns=self.close.columns), 26)
        return upper / (lower + 1e-10) * 100

    def alpha053(self) -> pd.DataFrame:
        return count(self.close > delay(self.close, 1), 12) / 12 * 100

    def alpha054(self) -> pd.DataFrame:
        return -1 * rank(_std((self.close - self.open).abs(), 10) + (self.close - self.open) + ts_corr(self.close, self.open, 10))

    def alpha055(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        d_open1 = delay(self.open, 1)
        abs_h_d1 = (self.high - d1).abs()
        abs_l_d1 = (self.low - d1).abs()
        abs_h_dl = (self.high - delay(self.low, 1)).abs()

        cond1 = (abs_h_d1 > abs_l_d1) & (abs_h_d1 > abs_h_dl)
        cond2 = (abs_l_d1 > abs_h_dl) & (abs_l_d1 > abs_h_d1)

        denom1 = abs_h_d1 + abs_l_d1 / 2 + (d1 - d_open1).abs() / 4
        denom2 = abs_l_d1 + abs_h_d1 / 2 + (d1 - d_open1).abs() / 4
        denom3 = abs_h_dl + (d1 - d_open1).abs() / 4

        denom = self._where(cond1, denom1, self._where(cond2, denom2, denom3))
        numerator = 16 * (self.close - d1 + (self.close - self.open) / 2 + d1 - d_open1)
        inner = numerator / (denom + 1e-10)
        max_abs = pd.DataFrame(
            np.maximum(abs_h_d1.values, abs_l_d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        return ts_sum(inner * max_abs, 20)

    def alpha056(self) -> pd.DataFrame:
        part1 = rank(self.open - ts_min(self.open, 12))
        part2 = rank(rank(ts_corr(ts_sum((self.high + self.low) / 2, 19), ts_sum(_sma(self.volume, 40), 19), 13)) ** 5)
        return self._where(part1 < part2, 1.0, 0.0)

    def alpha057(self) -> pd.DataFrame:
        inner = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9) + 1e-10) * 100
        return ema_sma(inner, n=3, m=1)

    def alpha058(self) -> pd.DataFrame:
        return count(self.close > delay(self.close, 1), 20) / 20 * 100

    def alpha059(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond_eq = self.close == d1
        cond_gt = self.close > d1
        inner_gt = self.close - pd.DataFrame(
            np.minimum(self.low.values, d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner_lt = self.close - pd.DataFrame(
            np.maximum(self.high.values, d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        inner = self._where(cond_eq, 0.0, self._where(cond_gt, inner_gt, inner_lt))
        return ts_sum(inner, 20)

    def alpha060(self) -> pd.DataFrame:
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10) * self.volume
        return ts_sum(inner, 20)

    # ==================================================================
    # Alpha #061 – #070
    # ==================================================================

    def alpha061(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.vwap, 1), 12))
        part2 = rank(decay_linear(rank(ts_corr(self.low, _sma(self.volume, 80), 8)), 17))
        return pd.DataFrame(
            np.maximum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        ) * -1

    def alpha062(self) -> pd.DataFrame:
        return -1 * ts_corr(self.high, rank(self.volume), 5)

    def alpha063(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        upper = ema_sma(pd.DataFrame(np.maximum(diff.values, 0), index=self.close.index, columns=self.close.columns), n=6, m=1)
        lower = ema_sma(diff.abs(), n=6, m=1)
        return upper / (lower + 1e-10) * 100

    def alpha064(self) -> pd.DataFrame:
        part1 = rank(decay_linear(ts_corr(rank(self.vwap), rank(self.volume), 4), 4))
        part2 = rank(decay_linear(ts_max(ts_corr(rank(self.close), rank(_sma(self.volume, 60)), 4), 13), 14))
        return pd.DataFrame(
            np.maximum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        ) * -1

    def alpha065(self) -> pd.DataFrame:
        return _sma(self.close, 6) / (self.close + 1e-10)

    def alpha066(self) -> pd.DataFrame:
        mean6 = _sma(self.close, 6)
        return (self.close - mean6) / (mean6 + 1e-10) * 100

    def alpha067(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        upper = ema_sma(pd.DataFrame(np.maximum(diff.values, 0), index=self.close.index, columns=self.close.columns), n=24, m=1)
        lower = ema_sma(diff.abs(), n=24, m=1)
        return upper / (lower + 1e-10) * 100

    def alpha068(self) -> pd.DataFrame:
        inner = (
            ((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2)
            * (self.high - self.low) / (self.volume + 1e-10)
        )
        return ema_sma(inner, n=15, m=2)

    def alpha069(self) -> pd.DataFrame:
        """DTM/DBM 条件因子"""
        dtm_inner = pd.DataFrame(
            np.maximum(0, self.open.values - delay(self.open, 1).values),
            index=self.close.index, columns=self.close.columns,
        )
        dtm = pd.DataFrame(
            np.maximum(dtm_inner.values, (self.high - self.open).values),
            index=self.close.index, columns=self.close.columns,
        )
        dbm_inner = pd.DataFrame(
            np.maximum(0, delay(self.open, 1).values - self.open.values),
            index=self.close.index, columns=self.close.columns,
        )
        dbm = pd.DataFrame(
            np.maximum(dbm_inner.values, (self.open - self.low).values),
            index=self.close.index, columns=self.close.columns,
        )
        sum_dtm = ts_sum(dtm, 20)
        sum_dbm = ts_sum(dbm, 20)
        cond_gt = sum_dtm > sum_dbm
        cond_eq = sum_dtm == sum_dbm
        result_gt = (sum_dtm - sum_dbm) / (sum_dtm + 1e-10)
        result_lt = (sum_dtm - sum_dbm) / (sum_dbm + 1e-10)
        return self._where(cond_gt, result_gt, self._where(cond_eq, 0.0, result_lt))

    def alpha070(self) -> pd.DataFrame:
        return _std(self.amount, 6)

    # ==================================================================
    # Alpha #071 – #080
    # ==================================================================

    def alpha071(self) -> pd.DataFrame:
        mean24 = _sma(self.close, 24)
        return (self.close - mean24) / (mean24 + 1e-10) * 100

    def alpha072(self) -> pd.DataFrame:
        inner = (ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6) + 1e-10) * 100
        return ema_sma(inner, n=15, m=1)

    def alpha073(self) -> pd.DataFrame:
        part1 = ts_rank(decay_linear(decay_linear(ts_corr(self.close, self.volume, 10), 16), 4), 5)
        part2 = rank(decay_linear(ts_corr(self.vwap, _sma(self.volume, 30), 4), 3))
        return (part1 - part2) * -1

    def alpha074(self) -> pd.DataFrame:
        part1 = rank(ts_corr(ts_sum(self.low * 0.35 + self.vwap * 0.65, 20), ts_sum(_sma(self.volume, 40), 20), 7))
        part2 = rank(ts_corr(rank(self.vwap), rank(self.volume), 6))
        return part1 + part2

    def alpha075(self) -> pd.DataFrame:
        bm_close = self._bm_close_panel()
        bm_open = self._bm_open_panel()
        cond_stock = self.close > self.open
        cond_bm = bm_close < bm_open
        both = cond_stock & cond_bm
        return count(both, 50) / (count(cond_bm, 50) + 1e-10)

    def alpha076(self) -> pd.DataFrame:
        ret_abs = (self.close / (delay(self.close, 1) + 1e-10) - 1).abs()
        ratio = ret_abs / (self.volume + 1e-10)
        return _std(ratio, 20) / (_sma(ratio, 20) + 1e-10)

    def alpha077(self) -> pd.DataFrame:
        part1 = rank(decay_linear(((self.high + self.low) / 2 + self.high - self.vwap - self.high), 20))
        part2 = rank(decay_linear(ts_corr((self.high + self.low) / 2, _sma(self.volume, 40), 3), 6))
        return pd.DataFrame(
            np.minimum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        )

    def alpha078(self) -> pd.DataFrame:
        tp = (self.high + self.low + self.close) / 3
        mean_tp = _sma(tp, 12)
        mean_dev = _sma((self.close - mean_tp).abs(), 12)
        return (tp - mean_tp) / (0.015 * mean_dev + 1e-10)

    def alpha079(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        upper = ema_sma(pd.DataFrame(np.maximum(diff.values, 0), index=self.close.index, columns=self.close.columns), n=12, m=1)
        lower = ema_sma(diff.abs(), n=12, m=1)
        return upper / (lower + 1e-10) * 100

    def alpha080(self) -> pd.DataFrame:
        d5 = delay(self.volume, 5)
        return (self.volume - d5) / (d5 + 1e-10) * 100

    # ==================================================================
    # Alpha #081 – #090
    # ==================================================================

    def alpha081(self) -> pd.DataFrame:
        return ema_sma(self.volume, n=21, m=2)

    def alpha082(self) -> pd.DataFrame:
        inner = (ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6) + 1e-10) * 100
        return ema_sma(inner, n=20, m=1)

    def alpha083(self) -> pd.DataFrame:
        return -1 * rank(ts_cov(rank(self.high), rank(self.volume), 5))

    def alpha084(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond_gt = self.close > d1
        cond_lt = self.close < d1
        inner = self._where(cond_gt, self.volume, self._where(cond_lt, -self.volume, 0.0))
        return ts_sum(inner, 20)

    def alpha085(self) -> pd.DataFrame:
        return ts_rank(self.volume / (_sma(self.volume, 20) + 1e-10), 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha086(self) -> pd.DataFrame:
        d10 = delay(self.close, 10)
        d20 = delay(self.close, 20)
        acc = (d20 - d10) / 10 - (d10 - self.close) / 10
        cond_gt = acc > 0.25
        cond_lt = acc < 0
        return self._where(cond_gt, -1.0, self._where(cond_lt, 1.0, -1 * delta(self.close, 1)))

    def alpha087(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.vwap, 4), 7))
        inner2 = (self.low * 0.9 + self.low * 0.1 - self.vwap) / (self.open - (self.high + self.low) / 2 + 1e-10)
        part2 = ts_rank(decay_linear(inner2, 11), 7)
        return (part1 + part2) * -1

    def alpha088(self) -> pd.DataFrame:
        d20 = delay(self.close, 20)
        return (self.close - d20) / (d20 + 1e-10) * 100

    def alpha089(self) -> pd.DataFrame:
        sma13 = ema_sma(self.close, n=13, m=2)
        sma27 = ema_sma(self.close, n=27, m=2)
        diff = sma13 - sma27
        return 2 * (diff - ema_sma(diff, n=10, m=2))

    def alpha090(self) -> pd.DataFrame:
        return rank(ts_corr(rank(self.vwap), rank(self.volume), 5)) * -1

    # ==================================================================
    # Alpha #091 – #100
    # ==================================================================

    def alpha091(self) -> pd.DataFrame:
        return (rank(self.close - ts_max(self.close, 5)) * rank(ts_corr(_sma(self.volume, 40), self.low, 5))) * -1

    def alpha092(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.close * 0.35 + self.vwap * 0.65, 2), 3))
        part2 = ts_rank(decay_linear(ts_corr(_sma(self.volume, 180), self.close, 13).abs(), 5), 15)
        return pd.DataFrame(
            np.maximum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        ) * -1

    def alpha093(self) -> pd.DataFrame:
        d_open1 = delay(self.open, 1)
        cond = self.open >= d_open1
        inner = pd.DataFrame(
            np.maximum((self.open - self.low).values, (self.open - d_open1).values),
            index=self.close.index, columns=self.close.columns,
        )
        return ts_sum(self._where(cond, 0.0, inner), 20)

    def alpha094(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond_gt = self.close > d1
        cond_lt = self.close < d1
        inner = self._where(cond_gt, self.volume, self._where(cond_lt, -self.volume, 0.0))
        return ts_sum(inner, 30)

    def alpha095(self) -> pd.DataFrame:
        return _std(self.amount, 20)

    def alpha096(self) -> pd.DataFrame:
        inner = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9) + 1e-10) * 100
        sma1 = ema_sma(inner, n=3, m=1)
        return ema_sma(sma1, n=3, m=1)

    def alpha097(self) -> pd.DataFrame:
        return _std(self.volume, 10)

    def alpha098(self) -> pd.DataFrame:
        mean100 = ts_sum(self.close, 100) / 100
        d100 = delay(self.close, 100)
        change = delta(mean100, 100) / (d100 + 1e-10)
        cond = change <= 0.05
        return self._where(cond, -1 * (self.close - ts_min(self.close, 100)), -1 * delta(self.close, 3))

    def alpha099(self) -> pd.DataFrame:
        return -1 * rank(ts_cov(rank(self.close), rank(self.volume), 5))

    def alpha100(self) -> pd.DataFrame:
        return _std(self.volume, 20)

    # ==================================================================
    # Alpha #101 – #110
    # ==================================================================

    def alpha101(self) -> pd.DataFrame:
        part1 = rank(ts_corr(self.close, ts_sum(_sma(self.volume, 30), 37), 15))
        part2 = rank(ts_corr(rank(self.high * 0.1 + self.vwap * 0.9), rank(self.volume), 11))
        return self._where(part1 < part2, 1.0, 0.0) * -1

    def alpha102(self) -> pd.DataFrame:
        d1 = delay(self.volume, 1)
        diff = self.volume - d1
        upper = ema_sma(pd.DataFrame(np.maximum(diff.values, 0), index=self.close.index, columns=self.close.columns), n=6, m=1)
        lower = ema_sma(diff.abs(), n=6, m=1)
        return upper / (lower + 1e-10) * 100

    def alpha103(self) -> pd.DataFrame:
        return ((20 - lowday(self.low, 20)) / 20) * 100

    def alpha104(self) -> pd.DataFrame:
        return -1 * delta(ts_corr(self.high, self.volume, 5), 5) * rank(_std(self.close, 20))

    def alpha105(self) -> pd.DataFrame:
        return -1 * ts_corr(rank(self.open), rank(self.volume), 10)

    def alpha106(self) -> pd.DataFrame:
        return self.close - delay(self.close, 20)

    def alpha107(self) -> pd.DataFrame:
        return (
            (-1 * rank(self.open - delay(self.high, 1)))
            * rank(self.open - delay(self.close, 1))
            * rank(self.open - delay(self.low, 1))
        )

    def alpha108(self) -> pd.DataFrame:
        return signedpower(rank(self.high - pd.DataFrame(np.minimum(self.high.values, delay(self.high, 2).values), index=self.close.index, columns=self.close.columns)), ts_corr(self.vwap, _sma(self.volume, 120), 6)) * -1

    def alpha109(self) -> pd.DataFrame:
        hl = self.high - self.low
        sma1 = ema_sma(hl, n=10, m=2)
        return sma1 / (ema_sma(sma1, n=10, m=2) + 1e-10)

    def alpha110(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        upper = ts_sum(pd.DataFrame(np.maximum(0, (self.high - d1).values), index=self.close.index, columns=self.close.columns), 20)
        lower = ts_sum(pd.DataFrame(np.maximum(0, (d1 - self.low).values), index=self.close.index, columns=self.close.columns), 20)
        return upper / (lower + 1e-10) * 100

    # ==================================================================
    # Alpha #111 – #120
    # ==================================================================

    def alpha111(self) -> pd.DataFrame:
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10)
        return ema_sma(clv * self.volume, n=11, m=2) - ema_sma(clv * self.volume, n=4, m=2)

    def alpha112(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        pos = self._where(diff > 0, diff, 0.0)
        neg = self._where(diff < 0, diff.abs(), 0.0)
        sum_pos = ts_sum(pos, 12)
        sum_neg = ts_sum(neg, 12)
        return (sum_pos - sum_neg) / (sum_pos + sum_neg + 1e-10) * 100

    def alpha113(self) -> pd.DataFrame:
        return (
            -1 * rank(ts_sum(delay(self.close, 5), 20) / 20)
            * ts_corr(self.close, self.volume, 2)
            * rank(ts_corr(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        )

    def alpha114(self) -> pd.DataFrame:
        mean5 = ts_sum(self.close, 5) / 5
        hl_ratio = (self.high - self.low) / (mean5 + 1e-10)
        return rank(delay(hl_ratio, 2)) * rank(rank(self.volume)) / (hl_ratio / (self.vwap - self.close + 1e-10) + 1e-10)

    def alpha115(self) -> pd.DataFrame:
        part1 = rank(ts_corr(self.high * 0.9 + self.close * 0.1, _sma(self.volume, 30), 10))
        part2 = rank(ts_corr(ts_rank((self.high + self.low) / 2, 4), ts_rank(self.volume, 10), 7))
        return signedpower(part1, part2)

    def alpha116(self) -> pd.DataFrame:
        seq = sequence(20)
        if isinstance(self.close, pd.DataFrame):
            result = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
            for col in self.close.columns:
                vals = self.close[col].values
                out = np.full(len(vals), np.nan)
                for i in range(19, len(vals)):
                    y = vals[i - 19: i + 1]
                    if np.any(np.isnan(y)):
                        continue
                    x_dm = seq - seq.mean()
                    denom = np.dot(x_dm, x_dm)
                    if denom == 0:
                        continue
                    out[i] = float(np.dot(x_dm, y - y.mean()) / denom)
                result[col] = out
            return result
        return regbeta(self.close, pd.Series(seq), 20)

    def alpha117(self) -> pd.DataFrame:
        return ts_rank(self.volume, 32) * (1 - ts_rank((self.close + self.high) - self.low, 16)) * (1 - ts_rank(self.returns, 32))

    def alpha118(self) -> pd.DataFrame:
        return ts_sum(self.high - self.open, 20) / (ts_sum(self.open - self.low, 20) + 1e-10) * 100

    def alpha119(self) -> pd.DataFrame:
        part1 = rank(decay_linear(ts_corr(self.vwap, ts_sum(_sma(self.volume, 5), 26), 5), 7))
        part2 = rank(decay_linear(ts_rank(pd.DataFrame(np.minimum(ts_corr(rank(self.open), rank(_sma(self.volume, 15)), 21).values, 9), index=self.close.index, columns=self.close.columns), 7), 8))
        return part1 - part2

    def alpha120(self) -> pd.DataFrame:
        return rank(self.vwap - self.close) / (rank(self.vwap + self.close) + 1e-10)

    # ==================================================================
    # Alpha #121 – #130
    # ==================================================================

    def alpha121(self) -> pd.DataFrame:
        min_vwap = pd.DataFrame(np.minimum(self.vwap.values, delay(self.vwap, 12).values), index=self.close.index, columns=self.close.columns)
        return signedpower(rank(self.vwap - min_vwap), ts_rank(ts_corr(ts_rank(self.vwap, 20), ts_rank(_sma(self.volume, 60), 2), 18), 3)) * -1

    def alpha122(self) -> pd.DataFrame:
        log_close = log(self.close)
        sma1 = ema_sma(log_close, n=13, m=2)
        sma2 = ema_sma(sma1, n=13, m=2)
        sma3 = ema_sma(sma2, n=13, m=2)
        d1 = delay(sma3, 1)
        return (sma3 - d1) / (d1 + 1e-10)

    def alpha123(self) -> pd.DataFrame:
        part1 = rank(ts_corr(ts_sum((self.high + self.low) / 2, 20), ts_sum(_sma(self.volume, 60), 20), 9))
        part2 = rank(ts_corr(self.low, self.volume, 6))
        return self._where(part1 < part2, 1.0, 0.0) * -1

    def alpha124(self) -> pd.DataFrame:
        return (self.close - self.vwap) / (decay_linear(rank(ts_max(self.close, 30)), 2) + 1e-10)

    def alpha125(self) -> pd.DataFrame:
        part1 = rank(decay_linear(ts_corr(self.vwap, _sma(self.volume, 80), 17), 20))
        part2 = rank(decay_linear(delta(self.close * 0.5 + self.vwap * 0.5, 3), 16))
        return part1 / (part2 + 1e-10)

    def alpha126(self) -> pd.DataFrame:
        return (self.close + self.high + self.low) / 3

    def alpha127(self) -> pd.DataFrame:
        max12 = ts_max(self.close, 12)
        inner = (100 * (self.close - max12) / (max12 + 1e-10)) ** 2
        return _sma(inner, 12) ** 0.5

    def alpha128(self) -> pd.DataFrame:
        tp = (self.high + self.low + self.close) / 3
        tp_lag = delay(tp, 1)
        cond_up = tp > tp_lag
        cond_dn = tp < tp_lag
        upper = ts_sum(self._where(cond_up, tp * self.volume, 0.0), 14)
        lower = ts_sum(self._where(cond_dn, tp * self.volume, 0.0), 14)
        return 100 - 100 / (1 + upper / (lower + 1e-10))

    def alpha129(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        return ts_sum(self._where(diff < 0, diff.abs(), 0.0), 12)

    def alpha130(self) -> pd.DataFrame:
        part1 = rank(decay_linear(ts_corr((self.high + self.low) / 2, _sma(self.volume, 40), 9), 10))
        part2 = rank(decay_linear(ts_corr(rank(self.vwap), rank(self.volume), 7), 3))
        return part1 / (part2 + 1e-10)

    # ==================================================================
    # Alpha #131 – #140
    # ==================================================================

    def alpha131(self) -> pd.DataFrame:
        return signedpower(rank(delta(self.vwap, 1)), ts_rank(ts_corr(self.close, _sma(self.volume, 50), 18), 18))

    def alpha132(self) -> pd.DataFrame:
        return _sma(self.amount, 20)

    def alpha133(self) -> pd.DataFrame:
        return ((20 - highday(self.high, 20)) / 20) * 100 - ((20 - lowday(self.low, 20)) / 20) * 100

    def alpha134(self) -> pd.DataFrame:
        d12 = delay(self.close, 12)
        return (self.close - d12) / (d12 + 1e-10) * self.volume

    def alpha135(self) -> pd.DataFrame:
        return ema_sma(delay(self.close / (delay(self.close, 20) + 1e-10), 1), n=20, m=1)

    def alpha136(self) -> pd.DataFrame:
        return -1 * rank(delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)

    def alpha137(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        d_open1 = delay(self.open, 1)
        abs_h_d1 = (self.high - d1).abs()
        abs_l_d1 = (self.low - d1).abs()
        abs_h_dl = (self.high - delay(self.low, 1)).abs()

        cond1 = (abs_h_d1 > abs_l_d1) & (abs_h_d1 > abs_h_dl)
        cond2 = (abs_l_d1 > abs_h_dl) & (abs_l_d1 > abs_h_d1)

        denom1 = abs_h_d1 + abs_l_d1 / 2 + (d1 - d_open1).abs() / 4
        denom2 = abs_l_d1 + abs_h_d1 / 2 + (d1 - d_open1).abs() / 4
        denom3 = abs_h_dl + (d1 - d_open1).abs() / 4

        denom = self._where(cond1, denom1, self._where(cond2, denom2, denom3))
        numerator = 16 * (self.close - d1 + (self.close - self.open) / 2 + d1 - d_open1)
        max_abs = pd.DataFrame(
            np.maximum(abs_h_d1.values, abs_l_d1.values),
            index=self.close.index, columns=self.close.columns,
        )
        return numerator / (denom + 1e-10) * max_abs

    def alpha138(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.low * 0.7 + self.vwap * 0.3, 3), 20))
        part2 = ts_rank(decay_linear(ts_rank(ts_corr(ts_rank(self.low, 8), ts_rank(_sma(self.volume, 60), 17), 5), 19), 16), 7)
        return (part1 - part2) * -1

    def alpha139(self) -> pd.DataFrame:
        return -1 * ts_corr(self.open, self.volume, 10)

    def alpha140(self) -> pd.DataFrame:
        part1 = rank(decay_linear((rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close)), 8))
        part2 = ts_rank(decay_linear(ts_corr(ts_rank(self.close, 8), ts_rank(_sma(self.volume, 60), 20), 8), 7), 3)
        return pd.DataFrame(
            np.minimum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        )

    # ==================================================================
    # Alpha #141 – #150
    # ==================================================================

    def alpha141(self) -> pd.DataFrame:
        return rank(ts_corr(rank(self.high), rank(_sma(self.volume, 15)), 9)) * -1

    def alpha142(self) -> pd.DataFrame:
        return (
            -1 * rank(ts_rank(self.close, 10))
            * rank(delta(delta(self.close, 1), 1))
            * rank(ts_rank(self.volume / (_sma(self.volume, 20) + 1e-10), 5))
        )

    def alpha143(self) -> pd.DataFrame:
        """递归因子: CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF"""
        d1 = delay(self.close, 1)
        ratio = (self.close - d1) / (d1 + 1e-10)
        result = pd.DataFrame(1.0, index=self.close.index, columns=self.close.columns)
        for i in range(1, len(self.close)):
            cond = self.close.iloc[i].values > d1.iloc[i].values
            prev = result.iloc[i - 1].values
            result.iloc[i] = np.where(cond, ratio.iloc[i].values * prev, prev)
        return result

    def alpha144(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond = self.close < d1
        ret_abs = (self.close / (d1 + 1e-10) - 1).abs()
        inner = ret_abs / (self.amount + 1e-10)
        return sumif(inner, 20, cond) / (count(cond, 20) + 1e-10)

    def alpha145(self) -> pd.DataFrame:
        return (_sma(self.volume, 9) - _sma(self.volume, 26)) / (_sma(self.volume, 12) + 1e-10) * 100

    def alpha146(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        ret = (self.close - d1) / (d1 + 1e-10)
        sma61 = ema_sma(ret, n=61, m=2)
        diff = ret - sma61
        mean_diff = _sma(diff, 20)
        var_diff = _sma((diff - diff) ** 2, 60)
        return mean_diff * diff / (var_diff + 1e-10)

    def alpha147(self) -> pd.DataFrame:
        mean12 = _sma(self.close, 12)
        seq = sequence(12)
        if isinstance(mean12, pd.DataFrame):
            result = pd.DataFrame(index=mean12.index, columns=mean12.columns, dtype=float)
            for col in mean12.columns:
                vals = mean12[col].values
                out = np.full(len(vals), np.nan)
                for i in range(11, len(vals)):
                    y = vals[i - 11: i + 1]
                    if np.any(np.isnan(y)):
                        continue
                    x_dm = seq - seq.mean()
                    denom = np.dot(x_dm, x_dm)
                    if denom == 0:
                        continue
                    out[i] = float(np.dot(x_dm, y - y.mean()) / denom)
                result[col] = out
            return result
        return regbeta(mean12, pd.Series(seq), 12)

    def alpha148(self) -> pd.DataFrame:
        part1 = rank(ts_corr(self.open, ts_sum(_sma(self.volume, 60), 9), 6))
        part2 = rank(self.open - ts_min(self.open, 14))
        return self._where(part1 < part2, 1.0, 0.0) * -1

    def alpha149(self) -> pd.DataFrame:
        bm_close = self._bm_close_panel()
        bm_d1 = delay(bm_close, 1)
        cond_bm_down = bm_close < bm_d1

        stock_ret = self.close / (delay(self.close, 1) + 1e-10) - 1
        bm_ret = bm_close / (bm_d1 + 1e-10) - 1

        filtered_stock = filter_cond(stock_ret, cond_bm_down)
        filtered_bm = filter_cond(bm_ret, cond_bm_down)

        if isinstance(filtered_stock, pd.DataFrame):
            result = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
            for col in self.close.columns:
                y = filtered_stock[col].values
                x_vals = filtered_bm[col].values if isinstance(filtered_bm, pd.DataFrame) else filtered_bm.values
                out = np.full(len(y), np.nan)
                for i in range(251, len(y)):
                    y_win = y[i - 251: i + 1]
                    x_win = x_vals[i - 251: i + 1]
                    mask = ~(np.isnan(y_win) | np.isnan(x_win))
                    if mask.sum() < 10:
                        continue
                    yv = y_win[mask]
                    xv = x_win[mask]
                    x_dm = xv - xv.mean()
                    d = np.dot(x_dm, x_dm)
                    if d == 0:
                        continue
                    out[i] = float(np.dot(x_dm, yv - yv.mean()) / d)
                result[col] = out
            return result
        return self.close * 0

    def alpha150(self) -> pd.DataFrame:
        return (self.close + self.high + self.low) / 3 * self.volume

    # ==================================================================
    # Alpha #151 – #160
    # ==================================================================

    def alpha151(self) -> pd.DataFrame:
        return ema_sma(self.close - delay(self.close, 20), n=20, m=1)

    def alpha152(self) -> pd.DataFrame:
        inner = delay(self.close / (delay(self.close, 9) + 1e-10), 1)
        sma1 = ema_sma(inner, n=9, m=1)
        a = _sma(delay(sma1, 1), 12)
        b = _sma(delay(sma1, 1), 26)
        return ema_sma(a - b, n=9, m=1)

    def alpha153(self) -> pd.DataFrame:
        return (_sma(self.close, 3) + _sma(self.close, 6) + _sma(self.close, 12) + _sma(self.close, 24)) / 4

    def alpha154(self) -> pd.DataFrame:
        cond = (self.vwap - pd.DataFrame(np.minimum(self.vwap.values, delay(self.vwap, 16).values), index=self.close.index, columns=self.close.columns)) < ts_corr(self.vwap, _sma(self.volume, 180), 18)
        return self._where(cond, 1.0, 0.0)

    def alpha155(self) -> pd.DataFrame:
        sma13 = ema_sma(self.volume, n=13, m=2)
        sma27 = ema_sma(self.volume, n=27, m=2)
        diff = sma13 - sma27
        return diff - ema_sma(diff, n=10, m=2)

    def alpha156(self) -> pd.DataFrame:
        part1 = rank(decay_linear(delta(self.vwap, 5), 3))
        inner2 = delta((self.open * 0.15 + self.low * 0.85), 2) / (self.open * 0.15 + self.low * 0.85 + 1e-10) * -1
        part2 = rank(decay_linear(inner2, 3))
        return pd.DataFrame(
            np.maximum(part1.values, part2.values),
            index=self.close.index, columns=self.close.columns,
        ) * -1

    def alpha157(self) -> pd.DataFrame:
        inner = -1 * rank(delta(self.close - 1, 5))
        min_rank = ts_min(rank(rank(inner)), 2)
        log_sum = log(ts_sum(min_rank, 1))
        part1 = pd.DataFrame(
            np.minimum(ts_product(rank(rank(log_sum)), 1).values, 5),
            index=self.close.index, columns=self.close.columns,
        )
        part2 = ts_rank(delay(-1 * self.returns, 6), 5)
        return part1 + part2

    def alpha158(self) -> pd.DataFrame:
        sma15 = ema_sma(self.close, n=15, m=2)
        return ((self.high - sma15) - (self.low - sma15)) / (self.close + 1e-10)

    def alpha159(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        min_lc = pd.DataFrame(np.minimum(self.low.values, d1.values), index=self.close.index, columns=self.close.columns)
        max_hc = pd.DataFrame(np.maximum(self.high.values, d1.values), index=self.close.index, columns=self.close.columns)
        range_hc = max_hc - min_lc

        p6 = (self.close - ts_sum(min_lc, 6)) / (ts_sum(range_hc, 6) + 1e-10)
        p12 = (self.close - ts_sum(min_lc, 12)) / (ts_sum(range_hc, 12) + 1e-10)
        p24 = (self.close - ts_sum(min_lc, 24)) / (ts_sum(range_hc, 24) + 1e-10)

        return (p6 * 12 * 24 + p12 * 6 * 24 + p24 * 6 * 24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)

    def alpha160(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond = self.close <= d1
        return ema_sma(self._where(cond, _std(self.close, 20), 0.0), n=20, m=1)

    # ==================================================================
    # Alpha #161 – #170
    # ==================================================================

    def alpha161(self) -> pd.DataFrame:
        abs_ch = (delay(self.close, 1) - self.high).abs()
        abs_cl = (delay(self.close, 1) - self.low).abs()
        hl = self.high - self.low
        tr = pd.DataFrame(
            np.maximum(np.maximum(hl.values, abs_ch.values), abs_cl.values),
            index=self.close.index, columns=self.close.columns,
        )
        return _sma(tr, 12)

    def alpha162(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        up = pd.DataFrame(np.maximum(diff.values, 0), index=self.close.index, columns=self.close.columns)
        rsi = ema_sma(up, n=12, m=1) / (ema_sma(diff.abs(), n=12, m=1) + 1e-10) * 100
        rsi_min = ts_min(rsi, 12)
        rsi_max = ts_max(rsi, 12)
        return (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    def alpha163(self) -> pd.DataFrame:
        return rank(-1 * self.returns * _sma(self.volume, 20) * self.vwap * (self.high - self.close))

    def alpha164(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond = self.close > d1
        inv = self._where(cond, 1 / (self.close - d1 + 1e-10), 1.0)
        min_inv = ts_min(inv, 12)
        return ema_sma((inv - min_inv) / (self.high - self.low + 1e-10) * 100, n=13, m=2)

    def alpha165(self) -> pd.DataFrame:
        diff = self.close - _sma(self.close, 48)
        cum = sumac(diff)
        max_cum = ts_max(cum, 48)
        min_cum = ts_min(cum, 48)
        return max_cum - min_cum / (_std(self.close, 48) + 1e-10)

    def alpha166(self) -> pd.DataFrame:
        ret = self.close / (delay(self.close, 1) + 1e-10) - 1
        mean_ret = _sma(ret, 20)
        n = 20
        sum_cubed = ts_sum((ret - mean_ret) ** 3, n)
        sum_sq = ts_sum((ret - mean_ret) ** 2, n)
        return -1 * n * (n - 1) ** 1.5 * sum_cubed / ((n - 1) * (n - 2) * (sum_sq ** 1.5 + 1e-10))

    def alpha167(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        return ts_sum(pd.DataFrame(np.maximum(diff.values, 0), index=self.close.index, columns=self.close.columns), 12)

    def alpha168(self) -> pd.DataFrame:
        return -1 * self.volume / (_sma(self.volume, 20) + 1e-10)

    def alpha169(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        diff = self.close - d1
        sma1 = ema_sma(diff, n=9, m=1)
        a = _sma(delay(sma1, 1), 12)
        b = _sma(delay(sma1, 1), 26)
        return ema_sma(a - b, n=10, m=1)

    def alpha170(self) -> pd.DataFrame:
        part1 = rank(1 / (self.close + 1e-10)) * self.volume / (_sma(self.volume, 20) + 1e-10)
        part2 = self.high * rank(self.high - self.close) / (ts_sum(self.high, 5) / 5 + 1e-10)
        part3 = rank(self.vwap - delay(self.vwap, 5))
        return part1 * part2 - part3

    # ==================================================================
    # Alpha #171 – #180
    # ==================================================================

    def alpha171(self) -> pd.DataFrame:
        return -1 * (self.low - self.close) * self.open ** 5 / ((self.close - self.high) * self.close ** 5 + 1e-10)

    def alpha172(self) -> pd.DataFrame:
        """ADX — LD/HD/TR 方向运动指标"""
        ld = delay(self.low, 1) - self.low
        hd = self.high - delay(self.high, 1)
        hl = self.high - self.low
        abs_ch = (delay(self.close, 1) - self.high).abs()
        abs_cl = (delay(self.close, 1) - self.low).abs()
        tr = pd.DataFrame(
            np.maximum(np.maximum(hl.values, abs_ch.values), abs_cl.values),
            index=self.close.index, columns=self.close.columns,
        )
        cond_ld = (ld > 0) & (ld > hd)
        cond_hd = (hd > 0) & (hd > ld)
        ld_plus = self._where(cond_ld, ld, 0.0)
        hd_plus = self._where(cond_hd, hd, 0.0)
        sum_ld = ts_sum(ld_plus, 14) * 100 / (ts_sum(tr, 14) + 1e-10)
        sum_hd = ts_sum(hd_plus, 14) * 100 / (ts_sum(tr, 14) + 1e-10)
        dx = (sum_ld - sum_hd).abs() / (sum_ld + sum_hd + 1e-10) * 100
        return _sma(dx, 6)

    def alpha173(self) -> pd.DataFrame:
        sma1 = ema_sma(self.close, n=13, m=2)
        sma2 = ema_sma(sma1, n=13, m=2)
        sma3 = ema_sma(ema_sma(log(self.close), n=13, m=2), n=13, m=2)
        return 3 * sma1 - 2 * sma2 + ema_sma(sma3, n=13, m=2)

    def alpha174(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        cond = self.close > d1
        return ema_sma(self._where(cond, _std(self.close, 20), 0.0), n=20, m=1)

    def alpha175(self) -> pd.DataFrame:
        abs_ch = (delay(self.close, 1) - self.high).abs()
        abs_cl = (delay(self.close, 1) - self.low).abs()
        hl = self.high - self.low
        tr = pd.DataFrame(
            np.maximum(np.maximum(hl.values, abs_ch.values), abs_cl.values),
            index=self.close.index, columns=self.close.columns,
        )
        return _sma(tr, 6)

    def alpha176(self) -> pd.DataFrame:
        hl_range = ts_max(self.high, 12) - ts_min(self.low, 12)
        inner = (self.close - ts_min(self.low, 12)) / (hl_range + 1e-10)
        return ts_corr(rank(inner), rank(self.volume), 6)

    def alpha177(self) -> pd.DataFrame:
        return ((20 - highday(self.high, 20)) / 20) * 100

    def alpha178(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        return (self.close - d1) / (d1 + 1e-10) * self.volume

    def alpha179(self) -> pd.DataFrame:
        return rank(ts_corr(self.vwap, self.volume, 4)) * rank(ts_corr(rank(self.low), rank(_sma(self.volume, 50)), 12))

    def alpha180(self) -> pd.DataFrame:
        adv20 = _sma(self.volume, 20)
        cond = adv20 < self.volume
        part_true = -1 * ts_rank(delta(self.close, 7).abs(), 60) * sign(delta(self.close, 7))
        return self._where(cond, part_true, -1 * self.volume)

    # ==================================================================
    # Alpha #181 – #191
    # ==================================================================

    def alpha181(self) -> pd.DataFrame:
        bm_close = self._bm_close_panel()
        ret = self.close / (delay(self.close, 1) + 1e-10) - 1
        mean_ret = _sma(ret, 20)
        bm_mean = _sma(bm_close, 20)
        diff_bm = bm_close - bm_mean
        numerator = ts_sum((ret - mean_ret) - diff_bm ** 2, 20)
        denominator = ts_sum(diff_bm ** 3, 20)
        return numerator / (denominator + 1e-10)

    def alpha182(self) -> pd.DataFrame:
        bm_close = self._bm_close_panel()
        bm_open = self._bm_open_panel()
        cond1 = (self.close > self.open) & (bm_close > bm_open)
        cond2 = (self.close < self.open) & (bm_close < bm_open)
        return count(cond1 | cond2, 20) / 20

    def alpha183(self) -> pd.DataFrame:
        diff = self.close - _sma(self.close, 24)
        cum = sumac(diff)
        max_cum = ts_max(cum, 24)
        min_cum = ts_min(cum, 24)
        return max_cum - min_cum / (_std(self.close, 24) + 1e-10)

    def alpha184(self) -> pd.DataFrame:
        return rank(ts_corr(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha185(self) -> pd.DataFrame:
        return rank(-1 * ((1 - self.open / (self.close + 1e-10)) ** 2))

    def alpha186(self) -> pd.DataFrame:
        """ADX + DELAY(ADX, 6) / 2"""
        ld = delay(self.low, 1) - self.low
        hd = self.high - delay(self.high, 1)
        hl = self.high - self.low
        abs_ch = (delay(self.close, 1) - self.high).abs()
        abs_cl = (delay(self.close, 1) - self.low).abs()
        tr = pd.DataFrame(
            np.maximum(np.maximum(hl.values, abs_ch.values), abs_cl.values),
            index=self.close.index, columns=self.close.columns,
        )
        cond_ld = (ld > 0) & (ld > hd)
        cond_hd = (hd > 0) & (hd > ld)
        ld_plus = self._where(cond_ld, ld, 0.0)
        hd_plus = self._where(cond_hd, hd, 0.0)
        sum_ld = ts_sum(ld_plus, 14) * 100 / (ts_sum(tr, 14) + 1e-10)
        sum_hd = ts_sum(hd_plus, 14) * 100 / (ts_sum(tr, 14) + 1e-10)
        dx = (sum_ld - sum_hd).abs() / (sum_ld + sum_hd + 1e-10) * 100
        adx = _sma(dx, 6)
        return (adx + delay(adx, 6)) / 2

    def alpha187(self) -> pd.DataFrame:
        d_open1 = delay(self.open, 1)
        cond = self.open <= d_open1
        inner = pd.DataFrame(
            np.maximum((self.high - self.open).values, (self.open - d_open1).values),
            index=self.close.index, columns=self.close.columns,
        )
        return ts_sum(self._where(cond, 0.0, inner), 20)

    def alpha188(self) -> pd.DataFrame:
        hl = self.high - self.low
        sma_hl = ema_sma(hl, n=11, m=2)
        return (hl - sma_hl) / (sma_hl + 1e-10) * 100

    def alpha189(self) -> pd.DataFrame:
        return _sma((self.close - _sma(self.close, 6)).abs(), 6)

    def alpha190(self) -> pd.DataFrame:
        d1 = delay(self.close, 1)
        d19 = delay(self.close, 19)
        daily_ret = self.close / (d1 + 1e-10) - 1
        geo_mean = (self.close / (d19 + 1e-10)) ** (1 / 20) - 1

        cond_up = daily_ret > geo_mean
        cond_dn = daily_ret < geo_mean
        diff_sq = (daily_ret - geo_mean) ** 2

        n_up = count(cond_up, 20)
        n_dn = count(cond_dn, 20)
        sum_up = sumif(diff_sq, 20, cond_up)
        sum_dn = sumif(diff_sq, 20, cond_dn)

        return log(((n_dn - 1) * sum_dn) / ((n_up) * sum_up + 1e-10) + 1e-10)

    def alpha191(self) -> pd.DataFrame:
        return ts_corr(_sma(self.volume, 20), self.low, 5) + (self.high + self.low) / 2 - self.close


IndicatorRegistry.register("alpha191", Alpha191Indicators)
