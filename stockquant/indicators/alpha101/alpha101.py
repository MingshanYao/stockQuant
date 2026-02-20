"""
WorldQuant Alpha#101 因子指标 — 继承 BaseIndicator，融入 indicators 体系。

基于 Kakushadze (2016) *"101 Formulaic Alphas"* 论文，
完整实现全部 101 个 Alpha 因子公式。

单股票模式 (BaseIndicator 接口)
-------------------------------
>>> from stockquant.indicators import Alpha101Indicators
>>> indicator = Alpha101Indicators()
>>> df = indicator.compute(df)            # 附加全部 alpha 因子列
>>> df = indicator.compute(df, alphas=[1, 6, 12, 101])  # 只计算指定因子

面板模式 (多股票截面计算)
--------------------------
>>> engine = Alpha101Indicators.panel(
...     open_=open_df, high=high_df, low=low_df,
...     close=close_df, volume=volume_df,
... )
>>> factor_1 = engine.alpha001()
>>> all_factors = engine.compute_all()

Notes
-----
- VWAP 优先使用用户传入的值；其次通过 ``amount / volume``
  （成交额 / 成交量）自动计算；最终 fallback 到 ``(high + low + close) / 3``。
- 涉及行业中性化 (IndNeutralize) 和市值 (cap) 的因子，当 ``industry`` / ``cap``
  参数缺失时，默认自动从本地 ``stock_info`` 数据库表加载（通过列名匹配股票代码）。
  若数据库也无数据，则行业中性化跳过，市值用 ``close`` 近似。
  可通过 ``auto_load_info=False`` 禁用自动加载。
- 论文中部分窗口参数为浮点数，实现时取整为 ``int``。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockquant.indicators.base import BaseIndicator
from stockquant.indicators.alpha101.operators import (
    adv,
    decay_linear,
    delay,
    delta,
    ind_neutralize,
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
from stockquant.utils.logger import get_logger

logger = get_logger("indicators.alpha101")

# 需要行业数据的 Alpha 编号
INDUSTRY_ALPHAS: set[int] = {
    48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100,
}

# 需要市值数据的 Alpha 编号
CAP_ALPHAS: set[int] = {56}


class Alpha101Indicators(BaseIndicator):
    """WorldQuant Alpha#101 因子指标。

    继承 :class:`BaseIndicator`，通过 ``compute(df)`` 方法可对单只股票
    OHLCV DataFrame 计算 Alpha 因子并附加为新列，与系统现有指标体系
    （Trend / Oscillator / Volume / Volatility）保持一致的接口。

    同时通过 ``panel()`` 工厂方法支持多股票面板数据的截面计算。

    Parameters
    ----------
    alphas : list[int], optional
        要计算的因子编号列表，默认全部。
    """

    def __init__(self, alphas: Sequence[int] | None = None) -> None:
        self._alphas = list(alphas) if alphas else list(range(1, 102))

    # ------------------------------------------------------------------
    # BaseIndicator 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Alpha101"

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算 Alpha101 因子，并将结果作为新列附加到 DataFrame 中。

        Parameters
        ----------
        df : DataFrame
            单只股票数据，至少包含 open, high, low, close, volume 列。
        alphas : list[int], optional
            要计算的因子编号列表（覆盖构造时的设置）。

        Returns
        -------
        DataFrame
            原始列 + ``alpha001`` ... ``alpha101`` 新增列。
        """
        df = df.copy()
        alphas = kwargs.get("alphas", self._alphas)

        # 构建面板引擎（单股票退化为 1 列 DataFrame）
        engine = self._build_engine_from_single(df)

        for alpha_id in alphas:
            method = getattr(engine, f"alpha{alpha_id:03d}", None)
            if method is None:
                continue
            try:
                result = method()
                # 面板结果只有 1 列，squeeze 回 Series
                col_name = f"alpha{alpha_id:03d}"
                if isinstance(result, pd.DataFrame):
                    df[col_name] = result.iloc[:, 0].values
                else:
                    df[col_name] = result.values
            except Exception as e:
                logger.warning(f"Alpha#{alpha_id} 计算失败: {e}")
                df[f"alpha{alpha_id:03d}"] = np.nan

        return df

    # ------------------------------------------------------------------
    # 面板模式工厂
    # ------------------------------------------------------------------

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
        cap: pd.DataFrame | None = None,
        industry: pd.DataFrame | None = None,
        auto_load_info: bool = True,
    ) -> "Alpha101Engine":
        """创建面板计算引擎（多股票截面计算）。

        Parameters
        ----------
        open_ : DataFrame
            开盘价，行=日期，列=股票代码。
        high, low, close, volume : DataFrame
            同结构。
        vwap : DataFrame, optional
            成交量加权平均价。若未提供，优先用 ``amount / volume`` 计算。
        amount : DataFrame, optional
            成交额。当 ``vwap`` 缺失时用于计算 VWAP。
        returns : DataFrame, optional
            收益率。默认用 ``close.pct_change()``。
        cap : DataFrame, optional
            流通市值（Alpha#56 需要）。缺失时自动从 stock_info 表加载。
        industry : DataFrame, optional
            行业分类代码（IndNeutralize 需要）。缺失时自动从 stock_info 表加载。
        auto_load_info : bool, default True
            当 ``industry`` / ``cap`` 缺失时，是否自动从本地 stock_info 表加载。

        Returns
        -------
        Alpha101Engine
            面板计算引擎，提供 ``alpha001()`` ... ``alpha101()`` 方法。
        """
        return Alpha101Engine(
            open_=open_, high=high, low=low, close=close,
            volume=volume, vwap=vwap, amount=amount, returns=returns,
            cap=cap, industry=industry, auto_load_info=auto_load_info,
        )

    @classmethod
    def from_stacked_df(
        cls,
        df: pd.DataFrame,
        date_col: str = "date",
        code_col: str = "code",
        auto_load_info: bool = True,
    ) -> "Alpha101Engine":
        """从堆叠格式 DataFrame 创建面板计算引擎。

        Parameters
        ----------
        df : DataFrame
            长格式，至少包含 date, code, open, high, low, close, volume 列。
        auto_load_info : bool, default True
            当 ``industry`` / ``cap`` 缺失时，是否自动从本地 stock_info 表加载。
        """
        df = df.copy()
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index([date_col, code_col])

        def _pivot(col: str) -> pd.DataFrame | None:
            if col in df.columns:
                return df[col].unstack()
            return None

        return Alpha101Engine(
            open_=_pivot("open"),  # type: ignore[arg-type]
            high=_pivot("high"),  # type: ignore[arg-type]
            low=_pivot("low"),  # type: ignore[arg-type]
            close=_pivot("close"),  # type: ignore[arg-type]
            volume=_pivot("volume"),  # type: ignore[arg-type]
            vwap=_pivot("vwap"),
            amount=_pivot("amount"),
            returns=_pivot("returns"),
            cap=_pivot("cap"),
            industry=_pivot("industry"),
            auto_load_info=auto_load_info,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _build_engine_from_single(df: pd.DataFrame) -> "Alpha101Engine":
        """从单只股票 DataFrame 构建面板引擎（列退化为 1）。"""
        code = df.get("code", pd.Series(["_stock"])).iloc[0] if "code" in df.columns else "_stock"

        def _col(col_name: str) -> pd.DataFrame | None:
            if col_name in df.columns:
                return df[[col_name]].rename(columns={col_name: code})
            return None

        # industry 为分类字段，需要特殊处理
        industry_panel = None
        if "industry" in df.columns:
            industry_panel = df[["industry"]].rename(columns={"industry": code})

        return Alpha101Engine(
            open_=df[["open"]].rename(columns={"open": code}),
            high=df[["high"]].rename(columns={"high": code}),
            low=df[["low"]].rename(columns={"low": code}),
            close=df[["close"]].rename(columns={"close": code}),
            volume=df[["volume"]].rename(columns={"volume": code}),
            vwap=_col("vwap"),
            amount=_col("amount"),
            returns=_col("returns"),
            cap=_col("cap"),
            industry=industry_panel,
            auto_load_info=(code != "_stock"),  # 有真实代码时尝试自动加载
        )


class Alpha101Engine:
    """Alpha#101 面板计算引擎。

    承载全部 101 个因子的具体计算逻辑。面板数据格式：行=日期，列=股票代码。

    通常不直接实例化，而是通过以下方式获取：

    - ``Alpha101Indicators.panel(...)``  — 从面板数据构建
    - ``Alpha101Indicators.from_stacked_df(...)``  — 从堆叠 DataFrame 构建
    - ``Alpha101Indicators().compute(df)``  — 内部自动创建（单股票模式）
    """

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
        cap: pd.DataFrame | None = None,
        industry: pd.DataFrame | None = None,
        auto_load_info: bool = True,
    ) -> None:
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        if vwap is not None:
            self.vwap = vwap
        elif amount is not None:
            # VWAP = 成交额 / 成交量
            self.vwap = amount / (volume + 1e-10)
        else:
            self.vwap = (high + low + close) / 3
        self.returns = returns if returns is not None else close.pct_change()
        self.cap = cap
        self.industry = industry
        self._cache: dict[str, Any] = {}

        # 自动从 stock_info 表加载行业 / 市值
        if auto_load_info and (self.industry is None or self.cap is None):
            self._auto_load_stock_info()

    # ==================================================================
    # 内部工具
    # ==================================================================

    def _auto_load_stock_info(self) -> None:
        """从本地 stock_info 表自动加载行业和流通市值，填充缺失的 industry / cap。"""
        codes = list(self.close.columns)
        try:
            from stockquant.data.database import Database

            db = Database(read_only=True)
            if not db.table_exists("stock_info"):
                return

            placeholders = ", ".join(f"'{c}'" for c in codes)
            info_df = db.query(
                f"SELECT code, industry, float_cap FROM stock_info "
                f"WHERE code IN ({placeholders})"
            )
            db.close()

            if info_df.empty:
                return

            ind_map = dict(zip(info_df["code"], info_df["industry"]))
            cap_map = dict(zip(info_df["code"], info_df["float_cap"]))

            if self.industry is None and any(
                pd.notna(v) for v in ind_map.values()
            ):
                self.industry = self._expand_static_to_panel(
                    ind_map, self.close.index, self.close.columns,
                )
                logger.info(
                    f"已从 stock_info 加载行业数据: "
                    f"{self.industry.iloc[0].dropna().nunique()} 个行业"
                )

            if self.cap is None and any(
                pd.notna(v) for v in cap_map.values()
            ):
                self.cap = self._expand_static_to_panel(
                    cap_map, self.close.index, self.close.columns,
                    dtype=float,
                )
                logger.info("已从 stock_info 加载流通市值数据")

        except Exception as e:
            logger.debug(f"自动加载 stock_info 失败（不影响计算）: {e}")

    @staticmethod
    def _expand_static_to_panel(
        mapping: dict[str, Any],
        index: pd.Index,
        columns: pd.Index,
        dtype: type | None = None,
    ) -> pd.DataFrame:
        """将静态映射 (code → value) 扩展为面板 DataFrame (dates × codes)。

        Parameters
        ----------
        mapping : dict
            股票代码 → 值（行业代码或市值）。
        index : Index
            日期索引。
        columns : Index
            股票代码索引。
        dtype : type, optional
            强制数据类型，如 ``float``。
        """
        values = pd.Series(mapping).reindex(columns)
        if dtype is not None:
            values = values.astype(dtype)
        return pd.DataFrame(
            np.tile(values.values, (len(index), 1)),
            index=index,
            columns=columns,
        )

    def _adv(self, d: int) -> pd.DataFrame:
        """带缓存的平均日成交量。"""
        key = f"adv{d}"
        if key not in self._cache:
            self._cache[key] = adv(self.volume, d)
        return self._cache[key]

    def _where(self, condition, x, y) -> pd.DataFrame:
        """向量化条件选择: condition ? x : y。"""
        return pd.DataFrame(
            np.where(condition, x, y),
            index=self.close.index,
            columns=self.close.columns,
        )

    def _ind_neutralize(self, x: pd.DataFrame) -> pd.DataFrame:
        """行业中性化（若行业数据缺失则返回原值）。"""
        if self.industry is not None:
            return ind_neutralize(x, self.industry)
        return x

    @staticmethod
    def _clean(result: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """清理 inf / NaN。"""
        return result.replace([np.inf, -np.inf], np.nan)

    # ==================================================================
    # 批量计算
    # ==================================================================

    def compute_factor(self, alpha_id: int) -> pd.DataFrame:
        """计算指定编号的 Alpha 因子。"""
        method = getattr(self, f"alpha{alpha_id:03d}", None)
        if method is None:
            raise ValueError(f"Alpha#{alpha_id} 未实现")
        try:
            return self._clean(method())
        except Exception as e:
            logger.warning(f"Alpha#{alpha_id} 计算失败: {e}")
            return pd.DataFrame(
                index=self.close.index, columns=self.close.columns, dtype=float
            )

    def compute_all(
        self,
        include_industry: bool = True,
    ) -> dict[int, pd.DataFrame]:
        """批量计算全部 Alpha 因子。

        Parameters
        ----------
        include_industry : bool
            是否包含需要行业数据的因子（在无行业数据时自动跳过中性化步骤）。
        """
        results: dict[int, pd.DataFrame] = {}
        for i in range(1, 102):
            method = getattr(self, f"alpha{i:03d}", None)
            if method is None:
                continue
            if not include_industry and i in INDUSTRY_ALPHAS:
                continue
            try:
                results[i] = self._clean(method())
            except Exception as e:
                logger.warning(f"Alpha#{i} 计算失败: {e}")
        logger.info(f"成功计算 {len(results)}/101 个 Alpha 因子")
        return results

    # ==================================================================
    # Alpha #001 – #020
    # ==================================================================

    def alpha001(self) -> pd.DataFrame:
        """rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) − 0.5"""
        inner = self._where(
            self.returns < 0, ts_stddev(self.returns, 20), self.close
        )
        return rank(ts_argmax(signedpower(inner, 2.0), 5)) - 0.5

    def alpha002(self) -> pd.DataFrame:
        """−1 × correlation(rank(delta(log(volume), 2)), rank(((close − open) / open)), 6)"""
        return -1 * ts_corr(
            rank(delta(log(self.volume), 2)),
            rank((self.close - self.open) / (self.open + 1e-10)),
            6,
        )

    def alpha003(self) -> pd.DataFrame:
        """−1 × correlation(rank(open), rank(volume), 10)"""
        return -1 * ts_corr(rank(self.open), rank(self.volume), 10)

    def alpha004(self) -> pd.DataFrame:
        """−1 × Ts_Rank(rank(low), 9)"""
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self) -> pd.DataFrame:
        """rank((open − (sum(vwap, 10) / 10))) × (−1 × abs(rank((close − vwap))))"""
        return rank(self.open - sma(self.vwap, 10)) * (
            -1 * rank(self.close - self.vwap).abs()
        )

    def alpha006(self) -> pd.DataFrame:
        """−1 × correlation(open, volume, 10)"""
        return -1 * ts_corr(self.open, self.volume, 10)

    def alpha007(self) -> pd.DataFrame:
        """(adv20 < volume) ? (−1 × ts_rank(abs(delta(close, 7)), 60) × sign(delta(close, 7))) : −1"""
        adv20 = self._adv(20)
        d = delta(self.close, 7)
        inner = -1 * ts_rank(d.abs(), 60) * sign(d)
        return self._where(adv20 < self.volume, inner, -1.0)

    def alpha008(self) -> pd.DataFrame:
        """−1 × rank(((sum(open, 5) × sum(returns, 5)) − delay((sum(open, 5) × sum(returns, 5)), 10)))"""
        x = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -1 * rank(x - delay(x, 10))

    def alpha009(self) -> pd.DataFrame:
        """条件差分动量因子"""
        d = delta(self.close, 1)
        cond1 = ts_min(d, 5) > 0
        cond2 = ts_max(d, 5) < 0
        return self._where(cond1, d, self._where(cond2, d, -d))

    def alpha010(self) -> pd.DataFrame:
        """rank(条件差分动量因子)"""
        d = delta(self.close, 1)
        cond1 = ts_min(d, 4) > 0
        cond2 = ts_max(d, 4) < 0
        inner = self._where(cond1, d, self._where(cond2, d, -d))
        return rank(inner)

    def alpha011(self) -> pd.DataFrame:
        """(rank(ts_max(vwap − close, 3)) + rank(ts_min(vwap − close, 3))) × rank(delta(volume, 3))"""
        diff = self.vwap - self.close
        return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(
            delta(self.volume, 3)
        )

    def alpha012(self) -> pd.DataFrame:
        """sign(delta(volume, 1)) × (−1 × delta(close, 1))"""
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self) -> pd.DataFrame:
        """−1 × rank(covariance(rank(close), rank(volume), 5))"""
        return -1 * rank(ts_cov(rank(self.close), rank(self.volume), 5))

    def alpha014(self) -> pd.DataFrame:
        """(−1 × rank(delta(returns, 3))) × correlation(open, volume, 10)"""
        return -1 * rank(delta(self.returns, 3)) * ts_corr(
            self.open, self.volume, 10
        )

    def alpha015(self) -> pd.DataFrame:
        """−1 × sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
        return -1 * ts_sum(
            rank(ts_corr(rank(self.high), rank(self.volume), 3)), 3
        )

    def alpha016(self) -> pd.DataFrame:
        """−1 × rank(covariance(rank(high), rank(volume), 5))"""
        return -1 * rank(ts_cov(rank(self.high), rank(self.volume), 5))

    def alpha017(self) -> pd.DataFrame:
        """(−1 × rank(ts_rank(close, 10))) × rank(delta(delta(close, 1), 1)) × rank(ts_rank(volume/adv20, 5))"""
        adv20 = self._adv(20)
        return (
            (-1 * rank(ts_rank(self.close, 10)))
            * rank(delta(delta(self.close, 1), 1))
            * rank(ts_rank(self.volume / (adv20 + 1e-10), 5))
        )

    def alpha018(self) -> pd.DataFrame:
        """−1 × rank((stddev(|close − open|, 5) + (close − open) + correlation(close, open, 10)))"""
        co = self.close - self.open
        return -1 * rank(
            ts_stddev(co.abs(), 5) + co + ts_corr(self.close, self.open, 10)
        )

    def alpha019(self) -> pd.DataFrame:
        """(−1 × sign((close − delay(close, 7)) + delta(close, 7))) × (1 + rank(1 + sum(returns, 250)))"""
        return (
            -1
            * sign(self.close - delay(self.close, 7) + delta(self.close, 7))
            * (1 + rank(1 + ts_sum(self.returns, 250)))
        )

    def alpha020(self) -> pd.DataFrame:
        """(−1 × rank(open − delay(high, 1))) × rank(open − delay(close, 1)) × rank(open − delay(low, 1))"""
        return (
            (-1 * rank(self.open - delay(self.high, 1)))
            * rank(self.open - delay(self.close, 1))
            * rank(self.open - delay(self.low, 1))
        )

    # ==================================================================
    # Alpha #021 – #040
    # ==================================================================

    def alpha021(self) -> pd.DataFrame:
        """均线/波动率条件 + 量比条件复合因子"""
        adv20 = self._adv(20)
        sma8 = sma(self.close, 8)
        std8 = ts_stddev(self.close, 8)
        sma2 = sma(self.close, 2)
        vol_ratio = self.volume / (adv20 + 1e-10)
        cond1 = (sma8 + std8) < sma2
        cond2 = sma2 < (sma8 - std8)
        cond3 = vol_ratio >= 1
        return self._where(
            cond1, -1.0, self._where(cond2, 1.0, self._where(cond3, 1.0, -1.0))
        )

    def alpha022(self) -> pd.DataFrame:
        """−1 × delta(correlation(high, volume, 5), 5) × rank(stddev(close, 20))"""
        return -1 * delta(ts_corr(self.high, self.volume, 5), 5) * rank(
            ts_stddev(self.close, 20)
        )

    def alpha023(self) -> pd.DataFrame:
        """(sma(high, 20) < high) ? −delta(high, 2) : 0"""
        cond = sma(self.high, 20) < self.high
        return self._where(cond, -1 * delta(self.high, 2), 0.0)

    def alpha024(self) -> pd.DataFrame:
        """长期均线偏离条件因子"""
        ratio = delta(sma(self.close, 100), 100) / (delay(self.close, 100) + 1e-10)
        cond = ratio <= 0.05
        return self._where(
            cond,
            -1 * (self.close - ts_min(self.close, 100)),
            -1 * delta(self.close, 3),
        )

    def alpha025(self) -> pd.DataFrame:
        """rank(−returns × adv20 × vwap × (high − close))"""
        return rank(
            -1 * self.returns * self._adv(20) * self.vwap * (self.high - self.close)
        )

    def alpha026(self) -> pd.DataFrame:
        """−1 × ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"""
        return -1 * ts_max(
            ts_corr(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3
        )

    def alpha027(self) -> pd.DataFrame:
        """(0.5 < rank(sum(corr(rank(volume), rank(vwap), 6), 2) / 2)) ? −1 : 1"""
        cond = (
            rank(ts_sum(ts_corr(rank(self.volume), rank(self.vwap), 6), 2) / 2.0)
            > 0.5
        )
        return self._where(cond, -1.0, 1.0)

    def alpha028(self) -> pd.DataFrame:
        """scale(corr(adv20, low, 5) + (high + low) / 2 − close)"""
        return scale(
            ts_corr(self._adv(20), self.low, 5)
            + (self.high + self.low) / 2
            - self.close
        )

    def alpha029(self) -> pd.DataFrame:
        """复合排名衰减因子"""
        inner = rank(rank(-1 * rank(delta(self.close - 1, 5))))
        inner = ts_sum(ts_min(inner, 2), 1)
        inner = rank(rank(scale(log(inner.clip(lower=1e-10)))))
        inner = ts_min(ts_product(inner, 1), 5)
        return inner + ts_rank(delay(-1 * self.returns, 6), 5)

    def alpha030(self) -> pd.DataFrame:
        """((1 − rank(sign(close − delay(close, 1)) + sign(delay... )) × sum(vol, 5)) / sum(vol, 20)"""
        d1 = sign(self.close - delay(self.close, 1))
        d2 = sign(delay(self.close, 1) - delay(self.close, 2))
        d3 = sign(delay(self.close, 2) - delay(self.close, 3))
        return (
            (1.0 - rank(d1 + d2 + d3))
            * ts_sum(self.volume, 5)
            / (ts_sum(self.volume, 20) + 1e-10)
        )

    def alpha031(self) -> pd.DataFrame:
        """rank^3(decay_linear(−rank^2(delta(close, 10)), 10)) + rank(−delta(close, 3)) + sign(scale(corr(adv20, low, 12)))"""
        part1 = rank(
            rank(rank(decay_linear(-1 * rank(rank(delta(self.close, 10))), 10)))
        )
        part2 = rank(-1 * delta(self.close, 3))
        part3 = sign(scale(ts_corr(self._adv(20), self.low, 12)))
        return part1 + part2 + part3

    def alpha032(self) -> pd.DataFrame:
        """scale(sma(close, 7) − close) + 20 × scale(corr(vwap, delay(close, 5), 230))"""
        return scale(sma(self.close, 7) - self.close) + 20 * scale(
            ts_corr(self.vwap, delay(self.close, 5), 230)
        )

    def alpha033(self) -> pd.DataFrame:
        """rank(−(1 − open / close))"""
        return rank(-1 * (1 - self.open / (self.close + 1e-10)))

    def alpha034(self) -> pd.DataFrame:
        """rank((1 − rank(stddev(ret, 2) / stddev(ret, 5))) + (1 − rank(delta(close, 1))))"""
        return rank(
            (1 - rank(ts_stddev(self.returns, 2) / (ts_stddev(self.returns, 5) + 1e-10)))
            + (1 - rank(delta(self.close, 1)))
        )

    def alpha035(self) -> pd.DataFrame:
        """Ts_Rank(volume, 32) × (1 − Ts_Rank(close + high − low, 16)) × (1 − Ts_Rank(returns, 32))"""
        return (
            ts_rank(self.volume, 32)
            * (1 - ts_rank(self.close + self.high - self.low, 16))
            * (1 - ts_rank(self.returns, 32))
        )

    def alpha036(self) -> pd.DataFrame:
        """多项排名加权复合因子"""
        p1 = 2.21 * rank(
            ts_corr(self.close - self.open, delay(self.volume, 1), 15)
        )
        p2 = 0.7 * rank(self.open - self.close)
        p3 = 0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5))
        p4 = rank(ts_corr(self.vwap, self._adv(20), 6).abs())
        p5 = 0.6 * rank(
            (sma(self.close, 200) - self.open) * (self.close - self.open)
        )
        return p1 + p2 + p3 + p4 + p5

    def alpha037(self) -> pd.DataFrame:
        """rank(corr(delay(open − close, 1), close, 200)) + rank(open − close)"""
        return rank(
            ts_corr(delay(self.open - self.close, 1), self.close, 200)
        ) + rank(self.open - self.close)

    def alpha038(self) -> pd.DataFrame:
        """(−1 × rank(Ts_Rank(close, 10))) × rank(close / open)"""
        return -1 * rank(ts_rank(self.close, 10)) * rank(
            self.close / (self.open + 1e-10)
        )

    def alpha039(self) -> pd.DataFrame:
        """(−1 × rank(delta(close, 7) × (1 − rank(decay_linear(vol/adv20, 9))))) × (1 + rank(sum(ret, 250)))"""
        adv20 = self._adv(20)
        return (
            -1
            * rank(
                delta(self.close, 7)
                * (1 - rank(decay_linear(self.volume / (adv20 + 1e-10), 9)))
            )
            * (1 + rank(ts_sum(self.returns, 250)))
        )

    def alpha040(self) -> pd.DataFrame:
        """(−1 × rank(stddev(high, 10))) × corr(high, volume, 10)"""
        return -1 * rank(ts_stddev(self.high, 10)) * ts_corr(
            self.high, self.volume, 10
        )

    # ==================================================================
    # Alpha #041 – #060
    # ==================================================================

    def alpha041(self) -> pd.DataFrame:
        """(high × low)^0.5 − vwap"""
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha042(self) -> pd.DataFrame:
        """rank(vwap − close) / rank(vwap + close)"""
        return rank(self.vwap - self.close) / (rank(self.vwap + self.close) + 1e-10)

    def alpha043(self) -> pd.DataFrame:
        """ts_rank(volume / adv20, 20) × ts_rank(−delta(close, 7), 8)"""
        return ts_rank(self.volume / (self._adv(20) + 1e-10), 20) * ts_rank(
            -1 * delta(self.close, 7), 8
        )

    def alpha044(self) -> pd.DataFrame:
        """−1 × correlation(high, rank(volume), 5)"""
        return -1 * ts_corr(self.high, rank(self.volume), 5)

    def alpha045(self) -> pd.DataFrame:
        """−1 × rank(sma(delay(close, 5), 20)) × corr(close, volume, 2) × rank(corr(sum(close, 5), sum(close, 20), 2))"""
        return (
            -1
            * rank(sma(delay(self.close, 5), 20))
            * ts_corr(self.close, self.volume, 2)
            * rank(ts_corr(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        )

    def alpha046(self) -> pd.DataFrame:
        """价格加速度条件因子"""
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (
            delay(self.close, 10) - self.close
        ) / 10
        cond1 = x > 0.25
        cond2 = x < 0
        return self._where(
            cond1,
            -1.0,
            self._where(cond2, 1.0, -1 * (self.close - delay(self.close, 1))),
        )

    def alpha047(self) -> pd.DataFrame:
        """量价排名复合因子"""
        adv20 = self._adv(20)
        part1 = (
            rank(1.0 / (self.close + 1e-10))
            * self.volume
            / (adv20 + 1e-10)
            * (self.high * rank(self.high - self.close) / (sma(self.high, 5) + 1e-10))
        )
        part2 = rank(self.vwap - delay(self.vwap, 5))
        return part1 - part2

    def alpha048(self) -> pd.DataFrame:
        """IndNeutralize 相关性衰减因子"""
        d1 = delta(self.close, 1)
        d2 = delta(delay(self.close, 1), 1)
        inner = ts_corr(d1, d2, 250) * d1 / (self.close + 1e-10)
        inner = self._ind_neutralize(inner)
        denom = ts_sum(
            (d1 / (delay(self.close, 1) + 1e-10)) ** 2, 250
        )
        return inner / (denom + 1e-10)

    def alpha049(self) -> pd.DataFrame:
        """价格加速度阈值因子 (−0.1)"""
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (
            delay(self.close, 10) - self.close
        ) / 10
        cond = x < -0.1
        return self._where(cond, 1.0, -1 * (self.close - delay(self.close, 1)))

    def alpha050(self) -> pd.DataFrame:
        """−1 × ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)"""
        return -1 * ts_max(
            rank(ts_corr(rank(self.volume), rank(self.vwap), 5)), 5
        )

    def alpha051(self) -> pd.DataFrame:
        """价格加速度阈值因子 (−0.05)"""
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (
            delay(self.close, 10) - self.close
        ) / 10
        cond = x < -0.05
        return self._where(cond, 1.0, -1 * (self.close - delay(self.close, 1)))

    def alpha052(self) -> pd.DataFrame:
        """低点反转 × 长期动量排名 × 量排名"""
        part1 = -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)
        part2 = rank(
            (ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220
        )
        return part1 * part2 * ts_rank(self.volume, 5)

    def alpha053(self) -> pd.DataFrame:
        """−delta(((close − low) − (high − close)) / (close − low), 9)"""
        inner = (
            (self.close - self.low) - (self.high - self.close)
        ) / (self.close - self.low + 1e-10)
        return -1 * delta(inner, 9)

    def alpha054(self) -> pd.DataFrame:
        """(−(low − close) × open^5) / ((low − high) × close^5)"""
        return (
            -1
            * (self.low - self.close)
            * self.open ** 5
            / ((self.low - self.high + 1e-10) * self.close ** 5 + 1e-10)
        )

    def alpha055(self) -> pd.DataFrame:
        """−corr(rank((close − ts_min(low, 12)) / (ts_max(high, 12) − ts_min(low, 12))), rank(volume), 6)"""
        inner = (self.close - ts_min(self.low, 12)) / (
            ts_max(self.high, 12) - ts_min(self.low, 12) + 1e-10
        )
        return -1 * ts_corr(rank(inner), rank(self.volume), 6)

    def alpha056(self) -> pd.DataFrame:
        """−rank(sum(ret, 10) / sum(sum(ret, 2), 3)) × rank(returns × cap)"""
        cap = self.cap if self.cap is not None else self.close
        return -1 * rank(
            ts_sum(self.returns, 10) / (ts_sum(ts_sum(self.returns, 2), 3) + 1e-10)
        ) * rank(self.returns * cap)

    def alpha057(self) -> pd.DataFrame:
        """−(close − vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)"""
        return -1 * (self.close - self.vwap) / (
            decay_linear(rank(ts_argmax(self.close, 30)), 2) + 1e-10
        )

    def alpha058(self) -> pd.DataFrame:
        """−Ts_Rank(decay_linear(corr(IndNeutralize(vwap), volume, 4), 8), 6)"""
        vwap_n = self._ind_neutralize(self.vwap)
        return -1 * ts_rank(
            decay_linear(ts_corr(vwap_n, self.volume, 4), 8), 6
        )

    def alpha059(self) -> pd.DataFrame:
        """−Ts_Rank(decay_linear(corr(IndNeutralize(vwap), volume, 4), 16), 8)"""
        vwap_n = self._ind_neutralize(self.vwap)
        return -1 * ts_rank(
            decay_linear(ts_corr(vwap_n, self.volume, 4), 16), 8
        )

    def alpha060(self) -> pd.DataFrame:
        """−(2 × scale(rank(((close−low)−(high−close))/(high−low) × volume)) − scale(rank(ts_argmax(close, 10))))"""
        inner = (
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low + 1e-10)
            * self.volume
        )
        return -1 * (
            2 * scale(rank(inner)) - scale(rank(ts_argmax(self.close, 10)))
        )

    # ==================================================================
    # Alpha #061 – #080
    # ==================================================================

    def alpha061(self) -> pd.DataFrame:
        """rank(vwap − ts_min(vwap, 16)) < rank(corr(vwap, adv180, 18))"""
        left = rank(self.vwap - ts_min(self.vwap, 16))
        right = rank(ts_corr(self.vwap, self._adv(180), 18))
        return (left < right).astype(float)

    def alpha062(self) -> pd.DataFrame:
        """(rank(corr(vwap, sum(adv20, 22), 10)) < rank(((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high))))) × −1"""
        left = rank(ts_corr(self.vwap, ts_sum(self._adv(20), 22), 10))
        inner_cond = (rank(self.open) + rank(self.open)) < (
            rank((self.high + self.low) / 2) + rank(self.high)
        )
        right = rank(inner_cond.astype(float))
        return (left < right).astype(float) * -1

    def alpha063(self) -> pd.DataFrame:
        """IndNeutralize delta/correlation 复合因子"""
        close_n = self._ind_neutralize(self.close)
        part1 = rank(decay_linear(delta(close_n, 2), 8))
        inner = self.vwap * 0.318108 + self.open * (1 - 0.318108)
        part2 = rank(
            decay_linear(
                ts_corr(inner, ts_sum(self._adv(180), 37), 14), 12
            )
        )
        return (part1 - part2) * -1

    def alpha064(self) -> pd.DataFrame:
        """量价相关性 vs 价格动量"""
        inner1 = self.open * 0.178404 + self.low * (1 - 0.178404)
        left = rank(
            ts_corr(ts_sum(inner1, 13), ts_sum(self._adv(120), 13), 17)
        )
        inner2 = (
            (self.high + self.low) / 2 * 0.178404 + self.vwap * (1 - 0.178404)
        )
        right = rank(delta(inner2, 4))
        return (left < right).astype(float) * -1

    def alpha065(self) -> pd.DataFrame:
        """open-vwap 混合量价相关性"""
        inner = self.open * 0.00817205 + self.vwap * (1 - 0.00817205)
        left = rank(ts_corr(inner, ts_sum(self._adv(60), 9), 6))
        right = rank(self.open - ts_min(self.open, 14))
        return (left < right).astype(float) * -1

    def alpha066(self) -> pd.DataFrame:
        """rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear((low − vwap)/(open − mid), 11), 7)) × −1"""
        part1 = rank(decay_linear(delta(self.vwap, 4), 7))
        inner = (self.low - self.vwap) / (
            self.open - (self.high + self.low) / 2 + 1e-10
        )
        part2 = ts_rank(decay_linear(inner, 11), 7)
        return (part1 + part2) * -1

    def alpha067(self) -> pd.DataFrame:
        """IndNeutralize 高点排名 × 量价相关性"""
        vwap_n = self._ind_neutralize(self.vwap)
        adv20_n = self._ind_neutralize(self._adv(20))
        part1 = rank(self.high - ts_min(self.high, 2))
        part2 = rank(ts_corr(vwap_n, adv20_n, 6))
        return -(part1 ** part2)

    def alpha068(self) -> pd.DataFrame:
        """Ts_Rank(corr(rank(high), rank(adv15), 9), 14) < rank(delta(close×0.52+low×0.48, 1)) × −1"""
        left = ts_rank(
            ts_corr(rank(self.high), rank(self._adv(15)), 9), 14
        )
        right = rank(
            delta(self.close * 0.518371 + self.low * (1 - 0.518371), 1)
        )
        return (left < right).astype(float) * -1

    def alpha069(self) -> pd.DataFrame:
        """IndNeutralize vwap 动量 × 量价相关性"""
        vwap_n = self._ind_neutralize(self.vwap)
        part1 = rank(ts_max(delta(vwap_n, 3), 5))
        inner = self.close * 0.490655 + self.vwap * (1 - 0.490655)
        part2 = ts_rank(ts_corr(inner, self._adv(20), 5), 9)
        return -(part1 ** part2)

    def alpha070(self) -> pd.DataFrame:
        """IndNeutralize close × adv50 相关性"""
        close_n = self._ind_neutralize(self.close)
        part1 = rank(delta(self.vwap, 1))
        part2 = ts_rank(ts_corr(close_n, self._adv(50), 18), 18)
        return -(part1 ** part2)

    def alpha071(self) -> pd.DataFrame:
        """max(ts_rank 衰减相关, ts_rank 衰减排名²)"""
        part1 = ts_rank(
            decay_linear(
                ts_corr(ts_rank(self.close, 3), ts_rank(self._adv(180), 12), 18),
                4,
            ),
            16,
        )
        inner = rank(self.low + self.open - 2 * self.vwap) ** 2
        part2 = ts_rank(decay_linear(inner, 16), 4)
        return pd.DataFrame(
            np.maximum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha072(self) -> pd.DataFrame:
        """rank(decay corr((H+L)/2, adv40)) / rank(decay corr(ts_rank(vwap), ts_rank(vol)))"""
        part1 = rank(
            decay_linear(
                ts_corr((self.high + self.low) / 2, self._adv(40), 9), 10
            )
        )
        part2 = rank(
            decay_linear(
                ts_corr(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3
            )
        )
        return part1 / (part2 + 1e-10)

    def alpha073(self) -> pd.DataFrame:
        """−max(rank(decay vwap delta), ts_rank(decay (−ret), 17))"""
        part1 = rank(decay_linear(delta(self.vwap, 5), 3))
        inner = self.open * 0.147155 + self.low * (1 - 0.147155)
        part2 = ts_rank(
            decay_linear(-1 * delta(inner, 2) / (inner + 1e-10), 3), 17
        )
        return -1 * pd.DataFrame(
            np.maximum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha074(self) -> pd.DataFrame:
        """rank(corr(close, sum(adv30, 37), 15)) < rank(corr(rank(high×0.03+vwap×0.97), rank(vol), 11)) × −1"""
        left = rank(ts_corr(self.close, ts_sum(self._adv(30), 37), 15))
        inner = self.high * 0.0261661 + self.vwap * (1 - 0.0261661)
        right = rank(ts_corr(rank(inner), rank(self.volume), 11))
        return (left < right).astype(float) * -1

    def alpha075(self) -> pd.DataFrame:
        """rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))"""
        left = rank(ts_corr(self.vwap, self.volume, 4))
        right = rank(ts_corr(rank(self.low), rank(self._adv(50)), 12))
        return (left < right).astype(float)

    def alpha076(self) -> pd.DataFrame:
        """−max(rank(decay vwap delta), ts_rank(decay(ts_rank(corr(IndNeutralize(low), adv81)))))"""
        low_n = self._ind_neutralize(self.low)
        part1 = rank(decay_linear(delta(self.vwap, 1), 12))
        part2 = ts_rank(
            decay_linear(
                ts_rank(ts_corr(low_n, self._adv(81), 8), 20), 17
            ),
            19,
        )
        return -1 * pd.DataFrame(
            np.maximum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha077(self) -> pd.DataFrame:
        """min(rank(decay((H+L)/2 − vwap, 20)), rank(decay(corr((H+L)/2, adv40, 3), 6)))"""
        part1 = rank(
            decay_linear((self.high + self.low) / 2 - self.vwap, 20)
        )
        part2 = rank(
            decay_linear(
                ts_corr((self.high + self.low) / 2, self._adv(40), 3), 6
            )
        )
        return pd.DataFrame(
            np.minimum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha078(self) -> pd.DataFrame:
        """rank(corr(sum(low×0.35+vwap×0.65, 20), sum(adv40, 20), 7)) ^ rank(corr(rank(vwap), rank(volume), 6))"""
        inner = self.low * 0.352233 + self.vwap * (1 - 0.352233)
        part1 = rank(ts_corr(ts_sum(inner, 20), ts_sum(self._adv(40), 20), 7))
        part2 = rank(ts_corr(rank(self.vwap), rank(self.volume), 6))
        return part1 ** part2

    def alpha079(self) -> pd.DataFrame:
        """IndNeutralize(close×0.61+open×0.39) 动量 < 量价 ts_rank 相关性"""
        inner = self.close * 0.60733 + self.open * (1 - 0.60733)
        inner_n = self._ind_neutralize(inner)
        left = rank(delta(inner_n, 1))
        right = rank(
            ts_corr(ts_rank(self.vwap, 4), ts_rank(self._adv(150), 9), 15)
        )
        return (left < right).astype(float)

    def alpha080(self) -> pd.DataFrame:
        """IndNeutralize(open×0.87+high×0.13) 符号变化 ^ 量价相关性排名"""
        inner = self.open * 0.868128 + self.high * (1 - 0.868128)
        inner_n = self._ind_neutralize(inner)
        part1 = rank(sign(delta(inner_n, 4)))
        part2 = ts_rank(ts_corr(self.high, self._adv(10), 5), 6)
        return -(part1 ** part2)

    # ==================================================================
    # Alpha #081 – #101
    # ==================================================================

    def alpha081(self) -> pd.DataFrame:
        """rank(log(product(rank(rank(corr(vwap, sum(adv10, 50), 8))^4), 15))) < rank(corr(rank(vwap), rank(vol), 5)) × −1"""
        inner = rank(ts_corr(self.vwap, ts_sum(self._adv(10), 50), 8)) ** 4
        left = rank(log(ts_product(rank(inner), 15)))
        right = rank(ts_corr(rank(self.vwap), rank(self.volume), 5))
        return (left < right).astype(float) * -1

    def alpha082(self) -> pd.DataFrame:
        """−min(rank(decay(delta(open, 1), 15)), ts_rank(decay(corr(IndNeutralize(vol), open, 17), 7), 13))"""
        vol_n = self._ind_neutralize(self.volume)
        part1 = rank(decay_linear(delta(self.open, 1), 15))
        part2 = ts_rank(
            decay_linear(ts_corr(vol_n, self.open, 17), 7), 13
        )
        return -1 * pd.DataFrame(
            np.minimum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha083(self) -> pd.DataFrame:
        """rank(delay((H−L)/(sum(close,5)/5), 2)) × rank(rank(volume)) / ((H−L)/(sum(close,5)/5) / (vwap−close))"""
        hl = (self.high - self.low) / (sma(self.close, 5) + 1e-10)
        num = rank(delay(hl, 2)) * rank(rank(self.volume))
        denom = hl / (self.vwap - self.close + 1e-10)
        return num / (denom + 1e-10)

    def alpha084(self) -> pd.DataFrame:
        """SignedPower(Ts_Rank(vwap − ts_max(vwap, 15), 21), delta(close, 5))"""
        return signedpower(
            ts_rank(self.vwap - ts_max(self.vwap, 15), 21),
            delta(self.close, 5),
        )

    def alpha085(self) -> pd.DataFrame:
        """rank(corr(high×0.88+close×0.12, adv30, 10)) ^ rank(corr(Ts_Rank((H+L)/2, 4), Ts_Rank(vol, 10), 7))"""
        inner1 = self.high * 0.876703 + self.close * (1 - 0.876703)
        part1 = rank(ts_corr(inner1, self._adv(30), 10))
        part2 = rank(
            ts_corr(
                ts_rank((self.high + self.low) / 2, 4),
                ts_rank(self.volume, 10),
                7,
            )
        )
        return part1 ** part2

    def alpha086(self) -> pd.DataFrame:
        """(Ts_Rank(corr(close, sum(adv20, 15), 6), 20) < rank(close − vwap)) × −1"""
        left = ts_rank(
            ts_corr(self.close, ts_sum(self._adv(20), 15), 6), 20
        )
        right = rank(self.close - self.vwap)
        return (left < right).astype(float) * -1

    def alpha087(self) -> pd.DataFrame:
        """−max(rank(decay delta(close×0.37+vwap×0.63)), ts_rank(decay |corr(IndNeutralize(adv81), close)|))"""
        adv81_n = self._ind_neutralize(self._adv(81))
        inner = self.close * 0.369701 + self.vwap * (1 - 0.369701)
        part1 = rank(decay_linear(delta(inner, 1), 3))
        part2 = ts_rank(
            decay_linear(ts_corr(adv81_n, self.close, 13).abs(), 5), 14
        )
        return -1 * pd.DataFrame(
            np.maximum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha088(self) -> pd.DataFrame:
        """min(rank(decay(rank(open)+rank(low)−rank(high)−rank(close))), ts_rank(decay(corr ts_rank)))"""
        part1 = rank(
            decay_linear(
                rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close),
                8,
            )
        )
        part2 = ts_rank(
            decay_linear(
                ts_corr(ts_rank(self.close, 8), ts_rank(self._adv(60), 21), 8), 7
            ),
            3,
        )
        return pd.DataFrame(
            np.minimum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha089(self) -> pd.DataFrame:
        """Ts_Rank(decay(corr(low, adv10, 7), 6), 4) − Ts_Rank(decay(delta(IndNeutralize(vwap), 3), 10), 15)"""
        vwap_n = self._ind_neutralize(self.vwap)
        part1 = ts_rank(
            decay_linear(ts_corr(self.low, self._adv(10), 7), 6), 4
        )
        part2 = ts_rank(decay_linear(delta(vwap_n, 3), 10), 15)
        return part1 - part2

    def alpha090(self) -> pd.DataFrame:
        """−(rank(close − ts_max(close, 5)) ^ Ts_Rank(corr(IndNeutralize(adv40), low, 5), 3))"""
        adv40_n = self._ind_neutralize(self._adv(40))
        part1 = rank(self.close - ts_max(self.close, 5))
        part2 = ts_rank(ts_corr(adv40_n, self.low, 5), 3)
        return -(part1 ** part2)

    def alpha091(self) -> pd.DataFrame:
        """(ts_rank(decay(decay(corr(IndNeutralize(close), volume, 10), 16), 4), 5) − rank(decay(corr(vwap, adv30, 4), 3))) × −1"""
        close_n = self._ind_neutralize(self.close)
        part1 = ts_rank(
            decay_linear(
                decay_linear(ts_corr(close_n, self.volume, 10), 16), 4
            ),
            5,
        )
        part2 = rank(
            decay_linear(ts_corr(self.vwap, self._adv(30), 4), 3)
        )
        return (part1 - part2) * -1

    def alpha092(self) -> pd.DataFrame:
        """min(Ts_Rank(decay(((H+L)/2+C < L+O), 15), 19), Ts_Rank(decay(corr(rank(low), rank(adv30), 8), 7), 7))"""
        cond = (
            (self.high + self.low) / 2 + self.close
        ) < (self.low + self.open)
        part1 = ts_rank(decay_linear(cond.astype(float), 15), 19)
        part2 = ts_rank(
            decay_linear(
                ts_corr(rank(self.low), rank(self._adv(30)), 8), 7
            ),
            7,
        )
        return pd.DataFrame(
            np.minimum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha093(self) -> pd.DataFrame:
        """Ts_Rank(decay(corr(IndNeutralize(vwap), adv81, 17), 20), 8) / rank(decay(delta(close×0.52+vwap×0.48, 3), 16))"""
        vwap_n = self._ind_neutralize(self.vwap)
        inner = self.close * 0.524434 + self.vwap * (1 - 0.524434)
        part1 = ts_rank(
            decay_linear(ts_corr(vwap_n, self._adv(81), 17), 20), 8
        )
        part2 = rank(decay_linear(delta(inner, 3), 16))
        return part1 / (part2 + 1e-10)

    def alpha094(self) -> pd.DataFrame:
        """−(rank(vwap − ts_min(vwap, 12)) ^ Ts_Rank(corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3))"""
        part1 = rank(self.vwap - ts_min(self.vwap, 12))
        part2 = ts_rank(
            ts_corr(ts_rank(self.vwap, 20), ts_rank(self._adv(60), 4), 18), 3
        )
        return -(part1 ** part2)

    def alpha095(self) -> pd.DataFrame:
        """rank(open − ts_min(open, 12)) < Ts_Rank(rank(corr(sum((H+L)/2, 19), sum(adv40, 19), 13))^5, 12)"""
        left = rank(self.open - ts_min(self.open, 12))
        inner = ts_corr(
            ts_sum((self.high + self.low) / 2, 19),
            ts_sum(self._adv(40), 19),
            13,
        )
        right = ts_rank(rank(inner) ** 5, 12)
        return (left < right).astype(float)

    def alpha096(self) -> pd.DataFrame:
        """−max(Ts_Rank(decay corr(rank(vwap), rank(vol)), Ts_Rank(decay Ts_ArgMax(corr ts_rank)))"""
        part1 = ts_rank(
            decay_linear(
                ts_corr(rank(self.vwap), rank(self.volume), 4), 4
            ),
            8,
        )
        inner = ts_argmax(
            ts_corr(ts_rank(self.close, 7), ts_rank(self._adv(60), 4), 4), 13
        )
        part2 = ts_rank(decay_linear(inner, 14), 13)
        return -1 * pd.DataFrame(
            np.maximum(part1, part2),
            index=self.close.index,
            columns=self.close.columns,
        )

    def alpha097(self) -> pd.DataFrame:
        """IndNeutralize(low×0.72+vwap×0.28) 衰减排名复合"""
        inner = self.low * 0.721001 + self.vwap * (1 - 0.721001)
        inner_n = self._ind_neutralize(inner)
        part1 = rank(decay_linear(delta(inner_n, 3), 20))
        inner2 = ts_corr(
            ts_rank(self.low, 8), ts_rank(self._adv(60), 17), 5
        )
        part2 = ts_rank(decay_linear(ts_rank(inner2, 19), 16), 7)
        return (part1 - part2) * -1

    def alpha098(self) -> pd.DataFrame:
        """rank(decay corr(vwap, sum(adv5, 26), 5)) − rank(decay Ts_Rank(Ts_ArgMin(corr(rank(open), rank(adv15)))))"""
        part1 = rank(
            decay_linear(
                ts_corr(self.vwap, ts_sum(self._adv(5), 26), 5), 7
            )
        )
        inner = ts_argmin(
            ts_corr(rank(self.open), rank(self._adv(15)), 21), 9
        )
        part2 = rank(decay_linear(ts_rank(inner, 7), 8))
        return part1 - part2

    def alpha099(self) -> pd.DataFrame:
        """(rank(corr(sum((H+L)/2, 20), sum(adv60, 20), 9)) < rank(corr(low, volume, 6))) × −1"""
        left = rank(
            ts_corr(
                ts_sum((self.high + self.low) / 2, 20),
                ts_sum(self._adv(60), 20),
                9,
            )
        )
        right = rank(ts_corr(self.low, self.volume, 6))
        return (left < right).astype(float) * -1

    def alpha100(self) -> pd.DataFrame:
        """双重 IndNeutralize 量价排名因子"""
        inner = (
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low + 1e-10)
            * self.volume
        )
        inner_ranked = rank(inner)
        if self.industry is not None:
            inner_ranked = ind_neutralize(
                ind_neutralize(inner_ranked, self.industry), self.industry
            )
        part1 = 1.5 * scale(inner_ranked)
        inner2 = ts_corr(self.close, rank(self._adv(20)), 5) - rank(
            ts_argmin(self.close, 30)
        )
        if self.industry is not None:
            inner2 = ind_neutralize(inner2, self.industry)
        part2 = scale(inner2)
        return -1 * (part1 - part2) * (self.volume / (self._adv(20) + 1e-10))

    def alpha101(self) -> pd.DataFrame:
        """(close − open) / ((high − low) + 0.001)"""
        return (self.close - self.open) / (self.high - self.low + 0.001)
