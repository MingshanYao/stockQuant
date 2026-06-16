"""
因子评价器 — 单因子四指标检验 + 多周期分析 + 因子体系评价。

借鉴国泰君安 Alpha191 报告的评价方法论：
  风格正交化 → IC / Factor Return / IR / T统计量 → 多周期扫描 → 体系相关性
"""

from __future__ import annotations

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from stockquant.analysis.factor import FactorAnalyzer
from stockquant.utils.logger import get_logger

logger = get_logger("analysis.evaluator")

TRADING_DAYS_PER_YEAR = 252

# 模块级全局用于 multiprocessing fork（避免 pickle evaluator）
_evaluator: "FactorEvaluator | None" = None


def _eval_one(item: tuple):
    """模块级函数，用于 ProcessPoolExecutor fork。"""
    name, panel, forward_period, kwargs = item
    return name, _evaluator.evaluate(panel, forward_period=forward_period, **kwargs)


def _neutralize_single_date(
    y: pd.Series,
    industry: pd.Series | None,
    market_cap: pd.Series | None,
    style_factors: dict[str, pd.Series] | None,
) -> pd.Series:
    """对单日截面做多元回归，取残差作为纯Alpha。

    回归模型：
        y = β_0 + β_ind·X_ind + β_size·X_size + β_beta·X_beta
              + β_vol·X_vol + β_mom·X_mom + ε
    """
    codes = y.dropna().index
    if len(codes) < 20:
        return y  # 截面太小不做中性化

    # 构建设计矩阵
    design_cols = []
    design = []

    # 行业哑变量
    if industry is not None:
        ind_series = industry.reindex(codes).dropna()
        if len(ind_series) > 0:
            ind_dummies = pd.get_dummies(ind_series, prefix="ind", drop_first=True)
            design.append(ind_dummies)

    # 市值（取对数后 Z-score）
    if market_cap is not None:
        if isinstance(market_cap, pd.Series):
            # code → cap 映射
            mc = market_cap.reindex(codes).dropna()
        else:
            mc = market_cap.dropna()
        if len(mc) > 0:
            log_mc = np.log(mc.replace(0, np.nan)).dropna()
            mc_z = (log_mc - log_mc.mean()) / log_mc.std()
            design.append(mc_z.to_frame("size"))

    # 其他风格因子（Beta / Volatility / Momentum）
    if style_factors is not None:
        for name, sf in style_factors.items():
            sf_reindexed = sf.reindex(codes).dropna()
            if len(sf_reindexed) > 0:
                # Z-score
                sf_z = (sf_reindexed - sf_reindexed.mean()) / sf_reindexed.std()
                design.append(sf_z.to_frame(name))

    if len(design) < 1:
        return y

    X = pd.concat(design, axis=1).dropna()
    y_aligned = y.reindex(X.index).dropna()
    common = y_aligned.index.intersection(X.index)
    if len(common) < 20:
        return y

    X = X.loc[common]
    y_vals = y_aligned.loc[common].values

    # 加截距列
    X_aug = np.column_stack([np.ones(len(X)), X.values])

    # OLS：β = (X'X)^{-1} X'y
    try:
        beta = np.linalg.lstsq(X_aug, y_vals, rcond=None)[0]
        residuals = y_vals - X_aug @ beta
    except Exception:
        return y

    result = pd.Series(np.nan, index=y.index)
    result.loc[common] = residuals
    return result


class FactorEvaluator:
    """因子评价器。

    Parameters
    ----------
    close_panel : DataFrame
        收盘价面板（行=日期，列=股票代码），用于计算前向收益。
    industry : Series, optional
        股票代码 → 行业分类的映射，用于风格正交化。
    market_cap : Series or DataFrame, optional
        股票代码 → 市值的映射（Series），或每日市值面板（DataFrame）。
    style_factors : dict, optional
        额外的风格因子，{name: DataFrame(行=日期, 列=股票代码)}。
        自动计算 beta, volatility, momentum（若未提供且 close_panel 可用）。
    benchmark_returns : Series, optional
        基准日收益率序列，用于计算滚动 Beta。
    """

    def __init__(
        self,
        close_panel: pd.DataFrame,
        industry: pd.Series | None = None,
        market_cap: pd.Series | pd.DataFrame | None = None,
        style_factors: dict[str, pd.DataFrame] | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> None:
        self.close = close_panel
        self.industry = industry
        self.market_cap = market_cap
        self._style_factors = style_factors or {}
        self.benchmark_returns = benchmark_returns

        # 自动计算未提供的风格因子
        self._compute_missing_style_factors()

    def _compute_missing_style_factors(self) -> None:
        """自动计算缺失的风格因子。"""
        # Beta
        if "beta" not in self._style_factors and self.benchmark_returns is not None:
            betas = {}
            for code in self.close.columns:
                stock_ret = self.close[code].pct_change().dropna()
                bench = self.benchmark_returns.reindex(stock_ret.index).dropna()
                common_idx = stock_ret.index.intersection(bench.index)
                if len(common_idx) < 30:
                    continue
                s = stock_ret.loc[common_idx]
                b = bench.loc[common_idx]
                cov = np.cov(s.values, b.values)
                if cov[1, 1] > 1e-12:
                    betas[code] = cov[0, 1] / cov[1, 1]
            if betas:
                self._style_factors["beta"] = pd.DataFrame(
                    {code: [v] for code, v in betas.items()},
                    index=[self.close.index[0]],
                )

        # Volatility（20日滚动标准差）
        if "volatility" not in self._style_factors:
            returns = self.close.pct_change()
            vol_df = returns.rolling(20).std() * np.sqrt(252)
            vol_df = vol_df.bfill(axis=0)
            self._style_factors["volatility"] = vol_df

        # Momentum（过去20日收益率）
        if "momentum" not in self._style_factors:
            mom_df = self.close.pct_change(20)
            mom_df = mom_df.bfill(axis=0)
            self._style_factors["momentum"] = mom_df

    def _get_date_style_factors(self, date) -> dict[str, pd.Series]:
        """获取某个日期的所有风格因子截面。"""
        result = {}
        for name, panel in self._style_factors.items():
            if date in panel.index:
                result[name] = panel.loc[date]
        return result

    def _get_date_market_cap(self, date) -> pd.Series | None:
        """获取某个日期的市值截面。"""
        if self.market_cap is None:
            return None
        if isinstance(self.market_cap, pd.Series):
            return self.market_cap  # 静态映射，所有日期相同
        if date in self.market_cap.index:
            return self.market_cap.loc[date]
        return None

    def _forward_returns(self, period: int) -> pd.DataFrame:
        return self.close.pct_change(periods=period).shift(-period)

    def _neutralize_panel(self, factor: pd.DataFrame) -> pd.DataFrame:
        """对每日截面做风格中性化，取残差作为纯Alpha因子值。"""
        result = factor.copy()
        for date in factor.index:
            row = factor.loc[date].dropna()
            if len(row) < 20:
                continue
            codes = row.index

            # 取该日期的行业归属
            ind = self.industry.reindex(codes) if self.industry is not None else None
            # 取该日期市值
            mc = self._get_date_market_cap(date)
            mc_s = mc.reindex(codes) if mc is not None else None
            # 取该日期风格因子
            style = self._get_date_style_factors(date)
            style_s = {name: sf.reindex(codes) for name, sf in style.items()}

            neutralized = _neutralize_single_date(row, ind, mc_s, style_s)
            result.loc[date, neutralized.index] = neutralized.values
        return result

    def evaluate(
        self,
        factor: pd.DataFrame,
        forward_period: int = 1,
        method: str = "spearman",
    ) -> dict[str, float]:
        """单因子完整评价 — IC / Factor Return / IR / T统计量。

        始终做风格中性化（alpha191 方法论要求）。

        Returns
        -------
        dict
            ic_mean, ic_std, ic_ir, ic_pos_ratio,
            fr_mean, fr_std, fr_ir, fr_annual, t_stat, n_periods
        """
        fwd = self._forward_returns(forward_period)

        # 1. 风格中性化因子值 → 纯Alpha预测值 E{ε}
        factor_neutral = self._neutralize_panel(factor)

        # 2. 风格中性化前向收益 → 实际Alpha收益 α
        fwd_neutral = self._neutralize_panel(fwd)

        common_idx = factor_neutral.index.intersection(fwd_neutral.index)
        common_cols = factor_neutral.columns.intersection(fwd_neutral.columns)
        f = factor_neutral.loc[common_idx, common_cols]
        r = fwd_neutral.loc[common_idx, common_cols]

        # 3. IC = corr(预测Alpha, 实际Alpha)
        ic_series = f.corrwith(r, axis=1, method=method).dropna()

        # 4. Factor Return
        fr_list = []
        for date in common_idx:
            f_row = f.loc[date].dropna()
            r_row = r.loc[date].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) >= 10:
                fr = FactorAnalyzer.calc_factor_return(f_row[common], r_row[common])
                fr_list.append(fr)
        fr_series = pd.Series(fr_list)

        ic_mean = ic_series.mean() if len(ic_series) else np.nan
        ic_std = ic_series.std() if len(ic_series) else np.nan
        ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        ic_pos = (ic_series > 0).mean() if len(ic_series) else np.nan

        fr_mean = fr_series.mean() if len(fr_series) else np.nan
        fr_std = fr_series.std() if len(fr_series) else np.nan
        fr_ir = fr_mean / fr_std if fr_std and fr_std > 0 else np.nan

        d = forward_period
        n = len(fr_series)
        fr_annual = fr_mean * (TRADING_DAYS_PER_YEAR / d) if not np.isnan(fr_mean) else np.nan
        t_stat = (
            ic_mean / (ic_std / np.sqrt(n))
            if n > 0 and ic_std and ic_std > 0
            else np.nan
        )

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_pos_ratio": ic_pos,
            "fr_mean": fr_mean,
            "fr_std": fr_std,
            "fr_ir": fr_ir,
            "fr_annual": fr_annual,
            "t_stat": t_stat,
            "n_periods": n,
        }

    def evaluate_multi_horizon(
        self,
        factor: pd.DataFrame,
        periods: list[int] | None = None,
        **kwargs,
    ) -> dict[int, dict[str, float]]:
        """多周期因子评价 — 扫描不同预测周期的因子表现。"""
        if periods is None:
            periods = [1, 2, 3, 4, 5]

        results = {}
        for d in periods:
            results[d] = self.evaluate(factor, forward_period=d, **kwargs)
        return results

    def evaluate_system(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """因子体系批量评价 — 返回所有因子的四指标汇总表。"""
        global _evaluator
        _evaluator = self

        items = [(name, panel, forward_period, kwargs) for name, panel in factors.items()]
        rows: list[dict] = []
        max_workers = min(max(os.cpu_count() - 2, 1), len(items))

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            total = len(items)
            futures = {executor.submit(_eval_one, item): item[0] for item in items}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    name, metrics = future.result()
                    metrics["factor"] = name
                    rows.append(metrics)
                except Exception as e:
                    name = futures[future]
                    logger.warning(f"[{i}/{total}] 评价失败 {name}: {e}")

        df = pd.DataFrame(rows).set_index("factor")
        return df.reindex(df["ic_mean"].abs().sort_values(ascending=False).index)

    def factor_correlation_matrix(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
    ) -> pd.DataFrame:
        """因子收益率相关性矩阵 — 用于去冗余。"""
        fwd = self._forward_returns(forward_period)
        fwd_neutral = self._neutralize_panel(fwd)
        fr_dict: dict[str, list[float]] = {}

        for name, panel in factors.items():
            panel_neutral = self._neutralize_panel(panel)
            common_idx = panel_neutral.index.intersection(fwd_neutral.index)
            common_cols = panel_neutral.columns.intersection(fwd_neutral.columns)
            f = panel_neutral.loc[common_idx, common_cols]
            r = fwd_neutral.loc[common_idx, common_cols]

            fr_list = []
            for date in common_idx:
                f_row = f.loc[date].dropna()
                r_row = r.loc[date].dropna()
                common = f_row.index.intersection(r_row.index)
                if len(common) >= 10:
                    fr = FactorAnalyzer.calc_factor_return(f_row[common], r_row[common])
                    fr_list.append(fr)
                else:
                    fr_list.append(np.nan)
            fr_dict[name] = fr_list

        fr_df = pd.DataFrame(fr_dict)
        return fr_df.corr()
