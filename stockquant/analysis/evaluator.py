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


class FactorEvaluator:
    """因子评价器。

    Parameters
    ----------
    close_panel : DataFrame
        收盘价面板（行=日期，列=股票代码），用于计算前向收益。
    industry : Series, optional
        股票代码 → 行业分类的映射，用于风格正交化。
    market_cap : Series, optional
        股票代码 → 市值的映射，用于风格正交化。
    """

    def __init__(
        self,
        close_panel: pd.DataFrame,
        industry: pd.Series | None = None,
        market_cap: pd.Series | None = None,
    ) -> None:
        self.close = close_panel
        self.industry = industry
        self.market_cap = market_cap

    def _forward_returns(self, period: int) -> pd.DataFrame:
        return self.close.pct_change(periods=period).shift(-period)

    def _neutralize_panel(self, factor: pd.DataFrame) -> pd.DataFrame:
        if self.industry is None and self.market_cap is None:
            return factor

        result = factor.copy()
        for date in factor.index:
            row = factor.loc[date].dropna()
            if len(row) < 10:
                continue
            codes = row.index
            ind = self.industry.reindex(codes) if self.industry is not None else None
            cap = self.market_cap.reindex(codes) if self.market_cap is not None else None
            neutralized = FactorAnalyzer.neutralize(row, market_cap=cap, industry=ind)
            result.loc[date, neutralized.index] = neutralized.values
        return result

    def evaluate(
        self,
        factor: pd.DataFrame,
        forward_period: int = 1,
        method: str = "spearman",
        neutralize: bool = True,
    ) -> dict[str, float]:
        """单因子完整评价 — IC / Factor Return / IR / T统计量。

        Returns
        -------
        dict
            ic_mean, ic_std, ic_ir, ic_pos_ratio,
            fr_mean, fr_std, fr_ir, fr_annual, t_stat, n_periods
        """
        fwd = self._forward_returns(forward_period)

        if neutralize and (self.industry is not None or self.market_cap is not None):
            factor = self._neutralize_panel(factor)

        common_idx = factor.index.intersection(fwd.index)
        common_cols = factor.columns.intersection(fwd.columns)
        f = factor.loc[common_idx, common_cols]
        r = fwd.loc[common_idx, common_cols]

        ic_series = f.corrwith(r, axis=1, method=method).dropna()

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
        fr_dict: dict[str, list[float]] = {}

        for name, panel in factors.items():
            common_idx = panel.index.intersection(fwd.index)
            common_cols = panel.columns.intersection(fwd.columns)
            f = panel.loc[common_idx, common_cols]
            r = fwd.loc[common_idx, common_cols]

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
