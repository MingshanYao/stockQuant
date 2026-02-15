"""
因子分析模块 — IC / IR / 分组回测 / 因子合成。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from stockquant.utils.logger import get_logger

logger = get_logger("analysis.factor")


class FactorAnalyzer:
    """因子分析工具集。"""

    # ------------------------------------------------------------------
    # IC / IR 检验
    # ------------------------------------------------------------------

    @staticmethod
    def calc_ic(
        factor: pd.Series,
        forward_returns: pd.Series,
        method: str = "spearman",
    ) -> float:
        """计算信息系数 (IC)。

        Parameters
        ----------
        factor : Series
            截面因子值。
        forward_returns : Series
            未来收益率（对齐索引）。
        method : str
            ``spearman``（Rank IC）或 ``pearson``。
        """
        valid = pd.concat([factor, forward_returns], axis=1).dropna()
        if len(valid) < 5:
            return 0.0
        return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))

    @staticmethod
    def calc_ic_series(
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: str = "spearman",
    ) -> pd.Series:
        """计算时间序列 IC。

        Parameters
        ----------
        factor_df : DataFrame
            行=日期，列=股票代码，值=因子值。
        returns_df : DataFrame
            同结构，值=未来收益率。
        """
        ic_list = []
        for date in factor_df.index:
            if date in returns_df.index:
                f = factor_df.loc[date].dropna()
                r = returns_df.loc[date].dropna()
                common = f.index.intersection(r.index)
                if len(common) >= 5:
                    ic = float(f[common].corr(r[common], method=method))
                    ic_list.append({"date": date, "ic": ic})
        return pd.DataFrame(ic_list).set_index("date")["ic"] if ic_list else pd.Series(dtype=float)

    @staticmethod
    def calc_ir(ic_series: pd.Series) -> float:
        """信息比率 IR = mean(IC) / std(IC)。"""
        if len(ic_series) == 0 or ic_series.std() == 0:
            return 0.0
        return float(ic_series.mean() / ic_series.std())

    # ------------------------------------------------------------------
    # 分组回测
    # ------------------------------------------------------------------

    @staticmethod
    def quantile_returns(
        factor: pd.Series,
        returns: pd.Series,
        n_groups: int = 5,
    ) -> pd.Series:
        """按因子分组并计算各组平均收益。"""
        valid = pd.concat(
            [factor.rename("factor"), returns.rename("returns")], axis=1
        ).dropna()

        if len(valid) < n_groups:
            return pd.Series(dtype=float)

        valid["group"] = pd.qcut(valid["factor"], q=n_groups, labels=False, duplicates="drop")
        return valid.groupby("group")["returns"].mean()

    # ------------------------------------------------------------------
    # 因子中性化
    # ------------------------------------------------------------------

    @staticmethod
    def neutralize(
        factor: pd.Series,
        market_cap: pd.Series | None = None,
        industry: pd.Series | None = None,
    ) -> pd.Series:
        """市值与行业中性化（线性回归残差法）。"""
        data = {"factor": factor}
        if market_cap is not None:
            data["ln_cap"] = np.log(market_cap)
        if industry is not None:
            dummies = pd.get_dummies(industry, prefix="ind", drop_first=True)
            for col in dummies.columns:
                data[col] = dummies[col]

        df = pd.DataFrame(data).dropna()
        if len(df) < 10:
            return factor

        y = df["factor"]
        X = df.drop(columns=["factor"])
        X["const"] = 1.0

        try:
            beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
            residuals = y.values - X.values @ beta
            return pd.Series(residuals, index=df.index, name="factor_neutral")
        except Exception:
            return factor

    # ------------------------------------------------------------------
    # 因子合成
    # ------------------------------------------------------------------

    @staticmethod
    def combine_factors(
        factors: dict[str, pd.Series],
        method: str = "equal",
        ic_values: dict[str, float] | None = None,
    ) -> pd.Series:
        """多因子合成。

        Parameters
        ----------
        factors : dict
            因子名 → 因子值 Series。
        method : str
            ``equal``（等权）/ ``icir``（IC/IR 加权）。
        ic_values : dict, optional
            各因子的 IC 均值（icir 加权时使用）。
        """
        df = pd.DataFrame(factors)
        # 标准化
        df = (df - df.mean()) / df.std()

        if method == "equal":
            return df.mean(axis=1)
        elif method == "icir" and ic_values:
            weights = pd.Series(ic_values)
            weights = weights / weights.abs().sum()
            return df.mul(weights, axis=1).sum(axis=1)
        else:
            return df.mean(axis=1)
