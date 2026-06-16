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

    @staticmethod
    def calc_factor_return(
        factor: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """截面回归因子收益率。

        对因子值标准化后，用 OLS 回归未来收益，回归系数即为因子收益率。
        """
        valid = pd.concat(
            [factor.rename("f"), forward_returns.rename("r")], axis=1
        ).dropna()
        if len(valid) < 10:
            return 0.0

        f = valid["f"].values
        r = valid["r"].values

        f_std = f.std()
        if f_std == 0:
            return 0.0
        f_norm = (f - f.mean()) / f_std

        X = np.column_stack([f_norm, np.ones(len(f_norm))])
        beta = np.linalg.lstsq(X, r, rcond=None)[0]
        return float(beta[0])

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

    # ------------------------------------------------------------------
    # 滚动因子收益率合成
    # ------------------------------------------------------------------

    @staticmethod
    def calc_rolling_factor_returns(
        factor_panels: dict[str, pd.DataFrame],
        returns_panel: pd.DataFrame,
        window: int = 250,
        forward_period: int = 1,
    ) -> dict[str, pd.Series]:
        """计算各因子的滚动因子收益率序列。

        对每个交易日，用截面 OLS 回归计算因子收益率：
            r_fwd = β_f · f_norm + α + ε
        其中 r_fwd 是前向收益，f_norm 是标准化因子值。

        Parameters
        ----------
        factor_panels : dict
            因子名 → 因子面板（行=日期，列=股票代码）。
        returns_panel : DataFrame
            前向收益面板（同结构）。
        window : int
            滚动窗口长度（交易日数）。
        forward_period : int
            预测周期。

        Returns
        -------
        dict[str, Series]
            因子名 → 滚动因子收益率序列。
        """
        fwd = returns_panel.shift(-forward_period)
        result: dict[str, list] = {name: [] for name in factor_panels}
        index: list = []

        for date in returns_panel.index:
            if date not in fwd.index:
                continue
            r_row = fwd.loc[date].dropna()
            if len(r_row) < 20:
                continue

            date_results = {}
            for name, panel in factor_panels.items():
                if date not in panel.index:
                    break
                f_row = panel.loc[date].dropna()
                common = f_row.index.intersection(r_row.index)
                if len(common) >= 20:
                    fr = FactorAnalyzer.calc_factor_return(f_row[common], r_row[common])
                    date_results[name] = fr
                else:
                    break
            else:
                index.append(date)
                for name in factor_panels:
                    result[name].append(date_results.get(name, np.nan))

        return {name: pd.Series(vals, index=pd.DatetimeIndex(index))
                for name, vals in result.items()}

    @staticmethod
    def calc_rolling_fr_means(
        fr_series_dict: dict[str, pd.Series],
        current_date,
        window: int = 60,
    ) -> dict[str, float]:
        """计算当前日期前 window 日的滚动因子收益率均值。

        Parameters
        ----------
        fr_series_dict : dict
            因子名 → 滚动因子收益率序列（来自 calc_rolling_factor_returns）。
        current_date : Timestamp
            当前日期。
        window : int
            回看窗口长度。

        Returns
        -------
        dict[str, float]
            因子名 → 滚动均值。
        """
        result = {}
        for name, series in fr_series_dict.items():
            past = series[series.index <= current_date].tail(window).dropna()
            if len(past) > 0:
                result[name] = float(past.mean())
            else:
                result[name] = 0.0
        return result

    @staticmethod
    def combine_by_rolling_fr(
        factors: dict[str, pd.Series],
        fr_means: dict[str, float],
    ) -> pd.Series:
        """按滚动因子收益率加权合成多因子。

        w_k = FR_k / Σ|FR_j|，权重与近期表现成正比。

        Parameters
        ----------
        factors : dict
            因子名 → 截面因子值 Series。
        fr_means : dict
            因子名 → 滚动因子收益率均值。

        Returns
        -------
        Series
            合成的 Alpha 预测值。
        """
        if not factors or not fr_means:
            return pd.Series(dtype=float)

        df = pd.DataFrame(factors)
        df = (df - df.mean()) / (df.std() + 1e-10)

        weights = pd.Series(fr_means)
        total_abs = weights.abs().sum()
        if total_abs == 0:
            return df.mean(axis=1)
        weights = weights / total_abs

        return df.mul(weights, axis=1).sum(axis=1)
