"""
数据清洗与预处理。
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from stockquant.utils.logger import get_logger

logger = get_logger("data.cleaner")


class DataCleaner:
    """股票数据清洗工具集。"""

    # ------------------------------------------------------------------
    # 缺失值处理
    # ------------------------------------------------------------------

    @staticmethod
    def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """填充缺失值。

        Parameters
        ----------
        df : DataFrame
            原始数据。
        method : str
            填充方式: ``ffill`` 前向 / ``bfill`` 后向 / ``interpolate`` 线性插值。
        """
        if df.empty:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        elif method == "ffill":
            df[numeric_cols] = df[numeric_cols].ffill()
        elif method == "bfill":
            df[numeric_cols] = df[numeric_cols].bfill()

        # 仍有缺失则用 bfill 补充
        df[numeric_cols] = df[numeric_cols].bfill()

        miss_count = df[numeric_cols].isna().sum().sum()
        if miss_count > 0:
            logger.warning(f"仍有 {miss_count} 个缺失值未填充")

        return df

    # ------------------------------------------------------------------
    # 异常值检测
    # ------------------------------------------------------------------

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str = "pct_change",
        threshold: float = 0.11,
    ) -> pd.DataFrame:
        """检测超出涨跌停幅度的异常值行。

        Parameters
        ----------
        column : str
            检测列名。
        threshold : float
            异常阈值（默认 11% ，略宽于 10% 涨跌停）。
        """
        if column not in df.columns:
            return pd.DataFrame()

        mask = df[column].abs() > threshold * 100 if df[column].max() > 1 else df[column].abs() > threshold
        outliers = df[mask]
        if not outliers.empty:
            logger.info(f"检测到 {len(outliers)} 条涨跌幅异常记录")
        return outliers

    # ------------------------------------------------------------------
    # 数据标准化
    # ------------------------------------------------------------------

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """确保 DataFrame 包含标准列名与数据类型。"""
        standard_cols = {
            "date": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
            "amount": "float64",
        }
        df = df.copy()

        for col, dtype in standard_cols.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    logger.warning(f"列 {col} 类型转换失败")

        return df

    # ------------------------------------------------------------------
    # 停牌日过滤
    # ------------------------------------------------------------------

    @staticmethod
    def remove_suspended(df: pd.DataFrame) -> pd.DataFrame:
        """移除成交量为 0 的停牌日数据。"""
        if "volume" not in df.columns:
            return df
        before = len(df)
        df = df[df["volume"] > 0].copy()
        removed = before - len(df)
        if removed > 0:
            logger.debug(f"移除 {removed} 条停牌日记录")
        return df

    # ------------------------------------------------------------------
    # 数据质量过滤
    # ------------------------------------------------------------------

    @staticmethod
    def drop_invalid_stocks(df: pd.DataFrame) -> pd.DataFrame:
        """剔除含异常数据的股票（以 code 为粒度，整支股票删除）。

        异常规则:
        - 任意价格列 (open / high / low / close) <= 0 或 NaN
        - volume < 0 或 NaN
        - amount < 0 或 NaN

        若 DataFrame 中不含 ``code`` 列，则直接删除异常行。

        Returns
        -------
        DataFrame
            已剔除异常股票后的干净数据。
        """
        if df.empty:
            return df

        df = df.copy()

        # 构建各规则的异常 mask
        invalid_mask = pd.Series(False, index=df.index)

        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            bad = s.le(0) | s.isna()
            if bad.any():
                invalid_mask |= bad

        if "volume" in df.columns:
            s = pd.to_numeric(df["volume"], errors="coerce")
            bad = s.lt(0) | s.isna()
            if bad.any():
                invalid_mask |= bad

        if "amount" in df.columns:
            s = pd.to_numeric(df["amount"], errors="coerce")
            bad = s.lt(0) | s.isna()
            if bad.any():
                invalid_mask |= bad

        if not invalid_mask.any():
            return df

        # 有 code 列时以股票为粒度整体剔除
        if "code" in df.columns:
            bad_codes = df.loc[invalid_mask, "code"].unique().tolist()
            # 逐只记录异常原因
            for code in bad_codes:
                code_mask = df["code"] == code
                reasons: list[str] = []
                for col in price_cols:
                    s = pd.to_numeric(df.loc[code_mask, col], errors="coerce")
                    n_le0 = int(s.le(0).sum())
                    n_nan = int(s.isna().sum())
                    if n_le0:
                        reasons.append(f"{col}<=0 ({n_le0}行)")
                    if n_nan:
                        reasons.append(f"{col}=NaN ({n_nan}行)")
                if "volume" in df.columns:
                    s = pd.to_numeric(df.loc[code_mask, "volume"], errors="coerce")
                    n_lt0, n_nan = int(s.lt(0).sum()), int(s.isna().sum())
                    if n_lt0:
                        reasons.append(f"volume<0 ({n_lt0}行)")
                    if n_nan:
                        reasons.append(f"volume=NaN ({n_nan}行)")
                if "amount" in df.columns:
                    s = pd.to_numeric(df.loc[code_mask, "amount"], errors="coerce")
                    n_lt0, n_nan = int(s.lt(0).sum()), int(s.isna().sum())
                    if n_lt0:
                        reasons.append(f"amount<0 ({n_lt0}行)")
                    if n_nan:
                        reasons.append(f"amount=NaN ({n_nan}行)")
                logger.warning(
                    f"[{code}] 数据异常，整支股票已剔除: {', '.join(reasons)}"
                )
            df = df[~df["code"].isin(bad_codes)].copy()
            logger.info(f"共剔除 {len(bad_codes)} 只异常股票: {bad_codes[:20]}{'...' if len(bad_codes) > 20 else ''}")
        else:
            # 无 code 列时直接删行
            n = int(invalid_mask.sum())
            logger.warning(f"无 code 列，直接剔除 {n} 条异常行")
            df = df[~invalid_mask].copy()

        return df

    # ------------------------------------------------------------------
    # 管道处理
    # ------------------------------------------------------------------

    @classmethod
    def clean_pipeline(
        cls,
        df: pd.DataFrame,
        fill_method: str = "ffill",
        remove_suspended: bool = True,
    ) -> pd.DataFrame:
        """执行标准清洗管道。"""
        df = cls.drop_invalid_stocks(df)   # 先剔除异常股票，再做后续处理
        df = cls.standardize_columns(df)
        df = cls.fill_missing(df, method=fill_method)
        if remove_suspended:
            df = cls.remove_suspended(df)
        return df.reset_index(drop=True)
