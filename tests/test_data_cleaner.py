"""测试 - 数据清洗模块。"""

import numpy as np
import pandas as pd
import pytest

from stockquant.data.data_cleaner import DataCleaner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "open": [10.0, 10.1, np.nan, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
        "high": [10.5] * 10,
        "low": [9.5] * 10,
        "close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
        "volume": [1000, 1100, 0, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        "amount": [10000.0] * 10,
    })


class TestDataCleaner:
    def test_fill_missing(self, sample_df):
        result = DataCleaner.fill_missing(sample_df)
        assert result["open"].isna().sum() == 0

    def test_remove_suspended(self, sample_df):
        result = DataCleaner.remove_suspended(sample_df)
        assert len(result) == 9  # volume=0 的那行被移除

    def test_clean_pipeline(self, sample_df):
        result = DataCleaner.clean_pipeline(sample_df)
        assert result["open"].isna().sum() == 0
        assert (result["volume"] > 0).all()

    def test_detect_outliers(self, sample_df):
        sample_df["pct_change"] = [0, 0.01, 0.02, 0.15, 0.01, -0.12, 0.01, 0.01, 0.01, 0.01]
        outliers = DataCleaner.detect_outliers(sample_df)
        assert len(outliers) == 2  # 0.15 和 -0.12 超出阈值
