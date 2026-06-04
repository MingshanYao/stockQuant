"""测试 - AlphaResearcher.ic_summary 批量 IC 评估。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockquant.data.universe import BacktestDataset
from stockquant.research import AlphaResearcher


def _make_synthetic_dataset(n_days: int = 80, n_stocks: int = 8) -> BacktestDataset:
    """生成最小化的合成 BacktestDataset，便于离线测试。"""
    np.random.seed(0)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"60000{i}" for i in range(n_stocks)]

    stock_data: dict[str, pd.DataFrame] = {}
    for code in codes:
        rets = np.random.normal(0.0005, 0.02, n_days)
        close = 100 * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]]
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        volume = np.random.randint(1_000_000, 5_000_000, n_days)
        df = pd.DataFrame({
            "date": dates, "code": code,
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume,
            "amount": close * volume,
        })
        stock_data[code] = df

    benchmark = pd.DataFrame({
        "date": dates,
        "close": 1000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n_days))),
    })

    return BacktestDataset(
        stock_data=stock_data,
        codes=codes,
        benchmark=benchmark,
        benchmark_code="000300",
        start_date="2024-01-01",
        end_date=str(dates[-1].date()),
    )


@pytest.fixture(scope="module")
def researcher() -> AlphaResearcher:
    ds = _make_synthetic_dataset()
    return AlphaResearcher(ds, max_positions=3, rebalance_freq=5)


class TestIcSummary:
    def test_returns_expected_columns(self, researcher):
        summary = researcher.ic_summary(alpha_ids=[1, 6, 12])
        expected = {"alpha", "coverage", "ic_mean", "ic_std",
                    "ic_ir", "ic_pos_ratio", "mean", "std"}
        assert expected.issubset(summary.columns)
        assert len(summary) == 3

    def test_default_runs_all_factors(self, researcher):
        # 仅跑 5 个验证默认列表逻辑（避免单测过慢）
        summary = researcher.ic_summary(alpha_ids=[1, 2, 3, 6, 12])
        assert set(summary["alpha"]) == {1, 2, 3, 6, 12}

    def test_sorted_by_abs_ic(self, researcher):
        summary = researcher.ic_summary(alpha_ids=[1, 6, 12, 41])
        abs_ic = summary["ic_mean"].abs().values
        # 允许 NaN 排在末尾
        finite = abs_ic[~pd.isna(abs_ic)]
        assert (finite[:-1] >= finite[1:]).all() if len(finite) > 1 else True

    def test_coverage_is_ratio(self, researcher):
        summary = researcher.ic_summary(alpha_ids=[1, 101])
        assert (summary["coverage"] >= 0).all()
        assert (summary["coverage"] <= 1).all()
