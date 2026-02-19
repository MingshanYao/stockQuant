"""
Alpha101 因子库单元测试。

测试策略：
- BaseIndicator 接口合规性（name、compute、isinstance）
- 使用小规模合成面板数据验证因子可计算性
- 验证输出形状与输入一致
- 验证代表性因子的数值合理性
- 验证工厂方法（from_stacked_df / panel）
- 验证单股票 compute() 模式
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockquant.indicators import Alpha101Indicators, BaseIndicator
from stockquant.indicators.alpha101 import Alpha101Engine, INDUSTRY_ALPHAS
from stockquant.indicators.alpha101.operators import (
    rank,
    scale,
    delay,
    delta,
    ts_min,
    ts_max,
    ts_argmin,
    ts_argmax,
    ts_rank,
    ts_sum,
    ts_stddev,
    ts_corr,
    ts_cov,
    sma,
    decay_linear,
    sign,
    signedpower,
    log,
    adv,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def panel_data():
    """生成小规模合成面板数据 (60 天 × 5 只股票)。"""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=60)
    codes = ["A", "B", "C", "D", "E"]

    close = pd.DataFrame(
        np.random.lognormal(mean=3, sigma=0.1, size=(60, 5)).cumsum(axis=0),
        index=dates,
        columns=codes,
    )
    open_ = close * np.random.uniform(0.98, 1.02, size=(60, 5))
    high = close * np.random.uniform(1.0, 1.05, size=(60, 5))
    low = close * np.random.uniform(0.95, 1.0, size=(60, 5))
    volume = pd.DataFrame(
        np.random.randint(1000, 100000, size=(60, 5)),
        index=dates,
        columns=codes,
        dtype=float,
    )
    return open_, high, low, close, volume


@pytest.fixture
def engine(panel_data):
    """Alpha101Engine 面板引擎实例。"""
    open_, high, low, close, volume = panel_data
    return Alpha101Indicators.panel(
        open_=open_, high=high, low=low, close=close, volume=volume,
    )


@pytest.fixture
def single_stock_df(panel_data):
    """单只股票 OHLCV DataFrame。"""
    open_, high, low, close, volume = panel_data
    return pd.DataFrame({
        "open": open_["A"],
        "high": high["A"],
        "low": low["A"],
        "close": close["A"],
        "volume": volume["A"],
    })


@pytest.fixture
def stacked_df(panel_data):
    """堆叠格式 DataFrame。"""
    open_, high, low, close, volume = panel_data
    records = []
    for date in close.index:
        for code in close.columns:
            records.append({
                "date": date,
                "code": code,
                "open": open_.loc[date, code],
                "high": high.loc[date, code],
                "low": low.loc[date, code],
                "close": close.loc[date, code],
                "volume": volume.loc[date, code],
            })
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# BaseIndicator 接口合规测试
# ------------------------------------------------------------------

class TestBaseIndicatorInterface:
    """确保 Alpha101Indicators 正确继承 BaseIndicator。"""

    def test_isinstance(self):
        ind = Alpha101Indicators()
        assert isinstance(ind, BaseIndicator)

    def test_name_property(self):
        ind = Alpha101Indicators()
        assert ind.name == "Alpha101"

    def test_repr(self):
        ind = Alpha101Indicators()
        assert "Alpha101" in repr(ind)

    def test_compute_returns_dataframe(self, single_stock_df):
        ind = Alpha101Indicators(alphas=[101])
        result = ind.compute(single_stock_df)
        assert isinstance(result, pd.DataFrame)
        # 原始列应保留
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_compute_appends_alpha_columns(self, single_stock_df):
        ind = Alpha101Indicators(alphas=[1, 6, 101])
        result = ind.compute(single_stock_df)
        for col in ["alpha001", "alpha006", "alpha101"]:
            assert col in result.columns

    def test_compute_all_alphas(self, single_stock_df):
        ind = Alpha101Indicators()
        result = ind.compute(single_stock_df)
        # 至少应有大部分 alpha 因子列
        alpha_cols = [c for c in result.columns if c.startswith("alpha")]
        assert len(alpha_cols) >= 80  # 允许部分因子因数据不足计算失败

    def test_compute_kwarg_override(self, single_stock_df):
        """compute 的 alphas kwarg 应覆盖构造时设置。"""
        ind = Alpha101Indicators(alphas=[1, 2, 3])
        result = ind.compute(single_stock_df, alphas=[101])
        assert "alpha101" in result.columns
        assert "alpha001" not in result.columns


# ------------------------------------------------------------------
# 运算符测试
# ------------------------------------------------------------------

class TestOperators:
    """核心运算符单元测试。"""

    def test_rank_series(self):
        s = pd.Series([3, 1, 4, 1, 5])
        r = rank(s)
        assert r.shape == s.shape
        assert r.max() <= 1.0
        assert r.min() >= 0.0

    def test_rank_dataframe(self, panel_data):
        _, _, _, close, _ = panel_data
        r = rank(close)
        assert r.shape == close.shape
        assert r.max().max() <= 1.0

    def test_scale(self):
        s = pd.Series([1, 2, 3, -1, -2])
        r = scale(s)
        assert abs(r.abs().sum() - 1.0) < 1e-10

    def test_delay(self, panel_data):
        _, _, _, close, _ = panel_data
        d = delay(close, 5)
        assert d.shape == close.shape
        assert d.iloc[:5].isna().all().all()

    def test_delta(self, panel_data):
        _, _, _, close, _ = panel_data
        d = delta(close, 1)
        expected = close - close.shift(1)
        pd.testing.assert_frame_equal(d, expected)

    def test_ts_min_max(self, panel_data):
        _, _, _, close, _ = panel_data
        mn = ts_min(close, 5)
        mx = ts_max(close, 5)
        diff = (mx - mn).stack().dropna()
        assert (diff >= 0).all()

    def test_ts_sum(self, panel_data):
        _, _, _, close, _ = panel_data
        s = ts_sum(close, 5)
        expected = close.rolling(5, min_periods=2).sum()
        pd.testing.assert_frame_equal(s, expected)

    def test_ts_corr(self, panel_data):
        _, _, _, close, volume = panel_data
        c = ts_corr(close, volume, 10)
        assert c.shape == close.shape
        valid = c.dropna(how="all")
        assert valid.max().max() <= 1.01
        assert valid.min().min() >= -1.01

    def test_decay_linear(self, panel_data):
        _, _, _, close, _ = panel_data
        d = decay_linear(close, 5)
        assert d.shape == close.shape

    def test_sma(self, panel_data):
        _, _, _, close, _ = panel_data
        m = sma(close, 10)
        expected = close.rolling(10, min_periods=5).mean()
        pd.testing.assert_frame_equal(m, expected)

    def test_sign(self):
        s = pd.Series([-2, 0, 3])
        np.testing.assert_array_equal(sign(s), [-1, 0, 1])

    def test_signedpower(self):
        s = pd.Series([-2, 0, 3])
        r = signedpower(s, 2)
        np.testing.assert_array_equal(r, [-4, 0, 9])

    def test_adv(self, panel_data):
        _, _, _, _, volume = panel_data
        a = adv(volume, 20)
        assert a.shape == volume.shape


# ------------------------------------------------------------------
# Alpha 因子测试 (面板模式)
# ------------------------------------------------------------------

class TestAlpha101Engine:
    """Alpha101Engine 面板计算测试。"""

    def test_shape_consistency(self, engine: Alpha101Engine):
        """验证各因子输出形状与输入一致。"""
        expected_shape = engine.close.shape
        for alpha_id in [1, 6, 12, 21, 33, 41, 53, 101]:
            result = engine.compute_factor(alpha_id)
            assert result.shape == expected_shape, (
                f"Alpha#{alpha_id} shape mismatch: "
                f"expected {expected_shape}, got {result.shape}"
            )

    def test_alpha001(self, engine: Alpha101Engine):
        result = engine.alpha001()
        assert not result.empty
        assert result.shape == engine.close.shape

    def test_alpha006(self, engine: Alpha101Engine):
        result = engine.alpha006()
        assert not result.empty

    def test_alpha012(self, engine: Alpha101Engine):
        """sign(delta(volume, 1)) × (−1 × delta(close, 1))"""
        result = engine.alpha012()
        assert not result.empty
        assert result.shape == engine.close.shape

    def test_alpha041(self, engine: Alpha101Engine):
        """(high * low)^0.5 − vwap"""
        result = engine.alpha041()
        assert not result.empty

    def test_alpha101_formula(self, engine: Alpha101Engine):
        """(close − open) / ((high − low) + 0.001)"""
        result = engine.alpha101()
        expected = (engine.close - engine.open) / (
            engine.high - engine.low + 0.001
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_compute_factor_method(self, engine: Alpha101Engine):
        """compute_factor(n) 应与 alpha{n:03d}() 一致 (清理后)。"""
        direct = engine.alpha012()
        via_compute = engine.compute_factor(12)
        pd.testing.assert_frame_equal(
            direct.replace([np.inf, -np.inf], np.nan),
            via_compute,
        )

    def test_compute_factor_invalid_id(self, engine: Alpha101Engine):
        with pytest.raises(ValueError):
            engine.compute_factor(999)

    def test_compute_all(self, engine: Alpha101Engine):
        """compute_all 应返回非空字典。"""
        results = engine.compute_all()
        assert isinstance(results, dict)
        assert len(results) > 0
        for k, v in results.items():
            assert isinstance(k, int)
            assert isinstance(v, (pd.DataFrame, pd.Series))

    def test_compute_all_exclude_industry(self, engine: Alpha101Engine):
        results = engine.compute_all(include_industry=False)
        for k in results:
            assert k not in INDUSTRY_ALPHAS


# ------------------------------------------------------------------
# 工厂方法测试
# ------------------------------------------------------------------

class TestFactory:
    """工厂方法 (from_stacked_df / panel) 测试。"""

    def test_from_stacked_df(self, stacked_df):
        engine = Alpha101Indicators.from_stacked_df(stacked_df)
        assert engine.close.shape[1] == 5
        assert engine.close.shape[0] == 60
        result = engine.alpha101()
        assert result.shape == engine.close.shape

    def test_panel_factory(self, panel_data):
        open_, high, low, close, volume = panel_data
        engine = Alpha101Indicators.panel(
            open_=open_, high=high, low=low, close=close, volume=volume,
        )
        assert isinstance(engine, Alpha101Engine)
        result = engine.alpha101()
        assert result.shape == close.shape

    def test_single_stock_via_compute(self, single_stock_df):
        """单股票通过 BaseIndicator.compute() 接口计算。"""
        ind = Alpha101Indicators(alphas=[1, 41, 101])
        result = ind.compute(single_stock_df)
        assert "alpha001" in result.columns
        assert "alpha041" in result.columns
        assert "alpha101" in result.columns
        assert result.shape[0] == single_stock_df.shape[0]
