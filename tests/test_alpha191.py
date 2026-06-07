"""测试 - Alpha191 因子指标。"""

import numpy as np
import pandas as pd
import pytest

from stockquant.indicators.alpha101.operators import delay, sma as _sma, ts_corr, ts_stddev


# ======================================================================
# Alpha191 专用算子测试
# ======================================================================

class TestAlpha191Operators:

    @pytest.fixture
    def sample_series(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.standard_normal(100))

    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=60)
        codes = [f"{i:06d}" for i in range(1, 11)]
        return pd.DataFrame(
            rng.standard_normal((60, 10)),
            index=dates, columns=codes,
        )

    def test_ema_sma_basic(self, sample_series):
        from stockquant.indicators.alpha191.operators import ema_sma
        result = ema_sma(sample_series, n=10, m=2)
        assert len(result) == len(sample_series)
        assert result.isna().sum() < len(sample_series)

    def test_ema_sma_dataframe(self, sample_df):
        from stockquant.indicators.alpha191.operators import ema_sma
        result = ema_sma(sample_df, n=10, m=2)
        assert result.shape == sample_df.shape

    def test_wma(self, sample_series):
        from stockquant.indicators.alpha191.operators import wma
        result = wma(sample_series, n=10)
        assert len(result) == len(sample_series)

    def test_regbeta(self, sample_series):
        from stockquant.indicators.alpha191.operators import regbeta
        x = pd.Series(range(len(sample_series)), dtype=float)
        result = regbeta(sample_series, x, n=20)
        assert len(result) == len(sample_series)

    def test_count(self, sample_series):
        from stockquant.indicators.alpha191.operators import count
        cond = sample_series > 0
        result = count(cond, n=10)
        assert result.max() <= 10
        assert result.min() >= 0

    def test_highday(self, sample_series):
        from stockquant.indicators.alpha191.operators import highday
        result = highday(sample_series, n=10)
        assert result.max() < 10

    def test_lowday(self, sample_series):
        from stockquant.indicators.alpha191.operators import lowday
        result = lowday(sample_series, n=10)
        assert result.max() < 10

    def test_sumif(self, sample_series):
        from stockquant.indicators.alpha191.operators import sumif
        cond = sample_series > 0
        result = sumif(sample_series, n=10, condition=cond)
        assert len(result) == len(sample_series)


# ======================================================================
# 辅助 — 构建测试面板
# ======================================================================

def _make_test_dataset(n_days: int = 60, n_stocks: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

    close = pd.DataFrame(
        10 + np.cumsum(rng.standard_normal((n_days, n_stocks)) * 0.2, axis=0),
        index=dates, columns=codes,
    )
    close = close.clip(lower=1.0)
    open_ = close * (1 + rng.uniform(-0.01, 0.01, (n_days, n_stocks)))
    high = pd.DataFrame(
        np.maximum(open_.values, close.values) * (1 + rng.uniform(0, 0.02, (n_days, n_stocks))),
        index=dates, columns=codes,
    )
    low = pd.DataFrame(
        np.minimum(open_.values, close.values) * (1 - rng.uniform(0, 0.02, (n_days, n_stocks))),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(100_000, 1_000_000, (n_days, n_stocks)),
        index=dates, columns=codes, dtype=float,
    )
    amount = close * volume * 100

    return {
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "amount": amount,
    }


# ======================================================================
# Alpha191Engine 基础测试
# ======================================================================

class TestAlpha191Engine:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset()
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    def test_panel_shapes(self, engine):
        assert engine.open.shape == engine.close.shape
        assert engine.vwap.shape == engine.close.shape
        assert engine.returns.shape == engine.close.shape

    def test_compute_factor_invalid_raises(self, engine):
        with pytest.raises(ValueError, match="未实现"):
            engine.compute_factor(999)

    def test_compute_factor_returns_dataframe(self, engine):
        result = engine.compute_factor(2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape


class TestAlpha191Indicators:

    def test_name_property(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Indicators
        ind = Alpha191Indicators()
        assert ind.name == "Alpha191"

    def test_compute_single_stock(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Indicators
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame({
            "open": 10 + rng.standard_normal(n) * 0.1,
            "high": 10.2 + rng.standard_normal(n) * 0.1,
            "low": 9.8 + rng.standard_normal(n) * 0.1,
            "close": 10 + rng.standard_normal(n) * 0.1,
            "volume": rng.integers(100_000, 1_000_000, n),
        })
        ind = Alpha191Indicators(alphas=[2, 13, 14])
        result = ind.compute(df)
        assert "alpha002" in result.columns
        assert "alpha013" in result.columns
        assert "alpha014" in result.columns

    def test_from_dataset(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Indicators
        from stockquant.data.universe import BacktestDataset

        rng = np.random.default_rng(42)
        n_days, n_stocks = 30, 5
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        stock_data = {}
        for code in codes:
            close = 10 + np.cumsum(rng.normal(0, 0.2, n_days))
            close = np.maximum(close, 1.0)
            stock_data[code] = pd.DataFrame({
                "code": code, "date": dates,
                "open": close * 0.99, "high": close * 1.02,
                "low": close * 0.98, "close": close,
                "volume": rng.integers(100_000, 500_000, n_days),
                "amount": close * rng.integers(100_000, 500_000, n_days),
            })
        benchmark = pd.DataFrame({
            "code": "000300", "date": dates,
            "open": 4000.0, "high": 4020.0, "low": 3980.0,
            "close": 4000 + np.cumsum(rng.normal(0, 5, n_days)),
            "volume": rng.integers(int(1e8), int(5e8), n_days),
            "amount": 4000 * rng.integers(int(1e8), int(5e8), n_days),
        })
        dataset = BacktestDataset(
            stock_data=stock_data, codes=codes,
            benchmark=benchmark, benchmark_code="000300",
            start_date="2024-01-02", end_date="2024-02-12",
        )
        engine = Alpha191Indicators.from_dataset(dataset)
        result = engine.compute_factor(14)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == n_stocks


# ======================================================================
# 因子 #001 – #050 测试
# ======================================================================

class TestAlpha191Factors001to050:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [1, 2, 5, 7, 10, 13, 14, 15, 20, 38, 41, 42, 46, 48, 50])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha014_is_5day_return(self, engine):
        result = engine.alpha014()
        expected = engine.close - delay(engine.close, 5)
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_alpha015_is_open_close_ratio(self, engine):
        result = engine.alpha015()
        expected = engine.open / (delay(engine.close, 1) + 1e-10) - 1
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_alpha013_formula(self, engine):
        result = engine.alpha013()
        expected = (engine.high * engine.low) ** 0.5 - engine.vwap
        pd.testing.assert_frame_equal(result, expected, check_names=False)


# ======================================================================
# 因子 #051 – #100 测试
# ======================================================================

class TestAlpha191Factors051to100:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [51, 53, 54, 60, 62, 65, 66, 70, 71, 80, 85, 86, 88, 90, 99, 100])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha065_is_mean_ratio(self, engine):
        result = engine.alpha065()
        expected = _sma(engine.close, 6) / (engine.close + 1e-10)
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result.values[mask.values], expected.values[mask.values], rtol=1e-5)


# ======================================================================
# 因子 #101 – #150 测试
# ======================================================================

class TestAlpha191Factors101to150:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [101, 102, 105, 106, 107, 108, 113, 114, 117, 120, 126, 132, 134, 136, 140, 141, 142, 148])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha106_is_20day_return(self, engine):
        result = engine.alpha106()
        expected = engine.close - delay(engine.close, 20)
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_alpha126_is_typical_price(self, engine):
        result = engine.alpha126()
        expected = (engine.close + engine.high + engine.low) / 3
        pd.testing.assert_frame_equal(result, expected, check_names=False)


# ======================================================================
# 因子 #151 – #191 测试
# ======================================================================

class TestAlpha191Factors151to191:

    @pytest.fixture
    def engine(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        return Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )

    @pytest.mark.parametrize("alpha_id", [153, 155, 158, 160, 161, 163, 167, 168, 170, 171, 176, 185, 188, 189, 191])
    def test_factor_shape_and_finite(self, engine, alpha_id):
        result = engine.compute_factor(alpha_id)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == engine.close.shape
        finite_ratio = np.isfinite(result.values).mean()
        assert finite_ratio > 0.3, f"Alpha191#{alpha_id} 有效值比例过低: {finite_ratio:.2%}"

    def test_alpha191_formula(self, engine):
        result = engine.alpha191()
        expected = ts_corr(_sma(engine.volume, 20), engine.low, 5) + (engine.high + engine.low) / 2 - engine.close
        mask = expected.notna() & result.notna()
        np.testing.assert_allclose(result.values[mask.values], expected.values[mask.values], rtol=1e-5)


# ======================================================================
# 注册与集成测试
# ======================================================================

class TestAlpha191Registry:

    def test_registered_in_indicator_registry(self):
        from stockquant.indicators import IndicatorRegistry
        assert "alpha191" in IndicatorRegistry.list()

    def test_create_via_registry(self):
        from stockquant.indicators import IndicatorRegistry
        ind = IndicatorRegistry.create("alpha191")
        assert ind.name == "Alpha191"

    def test_importable_from_indicators(self):
        from stockquant.indicators import Alpha191Indicators
        assert Alpha191Indicators is not None


class TestAlpha191ComputeAll:

    def test_compute_all_returns_dict(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        engine = Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )
        results = engine.compute_all()
        assert isinstance(results, dict)
        assert len(results) >= 180

    def test_compute_factors_subset(self):
        from stockquant.indicators.alpha191.alpha191 import Alpha191Engine
        data = _make_test_dataset(n_days=120, n_stocks=10)
        engine = Alpha191Engine(
            open_=data["open"], high=data["high"], low=data["low"],
            close=data["close"], volume=data["volume"], amount=data["amount"],
        )
        results = engine.compute_factors([1, 14, 65, 191])
        assert len(results) == 4
        assert all(isinstance(v, pd.DataFrame) for v in results.values())
