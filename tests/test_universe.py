"""测试 - 股票标的池 (StockUniverse) — 类型安全 Pool 枚举 + Builder API。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stockquant.data.universe import (
    BacktestDataset,
    Pool,
    StockUniverse,
)
from stockquant.utils.config import Config


@pytest.fixture(autouse=True)
def reset():
    Config.reset()
    yield
    Config.reset()


# ======================================================================
# 辅助数据
# ======================================================================

_FAKE_ALL_A = pd.DataFrame({
    "code": [
        "600000", "600036", "601318",       # 沪市主板
        "000001", "000002", "002230",       # 深市主板
        "300059", "301236",                 # 创业板
        "688001", "688111",                 # 科创板
        "430047", "830946", "871981",       # 北交所
    ],
    "name": [
        "浦发银行", "招商银行", "中国平安",
        "平安银行", "万科A",   "科大讯飞",
        "东方财富", "万邦医药",
        "华兴源创", "金山办公",
        "诺思兰德", "青矩技术", "锦好医疗",
    ],
})

_FAKE_CSI300 = pd.DataFrame({
    "品种代码": ["600000", "000001", "600036", "300059", "688001"],
    "品种名称": ["浦发银行", "平安银行", "招商银行", "东方财富", "华兴源创"],
})

_FAKE_CSI500 = pd.DataFrame({
    "品种代码": ["601318", "000002", "002230"],
    "品种名称": ["中国平安", "万科A",   "科大讯飞"],
})


def _make_daily(code: str, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "code": [code] * n,
        "date": pd.date_range("2020-01-01", periods=n),
        "open": [10.0] * n, "high": [11.0] * n,
        "low": [9.0] * n, "close": [10.5] * n,
        "volume": [1000] * n,
    })


def _make_mock_dm(valid_codes: list[str] | None = None) -> MagicMock:
    dm = MagicMock()
    dm.get_stock_list.return_value = _FAKE_ALL_A[["code", "name"]].copy()

    valid = set(valid_codes) if valid_codes else set(_FAKE_ALL_A["code"])

    def _fetch_daily(code, start_date=None, end_date=None, adjust=None):
        return _make_daily(code) if code in valid else pd.DataFrame()

    dm.fetch_daily.side_effect = _fetch_daily
    dm.db.query.return_value = _make_daily("000300", n=10)
    return dm


# ======================================================================
# Pool 枚举
# ======================================================================

class TestPool:

    def test_enum_count(self):
        assert len(Pool) == 11

    def test_index_pool_properties(self):
        assert Pool.CSI300.display_name == "沪深300"
        assert Pool.CSI300.index_code == "000300"
        assert Pool.CSI300.code_prefixes is None
        assert Pool.CSI300.is_index is True
        assert Pool.CSI300.is_board is False
        assert Pool.CSI300.is_all is False

    def test_board_pool_properties(self):
        assert Pool.STAR.display_name == "科创板"
        assert Pool.STAR.index_code is None
        assert Pool.STAR.code_prefixes == ("688",)
        assert Pool.STAR.is_index is False
        assert Pool.STAR.is_board is True

    def test_all_a_properties(self):
        assert Pool.ALL_A.display_name == "全部A股"
        assert Pool.ALL_A.is_all is True
        assert Pool.ALL_A.is_index is False
        assert Pool.ALL_A.is_board is False

    def test_str_returns_display_name(self):
        assert str(Pool.CSI300) == "沪深300"
        assert str(Pool.STAR) == "科创板"

    @pytest.mark.parametrize("pool", list(Pool))
    def test_all_pools_have_display_name(self, pool: Pool):
        assert isinstance(pool.display_name, str)
        assert len(pool.display_name) > 0


# ======================================================================
# scope() — 单个 Pool
# ======================================================================

class TestScopeSinglePool:

    def test_scope_all_a(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.ALL_A).codes()
        assert len(codes) == len(_FAKE_ALL_A)

    def test_scope_star(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.STAR).codes()
        assert set(codes) == {"688001", "688111"}

    def test_scope_chinext(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.CHINEXT).codes()
        assert set(codes) == {"300059", "301236"}

    def test_scope_bse(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.BSE).codes()
        assert set(codes) == {"430047", "830946", "871981"}

    def test_scope_sh_main(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.SH_MAIN).codes()
        assert set(codes) == {"600000", "600036", "601318"}

    def test_scope_sz_main(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.SZ_MAIN).codes()
        assert set(codes) == {"000001", "000002", "002230"}

    @patch("stockquant.data.universe.ak")
    def test_scope_csi300(self, mock_ak):
        mock_ak.index_stock_cons.return_value = _FAKE_CSI300.copy()
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.CSI300).codes()
        assert len(codes) == 5


# ======================================================================
# scope() — 多 Pool 并集
# ======================================================================

class TestScopeUnion:

    @patch("stockquant.data.universe.ak")
    def test_union_two_indexes(self, mock_ak):
        """沪深300 ∪ 中证500 → 去重并集。"""
        def _index_cons(symbol):
            if symbol == "000300":
                return _FAKE_CSI300.copy()
            if symbol == "000905":
                return _FAKE_CSI500.copy()
            raise ValueError(f"unknown index {symbol}")

        mock_ak.index_stock_cons.side_effect = _index_cons
        dm = _make_mock_dm()

        codes = StockUniverse(dm=dm).scope(Pool.CSI300, Pool.CSI500).codes()
        # CSI300: 600000, 000001, 600036, 300059, 688001
        # CSI500: 601318, 000002, 002230
        # union = 8
        assert len(codes) == 8

    def test_union_two_boards(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.STAR, Pool.CHINEXT).codes()
        assert set(codes) == {"688001", "688111", "300059", "301236"}

    def test_union_board_and_codes(self):
        """Pool + 代码列表混合。"""
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.STAR, ["600000"]).codes()
        assert set(codes) == {"688001", "688111", "600000"}

    def test_union_deduplicates(self):
        """并集自动去重。"""
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(Pool.STAR, ["688001"]).codes()
        assert codes.count("688001") == 1

    def test_chained_scope_union(self):
        """链式 scope() 调用累加为并集。"""
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.STAR)
            .scope(Pool.CHINEXT)
            .codes()
        )
        assert set(codes) == {"688001", "688111", "300059", "301236"}


# ======================================================================
# scope() — 代码列表
# ======================================================================

class TestScopeCodeList:

    def test_scope_code_list(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(["600000", "000001"]).codes()
        assert codes == ["600000", "000001"]

    def test_scope_normalizes_codes(self):
        dm = _make_mock_dm()
        codes = StockUniverse(dm=dm).scope(["sh600000", "1"]).codes()
        assert codes == ["600000", "000001"]

    def test_scope_type_error(self):
        dm = _make_mock_dm()
        with pytest.raises(TypeError, match="scope\\(\\) 参数类型错误"):
            StockUniverse(dm=dm).scope(12345)


# ======================================================================
# exclude() — Pool 排除
# ======================================================================

class TestExcludePool:

    @patch("stockquant.data.universe.ak")
    def test_exclude_board_from_index(self, mock_ak):
        mock_ak.index_stock_cons.return_value = _FAKE_CSI300.copy()
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.CSI300)
            .exclude(Pool.STAR)
            .codes()
        )
        assert "688001" not in codes
        assert "600000" in codes

    @patch("stockquant.data.universe.ak")
    def test_exclude_multiple_boards(self, mock_ak):
        mock_ak.index_stock_cons.return_value = _FAKE_CSI300.copy()
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.CSI300)
            .exclude(Pool.STAR, Pool.CHINEXT)
            .codes()
        )
        assert "688001" not in codes
        assert "300059" not in codes
        assert set(codes) == {"600000", "000001", "600036"}

    def test_exclude_boards_from_all_a(self):
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.ALL_A)
            .exclude(Pool.STAR, Pool.CHINEXT, Pool.BSE)
            .codes()
        )
        assert set(codes) == {"600000", "600036", "601318", "000001", "000002", "002230"}

    @patch("stockquant.data.universe.ak")
    def test_exclude_index_pool(self, mock_ak):
        """排除一个指数成分股（如排除沪深300的全部成分股）。"""
        mock_ak.index_stock_cons.return_value = _FAKE_CSI300.copy()
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.ALL_A)
            .exclude(Pool.CSI300)
            .codes()
        )
        # CSI300 成分: 600000, 000001, 600036, 300059, 688001
        for c in ["600000", "000001", "600036", "300059", "688001"]:
            assert c not in codes
        assert "601318" in codes  # 不在 CSI300 中

    def test_exclude_all_a_empties_list(self):
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.STAR)
            .exclude(Pool.ALL_A)
            .codes()
        )
        assert codes == []


# ======================================================================
# exclude() — 个股代码排除
# ======================================================================

class TestExcludeCode:

    def test_exclude_single_code(self):
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(["600000", "000001", "600036"])
            .exclude("600000")
            .codes()
        )
        assert "600000" not in codes
        assert set(codes) == {"000001", "600036"}

    def test_exclude_multiple_codes(self):
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.SH_MAIN)
            .exclude("600000", "601318")
            .codes()
        )
        assert "600000" not in codes
        assert "601318" not in codes
        assert "600036" in codes


# ======================================================================
# exclude() — 混合排除（Pool + 个股）
# ======================================================================

class TestExcludeMixed:

    @patch("stockquant.data.universe.ak")
    def test_exclude_pool_and_codes(self, mock_ak):
        mock_ak.index_stock_cons.return_value = _FAKE_CSI300.copy()
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.CSI300)
            .exclude(Pool.STAR, "300059")  # 排除科创板 + 东方财富
            .codes()
        )
        assert "688001" not in codes
        assert "300059" not in codes
        assert set(codes) == {"600000", "000001", "600036"}

    def test_chained_exclude(self):
        """链式 exclude() 调用累加。"""
        dm = _make_mock_dm()
        codes = (
            StockUniverse(dm=dm)
            .scope(Pool.ALL_A)
            .exclude(Pool.STAR)
            .exclude(Pool.CHINEXT)
            .exclude("000001")
            .codes()
        )
        assert "688001" not in codes
        assert "300059" not in codes
        assert "000001" not in codes
        assert "600000" in codes

    def test_exclude_type_error(self):
        dm = _make_mock_dm()
        with pytest.raises(TypeError, match="exclude\\(\\) 参数类型错误"):
            StockUniverse(dm=dm).scope(Pool.ALL_A).exclude(123)


# ======================================================================
# load()
# ======================================================================

class TestLoad:

    def test_load_custom_codes(self):
        dm = _make_mock_dm(["600000", "000001"])
        dataset = (
            StockUniverse(dm=dm)
            .scope(["600000", "000001", "999999"])
            .load("2020-01-01", "2025-12-31")
        )
        assert isinstance(dataset, BacktestDataset)
        assert set(dataset.codes) == {"600000", "000001"}
        assert "999999" in dataset.missing_codes
        assert not dataset.benchmark.empty

    def test_load_with_exclude(self):
        dm = _make_mock_dm()
        dataset = (
            StockUniverse(dm=dm)
            .scope(Pool.ALL_A)
            .exclude(Pool.STAR, Pool.CHINEXT, Pool.BSE)
            .load("2020-01-01", "2025-12-31")
        )
        for code in dataset.codes:
            assert not code.startswith(("688", "300", "301", "43", "83", "87"))

    def test_load_benchmark_pool(self):
        dm = _make_mock_dm(["600000"])
        dataset = (
            StockUniverse(dm=dm)
            .scope(["600000"])
            .load("2020-01-01", "2025-12-31", benchmark=Pool.CSI300)
        )
        assert dataset.benchmark_code == "000300"

    def test_load_benchmark_str(self):
        dm = _make_mock_dm(["600000"])
        dataset = (
            StockUniverse(dm=dm)
            .scope(["600000"])
            .load("2020-01-01", "2025-12-31", benchmark="000905")
        )
        assert dataset.benchmark_code == "000905"

    def test_load_benchmark_non_index_raises(self):
        dm = _make_mock_dm(["600000"])
        with pytest.raises(ValueError, match="基准必须是指数类型"):
            StockUniverse(dm=dm).scope(["600000"]).load(
                "2020-01-01", "2025-12-31", benchmark=Pool.STAR,
            )

    def test_load_dates(self):
        dm = _make_mock_dm(["600000"])
        dataset = (
            StockUniverse(dm=dm)
            .scope(["600000"])
            .load("2020-01-01", "2025-12-31")
        )
        assert dataset.start_date == "2020-01-01"
        assert dataset.end_date == "2025-12-31"


# ======================================================================
# BacktestDataset
# ======================================================================

class TestBacktestDataset:

    def test_repr(self):
        ds = BacktestDataset(
            stock_data={"600000": _make_daily("600000")},
            codes=["600000"],
            benchmark=pd.DataFrame(),
            benchmark_code="000300",
            start_date="2020-01-01",
            end_date="2025-12-31",
            missing_codes=["999999"],
        )
        assert "1/2 stocks loaded" in repr(ds)
        assert "000300" in repr(ds)

    def test_summary(self):
        ds = BacktestDataset(
            stock_data={"600000": _make_daily("600000")},
            codes=["600000"],
            benchmark=_make_daily("000300"),
            benchmark_code="000300",
            start_date="2020-01-01",
            end_date="2025-12-31",
            missing_codes=["999999"],
        )
        s = ds.summary()
        assert "1 只" in s
        assert "2020-01-01" in s
        assert "999999" in s


# ======================================================================
# Builder 链式调用
# ======================================================================

class TestBuilderChaining:

    def test_chain_returns_self(self):
        dm = _make_mock_dm()
        u = StockUniverse(dm=dm)
        assert u.scope(Pool.ALL_A) is u
        assert u.exclude(Pool.STAR) is u

    def test_scope_accumulates(self):
        dm = _make_mock_dm()
        u = StockUniverse(dm=dm)
        u.scope(Pool.STAR)
        u.scope(Pool.CHINEXT)
        codes = u.codes()
        assert set(codes) == {"688001", "688111", "300059", "301236"}

    def test_exclude_accumulates(self):
        dm = _make_mock_dm()
        u = StockUniverse(dm=dm)
        u.scope(Pool.ALL_A)
        u.exclude(Pool.STAR)
        u.exclude(Pool.CHINEXT)
        codes = u.codes()
        for c in codes:
            assert not c.startswith(("688", "300", "301"))
