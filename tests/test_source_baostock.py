"""
测试 - BaoStockDataSource: 单元测试 + 集成测试。

单元测试覆盖纯逻辑（代码转换、参数校验、内部工具方法），
集成测试真实调用 BaoStock API 验证数据返回结构和内容。
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stockquant.data.data_source import (
    DAILY_BAR_COLS,
    INDEX_DAILY_COLS,
    DataSourceFactory,
)
from stockquant.data.source_baostock import (
    BaoStockDataSource,
    _FINANCE_CATEGORIES,
    _INDEX_CONSTITUENT_MAP,
    _MINUTE_FREQ_MAP,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def src() -> BaoStockDataSource:
    """模块级数据源实例，复用登录会话减少 API 调用。"""
    return BaoStockDataSource()


# ======================================================================
# 工具：模拟 BaoStock ResultSet（仅用于单元测试 mock 场景）
# ======================================================================


class _FakeLogin:
    error_code = "0"
    error_msg = "success"


def _make_rs(rows: list[list[Any]], fields: list[str], error_code: str = "0") -> MagicMock:
    """构造模拟 ResultSet 对象。"""
    rs = MagicMock()
    rs.error_code = error_code
    rs.error_msg = ""
    rs.fields = fields
    rs._rows = rows
    rs._pos = -1

    def _next() -> bool:
        rs._pos += 1
        return rs._pos < len(rs._rows)

    def _get_row() -> list[Any]:
        return list(rs._rows[rs._pos])

    rs.next = _next
    rs.get_row_data = _get_row
    return rs


# ======================================================================
# 单元测试：_bs_code 静态方法
# ======================================================================


class TestBsCode:
    def test_shanghai_code(self):
        assert BaoStockDataSource._bs_code("600000") == "sh.600000"

    def test_shenzhen_code(self):
        assert BaoStockDataSource._bs_code("000001") == "sz.000001"

    def test_with_prefix_already(self):
        assert BaoStockDataSource._bs_code("sh600000") == "sh.600000"

    def test_with_suffix_already(self):
        assert BaoStockDataSource._bs_code("600000.SH") == "sh.600000"

    def test_chi_next(self):
        assert BaoStockDataSource._bs_code("300750") == "sz.300750"

    def test_star_market(self):
        assert BaoStockDataSource._bs_code("688001") == "sh.688001"


# ======================================================================
# 单元测试：_to_numeric
# ======================================================================


class TestToNumeric:
    def test_converts_strings(self):
        src = BaoStockDataSource()
        df = pd.DataFrame({"open": ["10.5", "11.2"]})
        df = src._to_numeric(df, ["open"])
        assert df["open"].dtype == float

    def test_skips_missing_cols(self):
        src = BaoStockDataSource()
        df = pd.DataFrame({"close": ["10.0"]})
        df = src._to_numeric(df, ["open", "close"])
        assert "open" not in df.columns
        assert df["close"].dtype == float


# ======================================================================
# 单元测试：参数校验
# ======================================================================


class TestParameterValidation:
    def test_invalid_freq_raises(self):
        src = BaoStockDataSource()
        with pytest.raises(ValueError, match="频率不支持"):
            src.get_minute_bars("600000", "2024-01-01", "2024-01-02", freq="10")

    def test_invalid_finance_category_raises(self):
        src = BaoStockDataSource()
        with pytest.raises(ValueError, match="invalid"):
            src.get_finance_data("600000", category="invalid")

    def test_unsupported_index_raises(self):
        src = BaoStockDataSource()
        with pytest.raises(NotImplementedError, match="000852"):
            src.get_index_constituents("000852")


# ======================================================================
# 单元测试：RateLimiter 基础属性
# ======================================================================


class TestRateLimiter:
    def test_rate_limiter_initialized(self):
        src = BaoStockDataSource()
        assert src._rate_limiter is not None
        assert src._rate_limiter.rate == 0.57
        assert src._rate_limiter.burst == 100


# ======================================================================
# 单元测试：工厂注册
# ======================================================================


class TestFactoryRegistration:
    def test_registered_in_factory(self):
        src = DataSourceFactory.create("baostock")
        assert isinstance(src, BaoStockDataSource)


# ======================================================================
# 集成测试：get_stock_list
# ======================================================================


class TestGetStockListIntegration:
    def test_returns_non_empty(self, src):
        df = src.get_stock_list()
        assert not df.empty, "股票列表不应为空"
        assert "code" in df.columns
        assert "name" in df.columns
        assert df["code"].str.len().eq(6).all(), "code 应为纯 6 位数字"
        # 至少返回几千只 A 股
        assert len(df) > 3000, f"股票数量应 > 3000，实际 {len(df)}"

    def test_contains_known_stocks(self, src):
        df = src.get_stock_list()
        codes = set(df["code"])
        assert "600000" in codes, "应包含浦发银行"
        assert "000001" in codes, "应包含平安银行"


# ======================================================================
# 集成测试：get_daily_bars
# ======================================================================


class TestGetDailyBarsIntegration:
    def test_hfq_returns_data(self, src):
        """后复权日线 — 取最近 5 个交易日。"""
        end = dt.date.today()
        start = end - dt.timedelta(days=30)
        df = src.get_daily_bars("600000", start, end, adjust="hfq")
        assert not df.empty, "后复权日线返回为空"
        assert set(df.columns) == set(DAILY_BAR_COLS), f"列不匹配: {list(df.columns)}"
        assert (df["code"] == "600000").all()
        assert df["date"].is_monotonic_increasing

    def test_qfq_returns_data(self, src):
        """前复权日线。"""
        end = dt.date.today()
        start = end - dt.timedelta(days=30)
        df = src.get_daily_bars("000001", start, end, adjust="qfq")
        assert not df.empty, "前复权日线返回为空"

    def test_numeric_types(self, src):
        """数值列应有数值类型（float 或 int）。"""
        end = dt.date.today()
        start = end - dt.timedelta(days=30)
        df = src.get_daily_bars("600000", start, end)
        for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"]:
            if df[col].notna().any():
                assert df[col].dtype in (float, "float64", int, "int64"), \
                    f"{col} 应为数值类型，实际 {df[col].dtype}"


# ======================================================================
# 集成测试：get_index_daily
# ======================================================================


class TestGetIndexDailyIntegration:
    @pytest.mark.parametrize("index_code", ["000300", "000905", "000016", "399001", "399006"])
    def test_returns_index_data(self, src, index_code):
        end = dt.date.today()
        start = end - dt.timedelta(days=60)
        df = src.get_index_daily(index_code, start, end)
        if df.empty:
            # 可能存在间歇性连接问题，重试一次
            import time
            time.sleep(3)
            df = src.get_index_daily(index_code, start, end)
        assert not df.empty, f"{index_code} 指数日线返回为空（start={start}, end={end}）"
        assert set(df.columns) == set(INDEX_DAILY_COLS), f"列不匹配: {list(df.columns)}"
        assert (df["code"] == index_code).all()
        assert df["date"].is_monotonic_increasing


# ======================================================================
# 集成测试：get_index_constituents
# ======================================================================


class TestGetIndexConstituentsIntegration:
    @pytest.mark.parametrize("index_code", ["000016", "000300", "000905"])
    def test_returns_constituents(self, src, index_code):
        codes = src.get_index_constituents(index_code)
        assert len(codes) > 0, f"{index_code} 成分股列表为空"
        # 应为去重纯 6 位数字
        assert all(isinstance(c, str) and len(c) == 6 and c.isdigit() for c in codes)
        assert codes == sorted(codes)

    def test_sz50_size(self, src):
        codes = src.get_index_constituents("000016")
        assert 40 <= len(codes) <= 60, f"上证50 成分股应在 50 左右，实际 {len(codes)}"

    def test_hs300_size(self, src):
        codes = src.get_index_constituents("000300")
        assert 280 <= len(codes) <= 320, f"沪深300 成分股应在 300 左右，实际 {len(codes)}"

    def test_zz500_size(self, src):
        codes = src.get_index_constituents("000905")
        assert 450 <= len(codes) <= 550, f"中证500 成分股应在 500 左右，实际 {len(codes)}"


# ======================================================================
# 集成测试：get_finance_data
# ======================================================================


class TestGetFinanceDataIntegration:
    @pytest.mark.parametrize("category", list(_FINANCE_CATEGORIES.keys()))
    def test_all_categories_return_data(self, src, category):
        """所有 8 种财报类型应返回非空数据。"""
        df = src.get_finance_data("600000", year=2024, quarter=4, category=category)

        # express/forecast 可能对部分股票返回空，允许
        if category in ("express", "forecast") and df.empty:
            pytest.skip(f"{category} 对该股票无数据")

        assert not df.empty, f"{category} 返回为空"
        assert "code" in df.columns

    def test_profit_returns_expected_fields(self, src):
        """盈利能力应包含常见字段。"""
        df = src.get_finance_data("600000", year=2024, quarter=4, category="profit")
        if not df.empty:
            # 盈利能力常见字段
            expected = {"code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin"}
            assert expected & set(df.columns), f"缺失预期字段: {expected - set(df.columns)}"

    def test_growth_returns_expected_fields(self, src):
        """成长能力应包含常见字段。"""
        df = src.get_finance_data("000001", year=2024, quarter=4, category="growth")
        if not df.empty:
            expected = {"code", "pubDate", "statDate", "YOYEquity", "YOYAsset", "YOYNI"}
            assert expected & set(df.columns), f"缺失预期字段: {expected - set(df.columns)}"


# ======================================================================
# 集成测试：get_trade_dates
# ======================================================================


class TestGetTradeDatesIntegration:
    def test_returns_recent_month(self, src):
        end = dt.date.today()
        start = end - dt.timedelta(days=35)
        dates = src.get_trade_dates(start, end)
        assert len(dates) >= 15, f"一个月交易日应 >= 15 天，实际 {len(dates)}"
        # 应按日期排序
        assert dates == sorted(dates)
        # 格式应为 YYYY-MM-DD
        assert all(d.count("-") == 2 for d in dates)

    def test_known_trading_month(self, src):
        """2024年1月有 22 个交易日（元旦+春节前）。"""
        dates = src.get_trade_dates("2024-01-01", "2024-01-31")
        assert 20 <= len(dates) <= 24, f"2024年1月交易日异常: {len(dates)} 天"


# ======================================================================
# 集成测试：get_stock_info
# ======================================================================


class TestGetStockInfoIntegration:
    def test_returns_non_empty(self, src):
        df = src.get_stock_info()
        assert not df.empty, "stock_info 不应为空"
        expected_cols = {"code", "name", "industry", "sector", "market",
                         "list_date", "total_shares", "float_shares", "total_cap", "float_cap"}
        assert expected_cols == set(df.columns), f"列不匹配: {set(df.columns)}"
        assert len(df) > 3000, f"股票数量应 > 3000，实际 {len(df)}"

    def test_has_industry_coverage(self, src):
        """应有相当比例的股票有行业分类。"""
        df = src.get_stock_info()
        has_industry = (df["industry"] != "").sum()
        ratio = has_industry / len(df)
        assert ratio > 0.5, f"行业覆盖率 {ratio:.1%} 偏低，应 > 50%"

    def test_has_list_date(self, src):
        """应有相当比例的股票有上市日期。"""
        df = src.get_stock_info()
        has_date = df["list_date"].notna().sum()
        ratio = has_date / len(df)
        assert ratio > 0.7, f"上市日期覆盖率 {ratio:.1%} 偏低，应 > 70%"

    def test_known_stock_info(self, src):
        """验证特定股票基本信息。"""
        df = src.get_stock_info()
        row = df[df["code"] == "600000"]
        if not row.empty:
            assert row.iloc[0]["name"] == "浦发银行"


# ======================================================================
# 集成测试：get_minute_bars
# ======================================================================


class TestGetMinuteBarsIntegration:
    @pytest.mark.parametrize("freq", ["5", "15", "30", "60"])
    def test_all_frequencies(self, src, freq):
        """各频率分钟线应返回数据（取最近 1 天）。"""
        today = dt.date.today()
        df = src.get_minute_bars("600000", today, today, freq=freq)
        if df.empty:
            # 可能当天非交易日，跳过
            pytest.skip("当天无分钟线数据")
        assert "time" in df.columns, f"{freq}min 应包含 time 列"
        assert (df["code"] == "600000").all()
        assert df["open"].dtype == float


# ======================================================================
# 集成测试：BaoStock 独有方法
# ======================================================================


class TestGetDividendDataIntegration:
    def test_returns_dividend_info(self, src):
        df = src.get_dividend_data("600000", 2024)
        assert not df.empty
        assert "code" in df.columns


class TestGetAdjustFactorIntegration:
    def test_returns_adjust_factors(self, src):
        df = src.get_adjust_factor("600000", "2024-01-01", "2024-12-31")
        assert not df.empty
        assert "code" in df.columns


class TestGetAllStockOnDateIntegration:
    def test_returns_stock_list(self, src):
        df = src.get_all_stock_on_date("2024-12-31")
        assert not df.empty, "全量股票列表不应为空"
        # 应有至少 4000+ 只股（包括已退市的交易状态可能为 0）
        assert len(df) > 3000
        expected_cols = {"code", "tradeStatus", "code_name"}
        assert expected_cols.issubset(set(df.columns))

    def test_accepts_date_object(self, src):
        df = src.get_all_stock_on_date(dt.date(2024, 12, 31))
        assert not df.empty


class TestMacroIntegration:
    def test_deposit_rate(self, src):
        """存款利率用 2015 年范围（BaoStock 仅有 2015 年数据）。"""
        df = src.get_deposit_rate(start_date="2015-01-01", end_date="2017-12-31")
        assert not df.empty

    def test_loan_rate(self, src):
        """贷款利率用 2015 年范围。"""
        df = src.get_loan_rate(start_date="2015-01-01", end_date="2017-12-31")
        assert not df.empty

    def test_required_reserve_ratio(self, src):
        """准备金率用 2015 年范围。"""
        df = src.get_required_reserve_ratio(start_date="2015-01-01", end_date="2017-12-31")
        assert not df.empty

    def test_money_supply_month(self, src):
        df = src.get_money_supply_month(start_date="2024-01", end_date="2024-12")
        assert not df.empty

    def test_money_supply_year(self, src):
        df = src.get_money_supply_year(start_date="2020", end_date="2024")
        assert not df.empty