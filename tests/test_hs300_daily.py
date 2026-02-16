"""
集成测试 — 沪深300成分股过去一个月 daily 数据采集。

验证 data 模块全链路:
    AkShare 远程拉取 → 数据清洗 → DuckDB 存储 → 本地查询 → 增量更新

运行方式:
    python -m pytest tests/test_hs300_daily.py -v -s
"""

from __future__ import annotations

import datetime as dt
import tempfile
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import pytest

from stockquant.utils.config import Config
from stockquant.data.data_source import DataSourceFactory
from stockquant.data.source_akshare import AkShareDataSource  # noqa: F401 触发注册
from stockquant.data.database import Database
from stockquant.data.data_cleaner import DataCleaner
from stockquant.data.data_manager import DataManager
from stockquant.utils.helpers import normalize_stock_code

# ======================================================================
# 常量
# ======================================================================
END_DATE = dt.date(2026, 2, 15)
START_DATE = END_DATE - dt.timedelta(days=30)  # 过去一个月
INDEX_CODE = "000300"  # 沪深300

# 标准日线列
REQUIRED_COLS = {"code", "date", "open", "high", "low", "close", "volume", "amount"}


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture(scope="module")
def hs300_codes() -> list[str]:
    """从 AkShare 获取当前沪深300成分股代码列表（去重）。"""
    df = ak.index_stock_cons(symbol=INDEX_CODE)
    codes = df["品种代码"].astype(str).str.zfill(6).drop_duplicates().tolist()
    # API 返回可能含重复，去重后实际数量可能少于 300
    assert len(codes) >= 250, f"预期约 300 只成分股，实际 {len(codes)}"
    return codes


@pytest.fixture(scope="module")
def tmp_db_path(tmp_path_factory) -> Path:
    """创建临时数据库文件路径。"""
    return tmp_path_factory.mktemp("data") / "test_hs300.duckdb"


@pytest.fixture(scope="module")
def config(tmp_db_path) -> Config:
    """创建使用临时数据库的配置实例。"""
    Config.reset()
    cfg = Config()
    cfg.set("database.path", str(tmp_db_path))
    cfg.set("data_source.primary", "akshare")
    cfg.set("data_fetch.start_date", str(START_DATE))
    cfg.set("data_fetch.adjust", "hfq")
    return cfg


@pytest.fixture(scope="module")
def db(tmp_db_path) -> Database:
    """初始化临时数据库。"""
    database = Database(db_path=tmp_db_path)
    database.init_tables()
    yield database
    database.close()


@pytest.fixture(scope="module")
def data_source() -> AkShareDataSource:
    return DataSourceFactory.create("akshare")


@pytest.fixture(scope="module")
def cleaner() -> DataCleaner:
    return DataCleaner()


@pytest.fixture(scope="module")
def all_daily_data(hs300_codes, data_source, cleaner, db) -> dict[str, pd.DataFrame]:
    """
    拉取全部沪深300成分股过去一个月的日线数据。

    这是核心 fixture —— 一次性拉取、清洗、入库，后续测试复用结果。
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    total = len(hs300_codes)

    print(f"\n{'='*60}", flush=True)
    print(f"开始拉取沪深300成分股日线数据", flush=True)
    print(f"区间: {START_DATE} ~ {END_DATE}  ({total} 只)", flush=True)
    print(f"{'='*60}", flush=True)

    t_start = time.time()

    for i, code in enumerate(hs300_codes, 1):
        code = normalize_stock_code(code)
        try:
            # 1. 远程拉取
            df = data_source.get_daily_bars(code, START_DATE, END_DATE, adjust="hfq")
            if df.empty:
                failed.append(code)
                print(f"  [{i:3d}/{total}] {code} ⚠ 空数据", flush=True)
                continue

            # 2. 清洗
            df = cleaner.clean_pipeline(df)

            # 3. 入库
            db.save_dataframe(df, "daily_bars")

            results[code] = df

            if i % 50 == 0 or i == total:
                elapsed = time.time() - t_start
                print(f"  [{i:3d}/{total}] 已完成 | 耗时 {elapsed:.1f}s | 速率 {i/elapsed:.1f} 只/s", flush=True)

        except Exception as e:
            failed.append(code)
            print(f"  [{i:3d}/{total}] {code} ✗ {e}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"拉取完成: 成功 {len(results)}, 失败 {len(failed)}, 耗时 {elapsed:.1f}s", flush=True)
    if failed:
        print(f"失败代码: {failed[:20]}{'...' if len(failed) > 20 else ''}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return results


# ======================================================================
# 测试用例
# ======================================================================

class TestHS300StockList:
    """测试1: 沪深300成分股获取。"""

    def test_stock_count(self, hs300_codes):
        """成分股应约300只（去重后 >= 250）。"""
        assert len(hs300_codes) >= 250

    def test_code_format(self, hs300_codes):
        """代码应为6位纯数字。"""
        for code in hs300_codes:
            assert len(code) == 6
            assert code.isdigit()

    def test_no_duplicates(self, hs300_codes):
        """不应有重复代码。"""
        assert len(set(hs300_codes)) == len(hs300_codes)


class TestDataFetch:
    """测试2: 数据拉取覆盖率与质量。"""

    def test_fetch_success_rate(self, all_daily_data, hs300_codes):
        """成功率应 >= 95%。"""
        success_rate = len(all_daily_data) / len(hs300_codes)
        print(f"  拉取成功率: {success_rate:.1%} ({len(all_daily_data)}/{len(hs300_codes)})")
        assert success_rate >= 0.95, f"成功率 {success_rate:.1%} 低于 95%"

    def test_data_not_empty(self, all_daily_data):
        """每只成功拉取的股票应有数据行。"""
        for code, df in all_daily_data.items():
            assert len(df) > 0, f"{code} 数据为空"

    def test_date_range(self, all_daily_data):
        """数据日期应在指定区间内。"""
        for code, df in list(all_daily_data.items())[:30]:  # 抽查30只
            if "date" in df.columns:
                min_date = pd.Timestamp(df["date"].min())
                max_date = pd.Timestamp(df["date"].max())
                assert min_date >= pd.Timestamp(START_DATE) - pd.Timedelta(days=3), \
                    f"{code} 起始日期 {min_date.date()} 早于预期"
                assert max_date <= pd.Timestamp(END_DATE) + pd.Timedelta(days=1), \
                    f"{code} 结束日期 {max_date.date()} 晚于预期"

    def test_trading_days_reasonable(self, all_daily_data):
        """每只股票交易日数应在合理范围（一个月约 20~23 个交易日）。"""
        day_counts = []
        for code, df in all_daily_data.items():
            day_counts.append(len(df))

        avg_days = sum(day_counts) / len(day_counts) if day_counts else 0
        print(f"  平均交易日数: {avg_days:.1f} (最少 {min(day_counts)}, 最多 {max(day_counts)})")
        assert 10 <= avg_days <= 25, f"平均交易日 {avg_days} 不在合理范围"


class TestDataQuality:
    """测试3: 数据质量校验。"""

    def test_required_columns(self, all_daily_data):
        """每只股票数据应包含标准列。"""
        for code, df in list(all_daily_data.items())[:50]:
            missing = REQUIRED_COLS - set(df.columns)
            assert not missing, f"{code} 缺少列: {missing}"

    def test_no_null_ohlc(self, all_daily_data):
        """OHLC 不应有空值。"""
        ohlc = ["open", "high", "low", "close"]
        null_stocks = []
        for code, df in all_daily_data.items():
            null_count = df[ohlc].isna().sum().sum()
            if null_count > 0:
                null_stocks.append((code, null_count))

        if null_stocks:
            print(f"  OHLC 含空值的股票: {null_stocks[:10]}")
        assert len(null_stocks) == 0, f"{len(null_stocks)} 只股票 OHLC 含空值"

    def test_price_positive(self, all_daily_data):
        """价格应为正数。"""
        for code, df in all_daily_data.items():
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    assert (df[col] > 0).all(), f"{code} 列 {col} 存在非正值"

    def test_high_low_consistency(self, all_daily_data):
        """最高价 >= 最低价，且 OHLC 在 [low, high] 区间。"""
        violations = []
        for code, df in all_daily_data.items():
            bad = df[df["high"] < df["low"]]
            if not bad.empty:
                violations.append((code, len(bad)))

            bad_open = df[(df["open"] > df["high"]) | (df["open"] < df["low"])]
            bad_close = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
            if not bad_open.empty:
                violations.append((code, f"open越界 {len(bad_open)}"))
            if not bad_close.empty:
                violations.append((code, f"close越界 {len(bad_close)}"))

        if violations:
            print(f"  价格不一致: {violations[:10]}")
        assert len(violations) == 0, f"{len(violations)} 条价格不一致记录"

    def test_volume_non_negative(self, all_daily_data):
        """成交量应 >= 0 (清洗后应 > 0)。"""
        for code, df in all_daily_data.items():
            if "volume" in df.columns:
                assert (df["volume"] >= 0).all(), f"{code} 成交量存在负值"

    def test_date_sorted(self, all_daily_data):
        """日期应为升序排列。"""
        for code, df in list(all_daily_data.items())[:50]:
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                assert dates.is_monotonic_increasing, f"{code} 日期未按升序排列"

    def test_no_duplicate_dates(self, all_daily_data):
        """同一股票不应有重复日期。"""
        dup_stocks = []
        for code, df in all_daily_data.items():
            if "date" in df.columns:
                dups = df["date"].duplicated().sum()
                if dups > 0:
                    dup_stocks.append((code, dups))
        assert len(dup_stocks) == 0, f"{len(dup_stocks)} 只股票存在重复日期: {dup_stocks[:10]}"


class TestDatabase:
    """测试4: DuckDB 存储与查询。"""

    def test_table_exists(self, db):
        """daily_bars 表应存在。"""
        assert db.table_exists("daily_bars")

    def test_total_rows(self, db, all_daily_data):
        """数据库中的总行数应与拉取数据一致。"""
        result = db.query("SELECT COUNT(*) AS cnt FROM daily_bars")
        db_count = result["cnt"].iloc[0]
        expected = sum(len(df) for df in all_daily_data.values())
        print(f"  数据库行数: {db_count}, 预期: {expected}")
        assert db_count == expected, f"数据库 {db_count} 行 ≠ 预期 {expected} 行"

    def test_stock_count_in_db(self, db, all_daily_data):
        """数据库中的股票数应与成功拉取数一致。"""
        result = db.query("SELECT COUNT(DISTINCT code) AS cnt FROM daily_bars")
        db_codes = result["cnt"].iloc[0]
        assert db_codes == len(all_daily_data), f"数据库 {db_codes} 只 ≠ 预期 {len(all_daily_data)} 只"

    def test_query_single_stock(self, db, all_daily_data):
        """能正确按代码查询单只股票。"""
        code = list(all_daily_data.keys())[0]
        result = db.query(
            "SELECT * FROM daily_bars WHERE code = ? ORDER BY date", [code]
        )
        assert len(result) > 0
        assert result["code"].iloc[0] == code

    def test_query_date_range(self, db, all_daily_data):
        """能正确按日期范围查询。"""
        mid_date = START_DATE + dt.timedelta(days=15)
        result = db.query(
            "SELECT * FROM daily_bars WHERE date >= ? AND date <= ?",
            [str(START_DATE), str(mid_date)],
        )
        assert len(result) > 0
        dates = pd.to_datetime(result["date"])
        assert dates.max() <= pd.Timestamp(mid_date)

    def test_no_duplicate_records(self, db):
        """数据库中不应有 (code, date) 重复记录。"""
        result = db.query("""
            SELECT code, date, COUNT(*) AS cnt
            FROM daily_bars
            GROUP BY code, date
            HAVING cnt > 1
        """)
        assert result.empty, f"发现 {len(result)} 条重复记录"

    def test_insert_idempotent(self, db, all_daily_data):
        """重复插入相同数据不应产生重复行。"""
        code = list(all_daily_data.keys())[0]
        df = all_daily_data[code]
        before = db.query("SELECT COUNT(*) AS cnt FROM daily_bars")["cnt"].iloc[0]

        # 重新插入
        db.save_dataframe(df, "daily_bars")

        after = db.query("SELECT COUNT(*) AS cnt FROM daily_bars")["cnt"].iloc[0]
        assert after == before, "重复插入导致数据增多"

    def test_aggregate_query(self, db):
        """能正确执行聚合查询（验证数据可分析性）。"""
        result = db.query("""
            SELECT code,
                   MIN(date) AS first_date,
                   MAX(date) AS last_date,
                   COUNT(*) AS days,
                   AVG(close) AS avg_close,
                   MAX(high) AS max_high,
                   MIN(low) AS min_low
            FROM daily_bars
            GROUP BY code
            ORDER BY avg_close DESC
            LIMIT 10
        """)
        assert len(result) == 10
        assert "avg_close" in result.columns
        print(f"  最贵10只股票 (均价):")
        for _, row in result.iterrows():
            print(f"    {row['code']}: 均价 {row['avg_close']:.2f}, "
                  f"最高 {row['max_high']:.2f}, {int(row['days'])} 个交易日")


class TestDataCleaner:
    """测试5: 数据清洗效果验证。"""

    def test_no_suspended_rows(self, all_daily_data):
        """清洗后不应有成交量为0的行。"""
        for code, df in all_daily_data.items():
            if "volume" in df.columns:
                zero_vol = (df["volume"] == 0).sum()
                assert zero_vol == 0, f"{code} 仍有 {zero_vol} 条停牌记录"

    def test_dtypes_correct(self, all_daily_data):
        """清洗后数据类型应正确。"""
        for code, df in list(all_daily_data.items())[:30]:
            if "close" in df.columns:
                assert df["close"].dtype in ("float64", "float32"), f"{code} close 类型 {df['close'].dtype}"
            if "volume" in df.columns:
                assert df["volume"].dtype in ("int64", "int32", "float64"), f"{code} volume 类型 {df['volume'].dtype}"


class TestStatsSummary:
    """测试6: 统计摘要（信息展示，不严格断言）。"""

    def test_print_summary(self, all_daily_data, hs300_codes, db):
        """打印数据采集统计摘要。"""
        total_rows = sum(len(df) for df in all_daily_data.values())
        codes_success = len(all_daily_data)
        codes_total = len(hs300_codes)

        # 涨跌幅分布
        all_returns = []
        for df in all_daily_data.values():
            if "pct_change" in df.columns:
                all_returns.extend(df["pct_change"].dropna().tolist())
            elif "close" in df.columns and len(df) > 1:
                rets = df["close"].pct_change().dropna() * 100
                all_returns.extend(rets.tolist())

        print(f"\n{'='*60}")
        print(f" 沪深300 数据采集统计摘要")
        print(f"{'='*60}")
        print(f"  日期区间   : {START_DATE} ~ {END_DATE}")
        print(f"  成分股总数 : {codes_total}")
        print(f"  成功拉取   : {codes_success} ({codes_success/codes_total:.1%})")
        print(f"  总记录数   : {total_rows:,}")
        print(f"  平均天数   : {total_rows/codes_success:.1f}" if codes_success else "")

        if all_returns:
            import numpy as np
            returns = np.array(all_returns)
            print(f"\n  日涨跌幅统计 (%):")
            print(f"    均值   : {returns.mean():.3f}%")
            print(f"    中位数 : {np.median(returns):.3f}%")
            print(f"    标准差 : {returns.std():.3f}%")
            print(f"    最大值 : {returns.max():.2f}%")
            print(f"    最小值 : {returns.min():.2f}%")

        # 数据库文件大小
        db_size = db._db_path.stat().st_size / 1024 / 1024
        print(f"\n  数据库大小 : {db_size:.2f} MB")
        print(f"{'='*60}\n")

        assert True  # 始终通过，这只是展示信息
