"""
数据更新器 — 批量更新日线数据。

典型用法::

    from stockquant.data.updater import DataUpdater

    updater = DataUpdater()

    # 最常用：更新全部 A 股日线
    updater.update_all_daily()

    # 最常用：更新 Benchmark 指数行情（沪深300/中证500/1000/2000）
    updater.update_benchmark_indices()

    # 更新股票基本信息（行业/市值等）
    updater.update_stock_info()

    # 更新某个指数成分股日线
    updater.update_index_daily("000300")

    # 更新指定代码
    updater.update_codes_daily(["000001", "600519"])

命令行::

    python -m stockquant.data.updater                  # 默认更新全部 A 股
    python -m stockquant.data.updater --mode stock_info  # 更新股票基本信息
    python -m stockquant.data.updater --mode benchmark # 更新 Benchmark 指数
    python -m stockquant.data.updater --mode index --index-code 000300
    python -m stockquant.data.updater --mode codes --codes 000001 600519
    python -m stockquant.data.updater --no-star --no-bse   # 排除科创板和北交所
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Sequence

import pandas as pd

from stockquant.data.data_cleaner import DataCleaner
from stockquant.data.data_source import BaseDataSource, DataSourceFactory
from stockquant.data.database import Database
from stockquant.utils.concurrent import parallel_fetch_serial_consume
from stockquant.utils.config import Config
from stockquant.utils.helpers import call_with_retries, ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("data.updater")

# ======================================================================
# 常用指数代码
# ======================================================================
INDEX_HS300 = "000300"
INDEX_CSI500 = "000905"
INDEX_CSI1000 = "000852"
INDEX_CSI2000 = "932000"

BENCHMARK_INDICES: dict[str, str] = {
    INDEX_HS300: "沪深300",
    INDEX_CSI500: "中证500",
    INDEX_CSI1000: "中证1000",
    INDEX_CSI2000: "中证2000",
}

# ======================================================================
# 板块前缀
# ======================================================================
_STAR_PREFIXES = ("688",)       # 科创板
_BSE_PREFIXES = ("4", "8")      # 北交所
_GEM_PREFIXES = ("30",)         # 创业板


# ======================================================================
# 工具函数
# ======================================================================

def filter_codes(
    codes: list[str],
    *,
    include_star: bool = True,
    include_bse: bool = True,
    include_gem: bool = True,
) -> list[str]:
    """根据板块过滤股票代码列表。"""
    result = []
    for c in codes:
        if not include_star and c.startswith(_STAR_PREFIXES):
            continue
        if not include_bse and c.startswith(_BSE_PREFIXES) and not c.startswith(_STAR_PREFIXES):
            continue
        if not include_gem and c.startswith(_GEM_PREFIXES):
            continue
        result.append(c)
    return result


# ======================================================================
# DataUpdater
# ======================================================================

class DataUpdater:
    """数据更新器 — 批量拉取 & 入库日线数据。

    Parameters
    ----------
    config : Config, optional
        配置实例，默认使用全局单例。
    db : Database, optional
        数据库实例，默认根据配置自动创建。
    """

    def __init__(
        self,
        config: Config | None = None,
        db: Database | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.db = db or Database()
        self.db.init_tables()
        self.cleaner = DataCleaner()

        primary = self.cfg.get("data_source.primary", "akshare")
        self._source: BaseDataSource = DataSourceFactory.create(primary)

    # ------------------------------------------------------------------
    # 核心公开接口
    # ------------------------------------------------------------------

    def update_all_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
        *,
        include_star: bool = True,
        include_bse: bool = True,
        include_gem: bool = True,
        include_delisted: bool = False,
    ) -> dict[str, int]:
        """更新全部 A 股日线数据。"""
        end_date_dt = ensure_date(end_date) if end_date else dt.date.today()
        codes = self._get_all_a_codes(include_delisted=include_delisted, end_date=end_date_dt)
        codes = filter_codes(
            codes,
            include_star=include_star,
            include_bse=include_bse,
            include_gem=include_gem,
        )
        logger.info(
            f"更新全部 A 股日线: {len(codes)} 只 "
            f"(star={include_star}, bse={include_bse}, gem={include_gem})"
        )
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_index_daily(
        self,
        index_code: str = INDEX_HS300,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
        *,
        include_star: bool = True,
        include_bse: bool = True,
        include_gem: bool = True,
    ) -> dict[str, int]:
        """更新某个指数成分股的日线数据。"""
        codes = self._source.get_index_constituents(index_code)
        codes = filter_codes(
            codes,
            include_star=include_star,
            include_bse=include_bse,
            include_gem=include_gem,
        )
        logger.info(
            f"更新指数 {index_code} 成分股日线: {len(codes)} 只 "
            f"(star={include_star}, bse={include_bse}, gem={include_gem})"
        )
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_codes_daily(
        self,
        codes: Sequence[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """更新指定代码列表的日线数据。"""
        codes = [normalize_stock_code(c) for c in codes]
        logger.info(f"更新自定义列表日线: {len(codes)} 只")
        return self._batch_update(list(codes), start_date, end_date, adjust)

    def update_benchmark_indices(
        self,
        index_codes: Sequence[str] | None = None,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> dict[str, int]:
        """更新 Benchmark 指数自身的日线行情（写入 index_daily 表）。"""
        if index_codes is None:
            index_codes = list(BENCHMARK_INDICES.keys())

        end_date = ensure_date(end_date) or dt.date.today()
        results: dict[str, int] = {}

        for code in index_codes:
            code = normalize_stock_code(code)
            label = BENCHMARK_INDICES.get(code, code)
            try:
                sd = self._resolve_index_start_date(code, start_date)
                if sd > end_date:
                    logger.info(f"指数 {label}({code}) 已是最新")
                    results[code] = 0
                    continue

                logger.info(f"拉取指数 {label}({code}) 日线 [{sd} ~ {end_date}]")
                df = call_with_retries(
                    lambda c=code, s=sd, e=end_date: self._source.get_index_daily(c, s, e),
                    label=f"index_daily {code}",
                )
                if df is None or df.empty:
                    logger.warning(f"指数 {label}({code}) 返回空数据")
                    results[code] = 0
                    continue

                rows = self.db.insert_or_ignore(df, "index_daily")
                results[code] = rows
                logger.info(f"指数 {label}({code}) 写入 {rows} 行")
            except Exception as e:
                logger.error(f"更新指数 {label}({code}) 失败: {e}")
                results[code] = 0

        total_rows = sum(results.values())
        logger.info(
            f"Benchmark 指数更新完成: {len(results)} 个指数, 共写入 {total_rows} 行"
        )
        return results

    def update_stock_info(self) -> int:
        """更新全市场股票基本信息（行业/市值等），写入 stock_info 表。

        全量替换：每次更新是当前时刻的快照。
        写完后自动调用 :meth:`update_market_cap` 补齐市值字段。
        """
        logger.info("开始更新股票基本信息")
        try:
            df = call_with_retries(
                self._source.get_stock_info,
                label="source.get_stock_info",
            )
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            return 0

        if df.empty:
            logger.warning("股票基本信息返回空数据")
            return 0

        if "list_date" in df.columns:
            df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce").dt.date
        if "out_date" in df.columns:
            df["out_date"] = pd.to_datetime(df["out_date"], errors="coerce").dt.date
        for col in ("total_shares", "float_shares", "total_cap", "float_cap"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["updated_at"] = pd.Timestamp.now()
        rows = 0
        try:
            self.db.truncate("stock_info")
            rows = self.db.insert_or_ignore(df, "stock_info")
            logger.info(f"股票基本信息更新完成: {rows} 只")
        except Exception as e:
            logger.error(f"写入股票基本信息失败: {e}")
            return 0

        # 自动补齐市值（当数据源未提供时）
        cap_cols = [c for c in ("total_cap", "float_cap") if c in df.columns]
        if cap_cols and df[cap_cols].isna().all().all():
            n_cap = self.update_market_cap()
            logger.info(f"市值补齐完成: {n_cap} 只")

        # truncate + insert 会清空 data_start/data_end，从 daily_bars 恢复
        n_range = self.db.refresh_daily_date_ranges()
        if n_range > 0:
            logger.info(f"日期范围恢复: {n_range} 只")

        return rows

    def update_market_cap(self) -> int:
        """从日线收盘价 × 股本计算并更新市值字段。

        仅更新 stock_info 表中 total_shares 或 float_shares 不为 NULL 的股票。
        """
        if not self.db.table_exists("daily_bars"):
            logger.warning("daily_bars 表不存在，无法计算市值")
            return 0
        if not self.db.table_exists("stock_info"):
            logger.warning("stock_info 表不存在，无法计算市值")
            return 0

        logger.info("计算股票市值（close × shares）...")
        try:
            # DuckDB UPDATE ... FROM 子查询（目标表列不加别名前缀）
            self.db.execute("""
                UPDATE stock_info
                SET
                    total_cap = stock_info.total_shares * d.close,
                    float_cap = stock_info.float_shares * d.close
                FROM (
                    SELECT code, close
                    FROM daily_bars
                    WHERE (code, date) IN (
                        SELECT code, MAX(date) FROM daily_bars GROUP BY code
                    )
                ) d
                WHERE stock_info.code = d.code
                  AND stock_info.total_shares IS NOT NULL
            """)

            result = self.db.query(
                "SELECT COUNT(*) AS n FROM stock_info WHERE total_cap IS NOT NULL"
            )
            n = int(result["n"].iloc[0]) if not result.empty else 0
            logger.info(f"市值更新完成: {n} 只")
            return n
        except Exception as e:
            logger.error(f"市值计算失败: {e}")
            return 0

    def update_adj_factors(self, codes: list[str] | None = None) -> int:
        """为 daily_bars 批量补齐后复权因子（backAdjustFactor）。

        对每只股票查询 ``query_adjust_factor``，通过 merge_asof
        前向填充到每个交易日，一次性 UPDATE 该股票的全部日线。
        """
        if not self.db.table_exists("daily_bars"):
            logger.warning("daily_bars 表不存在")
            return 0

        if codes is None:
            result = self.db.query("SELECT DISTINCT code FROM daily_bars")
            if result.empty:
                logger.warning("daily_bars 表为空")
                return 0
            codes = result["code"].astype(str).str.zfill(6).tolist()

        logger.info(f"开始补齐复权因子: {len(codes)} 只股票")
        updated = 0

        for i, code in enumerate(codes, 1):
            code = normalize_stock_code(code)
            try:
                adj_df = self._source.get_adjust_factor(
                    code, start_date="1990-01-01", end_date="2099-12-31",
                )
                if adj_df.empty or "backAdjustFactor" not in adj_df.columns:
                    continue

                adj_df["dividOperateDate"] = pd.to_datetime(adj_df["dividOperateDate"])
                adj_df["backAdjustFactor"] = pd.to_numeric(
                    adj_df["backAdjustFactor"], errors="coerce"
                )
                adj_df = adj_df.sort_values("dividOperateDate")

                daily = self.db.query(
                    "SELECT date FROM daily_bars WHERE code = ? ORDER BY date",
                    [code],
                )
                if daily.empty:
                    continue
                daily["date"] = pd.to_datetime(daily["date"])
                daily = daily.sort_values("date")

                daily = pd.merge_asof(
                    daily,
                    adj_df[["dividOperateDate", "backAdjustFactor"]],
                    left_on="date",
                    right_on="dividOperateDate",
                    direction="backward",
                )
                daily["adj_factor"] = daily["backAdjustFactor"].fillna(1.0)
                daily = daily[["date", "adj_factor"]]
                daily["date"] = daily["date"].dt.date

                # 注册临时表，一条 UPDATE FROM 完成该股全部日期更新
                self.db.conn.register("_tmp_adj", daily)
                self.db.execute(f"""
                    UPDATE daily_bars
                    SET adj_factor = _tmp_adj.adj_factor
                    FROM _tmp_adj
                    WHERE daily_bars.code = '{code}'
                      AND daily_bars.date = _tmp_adj.date
                """)
                self.db.conn.unregister("_tmp_adj")
                updated += 1
            except Exception as e:
                logger.warning(f"{code} 复权因子更新失败: {e}")

            if i % 300 == 0 or i == len(codes):
                logger.info(f"  [{i}/{len(codes)}] 复权因子进度")

        logger.info(f"复权因子补齐完成: {updated} 只")
        return updated

    def update_financials(
        self,
        codes: list[str] | None = None,
        years: list[int] | None = None,
        quarters: list[int] | None = None,
    ) -> int:
        """批量更新季频财务数据。

        Parameters
        ----------
        codes : list[str], optional
            股票代码列表。省略时取全部 A 股。
        years : list[int], optional
            年份列表。省略时取当前年份。
        quarters : list[int], optional
            季度列表。省略时取全部 4 个季度。

        Returns
        -------
        int
            写入的记录数。
        """
        if codes is None:
            codes = self._get_all_a_codes()
            if not codes:
                logger.warning("无法获取股票列表，跳过财报更新")
                return 0

        # 从 stock_info 过滤掉退市股（退市股无新财报）
        try:
            if self.db.table_exists("stock_info"):
                live = self.db.query(
                    "SELECT code FROM stock_info WHERE status = 1 OR status IS NULL"
                )
                if not live.empty:
                    live_codes = set(live["code"].astype(str).str.zfill(6))
                    codes = [c for c in codes if c in live_codes]
                    logger.info(f"过滤退市股后: {len(codes)} 只")
        except Exception as e:
            logger.debug(f"退市过滤跳过: {e}")

        logger.info(f"开始更新财报: {len(codes)} 只股票")
        df = self._source.get_financials(codes, years=years, quarters=quarters)

        if df.empty:
            logger.warning("未获取到财报数据")
            return 0

        # 类型转换
        for col in ("report_date", "pub_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        numeric_cols = [
            "roe", "eps", "net_profit", "revenue", "gp_margin", "np_margin",
            "total_shares", "float_shares",
            "growth_equity", "growth_asset", "growth_ni",
            "current_ratio", "debt_ratio",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Upsert（报告期数据可能被修订，覆盖写入）
        rows = self.db.upsert(df, "financials")
        logger.info(f"财报更新完成: {rows} 条记录, {df['code'].nunique()} 只股票")
        return rows

    # ------------------------------------------------------------------
    # 内部：代码列表
    # ------------------------------------------------------------------

    def _get_all_a_codes(
        self, include_delisted: bool = False, end_date: dt.date | None = None,
    ) -> list[str]:
        """获取全部 A 股代码（优先从本地 stock_info 读取）。

        Parameters
        ----------
        include_delisted : bool
            False: 仅当前上市股票。
            True:  含退市股。
        end_date : dt.date, optional
            目标区间结束日期，自动跳过上市日晚于此日期的股票。
        """
        # 优先从本地 stock_info 读取（已由 update_stock_info 写入）
        if self.db.table_exists("stock_info"):
            try:
                conditions = ["1=1"]
                if not include_delisted:
                    conditions.append("(status = 1 OR status IS NULL)")
                if end_date:
                    conditions.append(f"list_date <= '{end_date}'")
                where = " AND ".join(conditions)
                df = self.db.query(f"SELECT code, status FROM stock_info WHERE {where}")
                if not df.empty:
                    codes = df["code"].astype(str).str.zfill(6).tolist()
                    n_listed = (df["status"] == 1).sum() if "status" in df.columns else len(codes)
                    logger.info(
                        f"全部 A 股（本地 stock_info）: {len(codes)} 只 "
                        f"（上市: {n_listed}, "
                        f"退市: {len(codes) - n_listed}）"
                    )
                    return codes
            except Exception as e:
                logger.debug(f"本地 stock_info 读取失败: {e}")

        # fallback: 从数据源获取
        try:
            fn = self._source.get_stock_info if include_delisted else self._source.get_stock_list
            df = call_with_retries(fn, label=f"source.{fn.__name__}")
        except Exception as e:
            logger.error(f"获取 A 股列表失败: {e}")
            return []
        if df.empty:
            return []

        if include_delisted and "out_date" in df.columns:
            df["out_date"] = pd.to_datetime(df["out_date"], errors="coerce")
            cutoff = pd.Timestamp("2010-01-01")
            df = df[df["out_date"].isna() | (df["out_date"] >= cutoff)]
        if end_date and "list_date" in df.columns:
            df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
            df = df[df["list_date"] <= pd.Timestamp(str(end_date))]

        codes = df["code"].astype(str).str.zfill(6).tolist()
        logger.info(f"全部 A 股（数据源）: {len(codes)} 只")
        return codes

    # ------------------------------------------------------------------
    # 内部：日期决策
    # ------------------------------------------------------------------

    def _resolve_start_date(
        self,
        code: str,
        user_start: str | dt.date | None,
        date_ranges: dict[str, tuple] | None = None,
    ) -> dt.date:
        """决定某只股票的拉取起始日期（增量更新）。

        优先使用预加载的 date_ranges（O(1) 查 stock_info.data_end），
        fallback 到 SELECT MAX(date) 查询。
        """
        if user_start is None:
            # 快速路径：从 stock_info 预加载的日期范围取 data_end
            if date_ranges is not None:
                entry = date_ranges.get(code)
                if entry is not None:
                    _, de = entry
                    if de is not None:
                        de_date = ensure_date(de)
                        if de_date:
                            return de_date + dt.timedelta(days=1)

            # 慢速路径：直接查 daily_bars
            latest = self.db.get_latest_date("daily_bars", code)
            if latest:
                latest_date = ensure_date(latest)
                if latest_date:
                    return latest_date + dt.timedelta(days=1)
        fallback = user_start or self.cfg.get("data_fetch.start_date", "2020-01-01")
        return ensure_date(fallback) or dt.date(2020, 1, 1)

    def _resolve_index_start_date(
        self,
        code: str,
        user_start: str | dt.date | None,
    ) -> dt.date:
        """决定指数的拉取起始日期（增量更新）。"""
        if user_start is None:
            latest = self.db.get_latest_date("index_daily", code)
            if latest:
                latest_date = ensure_date(latest)
                if latest_date:
                    return latest_date + dt.timedelta(days=1)
        fallback = user_start or self.cfg.get("data_fetch.start_date", "2020-01-01")
        return ensure_date(fallback) or dt.date(2020, 1, 1)

    # ------------------------------------------------------------------
    # 内部：批量更新核心逻辑
    # ------------------------------------------------------------------

    def _batch_update(
        self,
        codes: list[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """批量拉取 & 入库。

        自动检测数据源能力：支持批量接口（get_daily_bars_batch）时走批量通道，
        否则走逐只并发通道。
        """
        adjust = adjust or self.cfg.get("data_fetch.adjust", "hfq")
        end_date = ensure_date(end_date) or dt.date.today()
        codes = [normalize_stock_code(c) for c in codes]

        date_ranges = self.db.get_date_ranges()

        # 过滤已是最新的股票
        pending_codes = []
        for code in codes:
            sd = self._resolve_start_date(code, start_date, date_ranges=date_ranges)
            if sd <= end_date:
                pending_codes.append(code)

        if not pending_codes:
            logger.info("所有股票已是最新，无需更新")
            return {}

        # 检测是否支持批量拉取
        if hasattr(self._source, "get_daily_bars_batch"):
            return self._batch_update_via_batch(
                pending_codes, start_date, end_date, adjust, date_ranges,
            )

        return self._batch_update_per_stock(
            pending_codes, start_date, end_date, adjust, date_ranges,
        )

    def _batch_update_via_batch(
        self,
        codes: list[str],
        start_date: str | dt.date | None,
        end_date: dt.date,
        adjust: str,
        date_ranges: dict[str, tuple],
    ) -> dict[str, int]:
        """通过数据源的 get_daily_bars_batch 批量拉取（100 只/批）。"""
        batch_size = 100
        results: dict[str, int] = {}
        failed: list[str] = []
        total = len(codes)
        t0 = __import__("time").monotonic()

        for i in range(0, total, batch_size):
            chunk = codes[i:i + batch_size]
            batch_no = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            # 每只股票的起始日期（增量更新）
            fetch_codes = []
            for code in chunk:
                sd = self._resolve_start_date(code, start_date, date_ranges=date_ranges)
                if sd <= end_date:
                    fetch_codes.append(code)

            if not fetch_codes:
                continue

            logger.info(
                f"[批量 {batch_no}/{total_batches}] 拉取 {len(fetch_codes)} 只..."
            )
            try:
                batch_result = self._source.get_daily_bars_batch(
                    fetch_codes, start_date, end_date, adjust,
                )
            except Exception as e:
                logger.error(f"批量拉取失败: {e}")
                failed.extend(fetch_codes)
                continue

            for code in fetch_codes:
                df = batch_result.get(code)
                if df is None or df.empty:
                    failed.append(code)
                    continue
                try:
                    df = self.cleaner.clean_pipeline(df)
                    rows = self.db.insert_or_ignore(df, "daily_bars")
                    results[code] = rows
                except Exception as e:
                    failed.append(code)
                    logger.warning(f"{code} 写入失败: {e}")

            written = len(results)
            elapsed = __import__("time").monotonic() - t0
            logger.info(
                f"[批量 {batch_no}/{total_batches}] 完成: {written}/{total} 只"
                f" ({written / elapsed:.1f}/s)"
            )

        if results:
            self.db.refresh_daily_date_ranges(list(results.keys()))

        elapsed = __import__("time").monotonic() - t0
        logger.info(
            f"批量更新完成: 共 {total} 只, 成功 {len(results)}, "
            f"失败 {len(failed)}, 耗时 {elapsed:.1f}s"
        )
        if failed:
            logger.warning(f"失败代码 (前20): {failed[:20]}")
        return results

    def _batch_update_per_stock(
        self,
        codes: list[str],
        start_date: str | dt.date | None,
        end_date: dt.date,
        adjust: str,
        date_ranges: dict[str, tuple],
    ) -> dict[str, int]:
        """逐只并发拉取（BaoStock 等不支持批量的源）。"""
        results: dict[str, int] = {}
        failed: list[str] = []

        fetch_timeout = int(self.cfg.get("data_fetch.timeout", 30))
        max_workers = int(self.cfg.get("data_fetch.max_workers", 1))
        q_maxsize = int(self.cfg.get("data_fetch.queue_maxsize", 200))

        def _fetch(code: str) -> pd.DataFrame:
            sd = self._resolve_start_date(code, start_date, date_ranges=date_ranges)
            if sd > end_date:
                return pd.DataFrame()
            return self._source.get_daily_bars(code, sd, end_date, adjust)

        def _consume(
            code: str, df: pd.DataFrame | None, err: Exception | None,
        ) -> None:
            if err is not None:
                failed.append(code)
                logger.warning(f"{code} 拉取出错: {err}")
                return
            if df is None or df.empty:
                failed.append(code)
                logger.warning(f"{code} 返回空数据")
                return
            try:
                df = self.cleaner.clean_pipeline(df)
                rows = self.db.insert_or_ignore(df, "daily_bars")
                results[code] = rows
            except Exception as e:
                failed.append(code)
                logger.warning(f"{code} 写入失败: {e}")

        total, elapsed = parallel_fetch_serial_consume(
            items=codes,
            fetch_fn=_fetch,
            consume_fn=_consume,
            max_workers=max_workers,
            queue_maxsize=q_maxsize,
            fetch_timeout=fetch_timeout,
            progress_interval=100,
            label="日线更新",
        )

        if results:
            self.db.refresh_daily_date_ranges(list(results.keys()))

        logger.info(
            f"批量更新完成: 共 {total} 只, 成功 {len(results)}, "
            f"失败 {len(failed)}, 耗时 {elapsed:.1f}s"
        )
        if failed:
            logger.warning(f"失败代码 (前20): {failed[:20]}")
        return results


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="stockQuant 日线数据更新器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  %(prog)s                                # 更新全部 A 股日线 (默认)
  %(prog)s --mode stock_info              # 更新股票基本信息（行业/市值/退市标记）
  %(prog)s --mode benchmark               # 更新 Benchmark 指数行情
  %(prog)s --mode index --index-code 000300
  %(prog)s --mode codes --codes 000001 600519
  %(prog)s --mode financials              # 更新季频财报数据
  %(prog)s --no-star --no-bse             # 排除科创板和北交所
""",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "stock_info", "benchmark", "index", "codes", "financials", "adj_factors"),
        default="all",
        help="更新模式 (默认: all)",
    )
    parser.add_argument(
        "--index-code",
        help="mode=index 时指定指数代码，如 000300",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        help="mode=codes 时指定股票代码列表",
    )
    parser.add_argument("--start-date", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--adjust",
        choices=("qfq", "hfq", "none"),
        help="复权方式",
    )
    parser.add_argument(
        "--no-star",
        dest="include_star",
        action="store_false",
        help="排除科创板",
    )
    parser.add_argument(
        "--no-bse",
        dest="include_bse",
        action="store_false",
        help="排除北交所",
    )
    parser.add_argument(
        "--no-gem",
        dest="include_gem",
        action="store_false",
        help="排除创业板",
    )
    parser.add_argument(
        "--include-delisted",
        action="store_true",
        help="纳入已退市股票（默认仅当前上市）。mode=all 时生效。",
    )

    args = parser.parse_args()
    updater = DataUpdater()

    try:
        include_delisted = getattr(args, "include_delisted", False)

        if args.mode == "all":
            res = updater.update_all_daily(
                start_date=args.start_date,
                end_date=args.end_date,
                adjust=args.adjust,
                include_star=args.include_star,
                include_bse=args.include_bse,
                include_gem=args.include_gem,
                include_delisted=include_delisted,
            )
        elif args.mode == "stock_info":
            rows = updater.update_stock_info()
            print(f"股票基本信息更新完成: {rows} 只")
            return
        elif args.mode == "benchmark":
            bm_codes = args.codes if args.codes else None
            res = updater.update_benchmark_indices(
                index_codes=bm_codes,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        elif args.mode == "index":
            if not args.index_code:
                parser.error("--index-code 在 mode=index 时必须提供")
            res = updater.update_index_daily(
                args.index_code,
                start_date=args.start_date,
                end_date=args.end_date,
                adjust=args.adjust,
                include_star=args.include_star,
                include_bse=args.include_bse,
                include_gem=args.include_gem,
            )
        elif args.mode == "codes":
            if not args.codes:
                parser.error("--codes 在 mode=codes 时不能为空")
            res = updater.update_codes_daily(
                args.codes,
                start_date=args.start_date,
                end_date=args.end_date,
                adjust=args.adjust,
            )
        elif args.mode == "financials":
            rows = updater.update_financials(codes=args.codes or None)
            print(f"财报更新完成: {rows} 条记录")
            return
        elif args.mode == "adj_factors":
            rows = updater.update_adj_factors(codes=args.codes or None)
            print(f"复权因子补齐完成: {rows} 只")
            return
        else:
            parser.error(f"未知 mode: {args.mode}")
            return

        total_written = sum(res.values()) if res else 0
        print(f"更新完成: {len(res)} 支, 写入 {total_written} 行")

    except Exception as e:
        logger.exception(f"更新过程中发生错误: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
