"""
数据更新器 — 批量更新日线数据。

典型用法::

    from stockquant.data.updater import DataUpdater

    updater = DataUpdater()

    # 最常用：更新全部 A 股日线
    updater.update_all_daily()

    # 最常用：更新 Benchmark 指数行情（沪深300/中证500/1000/2000）
    updater.update_benchmark_indices()

    # 更新某个指数成分股日线
    updater.update_index_daily("000300")

    # 更新指定代码
    updater.update_codes_daily(["000001", "600519"])

命令行::

    python -m stockquant.data.updater                  # 默认更新全部 A 股
    python -m stockquant.data.updater --mode benchmark # 更新 Benchmark 指数
    python -m stockquant.data.updater --mode index --index-code 000300
    python -m stockquant.data.updater --mode codes --codes 000001 600519
    python -m stockquant.data.updater --no-star --no-bse   # 排除科创板和北交所
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import queue
import sys
import threading
import time
from typing import Any, Callable, Sequence

import akshare as ak
import pandas as pd

from stockquant.data.data_cleaner import DataCleaner
from stockquant.data.data_source import BaseDataSource, DataSourceFactory
from stockquant.data.database import Database
from stockquant.utils.config import Config
from stockquant.utils.helpers import ensure_date, normalize_stock_code
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


def get_index_constituents(index_code: str) -> list[str]:
    """获取指数成分股代码列表。"""
    index_code = normalize_stock_code(index_code)
    logger.info(f"获取指数 {index_code} 成分股列表")
    try:
        df = _call_with_retries(lambda: ak.index_stock_cons(symbol=index_code))
        codes = df["品种代码"].astype(str).str.zfill(6).drop_duplicates().tolist()
        logger.info(f"指数 {index_code} 成分股: {len(codes)} 只")
        return codes
    except Exception as e:
        logger.error(f"获取指数 {index_code} 成分股失败: {e}")
        return []


def get_all_a_codes() -> list[str]:
    """获取全部 A 股代码列表。"""
    logger.info("获取全部 A 股代码列表")
    try:
        df = _call_with_retries(lambda: ak.stock_info_a_code_name())
        df.columns = ["code", "name"]
        codes = df["code"].astype(str).str.zfill(6).tolist()
        logger.info(f"全部 A 股: {len(codes)} 只")
        return codes
    except Exception as e:
        logger.error(f"获取 A 股列表失败: {e}")
        return []


def _call_with_retries(
    fn: Callable[[], Any],
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Any:
    """带重试的可调用对象执行。"""
    last_exc: Exception | None = None
    cur_delay = delay
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            logger.warning(f"尝试第 {i}/{attempts} 次失败: {e}")
            if i < attempts:
                time.sleep(cur_delay)
                cur_delay *= backoff
    raise last_exc  # type: ignore[misc]


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

        import stockquant.data.source_akshare  # noqa: F401
        primary = self.cfg.get("data_source.primary", "akshare")
        self._source: BaseDataSource = DataSourceFactory.create(primary)

    # ------------------------------------------------------------------
    # 核心公开接口（4 个）
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
    ) -> dict[str, int]:
        """更新全部 A 股日线数据。

        Parameters
        ----------
        start_date / end_date : str | date, optional
            起止日期，默认增量更新。
        adjust : str, optional
            复权方式，默认读取配置（hfq）。
        include_star / include_bse / include_gem : bool
            是否包含科创板 / 北交所 / 创业板，默认全部包含。

        Returns
        -------
        dict[str, int]
            {code: 写入行数}。
        """
        codes = get_all_a_codes()
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
        """更新某个指数成分股的日线数据。

        Parameters
        ----------
        index_code : str
            指数代码，默认沪深300。支持常见代码::

                000300 — 沪深300
                000905 — 中证500
                000852 — 中证1000
                932000 — 中证2000

        Returns
        -------
        dict[str, int]
            {code: 写入行数}。
        """
        codes = get_index_constituents(index_code)
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
        """更新指定代码列表的日线数据。

        Parameters
        ----------
        codes : list[str]
            股票代码列表。

        Returns
        -------
        dict[str, int]
            {code: 写入行数}。
        """
        codes = [normalize_stock_code(c) for c in codes]
        logger.info(f"更新自定义列表日线: {len(codes)} 只")
        return self._batch_update(list(codes), start_date, end_date, adjust)

    def update_benchmark_indices(
        self,
        index_codes: Sequence[str] | None = None,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> dict[str, int]:
        """更新 Benchmark 指数自身的日线行情（写入 index_daily 表）。

        用于回测时的基准对比。

        Parameters
        ----------
        index_codes : list[str], optional
            要更新的指数代码，默认沪深300 / 中证500 / 1000 / 2000。

        Returns
        -------
        dict[str, int]
            {指数代码: 写入行数}。
        """
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
                df = _call_with_retries(
                    lambda c=code, s=sd, e=end_date: self._source.get_index_daily(c, s, e)
                )
                if df is None or df.empty:
                    logger.warning(f"指数 {label}({code}) 返回空数据")
                    results[code] = 0
                    continue

                rows = self.db.save_dataframe(df, "index_daily")
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

    # ------------------------------------------------------------------
    # 内部：日期决策
    # ------------------------------------------------------------------

    def _resolve_start_date(
        self,
        code: str,
        user_start: str | dt.date | None,
    ) -> dt.date:
        """决定某只股票的拉取起始日期（增量更新）。"""
        if user_start is None:
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
        """批量拉取 & 入库（生产者-消费者模型，抓取并发 + 写入串行）。"""
        adjust = adjust or self.cfg.get("data_fetch.adjust", "hfq")
        end_date = ensure_date(end_date) or dt.date.today()

        total = len(codes)
        results: dict[str, int] = {}
        failed: list[str] = []
        t_start = time.time()

        fetch_timeout = int(self.cfg.get("data_fetch.timeout", 30))
        max_workers = int(self.cfg.get("data_fetch.max_workers", 5))
        q_maxsize = int(self.cfg.get("data_fetch.queue_maxsize", 200))

        write_q: queue.Queue[
            tuple[str, pd.DataFrame | None, Exception | None] | None
        ] = queue.Queue(maxsize=q_maxsize)
        lock = threading.Lock()
        processed = 0

        # ---- 写入线程 ----
        def _writer() -> None:
            nonlocal processed
            while True:
                item = write_q.get()
                if item is None:
                    write_q.task_done()
                    break
                w_code, df, err = item
                try:
                    if err is not None:
                        with lock:
                            failed.append(w_code)
                        logger.warning(f"{w_code} 拉取出错: {err}")
                    elif df is None or df.empty:
                        with lock:
                            results[w_code] = 0
                    else:
                        df = self.cleaner.clean_pipeline(df)
                        rows = self.db.save_dataframe(df, "daily_bars")
                        with lock:
                            results[w_code] = rows
                except Exception as e:
                    with lock:
                        failed.append(w_code)
                    logger.warning(f"{w_code} 写入失败: {e}")
                finally:
                    with lock:
                        processed += 1
                        cur = processed
                    write_q.task_done()
                    if cur % 100 == 0 or cur == total:
                        elapsed = time.time() - t_start
                        speed = cur / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"[{cur}/{total}] 成功 {len(results)} | "
                            f"失败 {len(failed)} | {speed:.1f} 只/s"
                        )

        writer_t = threading.Thread(target=_writer, daemon=True)
        writer_t.start()

        # ---- 抓取 ----
        def _fetch_one(code: str) -> None:
            code = normalize_stock_code(code)
            try:
                sd = self._resolve_start_date(code, start_date)
                if sd > end_date:
                    write_q.put((code, pd.DataFrame(), None))
                    return
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    fut = _ex.submit(
                        self._source.get_daily_bars, code, sd, end_date, adjust,
                    )
                    try:
                        df = fut.result(timeout=fetch_timeout)
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        raise TimeoutError(f"拉取 {code} 超时 ({fetch_timeout}s)")
                write_q.put((code, df, None))
            except Exception as e:
                write_q.put((code, None, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_one, c) for c in codes]
            concurrent.futures.wait(futures)

        write_q.join()
        write_q.put(None)
        writer_t.join()

        elapsed = time.time() - t_start
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
  %(prog)s --mode benchmark               # 更新 Benchmark 指数行情
  %(prog)s --mode index --index-code 000300
  %(prog)s --mode codes --codes 000001 600519
  %(prog)s --no-star --no-bse             # 排除科创板和北交所
""",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "benchmark", "index", "codes"),
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

    args = parser.parse_args()
    updater = DataUpdater()

    try:
        if args.mode == "all":
            res = updater.update_all_daily(
                start_date=args.start_date,
                end_date=args.end_date,
                adjust=args.adjust,
                include_star=args.include_star,
                include_bse=args.include_bse,
                include_gem=args.include_gem,
            )
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

