"""
数据更新器 — 提供按指数成分股 / 板块 / 全市场批量更新日线数据的能力。

典型用法::

    from stockquant.data.updater import DataUpdater

    updater = DataUpdater()

    # 更新沪深300成分股日线（排除科创板 & 北交所 & 创业板）
    updater.update_index_daily("000300", include_star=False, include_bse=False, include_gem=False)

    # 更新中证1000成分股日线
    updater.update_index_daily("000852")

    # 更新全部A股（含科创板 & 北交所）
    updater.update_all_daily()

    # 仅更新科创板
    updater.update_star_daily()

    # 仅更新北交所
    updater.update_bse_daily()
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Sequence

import akshare as ak
import pandas as pd
import argparse
import concurrent.futures
import queue
import threading
import sys
from typing import Callable, Any

from stockquant.data.data_source import DataSourceFactory, BaseDataSource
from stockquant.data.database import Database
from stockquant.data.data_cleaner import DataCleaner
from stockquant.utils.config import Config
from stockquant.utils.helpers import ensure_date, normalize_stock_code
from stockquant.utils.logger import get_logger

logger = get_logger("data.updater")

# ======================================================================
# 常用指数代码
# ======================================================================
INDEX_HS300 = "000300"   # 沪深300
INDEX_CSI500 = "000905"  # 中证500
INDEX_CSI1000 = "000852"  # 中证1000
INDEX_CSI2000 = "932000"  # 中证2000

# ======================================================================
# 板块代码前缀
# ======================================================================
_STAR_PREFIXES = ("688",)       # 科创板
_BSE_PREFIXES = ("4", "8")      # 北交所 (43xxxx / 83xxxx / 87xxxx 等)
_MAIN_SH_PREFIXES = ("60",)     # 沪市主板
_MAIN_SZ_PREFIXES = ("00",)     # 深市主板
_GEM_PREFIXES = ("30",)         # 创业板


# ======================================================================
# 工具函数
# ======================================================================

def _is_star(code: str) -> bool:
    """判断是否为科创板股票。"""
    return code.startswith(_STAR_PREFIXES)


def _is_bse(code: str) -> bool:
    """判断是否为北交所股票。"""
    return code.startswith(_BSE_PREFIXES) and not code.startswith(_STAR_PREFIXES)


def _is_gem(code: str) -> bool:
    """判断是否为创业板股票。"""
    return code.startswith(_GEM_PREFIXES)


def filter_codes(
    codes: list[str],
    *,
    include_star: bool = True,
    include_bse: bool = True,
    include_gem: bool = True,
) -> list[str]:
    """根据板块过滤股票代码列表。

    Parameters
    ----------
    codes : list[str]
        原始代码列表（6 位纯数字）。
    include_star : bool
        是否包含科创板 (688xxx)，默认 True。
    include_bse : bool
        是否包含北交所 (4xxxxx / 8xxxxx)，默认 True。
    include_gem : bool
        是否包含创业板 (30xxxx)，默认 True。

    Returns
    -------
    list[str]
        过滤后的代码列表。
    """
    result = []
    for c in codes:
        if not include_star and _is_star(c):
            continue
        if not include_bse and _is_bse(c):
            continue
        if not include_gem and _is_gem(c):
            continue
        result.append(c)
    return result


def get_index_constituents(index_code: str) -> list[str]:
    """获取指数成分股代码列表。

    Parameters
    ----------
    index_code : str
        指数代码，如 "000300"。

    Returns
    -------
    list[str]
        成分股代码列表（6 位纯数字，已去重）。
    """
    index_code = normalize_stock_code(index_code)
    logger.info(f"获取指数 {index_code} 成分股列表")

    def _call() -> pd.DataFrame:
        return ak.index_stock_cons(symbol=index_code)

    try:
        df = _call_with_retries(_call)
        codes = df["品种代码"].astype(str).str.zfill(6).drop_duplicates().tolist()
        logger.info(f"指数 {index_code} 成分股: {len(codes)} 只")
        return codes
    except Exception as e:
        logger.error(f"获取指数 {index_code} 成分股失败: {e}")
        return []


def get_all_a_codes() -> list[str]:
    """获取全部 A 股代码列表。"""
    logger.info("获取全部 A 股代码列表")

    def _call() -> pd.DataFrame:
        return ak.stock_info_a_code_name()

    try:
        df = _call_with_retries(_call)
        df.columns = ["code", "name"]
        codes = df["code"].astype(str).str.zfill(6).tolist()
        logger.info(f"全部 A 股: {len(codes)} 只")
        return codes
    except Exception as e:
        logger.error(f"获取 A 股列表失败: {e}")
        return []


def _call_with_retries(
    fn: Callable[[], Any], attempts: int = 3, delay: float = 1.0, backoff: float = 2.0
) -> Any:
    """在发生异常时重试调用可调用对象。

    在多数网络抖动场景下可减少临时错误导致的失败。
    """
    last_exc: Exception | None = None
    cur_delay = delay
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            logger.warning(f"尝试第 {i}/{attempts} 次失败: {e}")
            if i == attempts:
                break
            time.sleep(cur_delay)
            cur_delay *= backoff
    # 最后抛出最近一次异常以便上层处理和记录
    raise last_exc


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

        # 加载数据源
        import stockquant.data.source_akshare  # noqa: F401
        primary = self.cfg.get("data_source.primary", "akshare")
        self._source: BaseDataSource = DataSourceFactory.create(primary)

    # ------------------------------------------------------------------
    # 公开接口：按指数更新
    # ------------------------------------------------------------------

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
            指数代码，默认沪深300。
        start_date / end_date : str | date, optional
            起止日期，默认使用配置或增量更新。
        adjust : str, optional
            复权方式。
        include_star : bool
            是否包含科创板，默认 True。
        include_bse : bool
            是否包含北交所，默认 True。
        include_gem : bool
            是否包含创业板，默认 True。

        Returns
        -------
        dict[str, int]
            {code: 写入行数} 的映射。
        """
        codes = get_index_constituents(index_code)
        codes = filter_codes(codes, include_star=include_star, include_bse=include_bse, include_gem=include_gem)
        logger.info(
            f"更新指数 {index_code} 成分股日线: {len(codes)} 只 "
            f"(include_star={include_star}, include_bse={include_bse}, include_gem={include_gem})"
        )
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_hs300_daily(self, **kwargs) -> dict[str, int]:
        """更新沪深300成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_HS300, **kwargs)

    def update_csi500_daily(self, **kwargs) -> dict[str, int]:
        """更新中证500成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_CSI500, **kwargs)

    def update_csi1000_daily(self, **kwargs) -> dict[str, int]:
        """更新中证1000成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_CSI1000, **kwargs)

    def update_csi2000_daily(self, **kwargs) -> dict[str, int]:
        """更新中证2000成分股日线数据（快捷方法）。"""
        return self.update_index_daily(INDEX_CSI2000, **kwargs)

    # ------------------------------------------------------------------
    # 公开接口：全市场更新
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
        include_star : bool
            是否包含科创板，默认 True。
        include_bse : bool
            是否包含北交所，默认 True。
        include_gem : bool
            是否包含创业板，默认 True。
        """
        codes = get_all_a_codes()
        codes = filter_codes(codes, include_star=include_star, include_bse=include_bse, include_gem=include_gem)
        logger.info(
            f"更新全部 A 股日线: {len(codes)} 只 "
            f"(include_star={include_star}, include_bse={include_bse}, include_gem={include_gem})"
        )
        return self._batch_update(codes, start_date, end_date, adjust)

    # ------------------------------------------------------------------
    # 公开接口：按板块更新
    # ------------------------------------------------------------------

    def update_star_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新科创板 (688xxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [c for c in codes if _is_star(c)]
        logger.info(f"更新科创板日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_bse_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新北交所 (4xxxxx / 8xxxxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [c for c in codes if _is_bse(c)]
        logger.info(f"更新北交所日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_main_board_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新沪深主板 (60xxxx / 00xxxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [
            c for c in codes
            if c.startswith(_MAIN_SH_PREFIXES) or c.startswith(_MAIN_SZ_PREFIXES)
        ]
        logger.info(f"更新沪深主板日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    def update_gem_daily(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
    ) -> dict[str, int]:
        """仅更新创业板 (30xxxx) 股票日线数据。"""
        codes = get_all_a_codes()
        codes = [c for c in codes if c.startswith(_GEM_PREFIXES)]
        logger.info(f"更新创业板日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

    # ------------------------------------------------------------------
    # 公开接口：自定义代码列表
    # ------------------------------------------------------------------

    def update_codes_daily(
        self,
        codes: Sequence[str],
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        adjust: str | None = None,
        *,
        include_star: bool = True,
        include_bse: bool = True,
        include_gem: bool = True,
    ) -> dict[str, int]:
        """更新指定代码列表的日线数据。"""
        codes = [normalize_stock_code(c) for c in codes]
        codes = filter_codes(codes, include_star=include_star, include_bse=include_bse, include_gem=include_gem)
        logger.info(f"更新自定义列表日线: {len(codes)} 只")
        return self._batch_update(codes, start_date, end_date, adjust)

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
        """批量拉取 & 入库日线数据（支持增量更新）。

        Returns
        -------
        dict[str, int]
            {code: 实际写入行数}，失败的股票不包含在内。
        """
        adjust = adjust or self.cfg.get("data_fetch.adjust", "hfq")
        end_date = ensure_date(end_date) or dt.date.today()

        total = len(codes)
        results: dict[str, int] = {}
        failed: list[str] = []
        t_start = time.time()

        fetch_timeout = int(self.cfg.get("data_fetch.timeout", 30))
        max_workers = int(self.cfg.get("data_fetch.max_workers", 5))
        q_maxsize = int(self.cfg.get("data_fetch.queue_maxsize", 200))

        # ---- 结果队列：抓取线程 → 写入线程 ----
        # 元素: (code, DataFrame | None, Exception | None)  或  None 表示结束
        write_q: queue.Queue[tuple[str, pd.DataFrame | None, Exception | None] | None] = (
            queue.Queue(maxsize=q_maxsize)
        )
        lock = threading.Lock()
        processed = 0

        # ---- 写入线程：串行写 DB ----
        def _writer() -> None:
            nonlocal processed
            while True:
                item = write_q.get()
                if item is None:          # 毒丸：退出信号
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
                            f"[{cur}/{total}] 已完成 | 成功 {len(results)} | "
                            f"失败 {len(failed)} | 速率 {speed:.1f} 只/s"
                        )

        writer_t = threading.Thread(target=_writer, daemon=True)
        writer_t.start()

        # ---- 抓取函数（在线程池中执行） ----
        def _fetch_one(code: str) -> None:
            code = normalize_stock_code(code)
            try:
                sd = self._resolve_start_date(code, start_date)
                if sd > end_date:
                    write_q.put((code, pd.DataFrame(), None))
                    return
                # 内部用单独线程 + timeout 防止单次请求卡死
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    fut = _ex.submit(
                        self._source.get_daily_bars, code, sd, end_date, adjust
                    )
                    try:
                        df = fut.result(timeout=fetch_timeout)
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        raise TimeoutError(f"拉取 {code} 超时 ({fetch_timeout}s)")
                write_q.put((code, df, None))
            except Exception as e:
                write_q.put((code, None, e))

        # ---- 并发提交抓取 ----
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_one, c) for c in codes]
            # 等待所有抓取完成（异常已在 _fetch_one 内捕获并入队）
            concurrent.futures.wait(futures)

        # ---- 通知写线程退出并等待 ----
        write_q.join()          # 先等队列消费完
        write_q.put(None)       # 发送毒丸
        writer_t.join()

        elapsed = time.time() - t_start
        logger.info(
            f"批量更新完成: 共 {total} 只, 成功 {len(results)}, "
            f"失败 {len(failed)}, 耗时 {elapsed:.1f}s"
        )
        if failed:
            logger.warning(f"失败代码 (前20): {failed[:20]}")

        return results

    def _resolve_start_date(
        self,
        code: str,
        user_start: str | dt.date | None,
    ) -> dt.date:
        """决定某只股票的拉取起始日期。

        优先级:
            1. 增量更新 — 数据库中最新日期 + 1 天
            2. 用户指定的 start_date
            3. 配置文件中的 data_fetch.start_date
        """
        # 尝试增量
        if user_start is None:
            latest = self.db.get_latest_date("daily_bars", code)
            if latest:
                latest_date = ensure_date(latest)
                if latest_date:
                    return latest_date + dt.timedelta(days=1)

        # 回退到用户指定 / 配置默认
        fallback = user_start or self.cfg.get("data_fetch.start_date", "2020-01-01")
        return ensure_date(fallback) or dt.date(2020, 1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量更新日线数据 (stockquant 数据更新器)")
    parser.add_argument(
        "--mode",
        choices=(
            "hs300",
            "csi500",
            "csi1000",
            "csi2000",
            "index",
            "all",
            "star",
            "bse",
            "main_board",
            "gem",
            "codes",
        ),
        default="hs300",
        help="更新模式，默认 hs300",
    )
    parser.add_argument("--index-code", help="当 mode=index 时，指定指数代码，例如 000300")
    parser.add_argument(
        "--codes",
        nargs="+",
        help="当 mode=codes 时，指定要更新的股票代码列表（空格分隔，多个）",
    )
    parser.add_argument("--start-date", help="起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--adjust", choices=("qfq", "hfq", "none"), help="复权方式")
    parser.add_argument("--no-star", dest="include_star", action="store_false", help="排除科创板")
    parser.add_argument("--no-bse", dest="include_bse", action="store_false", help="排除北交所")
    parser.add_argument("--no-gem", dest="include_gem", action="store_false", help="排除创业板")

    args = parser.parse_args()

    updater = DataUpdater()

    try:
        mode = args.mode
        common_kwargs = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "adjust": args.adjust,
            "include_star": args.include_star,
            "include_bse": args.include_bse,
            "include_gem": args.include_gem,
        }

        if mode == "hs300":
            res = updater.update_hs300_daily(**common_kwargs)
        elif mode == "csi500":
            res = updater.update_csi500_daily(**common_kwargs)
        elif mode == "csi1000":
            res = updater.update_csi1000_daily(**common_kwargs)
        elif mode == "csi2000":
            res = updater.update_csi2000_daily(**common_kwargs)
        elif mode == "index":
            if not args.index_code:
                parser.error("--index-code 必须在 mode=index 时提供")
            res = updater.update_index_daily(args.index_code, **common_kwargs)
        elif mode == "all":
            res = updater.update_all_daily(**common_kwargs)
        elif mode == "star":
            res = updater.update_star_daily(**common_kwargs)
        elif mode == "bse":
            res = updater.update_bse_daily(**common_kwargs)
        elif mode == "main_board":
            res = updater.update_main_board_daily(**common_kwargs)
        elif mode == "gem":
            res = updater.update_gem_daily(**common_kwargs)
        elif mode == "codes":
            if not args.codes:
                parser.error("--codes 在 mode=codes 时不能为空")
            res = updater.update_codes_daily(args.codes, **common_kwargs)
        else:
            parser.error(f"未知 mode: {mode}")

        # 打印结果汇总
        total_written = sum(res.values()) if res else 0
        print(f"更新完成: {len(res)} 支股票, 写入行数总计: {total_written}")

    except Exception as e:
        logger.exception(f"更新过程中发生错误: {e}")
        sys.exit(2)
    
