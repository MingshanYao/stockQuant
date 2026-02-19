"""
并发工具 — 多线程拉取 + 单线程消费的通用模型。

典型用法::

    from stockquant.utils.concurrent import parallel_fetch_serial_consume

    results = []

    def fetch(url: str) -> bytes:
        return requests.get(url).content

    def consume(url: str, data: bytes | None, err: Exception | None) -> None:
        if err:
            print(f"失败: {url}")
        else:
            results.append(data)

    parallel_fetch_serial_consume(
        items=url_list,
        fetch_fn=fetch,
        consume_fn=consume,
        max_workers=8,
        label="下载",
    )
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from typing import Callable, Sequence, TypeVar

from stockquant.utils.logger import get_logger

logger = get_logger("utils.concurrent")

K = TypeVar("K")
V = TypeVar("V")


def parallel_fetch_serial_consume(
    items: Sequence[K],
    fetch_fn: Callable[[K], V],
    consume_fn: Callable[[K, V | None, Exception | None], None],
    *,
    max_workers: int = 5,
    queue_maxsize: int = 200,
    progress_interval: int = 100,
    label: str = "任务",
    fetch_timeout: float | None = None,
) -> tuple[int, float]:
    """多线程拉取 + 单线程消费的通用并发模型。

    生产者：线程池并发调用 *fetch_fn* 获取数据。
    消费者：单线程串行调用 *consume_fn* 处理结果。
    两者通过有界队列连接，自带背压控制。

    Parameters
    ----------
    items : Sequence[K]
        待处理的任务列表（如股票代码、板块名称等）。
    fetch_fn : (K) -> V
        拉取函数，在线程池中并发执行。抛异常视为该任务失败。
    consume_fn : (key, value | None, error | None) -> None
        消费函数，在独立线程中串行执行。
        三个参数分别为：任务 key、拉取结果（成功时）、异常（失败时）。
        消费函数内部无需加锁（天然单线程串行）。
    max_workers : int
        最大并发拉取线程数，默认 5。
    queue_maxsize : int
        结果队列最大容量（背压），默认 200。
    progress_interval : int
        每处理多少条打印一次进度日志，默认 100。
    label : str
        进度日志中的任务标签。
    fetch_timeout : float, optional
        单个 fetch 的超时时间（秒），为 ``None`` 则不限。

    Returns
    -------
    tuple[int, float]
        (已处理总数, 总耗时秒数)。
    """
    total = len(items)
    if total == 0:
        return 0, 0.0

    result_q: queue.Queue[
        tuple[K, V | None, Exception | None] | None
    ] = queue.Queue(maxsize=queue_maxsize)

    processed = 0
    lock = threading.Lock()
    t_start = time.time()

    # ---- 消费者线程（单线程串行处理） ----
    def _consumer() -> None:
        nonlocal processed
        while True:
            item = result_q.get()
            if item is None:  # 毒丸 → 退出
                result_q.task_done()
                break
            key, value, err = item
            try:
                consume_fn(key, value, err)
            except Exception as exc:
                logger.warning(f"[{label}] 消费失败 [{key}]: {exc}")
            finally:
                with lock:
                    processed += 1
                    cur = processed
                result_q.task_done()
                if cur % progress_interval == 0 or cur == total:
                    elapsed = time.time() - t_start
                    speed = cur / elapsed if elapsed > 0 else 0
                    logger.info(f"[{label}] {cur}/{total} ({speed:.1f}/s)")

    consumer_t = threading.Thread(target=_consumer, daemon=True)
    consumer_t.start()

    # ---- 生产者（线程池并发拉取） ----
    def _fetch_one(key: K) -> None:
        try:
            if fetch_timeout is not None:
                with concurrent.futures.ThreadPoolExecutor(1) as _ex:
                    fut = _ex.submit(fetch_fn, key)
                    try:
                        value = fut.result(timeout=fetch_timeout)
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        raise TimeoutError(f"拉取超时 ({fetch_timeout}s)")
            else:
                value = fetch_fn(key)
            result_q.put((key, value, None))
        except Exception as e:
            result_q.put((key, None, e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one, item) for item in items]
        concurrent.futures.wait(futs)

    # 等消费者处理完所有已入队的结果，再发毒丸终止
    result_q.join()
    result_q.put(None)
    consumer_t.join()

    elapsed = time.time() - t_start
    return total, elapsed
