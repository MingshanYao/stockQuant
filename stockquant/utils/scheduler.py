"""
调度器 — 基于 APScheduler 的定时任务管理。
"""

from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("scheduler")


class Scheduler:
    """定时任务调度器。"""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self._scheduler = BackgroundScheduler()
        self._jobs: dict[str, str] = {}

    def add_cron_job(
        self,
        job_id: str,
        func: Callable,
        cron_expr: str,
        **kwargs,
    ) -> None:
        """添加 Cron 定时任务。

        Parameters
        ----------
        job_id : str
            任务唯一标识。
        func : Callable
            任务函数。
        cron_expr : str
            Cron 表达式，如 "0 18 * * 1-5"。
        """
        parts = cron_expr.strip().split()
        trigger = CronTrigger(
            minute=parts[0] if len(parts) > 0 else "*",
            hour=parts[1] if len(parts) > 1 else "*",
            day=parts[2] if len(parts) > 2 else "*",
            month=parts[3] if len(parts) > 3 else "*",
            day_of_week=parts[4] if len(parts) > 4 else "*",
        )
        self._scheduler.add_job(func, trigger, id=job_id, **kwargs)
        self._jobs[job_id] = cron_expr
        logger.info(f"已添加定时任务: {job_id} [{cron_expr}]")

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("调度器已启动")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown()
            logger.info("调度器已停止")

    def list_jobs(self) -> dict[str, str]:
        return self._jobs.copy()
