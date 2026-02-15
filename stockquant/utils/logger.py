"""
统一日志模块 — 基于 loguru，提供开箱即用的日志能力。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# 移除 loguru 默认 handler，由框架统一管理
logger.remove()

_INITIALIZED = False


def _init_logger() -> None:
    """根据配置初始化日志输出（仅执行一次）。"""
    global _INITIALIZED
    if _INITIALIZED:
        return

    # 延迟导入，避免循环依赖
    from stockquant.utils.config import Config

    cfg = Config()
    log_level = cfg.get("project.log_level", "INFO")
    log_dir = Path(cfg.get("project.log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # 控制台输出
    logger.add(sys.stderr, level=log_level, format=fmt, colorize=True)

    # 文件输出（按天轮转，保留 30 天）
    logger.add(
        str(log_dir / "stockquant_{time:YYYY-MM-DD}.log"),
        level=log_level,
        format=fmt,
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )

    _INITIALIZED = True


def get_logger(name: str = "stockquant") -> Logger:
    """获取带模块名称绑定的 logger。

    Parameters
    ----------
    name : str
        模块名称，会附加到日志消息中以方便区分来源。

    Returns
    -------
    Logger
        loguru 绑定 logger 实例。
    """
    _init_logger()
    return logger.bind(name=name)
