"""
配置管理器 — 读取 YAML 配置并提供全局访问。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

# 项目根目录（config/default.yaml 所在目录的上一级）
PROJECT_ROOT: Path = _DEFAULT_CONFIG_PATH.parent.parent


class Config:
    """分层配置管理器。

    加载顺序:
        1. config/default.yaml        (框架默认值)
        2. 用户指定的 yaml 文件       (可选覆盖)
        3. 环境变量 SQ_*              (最高优先级)
    """

    _instance: Config | None = None
    _data: dict[str, Any] = {}

    def __new__(cls, *args, **kwargs) -> Config:  # noqa: D401
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str | Path | None = None) -> None:
        if self._data:
            return
        self._load_default()
        if config_path:
            self._merge(self._read_yaml(Path(config_path)))
        self._apply_env_overrides()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """用点分路径获取配置值，如 ``config.get("backtest.initial_capital")``。"""
        keys = key.split(".")
        node: Any = self._data
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k)
            else:
                return default
            if node is None:
                return default
        return node

    def resolve_path(self, key: str, default: str = ".") -> Path:
        """获取配置中的路径值，并将相对路径解析为相对于项目根目录的绝对路径。

        Parameters
        ----------
        key : str
            点分路径配置键，如 ``"database.path"``。
        default : str
            配置不存在时的默认值。

        Returns
        -------
        Path
            解析后的绝对路径。
        """
        raw = self.get(key, default)
        p = Path(raw)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def set(self, key: str, value: Any) -> None:
        """运行时动态设置配置项。"""
        keys = key.split(".")
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        """返回完整配置字典的深拷贝。"""
        import copy
        return copy.deepcopy(self._data)

    @classmethod
    def reset(cls) -> None:
        """重置单例（主要用于测试）。"""
        cls._instance = None
        cls._data = {}

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_default(self) -> None:
        if _DEFAULT_CONFIG_PATH.exists():
            self._data = self._read_yaml(_DEFAULT_CONFIG_PATH)

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _merge(self, override: dict) -> None:
        """递归合并配置。"""
        self._data = _deep_merge(self._data, override)

    def _apply_env_overrides(self) -> None:
        """以 ``SQ_`` 开头的环境变量可覆盖配置。

        例: ``SQ_BACKTEST__INITIAL_CAPITAL=500000`` → ``backtest.initial_capital = 500000``
        """
        prefix = "SQ_"
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix):
                cfg_key = env_key[len(prefix):].lower().replace("__", ".")
                # 尝试自动转换类型
                self.set(cfg_key, _auto_cast(env_val))


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 优先。"""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _auto_cast(value: str) -> Any:
    """将字符串自动转换为合适的 Python 类型。"""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
