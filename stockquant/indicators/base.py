"""
技术指标基类 — 所有自定义指标继承此类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseIndicator(ABC):
    """指标基类。

    子类需实现 ``compute`` 方法，接收包含 OHLCV 的 DataFrame，
    返回附加了指标列的 DataFrame。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称。"""

    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算指标。

        Parameters
        ----------
        df : DataFrame
            至少包含 date, open, high, low, close, volume 列。

        Returns
        -------
        DataFrame
            原始列 + 新增指标列。
        """

    def __repr__(self) -> str:
        return f"<Indicator: {self.name}>"


class IndicatorRegistry:
    """指标注册表 — 根据名称创建指标实例。"""

    _registry: dict[str, type[BaseIndicator]] = {}

    @classmethod
    def register(cls, name: str, indicator_cls: type[BaseIndicator]) -> None:
        cls._registry[name.lower()] = indicator_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseIndicator:
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"未注册的指标: {name}，可用: {cls.list()}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def get(cls, name: str) -> type[BaseIndicator]:
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"未注册的指标: {name}，可用: {cls.list()}")
        return cls._registry[name]
