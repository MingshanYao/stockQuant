"""回测 bar 数据结构。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class BarSnapshot:
    """单个交易日的轻量 bar 快照。

    Attributes
    ----------
    date : dt.date
        交易日。
    codes : set[str]
        当日有数据的股票代码集合。
    close : dict[str, float]
        股票代码 → 当日收盘价。
    """

    date: dt.date
    codes: set[str] = field(default_factory=set)
    close: dict[str, float] = field(default_factory=dict)

    def __contains__(self, code: str) -> bool:
        """支持 ``code in bar`` 用法（兼容旧接口）。"""
        return code in self.codes
