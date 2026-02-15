"""
事件系统 — 回测事件驱动核心。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from collections import defaultdict


class EventType(Enum):
    """事件类型枚举。"""
    MARKET = auto()       # 行情事件
    ORDER = auto()        # 订单事件
    FILL = auto()         # 成交事件
    RISK = auto()         # 风控事件
    TIMER = auto()        # 定时事件


@dataclass
class Event:
    """事件数据。"""

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


class EventBus:
    """简易事件总线。"""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list] = defaultdict(list)

    def register(self, event_type: EventType, handler) -> None:
        self._handlers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers.get(event.event_type, []):
            handler(event)
