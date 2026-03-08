"""
Alpha 因子研究模块 — 提供系统性研究 Alpha 因子表现的一站式工具。

主要导出
--------
- :class:`AlphaResearcher`  — 因子研究核心工具（因子计算/回测/分析/可视化）
- :class:`AlphaBacktestResult` — 单因子回测结果容器
"""

from .alpha_researcher import AlphaResearcher, AlphaBacktestResult

__all__ = ["AlphaResearcher", "AlphaBacktestResult"]
