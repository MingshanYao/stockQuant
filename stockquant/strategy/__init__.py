"""策略开发框架"""

from .base_strategy import BaseStrategy, StrategyRegistry
from .examples import DualMAStrategy
from .alpha_factor_strategy import AlphaFactorStrategy

__all__ = [
    "BaseStrategy",
    "StrategyRegistry",
    "DualMAStrategy",
    "AlphaFactorStrategy",
]
