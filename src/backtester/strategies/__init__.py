"""Trading strategies: the base interface and v1's two strategies."""

from backtester.strategies.base import Strategy
from backtester.strategies.buy_and_hold import BuyAndHold
from backtester.strategies.sma_crossover import SMACrossover

__all__ = ["BuyAndHold", "SMACrossover", "Strategy"]
