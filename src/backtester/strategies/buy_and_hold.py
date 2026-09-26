"""Buy and hold: the benchmark every other strategy is compared against."""

from __future__ import annotations

from backtester.data import MarketView
from backtester.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Fully invested in one symbol from the first decision onward.

    Returns ``{symbol: 1.0}`` at every close. The engine buys at the next open
    after the first decision and, since the target never changes, never trades
    again. It runs through the same engine, with the same fills, timing, and
    costs as every other strategy, so comparisons with it are like for like.

    ``warmup`` is 0. The symbol must be visible from the first decision; one
    that lists later is rejected by the engine's weight validation.
    """

    warmup = 0

    def __init__(self, symbol: str) -> None:
        if not isinstance(symbol, str) or not symbol:
            raise TypeError(f"symbol must be a non-empty string, got {symbol!r}")
        self.symbol = symbol
        self.name = f"buy_and_hold({symbol})"

    def target_weights(self, view: MarketView) -> dict[str, float]:
        return {self.symbol: 1.0}
