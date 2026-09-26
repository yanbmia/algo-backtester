"""Moving-average crossover: long when the short-term trend is above the long-term trend."""

from __future__ import annotations

import math

from backtester.data import MarketView
from backtester.strategies.base import Strategy


class SMACrossover(Strategy):
    """Long one symbol while its fast simple moving average is above its slow one; flat otherwise.

    At each close, with ``closes`` = the last ``slow`` closes (oldest first):

    - fast mean = mean of the last ``fast`` of them
    - slow mean = mean of all ``slow`` of them
    - weight = 1.0 if fast mean > slow mean, else 0.0

    **Ties are flat.** When the two means are exactly equal, the signal "fast
    above slow" is not true, so the strategy holds no position. That includes a
    tie right after being long: the rule is stateless and never looks at its
    previous decision. Means are compared exactly, with no tolerance, and are
    computed with ``math.fsum`` so that equal sums give equal means (with
    integer or other exactly representable prices, a mathematical tie is an
    exact tie). For real, noisy prices an exact tie is vanishingly rare.

    ``warmup`` is ``slow``, so the first decision comes once ``slow`` bars are
    visible. The symbol must trade from the start of the data; if it has fewer
    than ``slow`` bars when the first decision is due, the engine raises a
    ``StrategyError`` rather than trading on a shorter window.

    The defaults (50, 200) are the conventional "golden cross" windows, fixed in
    advance rather than tuned on the data being tested.
    """

    def __init__(self, symbol: str, fast: int = 50, slow: int = 200) -> None:
        if not isinstance(symbol, str) or not symbol:
            raise TypeError(f"symbol must be a non-empty string, got {symbol!r}")
        for label, value in (("fast", fast), ("slow", slow)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an int, got {value!r}")
            if value < 1:
                raise ValueError(f"{label} must be >= 1, got {value}")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")

        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.warmup = slow
        self.name = f"sma_crossover({symbol}, {fast}, {slow})"

    def target_weights(self, view: MarketView) -> dict[str, float]:
        closes = view.history(self.symbol, "close", lookback=self.slow).to_numpy()
        fast_mean = math.fsum(closes[-self.fast :]) / self.fast
        slow_mean = math.fsum(closes) / self.slow
        return {self.symbol: 1.0 if fast_mean > slow_mean else 0.0}
