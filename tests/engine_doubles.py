"""Small strategies and cost models used to drive the engine in tests.

These exist only to exercise the engine: fixed or scheduled weights, a recorder
that captures exactly what the engine handed it, deliberately broken outputs,
and a simple proportional cost model (v1 ships only ``ZeroCost``).
"""

from __future__ import annotations

import pandas as pd

from backtester.data import MarketView
from backtester.engine import Order
from backtester.strategies import Strategy


class Constant(Strategy):
    """Returns the same weights on every call."""

    def __init__(self, weights: dict[str, float], warmup: int = 0) -> None:
        self.weights = dict(weights)
        self.warmup = warmup

    def target_weights(self, view: MarketView) -> dict[str, float]:
        return dict(self.weights)


class Scheduled(Strategy):
    """Returns the most recent scheduled weights dated at or before ``view.now`` (flat before)."""

    def __init__(self, schedule: dict[str, dict[str, float]], warmup: int = 0) -> None:
        self.schedule = sorted((pd.Timestamp(d), dict(w)) for d, w in schedule.items())
        self.warmup = warmup

    def target_weights(self, view: MarketView) -> dict[str, float]:
        current: dict[str, float] = {}
        for date, weights in self.schedule:
            if date <= view.now:
                current = weights
        return dict(current)


class EqualWeight(Strategy):
    """1/n in every visible symbol."""

    def target_weights(self, view: MarketView) -> dict[str, float]:
        return {symbol: 1.0 / len(view.symbols) for symbol in view.symbols}


class Recorder(Strategy):
    """Stays flat and records what the engine passes it on every call.

    ``calls`` holds ``(argument type, view.now, bars visible)`` per decision and
    ``fits`` holds ``(argument type, view.now)`` per ``fit`` call. With
    ``lookback`` set, it also asks for that much history, to provoke a warmup error.
    """

    def __init__(self, warmup: int = 0, lookback: int | None = None) -> None:
        self.warmup = warmup
        self.lookback = lookback
        self.calls: list[tuple[type, pd.Timestamp, int]] = []
        self.fits: list[tuple[type, pd.Timestamp]] = []

    def fit(self, view: MarketView) -> None:
        self.fits.append((type(view), view.now))

    def target_weights(self, view: MarketView) -> dict[str, float]:
        symbol = view.symbols[0]
        self.calls.append((type(view), view.now, len(view.history(symbol))))
        if self.lookback is not None:
            view.history(symbol, lookback=self.lookback)
        return {}


class Returns(Strategy):
    """Returns ``output`` verbatim, valid or not."""

    def __init__(self, output: object) -> None:
        self.output = output

    def target_weights(self, view: MarketView) -> dict[str, float]:
        return self.output  # type: ignore[return-value]


class Raises(Strategy):
    """Raises ``RuntimeError`` on every call."""

    def target_weights(self, view: MarketView) -> dict[str, float]:
        raise RuntimeError("model file not found")


class BpsCost:
    """Charges ``bps`` basis points of traded notional. Test-only: v1 ships ``ZeroCost``."""

    def __init__(self, bps: float) -> None:
        self.bps = bps

    def cost(self, order: Order, price: float) -> float:
        return abs(order.quantity) * price * self.bps / 10_000

    def __repr__(self) -> str:
        return f"BpsCost({self.bps})"
