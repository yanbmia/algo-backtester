"""Portfolio accounting: cash, share positions, and rebalancing to target weights.

Share sizing: fractional shares
-------------------------------
v1 holds fractional shares. A target weight is reached exactly, which keeps the
engine's accounting identities exact (equity == cash + sum(shares * price)) and
means cash-conservation tests compare dollars to dollars without rounding
residue. Whole-share rounding would add a tracking error that depends on the
share price and account size (about 0.4% of a $100k account in a $400 stock),
which is noise for a strategy-level backtest. Many brokers now offer fractional
shares. If whole shares are needed later, rounding belongs in ``orders_for`` as
a lot-size parameter. Nothing else would change.

The portfolio is a ledger. ``apply`` records a fill: shares change by
``quantity`` and cash changes by ``-(quantity * price + cost)``. Rules about what
the account may do, such as no shorting and no leverage, are enforced where the
trades are decided: target weights are validated before any order is sized, and
the engine checks the account after every rebalance.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np

#: How far above 1 the sum of target weights may be, to absorb float noise
#: (e.g. three weights of 1/3). Any real overshoot is leverage and is rejected.
WEIGHT_TOLERANCE: float = 1e-9

#: Trades smaller than this fraction of portfolio value are float noise (e.g.
#: re-targeting an unchanged 100% position) and are not generated at all.
MIN_TRADE_FRACTION: float = 1e-9


@dataclass(frozen=True, slots=True)
class Order:
    """A request to change a position. ``quantity`` is in shares: positive buys, negative sells."""

    symbol: str
    quantity: float


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed order. ``cost`` is the total fee/slippage paid in cash (>= 0)."""

    symbol: str
    quantity: float
    price: float
    cost: float


def weight_problems(weights: object, allowed_symbols: Collection[str] | None = None) -> list[str]:
    """Everything wrong with a set of target weights under v1's long/flat, unlevered rules.

    Valid weights are a mapping of symbol -> real number, each in [0, 1], summing
    to at most 1 (plus ``WEIGHT_TOLERANCE``). If ``allowed_symbols`` is given,
    every key must be in it. Returns an empty list when the weights are valid.
    """
    if not isinstance(weights, Mapping):
        return [f"expected a dict of symbol -> weight, got {type(weights).__name__}"]

    problems: list[str] = []
    for symbol, weight in weights.items():
        if not isinstance(symbol, str):
            problems.append(f"key {symbol!r} is not a symbol string")
            continue
        if allowed_symbols is not None and symbol not in allowed_symbols:
            problems.append(
                f"{symbol!r} is not a visible symbol (visible: {sorted(allowed_symbols)})"
            )
        if isinstance(weight, bool | np.bool_) or not isinstance(weight, numbers.Real):
            problems.append(f"{symbol}: weight {weight!r} is not a number")
            continue
        w = float(weight)
        if not math.isfinite(w):
            problems.append(f"{symbol}: weight {w} is not finite")
        elif w < 0:
            problems.append(f"{symbol}: weight {w} is negative (shorting is not supported in v1)")
        elif w > 1 + WEIGHT_TOLERANCE:
            problems.append(f"{symbol}: weight {w} is above 1 (leverage is not supported in v1)")

    if not problems:
        total = math.fsum(float(w) for w in weights.values())
        if total > 1 + WEIGHT_TOLERANCE:
            problems.append(f"weights sum to {total!r}, above 1 (leverage is not supported in v1)")
    return problems


class Portfolio:
    """Cash plus fractional share positions.

    ``positions`` only contains symbols with a non-zero share count. Both
    ``cash`` and ``positions`` are read-only from outside, and change only
    through :meth:`apply`.
    """

    def __init__(self, cash: float, positions: Mapping[str, float] | None = None) -> None:
        cash = float(cash)
        if not math.isfinite(cash) or cash < 0:
            raise ValueError(f"starting cash must be finite and >= 0, got {cash}")
        self._cash = cash
        self._positions: dict[str, float] = {}
        for symbol, shares in (positions or {}).items():
            shares = float(shares)
            if not math.isfinite(shares) or shares < 0:
                raise ValueError(f"starting position {symbol}={shares} must be finite and >= 0")
            if shares:
                self._positions[symbol] = shares

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """A copy of the share positions."""
        return dict(self._positions)

    def __repr__(self) -> str:
        return f"Portfolio(cash={self._cash:.2f}, positions={self._positions})"

    def value(self, prices: Mapping[str, float]) -> float:
        """Cash plus every position valued at ``prices``. Every held symbol needs a price."""
        return self._cash + math.fsum(
            shares * _price(prices, symbol) for symbol, shares in self._positions.items()
        )

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        """Each held position's share of total value at ``prices``."""
        total = self.value(prices)
        if total <= 0:
            raise ValueError(f"cannot compute weights: portfolio value is {total}")
        return {s: shares * _price(prices, s) / total for s, shares in self._positions.items()}

    def orders_for(self, targets: Mapping[str, float], prices: Mapping[str, float]) -> list[Order]:
        """Orders that move the portfolio from its current weights to ``targets``, at ``prices``.

        Held symbols missing from ``targets`` are sold to zero. A target of 0 sells
        the exact share count, leaving no residue. Trades smaller than
        ``MIN_TRADE_FRACTION`` of portfolio value are skipped as float noise. Sells
        come before buys (each group sorted by symbol), so cash raised by sells
        funds the buys and cash never dips below zero partway through a rebalance.

        ``prices`` should be the prices the orders will fill at. The engine uses
        the execution bar's opens (see ``Backtester``).

        Raises:
            ValueError: ``targets`` break the v1 weight rules (see ``weight_problems``).
            KeyError: a held or targeted symbol has no price.
        """
        problems = weight_problems(targets)
        if problems:
            raise ValueError("invalid target weights: " + "; ".join(problems))

        total = self.value(prices)
        sells: list[Order] = []
        buys: list[Order] = []
        for symbol in sorted(set(self._positions) | set(targets)):
            target = float(targets.get(symbol, 0.0))
            held = self._positions.get(symbol, 0.0)
            if target == 0.0:
                if held:
                    sells.append(Order(symbol, -held))
                continue
            price = _price(prices, symbol)
            delta_value = target * total - held * price
            if abs(delta_value) <= MIN_TRADE_FRACTION * total:
                continue
            order = Order(symbol, delta_value / price)
            (sells if order.quantity < 0 else buys).append(order)
        return sells + buys

    def apply(self, fill: Fill) -> None:
        """Record a fill: shares change by ``quantity``; cash by ``-(quantity * price + cost)``."""
        quantity, price, cost = float(fill.quantity), float(fill.price), float(fill.cost)
        if not (math.isfinite(quantity) and math.isfinite(price) and math.isfinite(cost)):
            raise ValueError(f"fill has non-finite values: {fill}")
        if price <= 0:
            raise ValueError(f"fill price must be > 0: {fill}")
        if cost < 0:
            raise ValueError(f"fill cost must be >= 0: {fill}")

        shares = self._positions.get(fill.symbol, 0.0) + quantity
        if shares:
            self._positions[fill.symbol] = shares
        else:
            self._positions.pop(fill.symbol, None)
        self._cash -= quantity * price + cost


def _price(prices: Mapping[str, float], symbol: str) -> float:
    try:
        return float(prices[symbol])
    except KeyError:
        raise KeyError(f"no price for {symbol!r}; prices given for {sorted(prices)}") from None
