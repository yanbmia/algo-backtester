"""Execution: turning orders into fills, and what those fills cost.

The engine hands an execution model the bars of the date *after* the decision.
``NextOpenExecution`` fills every order at that bar's open, so a decision made
from the close of day t can only ever trade at the open of day t+1.

Costs are a separate, swappable ``CostModel``. v1 uses ``ZeroCost``, and the
caller always passes it explicitly: ``NextOpenExecution(ZeroCost())``. The cost
assumption is visible wherever a backtest is set up, never buried in a default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from backtester.data import Bar
from backtester.engine.portfolio import Fill, Order


class ExecutionError(RuntimeError):
    """An order could not be executed (e.g. its symbol has no bar on the execution date)."""


@runtime_checkable
class CostModel(Protocol):
    """Total cost, in cash, of filling ``order`` at ``price``. Must be finite and >= 0."""

    def cost(self, order: Order, price: float) -> float: ...


class ZeroCost:
    """No commissions, fees, spread, or slippage. v1's explicit, stated assumption."""

    def cost(self, order: Order, price: float) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "ZeroCost()"


@runtime_checkable
class ExecutionModel(Protocol):
    """How orders become fills on the execution bar."""

    def reference_prices(self, bars: Mapping[str, Bar]) -> dict[str, float]:
        """The prices orders will be sized at (the engine calls this before sizing)."""
        ...

    def execute(self, orders: Sequence[Order], bars: Mapping[str, Bar]) -> list[Fill]:
        """Fill ``orders`` against ``bars``, the bars of the execution date."""
        ...


class NextOpenExecution:
    """Fill every order at the open of the execution bar, charging ``cost_model``.

    The engine passes this model the bars of the date after the decision, so
    "the open" is always the next session's open. Orders are sized at those same
    opens (``reference_prices``), which amounts to submitting market-on-open
    orders for a dollar amount. The open is only used at the moment it is traded
    at, never earlier.
    """

    def __init__(self, cost_model: CostModel) -> None:
        if not isinstance(cost_model, CostModel):
            raise TypeError(
                f"cost_model must implement cost(order, price), got {type(cost_model).__name__}"
            )
        self.cost_model = cost_model

    def __repr__(self) -> str:
        return f"NextOpenExecution(cost_model={self.cost_model!r})"

    def reference_prices(self, bars: Mapping[str, Bar]) -> dict[str, float]:
        return {symbol: bar.open for symbol, bar in bars.items()}

    def execute(self, orders: Sequence[Order], bars: Mapping[str, Bar]) -> list[Fill]:
        fills = []
        for order in orders:
            try:
                bar = bars[order.symbol]
            except KeyError:
                raise ExecutionError(
                    f"cannot fill {order}: {order.symbol} has no bar on the execution date"
                ) from None
            cost = float(self.cost_model.cost(order, bar.open))
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(
                    f"{self.cost_model!r} returned cost {cost} for {order}; costs must be >= 0"
                )
            fills.append(Fill(order.symbol, order.quantity, bar.open, cost))
        return fills
