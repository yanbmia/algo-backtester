"""Execution models: next-open fills and pluggable costs."""

from __future__ import annotations

import pytest

from backtester.data import Bar
from backtester.engine import ExecutionError, NextOpenExecution, Order, ZeroCost
from engine_doubles import BpsCost

BARS = {"AAA": Bar(open=101.0, high=110.0, low=90.0, close=95.0, volume=1e6)}


class RecordingCost:
    def __init__(self) -> None:
        self.seen: list[tuple[Order, float]] = []

    def cost(self, order: Order, price: float) -> float:
        self.seen.append((order, price))
        return 2.5


def test_zero_cost_charges_nothing():
    assert ZeroCost().cost(Order("AAA", 100), 50.0) == 0.0


def test_fills_at_the_open_of_the_bar_it_is_given():
    (fill,) = NextOpenExecution(ZeroCost()).execute([Order("AAA", 10)], BARS)
    assert (fill.symbol, fill.quantity, fill.price, fill.cost) == ("AAA", 10, 101.0, 0.0)


def test_orders_are_sized_at_the_same_opens_they_fill_at():
    assert NextOpenExecution(ZeroCost()).reference_prices(BARS) == {"AAA": 101.0}


def test_cost_model_sees_each_order_and_its_fill_price():
    costs = RecordingCost()
    orders = [Order("AAA", 10), Order("AAA", -4)]

    fills = NextOpenExecution(costs).execute(orders, BARS)

    assert costs.seen == [(orders[0], 101.0), (orders[1], 101.0)]
    assert [f.cost for f in fills] == [2.5, 2.5]


def test_proportional_costs_scale_with_notional():
    (fill,) = NextOpenExecution(BpsCost(10)).execute([Order("AAA", -200)], BARS)
    assert fill.cost == pytest.approx(200 * 101.0 * 0.001)


def test_an_order_without_a_bar_cannot_fill():
    with pytest.raises(ExecutionError, match="BBB has no bar"):
        NextOpenExecution(ZeroCost()).execute([Order("BBB", 1)], BARS)


def test_negative_costs_are_rejected():
    class Rebate:
        def cost(self, order, price):
            return -1.0

    with pytest.raises(ValueError, match="costs must be >= 0"):
        NextOpenExecution(Rebate()).execute([Order("AAA", 1)], BARS)


def test_the_cost_model_is_required_and_checked():
    with pytest.raises(TypeError):
        NextOpenExecution()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="cost_model must implement"):
        NextOpenExecution(object())  # type: ignore[arg-type]


def test_repr_states_the_cost_assumption():
    assert repr(NextOpenExecution(ZeroCost())) == "NextOpenExecution(cost_model=ZeroCost())"
