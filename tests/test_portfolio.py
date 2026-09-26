"""Portfolio accounting: valuation, sizing orders to targets, and applying fills."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtester.engine import Fill, Order, Portfolio

PRICES = {"AAA": 50.0, "BBB": 200.0}


def rebalance(portfolio: Portfolio, targets: dict[str, float], prices: dict[str, float]) -> None:
    """Size orders at ``prices`` and fill them at the same prices, with no cost."""
    for order in portfolio.orders_for(targets, prices):
        portfolio.apply(Fill(order.symbol, order.quantity, prices[order.symbol], 0.0))


class TestState:
    def test_new_portfolio_is_all_cash(self):
        portfolio = Portfolio(100_000)
        assert portfolio.cash == 100_000
        assert portfolio.positions == {}
        assert portfolio.value(PRICES) == 100_000

    def test_value_and_weights(self):
        portfolio = Portfolio(10_000, {"AAA": 100, "BBB": 50})
        assert portfolio.value(PRICES) == 10_000 + 100 * 50 + 50 * 200  # 25,000
        assert portfolio.weights(PRICES) == {"AAA": 0.2, "BBB": 0.4}

    def test_valuing_a_holding_needs_its_price(self):
        with pytest.raises(KeyError, match="no price for 'BBB'"):
            Portfolio(0, {"BBB": 1}).value({"AAA": 50.0})

    def test_state_changes_only_through_apply(self):
        portfolio = Portfolio(1_000, {"AAA": 10})
        portfolio.positions["AAA"] = 999  # a copy
        assert portfolio.positions == {"AAA": 10}
        with pytest.raises(AttributeError):
            portfolio.cash = 5

    @pytest.mark.parametrize(
        ("cash", "positions"),
        [(-1, None), (math.nan, None), (100, {"AAA": -5}), (100, {"AAA": math.inf})],
        ids=["negative-cash", "nan-cash", "short-position", "inf-position"],
    )
    def test_invalid_starting_state_is_rejected(self, cash, positions):
        with pytest.raises(ValueError, match="starting"):
            Portfolio(cash, positions)


class TestOrdersFor:
    def test_from_cash_to_targets(self):
        orders = Portfolio(100_000).orders_for({"AAA": 0.6, "BBB": 0.4}, PRICES)

        assert [o.symbol for o in orders] == ["AAA", "BBB"]
        assert orders[0].quantity == pytest.approx(60_000 / 50, rel=1e-12)
        assert orders[1].quantity == pytest.approx(40_000 / 200, rel=1e-12)

    def test_sells_come_before_buys(self):
        orders = Portfolio(0, {"BBB": 500}).orders_for({"AAA": 1.0}, PRICES)
        assert orders == [Order("BBB", -500.0), Order("AAA", pytest.approx(2_000.0))]

    def test_a_zero_target_sells_the_exact_share_count(self):
        portfolio = Portfolio(0, {"AAA": 123.456789})

        orders = portfolio.orders_for({"AAA": 0.0}, PRICES)
        rebalance(portfolio, {"AAA": 0.0}, PRICES)

        assert orders == [Order("AAA", -123.456789)]
        assert portfolio.positions == {}  # no float residue left behind

    def test_symbols_left_out_of_the_targets_are_sold(self):
        orders = Portfolio(0, {"AAA": 10, "BBB": 5}).orders_for({"AAA": 0.2}, PRICES)
        assert Order("BBB", -5.0) in orders

    def test_unchanged_targets_generate_no_orders(self):
        portfolio = Portfolio(100_000)
        rebalance(portfolio, {"AAA": 0.6, "BBB": 0.4}, PRICES)

        assert portfolio.orders_for({"AAA": 0.6, "BBB": 0.4}, PRICES) == []

    @pytest.mark.parametrize(
        ("targets", "fragment"),
        [
            ({"AAA": -0.1}, "negative"),
            ({"AAA": 1.5}, "above 1"),
            ({"AAA": 0.7, "BBB": 0.7}, "sum to"),
            ({"AAA": math.nan}, "not finite"),
            ({"AAA": "0.5"}, "not a number"),
            (["AAA"], "expected a dict"),
        ],
        ids=["negative", "above-one", "levered-sum", "nan", "string", "not-a-dict"],
    )
    def test_invalid_targets_are_rejected(self, targets, fragment):
        with pytest.raises(ValueError, match=f"invalid target weights.*{fragment}"):
            Portfolio(100_000).orders_for(targets, PRICES)

    def test_a_target_needs_a_price(self):
        with pytest.raises(KeyError, match="no price for 'CCC'"):
            Portfolio(100_000).orders_for({"CCC": 0.5}, PRICES)


class TestAccounting:
    def test_rebalancing_hits_the_targets_and_conserves_value(self):
        portfolio = Portfolio(100_000)

        rebalance(portfolio, {"AAA": 0.6, "BBB": 0.4}, PRICES)
        assert portfolio.value(PRICES) == pytest.approx(100_000, abs=1e-9)
        assert portfolio.weights(PRICES) == pytest.approx({"AAA": 0.6, "BBB": 0.4}, abs=1e-12)

        moved = {"AAA": 55.0, "BBB": 180.0}
        before = portfolio.value(moved)
        rebalance(portfolio, {"AAA": 0.2, "BBB": 0.8}, moved)

        assert portfolio.value(moved) == pytest.approx(before, abs=1e-9)
        assert portfolio.weights(moved) == pytest.approx({"AAA": 0.2, "BBB": 0.8}, abs=1e-12)
        assert portfolio.cash == pytest.approx(0.0, abs=1e-9)

    def test_cash_is_conserved_across_a_sequence_of_fills(self):
        rng = np.random.default_rng(3)
        portfolio = Portfolio(1_000_000)
        fills = [
            Fill(
                symbol=str(rng.choice(["AAA", "BBB", "CCC"])),
                quantity=float(rng.normal(0, 100)),
                price=float(rng.uniform(10, 500)),
                cost=float(rng.uniform(0, 5)),
            )
            for _ in range(200)
        ]

        for fill in fills:
            portfolio.apply(fill)

        expected_cash = 1_000_000 - math.fsum(f.quantity * f.price + f.cost for f in fills)
        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-12)
        for symbol in ("AAA", "BBB", "CCC"):
            expected = math.fsum(f.quantity for f in fills if f.symbol == symbol)
            assert portfolio.positions.get(symbol, 0.0) == pytest.approx(expected, abs=1e-9)

    def test_a_zero_cost_round_trip_returns_the_starting_cash(self):
        portfolio = Portfolio(100_000)

        rebalance(portfolio, {"AAA": 0.73}, PRICES)
        assert portfolio.cash == pytest.approx(27_000, abs=1e-9)
        rebalance(portfolio, {}, PRICES)

        assert portfolio.cash == pytest.approx(100_000, abs=1e-9)
        assert portfolio.positions == {}

    def test_round_trip_profit_is_shares_times_the_price_change(self):
        portfolio = Portfolio(100_000)
        rebalance(portfolio, {"AAA": 1.0}, {"AAA": 50.0})
        shares = portfolio.positions["AAA"]

        rebalance(portfolio, {}, {"AAA": 55.0})

        assert portfolio.cash == pytest.approx(100_000 + shares * 5.0, abs=1e-9)

    def test_apply_charges_the_cost_to_cash(self):
        portfolio = Portfolio(1_000)
        portfolio.apply(Fill("AAA", 10, 50.0, 1.25))
        assert portfolio.cash == 1_000 - 500 - 1.25
        assert portfolio.positions == {"AAA": 10}

    @pytest.mark.parametrize(
        "fill",
        [
            Fill("AAA", math.nan, 50.0, 0.0),
            Fill("AAA", 1.0, math.inf, 0.0),
            Fill("AAA", 1.0, 0.0, 0.0),
            Fill("AAA", 1.0, 50.0, -0.01),
        ],
        ids=["nan-quantity", "inf-price", "zero-price", "negative-cost"],
    )
    def test_invalid_fills_are_rejected(self, fill):
        portfolio = Portfolio(1_000)
        with pytest.raises(ValueError, match="fill"):
            portfolio.apply(fill)
        assert (portfolio.cash, portfolio.positions) == (1_000, {})
