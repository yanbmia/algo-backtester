"""SMACrossover and BuyAndHold, run through the real engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data import MarketData
from backtester.engine import Backtester, InvalidWeightsError, NextOpenExecution, ZeroCost
from backtester.metrics import exposure
from backtester.strategies import BuyAndHold, SMACrossover
from synthetic import ohlcv_from_closes


def run(data, strategy, **window):
    engine = Backtester(data, strategy, execution=NextOpenExecution(ZeroCost()))
    return engine.run(**window)


# ---------------------------------------------------------------------------
# SMACrossover on a series engineered to cross on known dates
# ---------------------------------------------------------------------------

# fast = 3, slow = 6. Every mean below is an exact integer, so ties are exact.
#
#  day  close  fast mean (last 3)   slow mean (last 6)   decision
#  0-4  100    (warmup: fewer than 6 bars)
#  5-9  100    100                  100                  0  tie -> flat
#  10   106    102                  101                  1  crosses up
#  11   106    104                  102                  1
#  12   106    106                  103                  1
#  13   94     102                  102                  0  tie right after long -> flat
#  14   94     98                   101                  0
#  15   94     94                   100                  0
#  16   94     94                   98                   0
#  17   94     94                   96                   0
#  18   94     94                   94                   0  tie -> flat
#  19   94     94                   94                   0  tie -> flat
#  20   100    96                   95                   1  crosses up again
#  21   100    98                   96                   1
#  22   100    100                  97                   1
#  23   100    100                  98                   1  (final decision: never executed)
CLOSES = [100.0] * 10 + [106.0] * 3 + [94.0] * 7 + [100.0] * 4
EXPECTED = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # days 5..23
TIE_DAYS = [5, 6, 7, 8, 9, 13, 18, 19]


@pytest.fixture
def crossing() -> MarketData:
    return MarketData.from_frames({"XYZ": ohlcv_from_closes(CLOSES)})


class TestSMACrossover:
    def test_decisions_flip_exactly_on_the_engineered_dates(self, crossing):
        result = run(crossing, SMACrossover("XYZ", fast=3, slow=6))
        dates = crossing.calendar

        assert result.decisions.index[0] == dates[5]  # warmup = slow = 6 bars
        assert result.decisions["XYZ"].tolist() == EXPECTED

    def test_ties_are_flat_including_right_after_being_long(self, crossing):
        decisions = run(crossing, SMACrossover("XYZ", fast=3, slow=6)).decisions["XYZ"]
        dates = crossing.calendar

        assert decisions.loc[dates[12]] == 1.0  # long the day before the tie
        for day in TIE_DAYS:
            assert decisions.loc[dates[day]] == 0.0, f"tie on day {day} should be flat"

    def test_trades_happen_at_the_open_after_each_flip(self, crossing):
        result = run(crossing, SMACrossover("XYZ", fast=3, slow=6))
        dates = crossing.calendar
        fills = result.fills

        assert fills["date"].tolist() == [dates[11], dates[14], dates[21]]
        assert fills["price"].tolist() == [106.0, 94.0, 100.0]
        assert (np.sign(fills["qty"]) == [1, -1, 1]).all()
        # The round trip bought at 106 and sold at 94, then re-entered with what was left.
        assert result.equity.loc[dates[14]] == pytest.approx(100_000 * 94 / 106)

    def test_windows_are_exactly_fast_and_slow_bars_long(self):
        # A spike (130) on day 5 sits at the oldest edge of the 6-bar window on day 10
        # and has just left it on day 11. A 5-bar slow window would miss it on day 10;
        # a 7-bar one would still include it on day 11. Only 6 gives [.., 0, 1].
        #
        #  day  window (last 6)           fast (last 3)  slow  decision
        #  5    100 100 100 100 100 130   110            105   1
        #  6    100 100 100 100 130 100   110            105   1
        #  7    100 100 100 130 100 100   110            105   1
        #  8    100 100 130 100 100 100   100            105   0
        #  9    100 130 100 100 100 106   102            106   0
        #  10   130 100 100 100 106 106   104            107   0  (5-bar slow: 102.4 -> 1)
        #  11   100 100 100 106 106 106   106            103   1  (7-bar slow: 106.9 -> 0)
        closes = [100.0] * 5 + [130.0] + [100.0] * 3 + [106.0] * 3
        data = MarketData.from_frames({"XYZ": ohlcv_from_closes(closes)})

        decisions = run(data, SMACrossover("XYZ", fast=3, slow=6)).decisions["XYZ"]

        assert decisions.tolist() == [1, 1, 1, 0, 0, 0, 1]

    def test_default_windows_are_50_and_200(self, market_data):
        strategy = SMACrossover("AAA")

        result = run(market_data, strategy)

        assert (strategy.fast, strategy.slow, strategy.warmup) == (50, 200, 200)
        assert strategy.name == "sma_crossover(AAA, 50, 200)"
        assert result.decisions.index[0] == market_data.calendar[199]

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"fast": 200, "slow": 200}, ValueError, "shorter than slow"),
            ({"fast": 60, "slow": 50}, ValueError, "shorter than slow"),
            ({"fast": 0, "slow": 5}, ValueError, "fast must be >= 1"),
            ({"fast": 2.5, "slow": 5}, TypeError, "fast must be an int"),
            ({"fast": True, "slow": 5}, TypeError, "fast must be an int"),
        ],
        ids=["equal", "reversed", "zero", "float", "bool"],
    )
    def test_invalid_windows_are_rejected(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            SMACrossover("XYZ", **kwargs)

    def test_symbol_must_be_a_string(self):
        with pytest.raises(TypeError, match="symbol"):
            SMACrossover(["XYZ"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BuyAndHold
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_buys_once_at_the_first_open_after_its_first_decision(self, market_data):
        result = run(market_data, BuyAndHold("AAA"))
        first_open = market_data.bar(market_data.calendar[1])["AAA"].open

        assert len(result.fills) == 1
        fill = result.fills.iloc[0]
        assert (fill["date"], fill["symbol"], fill["price"]) == (
            market_data.calendar[1],
            "AAA",
            first_open,
        )
        assert fill["qty"] == pytest.approx(100_000 / first_open, rel=1e-12)

    def test_fully_invested_from_the_first_fill_to_the_end(self, market_data):
        result = run(market_data, BuyAndHold("AAA"))
        held = result.positions["AAA"]

        assert held.iloc[0] == 0.0  # decision day: nothing can have filled yet
        assert (held.iloc[1:] == held.iloc[1]).all()  # one position, never touched
        assert (result.positions[["BBB", "CCC"]] == 0).all().all()
        assert exposure(result.positions) == (len(market_data) - 1) / len(market_data)
        assert exposure(result.positions.iloc[1:]) == 1.0

    def test_equity_moves_exactly_with_the_price(self, market_data):
        result = run(market_data, BuyAndHold("AAA"))
        closes = market_data.frame("AAA")["close"]
        first_open = market_data.bar(market_data.calendar[1])["AAA"].open

        # Day 1: bought at the open, marked at the close.
        assert result.returns.iloc[1] == pytest.approx(closes.iloc[1] / first_open - 1, rel=1e-12)
        # From day 2 on, equity tracks the close exactly.
        np.testing.assert_allclose(
            result.returns.iloc[2:], closes.pct_change().iloc[2:], rtol=1e-10, atol=1e-15
        )
        assert result.equity.iloc[-1] == pytest.approx(
            100_000 * closes.iloc[-1] / first_open, rel=1e-12
        )

    def test_can_start_on_any_date(self, market_data):
        start = market_data.calendar[100]
        result = run(market_data, BuyAndHold("AAA"), start=start)
        assert result.fills["date"].tolist() == [market_data.calendar[101]]

    def test_name_and_warmup(self):
        strategy = BuyAndHold("SPY")
        assert (strategy.name, strategy.warmup) == ("buy_and_hold(SPY)", 0)

    def test_a_symbol_outside_the_data_is_rejected_by_the_engine(self, market_data):
        with pytest.raises(InvalidWeightsError, match="'SPY' is not a visible symbol"):
            run(market_data, BuyAndHold("SPY"))


def test_crossover_and_benchmark_compare_over_the_same_window(market_data):
    # How the two are meant to be compared: start the benchmark on the
    # crossover's first decision date, so both are judged over the same days.
    sma = run(market_data, SMACrossover("AAA", fast=10, slow=40))
    benchmark = run(market_data, BuyAndHold("AAA"), start=sma.decisions.index[0])

    assert benchmark.decisions.index[0] == sma.decisions.index[0]
    assert pd.Index(benchmark.equity.index).equals(sma.equity.loc[sma.decisions.index[0] :].index)
