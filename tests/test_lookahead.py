"""The lookahead guard, proven from the outside, through the real engine.

1. Future-perturbation invariance: rewriting everything after a cutoff must not
   change anything the engine produced at or before it: decisions, equity, cash,
   positions, fills, or anything a probe observed.
2. Negative control: the same checks must FAIL against a deliberately leaky view.
   If they did not, they would not be checking anything.
3. Boundary: at every step the latest bar a strategy can see is dated exactly
   ``view.now`` (or earlier, for a symbol that did not trade that day).
4. Read-only: data handed to strategies cannot be written, and cannot reach
   the parent MarketData or later views.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from backtester.data import MarketData, YFinanceLoader
from backtester.strategies import BuyAndHold, SMACrossover
from lookahead_harness import (
    LeakyMarketData,
    MomentumStrategy,
    Observation,
    PanelProbe,
    ProbeStrategy,
    assert_no_lookahead,
    first_divergence,
    make_leaky,
    run_backtest,
)
from synthetic import replace_after

# Each entry builds a fresh strategy for a dataset. The probes can't pass by luck;
# SMACrossover and BuyAndHold are the real v1 strategies (SMA with short windows
# so it trades within the synthetic runs).
STRATEGIES = {
    "ProbeStrategy": lambda data: ProbeStrategy(),
    "PanelProbe": lambda data: PanelProbe(),
    "MomentumStrategy": lambda data: MomentumStrategy(),
    "SMACrossover": lambda data: SMACrossover(data.symbols[0], fast=5, slow=20),
    "BuyAndHold": lambda data: BuyAndHold(data.symbols[0]),
}
CUTOFF_FRACTIONS = [0.3, 0.5, 0.8]


@dataclass(frozen=True)
class Scenario:
    """A dataset plus the run window the engine accepts for it."""

    data: MarketData
    end: pd.Timestamp | None = None

    @property
    def run_dates(self) -> pd.DatetimeIndex:
        calendar = self.data.calendar
        return calendar if self.end is None else calendar[calendar <= self.end]

    def cutoff(self, fraction: float) -> pd.Timestamp:
        return self.run_dates[int(len(self.run_dates) * fraction)]

    def next_date(self, t: pd.Timestamp) -> pd.Timestamp:
        return self.run_dates[self.run_dates.get_loc(t) + 1]


@pytest.fixture(params=["shared_calendar", "staggered_calendars"])
def scenario(request, market_data, staggered_market_data) -> Scenario:
    if request.param == "shared_calendar":
        return Scenario(market_data)
    # The v1 engine rejects a symbol that stops trading mid-run, so this run ends on
    # GONE's last day. MID still lists partway through (2020-04-01), and LATE lists
    # after the run ends, so it is never visible.
    return Scenario(staggered_market_data, end=pd.Timestamp("2020-07-31"))


# ---------------------------------------------------------------------------
# 1. Future-perturbation invariance
# ---------------------------------------------------------------------------


class TestFuturePerturbation:
    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    @pytest.mark.parametrize("build", STRATEGIES.values(), ids=STRATEGIES.keys())
    def test_nothing_at_or_before_the_cutoff_depends_on_the_future(self, scenario, build, fraction):
        data = scenario.data
        assert_no_lookahead(data, lambda: build(data), scenario.cutoff(fraction), end=scenario.end)

    def test_the_crossover_really_trades_in_these_scenarios(self, scenario):
        # Otherwise its invariance check above would be vacuous (always flat).
        strategy = STRATEGIES["SMACrossover"](scenario.data)
        result = run_backtest(scenario.data, strategy, end=scenario.end)

        assert set(result.decisions[strategy.symbol]) == {0.0, 1.0}
        assert len(result.fills) >= 4

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_perturbation_rewrites_only_the_future(self, scenario, fraction):
        data, cutoff = scenario.data, scenario.cutoff(fraction)
        perturbed = replace_after(data, cutoff, seed=7)

        assert perturbed.calendar.equals(data.calendar)
        for symbol in data.symbols:
            original, rewritten = data.frame(symbol), perturbed.frame(symbol)
            pd.testing.assert_frame_equal(original.loc[:cutoff], rewritten.loc[:cutoff])
            future = original.index > cutoff
            # Every future value differs, so an invariance pass cannot be a coincidence.
            assert (original[future].to_numpy() != rewritten[future].to_numpy()).all()

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_perturbation_shows_up_on_the_very_next_bar(self, scenario, fraction):
        # The flip side of invariance: the first perturbed bar changes the probe's
        # decision, fills, positions and equity immediately. So when they stay
        # unchanged up to the cutoff, it is because nothing could see the future.
        cutoff = scenario.cutoff(fraction)
        following = scenario.next_date(cutoff)

        base, alt = assert_no_lookahead(scenario.data, ProbeStrategy, cutoff, end=scenario.end)

        assert first_divergence(base.decisions, alt.decisions) == following
        assert first_divergence(base.positions, alt.positions) == following
        assert first_divergence(base.equity, alt.equity) == following

    def test_a_symbol_that_lists_after_the_cutoff_is_invisible(self, staggered_market_data):
        # Knowing a ticker WILL exist is future information. Everything up to the
        # cutoff must be identical whether or not the later listing is in the data.
        data, end = staggered_market_data, pd.Timestamp("2020-07-31")
        cutoff = pd.Timestamp("2020-03-20")  # MID lists on 2020-04-01
        without_mid = MarketData.from_frames({s: data.frame(s) for s in data.symbols if s != "MID"})
        with_probe, without_probe = ProbeStrategy(), ProbeStrategy()

        with_mid = run_backtest(data, with_probe, end=end)
        without = run_backtest(without_mid, without_probe, end=end)

        assert (with_mid.decisions.loc[:cutoff, "MID"] == 0.0).all()
        assert (with_mid.positions.loc[:cutoff, "MID"] == 0.0).all()
        for frame in ("decisions", "positions"):
            pd.testing.assert_frame_equal(
                getattr(with_mid, frame).loc[:cutoff].drop(columns="MID"),
                getattr(without, frame).loc[:cutoff],
                check_exact=True,
            )
        pd.testing.assert_series_equal(
            with_mid.equity.loc[:cutoff], without.equity.loc[:cutoff], check_exact=True
        )
        seen_with = [o for o in with_probe.observations if o.now <= cutoff]
        seen_without = [o for o in without_probe.observations if o.now <= cutoff]
        assert seen_with == seen_without


# ---------------------------------------------------------------------------
# 2. Negative control: a leaky view must make the checks fail
# ---------------------------------------------------------------------------


class TestNegativeControl:
    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    @pytest.mark.parametrize("factory", [ProbeStrategy, PanelProbe], ids=lambda f: f.__name__)
    def test_perturbation_check_fails_against_a_leaky_view(self, market_data, factory, fraction):
        leaky = make_leaky(market_data)

        with pytest.raises(AssertionError, match="decisions at or before"):
            assert_no_lookahead(leaky, factory, Scenario(market_data).cutoff(fraction))

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_leak_changes_the_decision_at_the_cutoff_itself(self, market_data, fraction):
        # Honest view: the first changed decision is the first perturbed bar (T+1).
        # Leaky view: that bar is already visible at the cutoff T.
        cutoff = Scenario(market_data).cutoff(fraction)
        leaky = make_leaky(market_data)

        base = run_backtest(leaky, ProbeStrategy())
        alt = run_backtest(replace_after(leaky, cutoff, seed=7), ProbeStrategy())

        assert first_divergence(base.decisions, alt.decisions) == cutoff

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_equity_alone_cannot_see_a_one_bar_leak(self, market_data, fraction):
        # Why assert_no_lookahead checks decisions, not just equity: the leaked
        # decision at T trades at T+1's open, and T+1 is perturbed anyway. So the
        # leaky run's equity diverges exactly where an honest run's does.
        scenario = Scenario(market_data)
        cutoff = scenario.cutoff(fraction)
        leaky = make_leaky(market_data)

        base = run_backtest(leaky, ProbeStrategy())
        alt = run_backtest(replace_after(leaky, cutoff, seed=7), ProbeStrategy())

        assert first_divergence(base.equity, alt.equity) == scenario.next_date(cutoff)

    def test_the_leak_is_subtle(self, market_data):
        # The leaky view reports the correct date and looks normal. Only its
        # contents are wrong, which is exactly why this needs an outside test.
        leaky = make_leaky(market_data)
        t = market_data.calendar[100]

        honest_view, leaky_view = market_data.view(t), leaky.view(t)

        assert isinstance(leaky, LeakyMarketData)
        assert leaky_view.now == honest_view.now == t
        assert leaky_view.symbols == honest_view.symbols
        assert len(leaky_view.history("AAA")) == len(honest_view.history("AAA")) + 1


# ---------------------------------------------------------------------------
# 3. Boundary: the latest visible bar is dated now (or earlier)
# ---------------------------------------------------------------------------


def assert_seen_only_up_to_now(data: MarketData, observations: list[Observation]) -> None:
    """Each probe observation must match, exactly, what was knowable at ``now``.

    The expected last date for each symbol is recomputed independently from
    ``data.frame()``: its latest bar at or before ``now``. Symbols with no such
    bar must not have been visible at all.
    """
    for obs in observations:
        expected = {}
        for symbol in data.symbols:
            dates = data.frame(symbol).index
            dates = dates[dates <= obs.now]
            if len(dates):
                expected[symbol] = dates[-1]
        assert obs.last_seen == expected, f"on {obs.now:%Y-%m-%d} the probe saw {obs.last_seen}"
        assert max(obs.last_seen.values()) == obs.now


class TestBoundary:
    def test_probe_sees_exactly_up_to_now_at_every_step(self, scenario):
        probe = ProbeStrategy()

        result = run_backtest(scenario.data, probe, end=scenario.end)

        assert [o.now for o in probe.observations] == list(scenario.run_dates)
        assert list(result.decisions.index) == list(scenario.run_dates)
        assert_seen_only_up_to_now(scenario.data, probe.observations)

    def test_boundary_check_fails_against_a_leaky_view(self, market_data):
        leaky = make_leaky(market_data)
        probe = ProbeStrategy()
        run_backtest(leaky, probe)

        with pytest.raises(AssertionError, match="the probe saw"):
            assert_seen_only_up_to_now(leaky, probe.observations)

    def test_latest_visible_bar_is_the_bar_the_engine_sees_on_that_date(self, scenario):
        data = scenario.data
        for t in scenario.run_dates[::5]:
            view, bars = data.view(t), data.bar(t)
            for symbol, bar in bars.items():
                for field in ("open", "high", "low", "close", "volume"):
                    history = view.history(symbol, field)
                    assert history.index[-1] == t
                    assert history.iloc[-1] == getattr(bar, field)


# ---------------------------------------------------------------------------
# 4. Read-only: returned data cannot be written or reach the parent
# ---------------------------------------------------------------------------

SERIES_WRITES = {
    "setitem": lambda s: s.__setitem__(s.index[-1], 0.0),
    "iloc": lambda s: s.iloc.__setitem__(-1, 0.0),
    "loc": lambda s: s.loc.__setitem__(s.index[-1], 0.0),
    "iloc_slice": lambda s: s.iloc.__setitem__(slice(0, 3), 0.0),
    "values": lambda s: s.values.__setitem__(-1, 0.0),
    "to_numpy": lambda s: s.to_numpy().__setitem__(-1, 0.0),
    "array": lambda s: s.array.__setitem__(-1, 0.0),
}

FRAME_WRITES = {
    "iloc": lambda df: df.iloc.__setitem__((0, 0), 0.0),
    "loc": lambda df: df.loc.__setitem__((df.index[0], "close"), 0.0),
    "at": lambda df: df.at.__setitem__((df.index[0], "close"), 0.0),
    "iat": lambda df: df.iat.__setitem__((0, 0), 0.0),
    "iloc_row": lambda df: df.iloc.__setitem__(0, 0.0),
    "values": lambda df: df.values.__setitem__((0, 0), 0.0),
    "to_numpy": lambda df: df.to_numpy().__setitem__((0, 0), 0.0),
}


class TestReadOnly:
    @pytest.mark.parametrize("write", SERIES_WRITES.values(), ids=SERIES_WRITES.keys())
    def test_writing_into_history_raises(self, market_data, write):
        series = market_data.view(market_data.calendar[50]).history("AAA")
        before = series.to_numpy().copy()

        with pytest.raises(ValueError, match="read-only"):
            write(series)

        np.testing.assert_array_equal(series.to_numpy(), before)

    @pytest.mark.parametrize("write", FRAME_WRITES.values(), ids=FRAME_WRITES.keys())
    def test_writing_into_panel_raises(self, market_data, write):
        panel = market_data.view(market_data.calendar[50]).panel()
        before = panel.to_numpy().copy()

        with pytest.raises(ValueError, match="read-only"):
            write(panel)

        np.testing.assert_array_equal(panel.to_numpy(), before)

    def test_returned_data_shares_no_memory_with_the_parent(self, market_data):
        view = market_data.view(market_data.calendar[50])
        parent_values = market_data._values["AAA"]
        parent_dates = market_data._index["AAA"].to_numpy()

        history, panel = view.history("AAA"), view.panel()

        assert not history.to_numpy().flags.writeable
        assert not panel.to_numpy().flags.writeable
        assert not np.shares_memory(history.to_numpy(), parent_values)
        assert not np.shares_memory(history.index.to_numpy(), parent_dates)
        assert not np.shares_memory(panel.to_numpy(), parent_values)
        assert not np.shares_memory(panel.index.get_level_values("date").to_numpy(), parent_dates)

    def test_rebinding_the_returned_object_cannot_reach_the_view_or_parent(self, market_data):
        # Some pandas operations don't write into the returned array. Instead they
        # rebind the caller's object to NEW arrays: pandas 3's `s += 1` and
        # `inplace=True`, and in any version `df["col"] = ...`. Whether they raise
        # depends on the pandas version. Either way, they must not change what the
        # view or the parent returns afterwards.
        t = market_data.calendar[50]
        view = market_data.view(t)
        expected_history = view.history("AAA").copy()
        expected_panel = view.panel().copy()
        expected_frame = market_data.frame("AAA")

        series, panel = view.history("AAA"), view.panel()
        for rebind in (
            lambda: series.__iadd__(1.0),
            lambda: series.clip(0.0, 1.0, inplace=True),
            lambda: panel.__setitem__("close", 0.0),
            lambda: panel.clip(0.0, 1.0, inplace=True),
        ):
            with contextlib.suppress(ValueError, TypeError):
                rebind()

        pd.testing.assert_series_equal(view.history("AAA"), expected_history)
        pd.testing.assert_frame_equal(view.panel(), expected_panel)
        pd.testing.assert_series_equal(market_data.view(t).history("AAA"), expected_history)
        pd.testing.assert_frame_equal(market_data.frame("AAA"), expected_frame)


# ---------------------------------------------------------------------------
# Real data (opt-in: needs network access to Yahoo)
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
def test_live_spy_backtest_ignores_the_future(tmp_path, fraction):
    spy = Scenario(YFinanceLoader(tmp_path).load(["SPY"], "2020-01-01", "2020-12-31"))
    cutoff = spy.cutoff(fraction)

    base, alt = assert_no_lookahead(spy.data, ProbeStrategy, cutoff)
    assert first_divergence(base.decisions, alt.decisions) == spy.next_date(cutoff)

    with pytest.raises(AssertionError):
        assert_no_lookahead(make_leaky(spy.data), ProbeStrategy, cutoff)
