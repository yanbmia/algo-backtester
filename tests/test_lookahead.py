"""The lookahead guard, proven from the outside.

1. Future-perturbation invariance: rewriting everything after a cutoff must not
   change any decision (or anything a probe observed) at or before it.
2. Negative control: the same checks must FAIL against a deliberately leaky view.
   If they did not, they would not be checking anything.
3. Boundary: at every step the latest bar a strategy can see is dated exactly
   ``view.now`` (or earlier, for a symbol that did not trade that day).
4. Read-only: data handed to strategies cannot be written, and cannot reach
   the parent MarketData or later views.

There is no engine yet, so "decisions" are the target weights a strategy returns
at each date (see ``lookahead_harness.run_decisions``). Equity and positions will
be added to the same invariance check when the engine exists.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from backtester.data import MarketData, YFinanceLoader
from lookahead_harness import (
    LeakyMarketData,
    MomentumStrategy,
    Observation,
    PanelProbe,
    ProbeStrategy,
    assert_no_lookahead,
    first_divergence,
    make_leaky,
    run_decisions,
)
from synthetic import replace_after

STRATEGIES = [ProbeStrategy, PanelProbe, MomentumStrategy]
CUTOFF_FRACTIONS = [0.3, 0.5, 0.8]


def cutoff_at(data: MarketData, fraction: float) -> pd.Timestamp:
    return data.calendar[int(len(data) * fraction)]


def next_date(data: MarketData, t: pd.Timestamp) -> pd.Timestamp:
    return data.calendar[data.calendar.get_loc(t) + 1]


@pytest.fixture(params=["shared_calendar", "staggered_calendars"])
def dataset(request, market_data, staggered_market_data) -> MarketData:
    return market_data if request.param == "shared_calendar" else staggered_market_data


# ---------------------------------------------------------------------------
# 1. Future-perturbation invariance
# ---------------------------------------------------------------------------


class TestFuturePerturbation:
    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    @pytest.mark.parametrize("factory", STRATEGIES, ids=lambda f: f.__name__)
    def test_decisions_at_or_before_cutoff_ignore_the_future(self, dataset, factory, fraction):
        assert_no_lookahead(dataset, factory, cutoff_at(dataset, fraction))

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_perturbation_rewrites_only_the_future(self, dataset, fraction):
        cutoff = cutoff_at(dataset, fraction)
        perturbed = replace_after(dataset, cutoff, seed=7)

        assert perturbed.calendar.equals(dataset.calendar)
        for symbol in dataset.symbols:
            original, rewritten = dataset.frame(symbol), perturbed.frame(symbol)
            pd.testing.assert_frame_equal(original.loc[:cutoff], rewritten.loc[:cutoff])
            future = original.index > cutoff
            # Every future value differs, so an invariance pass cannot be a coincidence.
            assert (original[future].to_numpy() != rewritten[future].to_numpy()).all()

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_probe_reacts_to_the_perturbation_on_the_very_next_bar(self, dataset, fraction):
        # The flip side of invariance: the probe is sensitive enough that the first
        # perturbed bar changes its decision immediately. So when it stays unchanged
        # up to the cutoff, that is because it could not see the future.
        cutoff = cutoff_at(dataset, fraction)

        base, alt = assert_no_lookahead(dataset, ProbeStrategy, cutoff)

        assert first_divergence(base, alt) == next_date(dataset, cutoff)

    def test_a_symbol_that_lists_after_the_cutoff_is_invisible(self, staggered_market_data):
        # Knowing a ticker WILL exist is future information. Decisions and everything
        # the probe observed up to the cutoff must be identical whether or not the
        # later listing is in the dataset at all.
        cutoff = pd.Timestamp("2020-08-14")  # LATE lists on 2020-09-01
        kept = [s for s in staggered_market_data.symbols if s != "LATE"]
        without_late = MarketData.from_frames({s: staggered_market_data.frame(s) for s in kept})
        with_probe, without_probe = ProbeStrategy(), ProbeStrategy()

        with_late = run_decisions(staggered_market_data, with_probe)
        without = run_decisions(without_late, without_probe)

        assert (with_late.loc[:cutoff, "LATE"] == 0.0).all()
        pd.testing.assert_frame_equal(
            with_late.loc[:cutoff].drop(columns="LATE"), without.loc[:cutoff], check_exact=True
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
            assert_no_lookahead(leaky, factory, cutoff_at(market_data, fraction))

    @pytest.mark.parametrize("fraction", CUTOFF_FRACTIONS)
    def test_leak_is_detected_exactly_one_bar_early(self, market_data, fraction):
        # Honest view: the first changed decision is the first perturbed bar (T+1).
        # Leaky view: it is already visible at the cutoff T itself.
        cutoff = cutoff_at(market_data, fraction)
        leaky = make_leaky(market_data)

        base = run_decisions(leaky, ProbeStrategy())
        alt = run_decisions(replace_after(leaky, cutoff, seed=7), ProbeStrategy())

        assert first_divergence(base, alt) == cutoff

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
    def test_probe_sees_exactly_up_to_now_at_every_step(self, dataset):
        probe = ProbeStrategy()

        run_decisions(dataset, probe)

        assert [o.now for o in probe.observations] == list(dataset.calendar)
        assert_seen_only_up_to_now(dataset, probe.observations)

    def test_boundary_check_fails_against_a_leaky_view(self, market_data):
        leaky = make_leaky(market_data)
        probe = ProbeStrategy()
        run_decisions(leaky, probe)

        with pytest.raises(AssertionError, match="the probe saw"):
            assert_seen_only_up_to_now(leaky, probe.observations)

    def test_latest_visible_bar_is_the_bar_the_engine_sees_on_that_date(self, dataset):
        for t in dataset.calendar[::5]:
            view, bars = dataset.view(t), dataset.bar(t)
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
def test_live_spy_decisions_ignore_the_future(tmp_path, fraction):
    spy = YFinanceLoader(tmp_path).load(["SPY"], "2020-01-01", "2020-12-31")
    cutoff = cutoff_at(spy, fraction)

    base, alt = assert_no_lookahead(spy, ProbeStrategy, cutoff)
    assert first_divergence(base, alt) == next_date(spy, cutoff)

    with pytest.raises(AssertionError):
        assert_no_lookahead(make_leaky(spy), ProbeStrategy, cutoff)
