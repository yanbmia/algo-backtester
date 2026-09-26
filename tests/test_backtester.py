"""The engine loop: timing, accounting identities, warmup, weight validation, and setup checks."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from backtester.data import MarketData, MarketView
from backtester.engine import (
    Backtester,
    EngineError,
    InvalidWeightsError,
    NextOpenExecution,
    StrategyError,
    ZeroCost,
)
from backtester.results import FILL_COLUMNS, BacktestResult
from engine_doubles import BpsCost, Constant, EqualWeight, Raises, Recorder, Returns, Scheduled
from lookahead_harness import MomentumStrategy
from synthetic import make_gapped_ohlcv

ZERO_COST = NextOpenExecution(ZeroCost())


def run(data, strategy, execution=ZERO_COST, initial_cash=100_000.0, **window) -> BacktestResult:
    return Backtester(data, strategy, execution=execution, initial_cash=initial_cash).run(**window)


@pytest.fixture
def gapped() -> MarketData:
    """One symbol whose open is always 500 above its close (see make_gapped_ohlcv)."""
    return MarketData.from_frames({"GAP": make_gapped_ohlcv(30)})


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class TestSetup:
    def test_execution_must_be_passed_explicitly(self, market_data):
        with pytest.raises(TypeError, match="execution"):
            Backtester(market_data, EqualWeight())  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"data": "not data"}, TypeError, "data must be MarketData"),
            ({"strategy": object()}, TypeError, "strategy must be a Strategy"),
            ({"execution": ZeroCost()}, TypeError, "execution must implement"),
            ({"initial_cash": 0}, ValueError, "initial_cash"),
            ({"initial_cash": math.inf}, ValueError, "initial_cash"),
        ],
        ids=["data", "strategy", "execution", "zero-cash", "inf-cash"],
    )
    def test_bad_arguments_are_rejected(self, market_data, kwargs, error, match):
        args = {"data": market_data, "strategy": EqualWeight(), "execution": ZERO_COST}
        args.update(kwargs)
        with pytest.raises(error, match=match):
            Backtester(args.pop("data"), args.pop("strategy"), **args)

    @pytest.mark.parametrize(
        ("warmup", "error"), [(-1, ValueError), (2.5, TypeError), (True, TypeError)]
    )
    def test_invalid_warmup_is_rejected_up_front(self, market_data, warmup, error):
        strategy = EqualWeight()
        strategy.warmup = warmup
        with pytest.raises(error, match=r"EqualWeight\.warmup"):
            Backtester(market_data, strategy, execution=ZERO_COST)

    def test_run_window_must_contain_trading_dates(self, market_data):
        with pytest.raises(ValueError, match="no trading dates between 2021-01-04"):
            run(market_data, EqualWeight(), start="2021-01-04", end="2021-02-01")
        with pytest.raises(ValueError, match="is after end"):
            run(market_data, EqualWeight(), start="2020-06-01", end="2020-05-01")


# ---------------------------------------------------------------------------
# Execution timing
# ---------------------------------------------------------------------------


class TestTiming:
    def test_a_decision_at_t_fills_at_the_next_open_never_the_close(self, gapped):
        # Rows have open = 1000 + i, close = 500 + i. Buying on row 5's decision must
        # fill at row 6's open (1006): not row 5's close (505), not row 5's open (1005).
        dates = gapped.calendar
        decided, filled = dates[5], dates[6]

        result = run(gapped, Scheduled({decided: {"GAP": 1.0}}))

        assert len(result.fills) == 1
        fill = result.fills.iloc[0]
        assert fill["date"] == filled
        assert fill["price"] == 1006.0
        assert fill["qty"] == pytest.approx(100_000 / 1006.0, rel=1e-12)
        # Nothing happens to the account on the decision date itself...
        assert result.equity.loc[:decided].eq(100_000).all()
        assert result.positions.loc[decided, "GAP"] == 0.0
        # ...and the next close marks the new position at that bar's close (506).
        assert result.equity.loc[filled] == pytest.approx(100_000 / 1006.0 * 506.0, rel=1e-12)

    def test_every_fill_happens_at_its_own_dates_open_one_bar_after_a_decision(self, market_data):
        result = run(market_data, MomentumStrategy(lookback=5))
        run_dates = list(result.equity.index)

        assert len(result.fills) > 50
        for fill in result.fills.itertuples():
            assert fill.price == market_data.bar(fill.date)[fill.symbol].open
            decided_on = run_dates[run_dates.index(fill.date) - 1]
            assert decided_on in result.decisions.index

    def test_weights_right_after_each_fill_match_the_previous_decision(self, market_data):
        result = run(market_data, MomentumStrategy(lookback=5))
        symbols = list(market_data.symbols)
        dates = list(result.equity.index)

        for decided, filled in zip(dates[:-1], dates[1:], strict=True):
            if decided not in result.decisions.index:
                continue
            opens = np.array([market_data.bar(filled)[s].open for s in symbols])
            held = result.positions.loc[filled].to_numpy()
            value_at_open = result.cash.loc[filled] + held @ opens
            np.testing.assert_allclose(
                held * opens / value_at_open, result.decisions.loc[decided], atol=1e-9
            )

    def test_the_final_decision_is_recorded_but_never_executed(self, market_data):
        end = market_data.calendar[-10]
        result = run(market_data, EqualWeight(), end=end)

        assert result.decisions.index[-1] == end
        assert result.fills["date"].max() <= end
        assert result.equity.index[-1] == end


# ---------------------------------------------------------------------------
# Accounting identities
# ---------------------------------------------------------------------------


class TestAccounting:
    def test_equity_is_cash_plus_positions_at_the_close(self, market_data):
        result = run(market_data, MomentumStrategy(lookback=5))
        closes = pd.DataFrame({s: market_data.frame(s)["close"] for s in market_data.symbols})

        rebuilt = result.cash + (result.positions * closes).sum(axis=1)

        np.testing.assert_allclose(rebuilt, result.equity, rtol=1e-12)

    def test_cash_is_starting_cash_minus_everything_spent(self, market_data):
        partly_invested = Constant({"AAA": 0.3, "BBB": 0.3})  # costs need cash headroom in v1
        result = run(market_data, partly_invested, execution=NextOpenExecution(BpsCost(5)))
        fills = result.fills

        spent = (fills["qty"] * fills["price"] + fills["cost"]).sum()

        assert result.cash.iloc[-1] == pytest.approx(100_000 - spent, rel=1e-12)

    def test_positions_are_the_running_sum_of_fills(self, market_data):
        result = run(market_data, MomentumStrategy(lookback=5))
        by_day = result.fills.pivot_table(
            index="date", columns="symbol", values="qty", aggfunc="sum"
        )
        running = by_day.reindex(result.positions.index).fillna(0.0).cumsum()

        np.testing.assert_allclose(running[list(market_data.symbols)], result.positions, atol=1e-9)

    def test_returns_are_close_to_close_starting_from_initial_cash(self, market_data):
        result = run(market_data, EqualWeight(), initial_cash=50_000)

        assert result.returns.iloc[0] == result.equity.iloc[0] / 50_000 - 1
        np.testing.assert_allclose(
            result.returns.iloc[1:], result.equity.pct_change().iloc[1:], rtol=1e-12
        )

    def test_costs_are_charged_and_recorded(self, market_data):
        strategy = Constant({"AAA": 0.3, "BBB": 0.3})
        free = run(market_data, strategy)
        costly = run(market_data, strategy, execution=NextOpenExecution(BpsCost(10)))
        fills = costly.fills

        np.testing.assert_allclose(fills["cost"], (fills["qty"] * fills["price"]).abs() * 0.001)
        assert costly.equity.iloc[-1] < free.equity.iloc[-1]

    def test_a_cost_that_overdraws_a_fully_invested_account_is_an_error(self, market_data):
        with pytest.raises(EngineError, match=r"cash is -\d"):
            run(market_data, Constant({"AAA": 1.0}), execution=NextOpenExecution(BpsCost(10)))


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_first_call_comes_once_warmup_bars_are_visible_and_never_before(self, market_data):
        recorder = Recorder(warmup=20)

        result = run(market_data, recorder)

        assert recorder.calls[0][1:] == (market_data.calendar[19], 20)
        assert len(recorder.calls) == len(market_data) - 19
        assert result.decisions.index[0] == market_data.calendar[19]
        assert result.config["first_decision"] == "2020-01-29"

    def test_nothing_trades_before_the_first_decision(self, market_data):
        result = run(market_data, Constant({"AAA": 1.0}, warmup=20))

        assert result.fills["date"].min() == market_data.calendar[20]
        assert result.equity.loc[: market_data.calendar[19]].eq(100_000).all()

    def test_history_before_start_counts_toward_warmup(self, market_data):
        start = market_data.calendar[30]
        recorder = Recorder(warmup=20)

        run(market_data, recorder, start=start)

        assert recorder.calls[0][1:] == (start, 31)

    @pytest.mark.parametrize("warmup", [0, 1])
    def test_warmup_of_zero_or_one_decides_on_the_first_date(self, market_data, warmup):
        result = run(market_data, Recorder(warmup=warmup))
        assert result.decisions.index[0] == market_data.start

    def test_warmup_longer_than_the_data_is_rejected(self, market_data):
        with pytest.raises(ValueError, match=r"needs warmup=261 bars .* would never trade"):
            run(market_data, Recorder(warmup=261))

    def test_asking_for_more_history_than_warmup_names_strategy_date_and_warmup(self, market_data):
        with pytest.raises(StrategyError) as info:
            run(market_data, Recorder(warmup=5, lookback=20))

        err = info.value
        assert err.strategy_name == "Recorder"
        assert err.date == market_data.calendar[4]
        assert str(err).startswith(
            "Recorder.target_weights failed on 2020-01-08 (5 bars visible, warmup=5): lookback=20"
        )
        assert "warmup should cover its longest lookback" in str(err)
        assert isinstance(err.__cause__, ValueError)

    def test_fit_is_called_once_with_the_first_decision_view(self, market_data):
        recorder = Recorder(warmup=10)
        run(market_data, recorder)
        assert recorder.fits == [(MarketView, market_data.calendar[9])]

    def test_strategy_failures_are_wrapped_with_context(self, market_data):
        with pytest.raises(
            StrategyError, match="Raises.target_weights failed on 2020-01-02"
        ) as info:
            run(market_data, Raises())
        assert isinstance(info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

INVALID_OUTPUTS = {
    "negative": ({"AAA": -0.1}, "AAA: weight -0.1 is negative"),
    "above-one": ({"AAA": 1.2}, "AAA: weight 1.2 is above 1"),
    "levered-sum": ({"AAA": 0.6, "BBB": 0.6}, "weights sum to 1.2"),
    "nan": ({"AAA": math.nan}, "AAA: weight nan is not finite"),
    "inf": ({"AAA": math.inf}, "AAA: weight inf is not finite"),
    "string": ({"AAA": "0.5"}, "AAA: weight '0.5' is not a number"),
    "bool": ({"AAA": True}, "AAA: weight True is not a number"),
    "unknown-symbol": ({"ZZZ": 0.1}, "'ZZZ' is not a visible symbol"),
    "not-a-dict": ([("AAA", 1.0)], "expected a dict of symbol -> weight, got list"),
    "none": (None, "expected a dict of symbol -> weight, got NoneType"),
}


class TestWeightValidation:
    @pytest.mark.parametrize(
        ("output", "problem"), INVALID_OUTPUTS.values(), ids=INVALID_OUTPUTS.keys()
    )
    def test_invalid_weights_stop_the_run_with_a_clear_error(self, market_data, output, problem):
        strategy = Returns(output)
        strategy.name = "bad-strategy"

        with pytest.raises(InvalidWeightsError) as info:
            run(market_data, strategy)

        err = info.value
        message = str(err)
        assert message.startswith("bad-strategy returned invalid target weights on 2020-01-02")
        assert problem in message
        assert repr(output)[:20] in message
        assert err.strategy_name == "bad-strategy"
        assert err.date == market_data.start
        assert err.weights is output

    def test_a_symbol_that_has_not_listed_yet_is_rejected(self, staggered_market_data):
        early = Scheduled({"2020-03-02": {"MID": 0.5}})  # MID lists on 2020-04-01
        with pytest.raises(InvalidWeightsError, match="'MID' is not a visible symbol"):
            run(staggered_market_data, early, end="2020-07-31")

    @pytest.mark.parametrize(
        "weights",
        [{"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}, {"AAA": 0.5, "BBB": 0.5 + 1e-12}],
        ids=["thirds", "float-noise"],
    )
    def test_float_noise_in_a_fully_invested_target_is_accepted(self, market_data, weights):
        result = run(market_data, Constant(weights))
        assert result.cash.min() >= -1e-6

    def test_valid_mixed_number_types_are_accepted(self, market_data):
        result = run(market_data, Constant({"AAA": np.float32(0.25), "BBB": 0, "CCC": 0.5}))
        assert result.decisions.iloc[0].tolist() == [0.25, 0.0, 0.5]


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------


class TestCalendars:
    def test_a_symbol_that_stops_trading_mid_run_is_rejected(self, staggered_market_data):
        with pytest.raises(ValueError, match=r"GONE has no bar on .*\(first: 2020-08-03\)"):
            run(staggered_market_data, EqualWeight())

    def test_a_symbol_listing_mid_run_becomes_tradable(self, staggered_market_data):
        result = run(staggered_market_data, EqualWeight(), end="2020-07-31")
        mid = result.positions["MID"]

        assert (mid.loc[:"2020-04-01"] == 0).all()  # listed 04-01, first decision that close
        assert (mid.loc["2020-04-02":] > 0).all()  # filled at the next open
        assert (result.positions["LATE"] == 0).all()  # lists after the run ends


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class TestResult:
    def test_frames_are_labelled_and_aligned(self, market_data):
        result = run(market_data, MomentumStrategy(lookback=5))

        assert result.strategy_name == "momentum(5)"
        assert result.equity.index.equals(market_data.calendar)
        assert result.equity.index.name == "date"
        for frame in (result.positions, result.decisions):
            assert list(frame.columns) == list(market_data.symbols)
            assert frame.columns.name == "symbol"
        assert result.positions.index.equals(result.equity.index)
        assert tuple(result.fills.columns) == FILL_COLUMNS
        assert result.fills["date"].dtype == "datetime64[ns]"

    def test_a_run_with_no_trades_has_an_empty_but_typed_fills_table(self, market_data):
        result = run(market_data, Recorder())
        assert result.fills.empty
        assert tuple(result.fills.columns) == FILL_COLUMNS
        assert result.equity.eq(100_000).all()

    def test_config_records_the_setup_and_is_read_only(self, market_data):
        result = run(market_data, Recorder(warmup=3), start="2020-02-03", end="2020-11-30")

        assert dict(result.config) == {
            "strategy": "Recorder",
            "strategy_class": "Recorder",
            "warmup": 3,
            "initial_cash": 100_000.0,
            "execution": "NextOpenExecution(cost_model=ZeroCost())",
            "symbols": ["AAA", "BBB", "CCC"],
            "start": "2020-02-03",
            "end": "2020-11-30",
            "first_decision": "2020-02-03",
            "timing": "decide at close t; size and fill at open t+1",
            "share_sizing": "fractional",
        }
        assert isinstance(result.config, MappingProxyType)
        with pytest.raises(TypeError):
            result.config["warmup"] = 0  # type: ignore[index]

    def test_result_is_frozen_and_has_a_short_repr(self, market_data):
        result = run(market_data, Recorder())
        with pytest.raises(AttributeError):
            result.strategy_name = "other"  # type: ignore[misc]
        assert repr(result) == (
            "BacktestResult(strategy='Recorder', 2020-01-02 to 2020-12-30, 260 bars, 0 fills, "
            "final equity 100,000.00)"
        )

    def test_the_strategy_only_ever_receives_a_view(self, market_data):
        recorder = Recorder()
        run(market_data, recorder)
        assert {kind for kind, _, _ in recorder.calls} == {MarketView}
