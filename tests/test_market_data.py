"""MarketData: construction, accessors, and one test per validation rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data import Bar, DataValidationError, MarketData, Rule
from backtester.data.market_data import MAX_EXAMPLES, validate_frame
from synthetic import BAD_ROW, SYMBOLS, InvalidCase, corrupt, make_ohlcv


def _date(frame: pd.DataFrame, row: int = BAD_ROW) -> str:
    return frame.index[row].strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_frames_build(self, universe):
        md = MarketData.from_frames(universe)

        assert md.symbols == SYMBOLS
        assert len(md) == 260
        assert md.calendar.name == "date"
        assert md.calendar.dtype == "datetime64[ns]"
        assert md.calendar.is_monotonic_increasing
        assert md.calendar.is_unique
        assert md.start == pd.Timestamp("2020-01-02")

    def test_direct_construction_is_blocked(self, frame):
        with pytest.raises(TypeError, match="from_frames"):
            MarketData({"AAA": frame})

    def test_empty_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="at least one symbol"):
            MarketData.from_frames({})

    def test_non_mapping_is_rejected(self, frame):
        with pytest.raises(TypeError, match="mapping of symbol"):
            MarketData.from_frames([frame])

    def test_non_dataframe_value_is_rejected(self, frame):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            MarketData.from_frames({"AAA": frame.to_numpy()})

    def test_input_frame_is_not_mutated(self, frame):
        before = frame.copy()
        MarketData.from_frames({"AAA": frame})
        pd.testing.assert_frame_equal(frame, before)

    def test_later_changes_to_input_do_not_leak_in(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        original_close = frame["close"].iloc[0]
        frame.iloc[0, frame.columns.get_loc("close")] = 999.0

        assert md.frame("AAA")["close"].iloc[0] == original_close
        assert md.bar(frame.index[0])["AAA"].close == original_close

    def test_integer_volume_is_accepted_and_stored_as_float(self, frame):
        md = MarketData.from_frames({"AAA": frame.astype({"volume": "int64"})})
        assert md.frame("AAA").dtypes.eq("float64").all()

    def test_column_order_is_normalized(self, frame):
        shuffled = frame[["volume", "close", "low", "high", "open"]]
        md = MarketData.from_frames({"AAA": shuffled})
        assert list(md.frame("AAA").columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_bar_returns_that_dates_values(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        t = frame.index[5]

        bar = md.bar(t)["AAA"]

        row = frame.iloc[5]
        assert bar == Bar(row["open"], row["high"], row["low"], row["close"], row["volume"])

    def test_bar_accepts_date_strings(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        assert md.bar(_date(frame, 5)) == md.bar(frame.index[5])

    def test_bar_on_a_non_trading_date_raises(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        with pytest.raises(KeyError, match="2020-01-04 is not a trading date"):
            md.bar("2020-01-04")  # a Saturday

    def test_bar_rejects_timezone_aware_dates(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        with pytest.raises(ValueError, match="timezone-naive"):
            md.bar(pd.Timestamp("2020-01-02", tz="UTC"))

    def test_calendar_is_union_and_bar_omits_symbols_without_a_bar(self):
        early = make_ohlcv(n_days=20, seed=1)
        late = make_ohlcv(n_days=15, start="2020-01-09", seed=2)  # "lists" 5 days later
        md = MarketData.from_frames({"EARLY": early, "LATE": late})

        assert md.calendar.equals(early.index.union(late.index).rename("date"))
        assert set(md.bar(early.index[0])) == {"EARLY"}
        assert set(md.bar(early.index[10])) == {"EARLY", "LATE"}

    def test_frame_returns_an_independent_copy(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        copy = md.frame("AAA")
        copy.iloc[:, :] = 0.0
        assert md.frame("AAA")["close"].iloc[0] == frame["close"].iloc[0]

    def test_frame_for_unknown_symbol_names_the_available_ones(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        with pytest.raises(KeyError, match=r"unknown symbol 'ZZZ'; available: \['AAA'\]"):
            md.frame("ZZZ")

    def test_to_panel_is_indexed_by_date_then_symbol(self, universe):
        panel = MarketData.from_frames(universe).to_panel()

        assert panel.index.names == ["date", "symbol"]
        assert len(panel) == sum(len(f) for f in universe.values())
        assert panel.index.is_monotonic_increasing
        assert (
            panel.loc[(universe["BBB"].index[3], "BBB"), "close"]
            == (universe["BBB"]["close"].iloc[3])
        )

    def test_repr_summarizes_the_dataset(self, frame):
        md = MarketData.from_frames({"AAA": frame})
        expected = f"MarketData(symbols=['AAA'], 2020-01-02 to {_date(frame, -1)}, 260 dates)"
        assert repr(md) == expected


# ---------------------------------------------------------------------------
# Validation: one test per rule (parametrized over the cases in synthetic.py)
# ---------------------------------------------------------------------------


class TestValidationRules:
    def test_valid_frame_has_no_violations(self, frame):
        assert validate_frame("AAA", frame) == []

    def test_bad_values_are_rejected_with_symbol_rule_and_date(
        self, frame, value_case: InvalidCase
    ):
        bad = value_case.mutate(frame)

        with pytest.raises(DataValidationError) as info:
            MarketData.from_frames({"AAA": bad})

        err = info.value
        # Exactly the intended rule fired, so each case tests one rule in isolation.
        assert err.rules == {value_case.rule}
        assert err.symbols == {"AAA"}
        message = str(err)
        assert f"[AAA] {value_case.rule}" in message
        assert value_case.fragment in message
        assert _date(frame) in message
        assert frame.index[BAD_ROW] in err.violations[0].dates

    def test_malformed_frames_are_rejected(self, frame, structural_case: InvalidCase):
        bad = structural_case.mutate(frame)

        with pytest.raises(DataValidationError) as info:
            MarketData.from_frames({"AAA": bad})

        assert info.value.rules == {structural_case.rule}
        assert f"[AAA] {structural_case.rule}" in str(info.value)
        assert structural_case.fragment in str(info.value)

    @pytest.mark.parametrize("symbol", ["", " SPY", "SPY ", 42])
    def test_invalid_symbol_keys_are_rejected(self, frame, symbol):
        with pytest.raises(DataValidationError) as info:
            MarketData.from_frames({symbol: frame})
        assert info.value.rules == {Rule.SYMBOL}

    def test_all_violations_across_symbols_are_reported_together(self, universe):
        universe["AAA"] = corrupt(universe["AAA"], close=np.nan)
        universe["BBB"] = corrupt(universe["BBB"], row=20, volume=-5.0)

        with pytest.raises(DataValidationError) as info:
            MarketData.from_frames(universe)

        err = info.value
        assert err.rules == {Rule.FINITE_VALUES, Rule.NON_NEGATIVE_VOLUME}
        assert err.symbols == {"AAA", "BBB"}
        assert "2 violation(s)" in str(err)

    def test_every_broken_rule_in_one_frame_is_reported(self, frame):
        bad = corrupt(frame, row=3, close=np.nan)
        bad = corrupt(bad, row=7, volume=-1.0)
        bad = corrupt(bad, row=9, low=bad["close"].iloc[9] * 1.05)

        with pytest.raises(DataValidationError) as info:
            MarketData.from_frames({"AAA": bad})

        assert info.value.rules == {
            Rule.FINITE_VALUES,
            Rule.NON_NEGATIVE_VOLUME,
            Rule.LOW_BOUND,
        }

    def test_message_quotes_a_few_rows_but_records_every_date(self, frame):
        bad = frame.copy()
        rows = list(range(10, 10 + MAX_EXAMPLES + 3))
        bad.iloc[rows, bad.columns.get_loc("close")] = np.nan

        with pytest.raises(DataValidationError) as info:
            MarketData.from_frames({"AAA": bad})

        (violation,) = info.value.violations
        assert len(violation.dates) == len(rows)
        assert len(violation.examples) == MAX_EXAMPLES
        assert "... and 3 more" in str(info.value)

    def test_bound_tolerance_absorbs_float_rounding(self, frame):
        body_top = max(frame["open"].iloc[BAD_ROW], frame["close"].iloc[BAD_ROW])
        noisy = corrupt(frame, high=body_top * (1 - 1e-12))
        assert validate_frame("AAA", noisy) == []

    def test_bound_tolerance_does_not_hide_a_one_cent_error(self, frame):
        body_top = max(frame["open"].iloc[BAD_ROW], frame["close"].iloc[BAD_ROW])
        bad = corrupt(frame, high=body_top - 0.01)
        assert {v.rule for v in validate_frame("AAA", bad)} == {Rule.HIGH_BOUND}
