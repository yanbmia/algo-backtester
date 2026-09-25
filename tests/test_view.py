"""MarketView's API: construction, truncation, history(), panel(), and its public surface.

The lookahead guarantees themselves are proven in test_lookahead.py. These tests
pin down the behaviour those guarantees are built on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data import MarketData, MarketView
from backtester.data.market_data import OHLCV_COLUMNS


class TestConstruction:
    def test_only_marketdata_can_create_a_view(self, market_data):
        with pytest.raises(TypeError, match=r"MarketData.view\(cutoff\)"):
            MarketView(market_data.start, {})

    def test_now_is_the_cutoff(self, market_data):
        t = market_data.calendar[42]
        assert market_data.view(t).now == t
        assert market_data.view(t.strftime("%Y-%m-%d")).now == t

    @pytest.mark.parametrize(
        "cutoff",
        ["2020-01-04", "2019-12-31", "2021-06-01"],
        ids=["weekend", "before-start", "after-end"],
    )
    def test_cutoff_must_be_a_trading_date(self, market_data, cutoff):
        with pytest.raises(KeyError, match=f"{cutoff} is not a trading date"):
            market_data.view(cutoff)

    def test_cutoff_must_be_timezone_naive(self, market_data):
        with pytest.raises(ValueError, match="timezone-naive"):
            market_data.view(pd.Timestamp("2020-01-02", tz="UTC"))

    def test_repr(self, market_data):
        assert repr(market_data.view("2020-01-02")) == (
            "MarketView(now=2020-01-02, symbols=['AAA', 'BBB', 'CCC'])"
        )


class TestTruncation:
    def test_first_date_shows_exactly_one_bar(self, market_data):
        view = market_data.view(market_data.start)
        for symbol in market_data.symbols:
            assert len(view.history(symbol)) == 1

    def test_cutoff_is_inclusive(self, market_data):
        t = market_data.calendar[42]
        history = market_data.view(t).history("AAA")
        assert len(history) == 43
        assert history.index[-1] == t

    def test_last_date_shows_everything(self, market_data):
        view = market_data.view(market_data.end)
        for symbol in market_data.symbols:
            for field in OHLCV_COLUMNS:
                pd.testing.assert_series_equal(
                    view.history(symbol, field),
                    market_data.frame(symbol)[field],
                    check_freq=False,
                )

    def test_symbols_lists_only_what_has_traded(self, staggered_market_data):
        data = staggered_market_data
        assert data.view("2020-03-31").symbols == ("EARLY", "GONE")
        assert data.view("2020-04-01").symbols == ("EARLY", "MID", "GONE")
        assert data.view("2020-09-01").symbols == ("EARLY", "MID", "LATE", "GONE")

    def test_a_stopped_symbol_stays_visible_with_its_old_history(self, staggered_market_data):
        history = staggered_market_data.view("2020-10-01").history("GONE")
        assert history.index[-1] == pd.Timestamp("2020-07-31")

    def test_unlisted_and_nonexistent_symbols_are_indistinguishable(self, staggered_market_data):
        view = staggered_market_data.view("2020-03-31")

        with pytest.raises(KeyError) as not_yet:
            view.history("LATE")
        with pytest.raises(KeyError) as never:
            view.history("ZZZ")

        assert str(not_yet.value).replace("LATE", "X") == str(never.value).replace("ZZZ", "X")
        assert "no data for 'LATE' at or before 2020-03-31" in str(not_yet.value)


class TestHistory:
    def test_default_field_is_close(self, market_data):
        view = market_data.view(market_data.calendar[20])
        pd.testing.assert_series_equal(view.history("BBB"), view.history("BBB", "close"))

    def test_series_is_named_and_date_indexed(self, market_data):
        history = market_data.view(market_data.calendar[20]).history("BBB", "volume")
        assert history.name == "volume"
        assert history.index.name == "date"
        assert history.dtype == "float64"

    def test_lookback_returns_the_last_n_bars(self, market_data):
        view = market_data.view(market_data.calendar[20])
        full, last5 = view.history("AAA"), view.history("AAA", lookback=5)
        pd.testing.assert_series_equal(last5, full.iloc[-5:])

    def test_lookback_of_everything_visible_is_allowed(self, market_data):
        view = market_data.view(market_data.calendar[20])
        assert len(view.history("AAA", lookback=21)) == 21

    def test_lookback_beyond_visible_history_raises(self, market_data):
        view = market_data.view(market_data.calendar[20])
        with pytest.raises(ValueError, match=r"lookback=22 .* only 21 bar\(s\) are visible"):
            view.history("AAA", lookback=22)

    @pytest.mark.parametrize(
        ("lookback", "error"),
        [(0, ValueError), (-3, ValueError), (True, TypeError), (2.0, TypeError)],
        ids=["zero", "negative", "bool", "float"],
    )
    def test_invalid_lookback_raises(self, market_data, lookback, error):
        with pytest.raises(error, match="lookback"):
            market_data.view(market_data.calendar[20]).history("AAA", lookback=lookback)

    def test_numpy_integer_lookback_is_accepted(self, market_data):
        view = market_data.view(market_data.calendar[20])
        assert len(view.history("AAA", lookback=np.int64(3))) == 3

    def test_unknown_field_raises(self, market_data):
        with pytest.raises(ValueError, match=r"unknown field\(s\) \['adj_close'\]"):
            market_data.view(market_data.start).history("AAA", "adj_close")

    def test_field_must_be_a_string(self, market_data):
        with pytest.raises(TypeError, match="field must be a string"):
            market_data.view(market_data.start).history("AAA", ["close"])


class TestPanel:
    def test_full_panel_at_the_last_date_equals_to_panel(self, market_data, staggered_market_data):
        for data in (market_data, staggered_market_data):
            pd.testing.assert_frame_equal(data.view(data.end).panel(), data.to_panel())

    def test_panel_is_truncated_at_now(self, staggered_market_data):
        t = pd.Timestamp("2020-05-15")
        panel = staggered_market_data.view(t).panel()
        expected = staggered_market_data.to_panel().loc[:t]
        pd.testing.assert_frame_equal(panel, expected)

    def test_field_selection_and_order(self, market_data):
        panel = market_data.view(market_data.calendar[20]).panel(["close", "open"])
        assert list(panel.columns) == ["close", "open"]

    def test_lookback_counts_calendar_dates_and_never_fills_gaps(self, staggered_market_data):
        # 2020-04-01 is MID's first day, so over the 3 dates ending there, MID has
        # one row, not three rows padded with NaN.
        panel = staggered_market_data.view("2020-04-01").panel(["close"], lookback=3)

        dates = panel.index.get_level_values("date").unique()
        assert list(dates) == list(pd.to_datetime(["2020-03-30", "2020-03-31", "2020-04-01"]))
        assert panel.xs("MID", level="symbol").index.tolist() == [pd.Timestamp("2020-04-01")]
        assert not panel.isna().any().any()

    def test_lookback_drops_symbols_with_no_bars_in_the_window(self, staggered_market_data):
        panel = staggered_market_data.view("2020-10-01").panel(lookback=5)
        assert "GONE" not in panel.index.get_level_values("symbol")

    def test_lookback_beyond_visible_dates_raises(self, market_data):
        view = market_data.view(market_data.calendar[4])
        with pytest.raises(ValueError, match=r"only 5 bar\(s\) are visible"):
            view.panel(lookback=6)

    @pytest.mark.parametrize(
        ("fields", "error", "match"),
        [
            ("close", TypeError, "not a string"),
            ([], ValueError, "at least one field"),
            (["close", "close"], ValueError, "duplicated"),
            (["close", "vwap"], ValueError, "unknown field"),
        ],
        ids=["bare-string", "empty", "duplicate", "unknown"],
    )
    def test_invalid_fields_raise(self, market_data, fields, error, match):
        with pytest.raises(error, match=match):
            market_data.view(market_data.start).panel(fields)


class TestSurface:
    def test_public_api_is_exactly_now_symbols_history_panel(self, market_data):
        view = market_data.view(market_data.start)
        public = {name for name in dir(view) if not name.startswith("_")}
        assert public == {"now", "symbols", "history", "panel"}

    def test_view_holds_no_reference_to_its_parent(self, market_data):
        view = market_data.view(market_data.calendar[20])
        held = [getattr(view, slot) for slot in MarketView.__slots__]
        held += list(view._windows.values())
        assert not any(isinstance(obj, MarketData) for obj in held)

    def test_view_is_immutable(self, market_data):
        view = market_data.view(market_data.start)
        with pytest.raises(AttributeError):
            view.now = market_data.end
        with pytest.raises(AttributeError):
            view.extra = "anything"

    def test_views_are_independent(self, market_data):
        early = market_data.view(market_data.calendar[10])
        market_data.view(market_data.end)  # creating a later view changes nothing earlier
        assert early.history("AAA").index[-1] == market_data.calendar[10]
