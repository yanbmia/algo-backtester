"""Metrics against values worked out by hand.

Each expected value below comes from arithmetic written out in the comments,
not from calling a second copy of the same formula.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    annualized_vol,
    cagr,
    drawdown_series,
    exposure,
    max_drawdown,
    sharpe,
    summarize,
    total_return,
)
from backtester.results import FILL_COLUMNS, BacktestResult

DATES = pd.bdate_range("2021-01-04", periods=12, name="date")


def series(values: list[float], name: str = "equity") -> pd.Series:
    return pd.Series(values, index=DATES[: len(values)], name=name, dtype="float64")


class TestReturns:
    def test_total_return(self):
        # 100 -> 121 is +21%, whatever happened in between.
        assert total_return(series([100, 110, 99, 121])) == pytest.approx(0.21)

    @pytest.mark.parametrize(
        ("periods_per_year", "expected"),
        [
            (4, 0.4641),  # 4 periods = 1 year: 1.1^4 - 1
            (2, 0.21),  # 4 periods = 2 years: sqrt(1.4641) - 1
            (8, 1.14358881),  # 4 periods = half a year: 1.4641^2 - 1
        ],
    )
    def test_cagr_counts_periods_not_calendar_days(self, periods_per_year, expected):
        equity = series([100, 110, 121, 133.1, 146.41])  # +10% per period
        assert cagr(equity, periods_per_year) == pytest.approx(expected)

    def test_cagr_over_exactly_one_trading_year(self):
        # 253 closes span 252 periods = 1 year at the default 252/year.
        dates = pd.bdate_range("2021-01-04", periods=253)
        values = np.full(253, 100.0)
        values[-1] = 110.0
        assert cagr(pd.Series(values, index=dates)) == pytest.approx(0.10)

    def test_cagr_of_a_total_loss_is_minus_one(self):
        assert cagr(series([100, 50, 0]), periods_per_year=2) == -1.0


class TestVolatilityAndSharpe:
    def test_annualized_vol_uses_sample_std(self):
        # mean 0; squared deviations 4 x 0.0001; sample variance 0.0004 / 3
        # => std 0.02 / sqrt(3); x sqrt(252) => 0.02 * sqrt(84)
        returns = series([0.01, -0.01, 0.01, -0.01], "returns")
        assert annualized_vol(returns) == pytest.approx(0.02 * math.sqrt(84))

    def test_sharpe_by_hand(self):
        # mean 0.015; deviations .005, -.015, -.005, .015; squares sum to 5e-4
        # sample std = sqrt(5e-4 / 3) = 0.01 * sqrt(5/3)
        # Sharpe = 0.015 / (0.01 * sqrt(5/3)) * sqrt(4) = 3 * sqrt(3/5)
        returns = series([0.02, 0.00, 0.01, 0.03], "returns")
        assert sharpe(returns, periods_per_year=4) == pytest.approx(3 * math.sqrt(0.6))

    def test_sharpe_converts_the_risk_free_rate_geometrically(self):
        # rf_annual = 1.01^4 - 1 is exactly 1% per period at 4 periods/year, so the
        # excess returns are .01, -.01, 0, .02: mean .005, same std as above
        # => 0.005 / (0.01 * sqrt(5/3)) * 2 = sqrt(3/5)
        returns = series([0.02, 0.00, 0.01, 0.03], "returns")
        rf_annual = 1.01**4 - 1
        assert sharpe(returns, rf_annual, periods_per_year=4) == pytest.approx(math.sqrt(0.6))

    def test_sharpe_matches_the_statistics_module(self):
        rng = np.random.default_rng(11)
        values = rng.normal(0.0005, 0.01, 500)
        rf_period = 1.03 ** (1 / 252) - 1
        excess = [v - rf_period for v in values]
        expected = statistics.mean(excess) / statistics.stdev(excess) * math.sqrt(252)

        returns = pd.Series(values, index=pd.bdate_range("2021-01-04", periods=500))
        assert sharpe(returns, rf_annual=0.03) == pytest.approx(expected, rel=1e-10)

    def test_sharpe_of_a_flat_strategy_is_undefined(self):
        assert math.isnan(sharpe(series([0.0] * 5, "returns")))


class TestDrawdown:
    def test_drawdown_series(self):
        # running peak: 100, 120, 120, 120, 130, 130
        equity = series([100, 120, 90, 110, 130, 65])
        expected = [0.0, 0.0, -0.25, -1 / 12, 0.0, -0.5]
        np.testing.assert_allclose(drawdown_series(equity), expected)
        assert drawdown_series(equity).name == "drawdown"

    def test_max_drawdown_returns_depth_peak_and_trough(self):
        # deepest fall is 130 -> 65, from DATES[4] to DATES[5]
        assert max_drawdown(series([100, 120, 90, 110, 130, 65])) == (-0.5, DATES[4], DATES[5])

    def test_max_drawdown_picks_the_deepest_not_the_latest(self):
        # 120 -> 90 (-25%) beats the later 120 -> 100 (-16.7%)
        assert max_drawdown(series([100, 120, 90, 120, 100])) == (-0.25, DATES[1], DATES[2])

    def test_peak_is_the_last_date_at_the_high_before_the_fall(self):
        depth, peak, trough = max_drawdown(series([100, 120, 120, 90, 95]))
        assert (depth, peak, trough) == (-0.25, DATES[2], DATES[3])

    def test_no_drawdown(self):
        depth, peak, trough = max_drawdown(series([100, 100, 110]))
        assert depth == 0.0
        assert peak is pd.NaT
        assert trough is pd.NaT


class TestExposure:
    def test_fraction_of_days_with_any_position(self):
        positions = pd.DataFrame(
            {"AAA": [0.0, 5.0, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0, 2.0]}, index=DATES[:4]
        )
        assert exposure(positions) == 0.5

    def test_needs_positions(self):
        with pytest.raises(ValueError, match="non-empty DataFrame"):
            exposure(pd.DataFrame())


class TestInputValidation:
    @pytest.mark.parametrize(
        ("call", "error", "match"),
        [
            (lambda: total_return(series([])), ValueError, "at least 1"),
            (lambda: total_return(series([100, math.nan])), ValueError, "NaN"),
            (lambda: total_return(series([0, 10])), ValueError, "start positive"),
            (lambda: total_return([100, 110]), TypeError, "pandas Series"),
            (lambda: cagr(series([100])), ValueError, "at least 2"),
            (lambda: cagr(series([100, -5])), ValueError, "negative final equity"),
            (lambda: cagr(series([100, 110]), 0), ValueError, "periods_per_year"),
            (lambda: annualized_vol(series([0.01], "r")), ValueError, "at least 2 returns"),
            (lambda: sharpe(series([0.01, math.inf], "r")), ValueError, "NaN or infinite"),
            (lambda: sharpe(series([0.01, 0.02], "r"), rf_annual=-1), ValueError, "rf_annual"),
        ],
        ids=[
            "empty",
            "nan",
            "non-positive-start",
            "not-a-series",
            "one-value-cagr",
            "negative-final",
            "zero-periods",
            "one-return",
            "inf-return",
            "rf-minus-one",
        ],
    )
    def test_invalid_inputs_raise(self, call, error, match):
        with pytest.raises(error, match=match):
            call()


def hand_built_result() -> BacktestResult:
    """Seven days; the strategy's first decision is on day 2 (two warmup days before it)."""
    dates = DATES[:7]
    equity = pd.Series([100, 100, 100, 110, 99, 121, 121], index=dates, name="equity", dtype=float)
    positions = pd.DataFrame({"AAA": [0, 0, 0, 1.0, 1.0, 1.0, 0]}, index=dates)
    decisions = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0, 0.0]}, index=dates[2:])
    fills = pd.DataFrame(
        [(dates[3], "AAA", 1.0, 100.0, 0.0), (dates[6], "AAA", -1.0, 121.0, 0.0)],
        columns=list(FILL_COLUMNS),
    )
    return BacktestResult(
        strategy_name="hand-built",
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        cash=pd.Series(0.0, index=dates),
        positions=positions,
        decisions=decisions,
        fills=fills,
        config={},
    )


class TestSummarize:
    def test_measures_the_evaluation_window_from_the_first_decision(self):
        row = summarize(hand_built_result(), periods_per_year=4)

        # Window: DATES[2]..DATES[6], equity 100, 110, 99, 121, 121 (warmup excluded).
        assert row.name == "hand-built"
        assert (row["start"], row["end"], row["periods"]) == (DATES[2], DATES[6], 4)
        assert row["total_return"] == pytest.approx(0.21)
        assert row["cagr"] == pytest.approx(0.21)  # 4 periods = 1 year at 4/year
        # Worst fall: 110 on DATES[3] to 99 on DATES[4] = -10%
        assert row["max_drawdown"] == pytest.approx(-0.1)
        assert (row["max_drawdown_peak"], row["max_drawdown_trough"]) == (DATES[3], DATES[4])
        assert row["exposure"] == pytest.approx(3 / 5)  # positions held on 3 of 5 window days
        assert row["n_fills"] == 2
        assert row["final_equity"] == 121.0

    def test_uses_the_window_returns_for_vol_and_sharpe(self):
        row = summarize(hand_built_result(), periods_per_year=4)
        window_returns = series([0.1, -0.1, 121 / 99 - 1, 0.0], "returns")

        assert row["sharpe"] == pytest.approx(sharpe(window_returns, periods_per_year=4))
        assert row["annualized_vol"] == pytest.approx(
            annualized_vol(window_returns, periods_per_year=4)
        )

    def test_rows_combine_into_a_comparison_table(self):
        first = summarize(hand_built_result())
        second = first.rename("another")

        table = pd.DataFrame([first, second])

        assert list(table.index) == ["hand-built", "another"]
        assert {"sharpe", "max_drawdown", "cagr"} <= set(table.columns)
