"""Deterministic synthetic market data for tests. No network access, ever.

- :func:`make_ohlcv` builds a realistic, *valid* frame from a fixed seed.
- :func:`corrupt` and the ``*_CASES`` tables break exactly one invariant at a
  time, so every validation rule can be tested in isolation.
- :class:`FakeDownloader` stands in for yfinance: it serves frames in yfinance's
  raw response shape and records every call, so tests can tell cache hits from
  misses.

Fixtures built from these helpers live in ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.data import Rule

SYMBOLS: tuple[str, ...] = ("AAA", "BBB", "CCC")

#: The row every corruption case breaks (a date well inside the fixture range).
BAD_ROW = 10


def make_ohlcv(
    n_days: int = 260,
    start: str = "2020-01-02",
    seed: int = 0,
    start_price: float = 100.0,
    daily_vol: float = 0.012,
) -> pd.DataFrame:
    """Valid daily bars following a seeded geometric random walk.

    Business-day dates; each open gaps slightly from the previous close; the
    high/low wicks extend beyond the open-close body; volume is integer-valued.
    Identical arguments always produce an identical frame.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    close = start_price * np.exp(np.cumsum(rng.normal(0.0003, daily_vol, n_days)))
    prev_close = np.concatenate([[start_price], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, daily_vol / 3, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype("float64")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def corrupt(frame: pd.DataFrame, row: int = BAD_ROW, **values: float) -> pd.DataFrame:
    """A copy of ``frame`` with the given column values overwritten at position ``row``."""
    out = frame.copy()
    for column, value in values.items():
        out.iloc[row, out.columns.get_loc(column)] = value
    return out


def to_yfinance_format(
    frame: pd.DataFrame, symbol: str, *, multi_level: bool = True, tz: str | None = None
) -> pd.DataFrame:
    """Reshape a clean frame into what ``yf.download`` returns.

    Capitalized columns in yfinance's order, a "Date" index, optionally a
    (Price, Ticker) column MultiIndex (the default since yfinance 0.2.48) and a
    timezone-aware index.
    """
    raw = frame.rename(columns=str.title)[["Close", "High", "Low", "Open", "Volume"]]
    raw.index = raw.index.rename("Date")
    if tz is not None:
        raw.index = raw.index.tz_localize(tz)
    if multi_level:
        raw.columns = pd.MultiIndex.from_product([raw.columns, [symbol]], names=["Price", "Ticker"])
    return raw


class FakeDownloader:
    """Drop-in replacement for ``yfinance_download`` that records its calls.

    Returns the rows of ``frames[symbol]`` within ``[start, end]`` (inclusive),
    preserving their order and any duplicates, in yfinance's raw shape. An
    unknown symbol yields an empty frame, as yfinance does.
    """

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        multi_level: bool = True,
        tz: str | None = None,
    ) -> None:
        self.frames = dict(frames)
        self.multi_level = multi_level
        self.tz = tz
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def __call__(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if symbol not in self.frames:
            return pd.DataFrame()
        frame = self.frames[symbol]
        in_range = (frame.index >= start) & (frame.index <= end)
        return to_yfinance_format(
            frame.loc[in_range], symbol, multi_level=self.multi_level, tz=self.tz
        )

    def calls_for(self, symbol: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        return [c for c in self.calls if c[0] == symbol]


# ---------------------------------------------------------------------------
# Invalid-frame cases: each breaks exactly one rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvalidCase:
    id: str
    mutate: Callable[[pd.DataFrame], pd.DataFrame]
    rule: Rule
    fragment: str  # text the error message must contain


def _duplicate_row(f: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([f.iloc[: BAD_ROW + 1], f.iloc[BAD_ROW:]])


def _swap_rows(f: pd.DataFrame) -> pd.DataFrame:
    order = list(range(len(f)))
    order[BAD_ROW], order[BAD_ROW + 1] = order[BAD_ROW + 1], order[BAD_ROW]
    return f.iloc[order]


def _high_below_close(f: pd.DataFrame) -> pd.DataFrame:
    return corrupt(f, high=f["close"].iloc[BAD_ROW] * 0.99)


def _high_below_open(f: pd.DataFrame) -> pd.DataFrame:
    return corrupt(f, open=f["high"].iloc[BAD_ROW] * 1.01)


def _low_above_close(f: pd.DataFrame) -> pd.DataFrame:
    return corrupt(f, low=f["close"].iloc[BAD_ROW] * 1.01)


def _low_above_open(f: pd.DataFrame) -> pd.DataFrame:
    return corrupt(f, open=f["low"].iloc[BAD_ROW] * 0.99)


#: Bad *values* in an otherwise well-formed frame. These can arrive from a real
#: data source, so they are tested both on MarketData and through the loader.
VALUE_CASES: tuple[InvalidCase, ...] = (
    InvalidCase(
        "nan_close",
        lambda f: corrupt(f, close=np.nan),
        Rule.FINITE_VALUES,
        "close contains 1 NaN/inf",
    ),
    InvalidCase(
        "nan_open",
        lambda f: corrupt(f, open=np.nan),
        Rule.FINITE_VALUES,
        "open contains 1 NaN/inf",
    ),
    InvalidCase(
        "inf_high",
        lambda f: corrupt(f, high=np.inf),
        Rule.FINITE_VALUES,
        "high contains 1 NaN/inf",
    ),
    InvalidCase(
        "nan_volume",
        lambda f: corrupt(f, volume=np.nan),
        Rule.FINITE_VALUES,
        "volume contains 1 NaN/inf",
    ),
    InvalidCase(
        "negative_low",
        lambda f: corrupt(f, low=-1.0),
        Rule.POSITIVE_PRICES,
        "must all be > 0",
    ),
    InvalidCase(
        "zero_prices",
        lambda f: corrupt(f, open=0.0, high=0.0, low=0.0, close=0.0),
        Rule.POSITIVE_PRICES,
        "must all be > 0",
    ),
    InvalidCase(
        "high_below_close",
        _high_below_close,
        Rule.HIGH_BOUND,
        "high must be >= max(open, close)",
    ),
    InvalidCase(
        "high_below_open",
        _high_below_open,
        Rule.HIGH_BOUND,
        "high must be >= max(open, close)",
    ),
    InvalidCase(
        "low_above_close",
        _low_above_close,
        Rule.LOW_BOUND,
        "low must be <= min(open, close)",
    ),
    InvalidCase(
        "low_above_open",
        _low_above_open,
        Rule.LOW_BOUND,
        "low must be <= min(open, close)",
    ),
    InvalidCase(
        "negative_volume",
        lambda f: corrupt(f, volume=-100.0),
        Rule.NON_NEGATIVE_VOLUME,
        "volume must be >= 0",
    ),
    InvalidCase(
        "duplicate_date",
        _duplicate_row,
        Rule.UNIQUE_DATES,
        "appear more than once",
    ),
    InvalidCase(
        "unsorted_dates",
        _swap_rows,
        Rule.SORTED_DATES,
        "must be in ascending order",
    ),
)


def _with_nat(f: pd.DataFrame) -> pd.DataFrame:
    index = f.index.to_series()
    index.iloc[BAD_ROW] = pd.NaT
    return f.set_axis(pd.DatetimeIndex(index), axis=0)


#: Malformed frames (wrong shape or types). The loader normalizes yfinance's
#: shape before validating, so these are tested on MarketData directly.
STRUCTURAL_CASES: tuple[InvalidCase, ...] = (
    InvalidCase("empty", lambda f: f.iloc[:0], Rule.NON_EMPTY, "no rows"),
    InvalidCase(
        "missing_column",
        lambda f: f.drop(columns="volume"),
        Rule.COLUMNS,
        "missing ['volume']",
    ),
    InvalidCase(
        "extra_column",
        lambda f: f.assign(adj_close=f["close"]),
        Rule.COLUMNS,
        "unexpected ['adj_close']",
    ),
    InvalidCase(
        "capitalized_columns",
        lambda f: f.rename(columns=str.title),
        Rule.COLUMNS,
        "missing ['open', 'high', 'low', 'close', 'volume']",
    ),
    InvalidCase(
        "integer_index",
        lambda f: f.reset_index(drop=True),
        Rule.DATETIME_INDEX,
        "got RangeIndex",
    ),
    InvalidCase(
        "string_dates",
        lambda f: f.set_axis(f.index.strftime("%Y-%m-%d"), axis=0),
        Rule.DATETIME_INDEX,
        "must be a pandas DatetimeIndex",
    ),
    InvalidCase(
        "tz_aware_index",
        lambda f: f.tz_localize("America/New_York"),
        Rule.DATETIME_INDEX,
        "must be timezone-naive",
    ),
    InvalidCase("nat_in_index", _with_nat, Rule.DATETIME_INDEX, "1 NaT"),
    InvalidCase(
        "string_prices",
        lambda f: f.assign(close=f["close"].astype(str)),
        Rule.NUMERIC_DTYPE,
        "non-numeric dtypes",
    ),
    InvalidCase(
        "bool_volume",
        lambda f: f.assign(volume=f["volume"] > 0),
        Rule.NUMERIC_DTYPE,
        "'volume': 'bool'",
    ),
)
