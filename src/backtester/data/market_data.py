"""Validated container for daily OHLCV market data.

``MarketData`` holds the complete (date, symbol) panel. Only the backtest engine
should ever hold one: strategies receive a truncated, point-in-time
:class:`~backtester.data.view.MarketView` from :meth:`MarketData.view`, which is
what makes lookahead structurally hard.

The only supported constructor is :meth:`MarketData.from_frames`. It validates
every input frame and raises :class:`DataValidationError` listing *every* broken
invariant, with the symbol, rule, and offending dates. It never repairs data:
no sorting, de-duplication, forward-filling, clipping, or type coercion. If the
input is wrong, the caller finds out immediately and exactly where.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:
    from backtester.data.view import MarketView

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: Relative tolerance for the high/low bound checks. It exists only to absorb
#: floating-point rounding from yfinance's auto-adjustment (which multiplies all
#: four prices by the same factor, so errors are ~1e-16 relative). It is far too
#: small to hide a genuinely bad bar: a one-cent error on a $100 stock is 1e-4.
BOUND_RTOL: float = 1e-9

#: How many offending rows to quote in an error message per violation.
MAX_EXAMPLES: int = 5


class Rule(StrEnum):
    """Every invariant ``MarketData`` enforces. Values appear in error messages."""

    SYMBOL = "symbol"  # symbol is a non-empty string without surrounding whitespace
    NON_EMPTY = "non_empty"  # frame has at least one row
    COLUMNS = "columns"  # exactly open/high/low/close/volume, no duplicates
    DATETIME_INDEX = "datetime_index"  # tz-naive DatetimeIndex without NaT
    NUMERIC_DTYPE = "numeric_dtype"  # every OHLCV column is numeric (not bool/str/object)
    UNIQUE_DATES = "unique_dates"  # no date appears twice
    SORTED_DATES = "sorted_dates"  # dates ascending
    FINITE_VALUES = "finite_values"  # no NaN or +/-inf in any OHLCV column
    POSITIVE_PRICES = "positive_prices"  # open/high/low/close > 0
    HIGH_BOUND = "high_bound"  # high >= max(open, close)
    LOW_BOUND = "low_bound"  # low <= min(open, close)
    NON_NEGATIVE_VOLUME = "non_negative_volume"  # volume >= 0


@dataclass(frozen=True, slots=True)
class Bar:
    """One symbol's OHLCV values for one date."""

    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Violation:
    """One broken invariant for one symbol.

    ``dates`` holds every offending date (for programmatic use); ``examples``
    holds human-readable descriptions of the first few, for the error message.
    """

    symbol: str
    rule: Rule
    message: str
    dates: tuple[pd.Timestamp, ...] = ()
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        text = f"[{self.symbol}] {self.rule}: {self.message}"
        if self.examples:
            shown = "; ".join(self.examples)
            remaining = len(self.dates) - len(self.examples)
            if remaining > 0:
                shown += f"; ... and {remaining} more"
            text += f". First offending: {shown}"
        return text


class DataValidationError(ValueError):
    """Raised when market data breaks one or more invariants.

    Carries every violation found (not just the first), so one run tells you
    everything that is wrong with a dataset.
    """

    def __init__(self, violations: list[Violation]) -> None:
        if not violations:
            raise ValueError("DataValidationError requires at least one violation")
        self.violations: tuple[Violation, ...] = tuple(violations)
        super().__init__(self._render())

    @property
    def rules(self) -> set[Rule]:
        return {v.rule for v in self.violations}

    @property
    def symbols(self) -> set[str]:
        return {v.symbol for v in self.violations}

    def _render(self) -> str:
        n = len(self.violations)
        symbols = ", ".join(sorted(self.symbols))
        lines = [f"Market data failed validation ({n} violation(s); symbols: {symbols}):"]
        lines += [f"  - {v}" for v in self.violations]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_frame(symbol: str, frame: pd.DataFrame) -> list[Violation]:
    """Return every invariant ``frame`` breaks (an empty list means it is valid).

    Structural checks (symbol, rows, columns, index type, dtypes) run first and
    stop early, because value checks are meaningless on a malformed frame.
    Value checks all run, so every data problem is reported at once.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame for {symbol!r} must be a pandas DataFrame, got {type(frame).__name__}"
        )

    structural = _structural_violations(symbol, frame)
    if structural:
        return structural
    return _value_violations(symbol, frame)


def _structural_violations(symbol: object, frame: pd.DataFrame) -> list[Violation]:
    if not isinstance(symbol, str) or not symbol or symbol != symbol.strip():
        return [
            Violation(
                repr(symbol),
                Rule.SYMBOL,
                "symbol must be a non-empty string without surrounding whitespace",
            )
        ]

    if len(frame) == 0:
        return [Violation(symbol, Rule.NON_EMPTY, "frame has no rows")]

    columns = [str(c) for c in frame.columns]
    missing = [c for c in OHLCV_COLUMNS if c not in columns]
    unexpected = [c for c in columns if c not in OHLCV_COLUMNS]
    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    if missing or unexpected or duplicated:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unexpected:
            parts.append(f"unexpected {unexpected}")
        if duplicated:
            parts.append(f"duplicated {duplicated}")
        return [
            Violation(
                symbol,
                Rule.COLUMNS,
                f"columns must be exactly {list(OHLCV_COLUMNS)}; " + ", ".join(parts),
            )
        ]

    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return [
            Violation(
                symbol,
                Rule.DATETIME_INDEX,
                f"index must be a pandas DatetimeIndex, got {type(index).__name__} "
                f"(dtype {index.dtype})",
            )
        ]
    if index.tz is not None:
        return [
            Violation(
                symbol,
                Rule.DATETIME_INDEX,
                f"index must be timezone-naive (daily bars keyed by exchange-local date), "
                f"got tz={index.tz}",
            )
        ]
    if index.hasnans:
        n_nat = int(index.isna().sum())
        return [Violation(symbol, Rule.DATETIME_INDEX, f"index contains {n_nat} NaT value(s)")]

    bad_dtypes = {
        c: str(frame[c].dtype)
        for c in OHLCV_COLUMNS
        if not is_numeric_dtype(frame[c]) or is_bool_dtype(frame[c])
    }
    if bad_dtypes:
        return [
            Violation(
                symbol,
                Rule.NUMERIC_DTYPE,
                f"OHLCV columns must be numeric; got non-numeric dtypes {bad_dtypes}",
            )
        ]

    return []


def _value_violations(symbol: str, frame: pd.DataFrame) -> list[Violation]:
    violations: list[Violation] = []
    index = frame.index
    values = frame.loc[:, list(OHLCV_COLUMNS)].to_numpy(dtype="float64")
    o, h, lo, c, v = values.T

    def row_violation(rule: Rule, mask: np.ndarray, message: str) -> Violation:
        positions = np.flatnonzero(mask)
        return Violation(
            symbol,
            rule,
            message,
            dates=tuple(index[positions]),
            examples=tuple(_describe_row(index[p], values[p]) for p in positions[:MAX_EXAMPLES]),
        )

    # --- dates -----------------------------------------------------------
    dup_mask = index.duplicated(keep=False)
    if dup_mask.any():
        counts = pd.Series(index[dup_mask]).value_counts(sort=False)
        dup_dates = tuple(sorted(counts.index))
        violations.append(
            Violation(
                symbol,
                Rule.UNIQUE_DATES,
                f"dates must be unique; {len(dup_dates)} date(s) appear more than once",
                dates=dup_dates,
                examples=tuple(f"{_fmt(d)} ({counts[d]} rows)" for d in dup_dates[:MAX_EXAMPLES]),
            )
        )

    if not index.is_monotonic_increasing:
        # Positions where the date goes backwards relative to the previous row.
        breaks = np.flatnonzero(index[1:] < index[:-1]) + 1
        violations.append(
            Violation(
                symbol,
                Rule.SORTED_DATES,
                f"dates must be in ascending order; order breaks at {len(breaks)} position(s)",
                dates=tuple(index[breaks]),
                examples=tuple(
                    f"row {p}: {_fmt(index[p])} follows {_fmt(index[p - 1])}"
                    for p in breaks[:MAX_EXAMPLES]
                ),
            )
        )

    # --- finiteness (NaN / inf) --------------------------------------------
    for j, column in enumerate(OHLCV_COLUMNS):
        bad = ~np.isfinite(values[:, j])
        if bad.any():
            violations.append(
                row_violation(
                    Rule.FINITE_VALUES,
                    bad,
                    f"{column} contains {int(bad.sum())} NaN/inf value(s)",
                )
            )

    # Price relationships are only checked on rows whose prices are all finite,
    # so a NaN is reported once (as FINITE_VALUES), not again as a bound failure.
    finite_prices = np.isfinite(values[:, :4]).all(axis=1)

    non_positive = finite_prices & (values[:, :4] <= 0).any(axis=1)
    if non_positive.any():
        violations.append(
            row_violation(
                Rule.POSITIVE_PRICES,
                non_positive,
                f"open/high/low/close must all be > 0; {int(non_positive.sum())} bar(s) "
                "have a zero or negative price",
            )
        )

    high_bad = finite_prices & (h < np.maximum(o, c) * (1 - BOUND_RTOL))
    if high_bad.any():
        violations.append(
            row_violation(
                Rule.HIGH_BOUND,
                high_bad,
                f"high must be >= max(open, close); violated on {int(high_bad.sum())} bar(s)",
            )
        )

    low_bad = finite_prices & (lo > np.minimum(o, c) * (1 + BOUND_RTOL))
    if low_bad.any():
        violations.append(
            row_violation(
                Rule.LOW_BOUND,
                low_bad,
                f"low must be <= min(open, close); violated on {int(low_bad.sum())} bar(s)",
            )
        )

    volume_bad = np.isfinite(v) & (v < 0)
    if volume_bad.any():
        violations.append(
            row_violation(
                Rule.NON_NEGATIVE_VOLUME,
                volume_bad,
                f"volume must be >= 0; {int(volume_bad.sum())} negative value(s)",
            )
        )

    return violations


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _describe_row(ts: pd.Timestamp, row: np.ndarray) -> str:
    fields = ", ".join(
        f"{name}={value:.10g}" for name, value in zip(OHLCV_COLUMNS, row, strict=True)
    )
    return f"{_fmt(ts)} ({fields})"


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

_FROM_FRAMES = object()  # construction key: only from_frames may call __init__


class MarketData:
    """A validated, immutable (date, symbol) panel of daily OHLCV bars.

    - ``calendar``: the sorted union of every symbol's trading dates.
    - ``symbols``: symbols in the order they were supplied.
    - ``bar(t)``: the engine's per-date accessor.
    - ``view(t)``: the point-in-time view handed to strategies.

    With multiple symbols, calendars may differ (e.g. different listing dates);
    ``bar(t)`` then returns only the symbols that traded on ``t``.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], *, _key: object = None) -> None:
        if _key is not _FROM_FRAMES:
            raise TypeError(
                "Build MarketData with MarketData.from_frames(frames), which validates the data."
            )
        self._frames = frames
        self._symbols: tuple[str, ...] = tuple(frames)
        self._index = {s: f.index for s, f in frames.items()}
        self._values: dict[str, np.ndarray] = {}
        for symbol, frame in frames.items():
            arr = frame.to_numpy(dtype="float64", copy=True)
            arr.flags.writeable = False
            self._values[symbol] = arr
        calendar = reduce(lambda a, b: a.union(b), self._index.values())
        self._calendar = pd.DatetimeIndex(calendar, name="date")

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> MarketData:
        """Validate per-symbol OHLCV frames and build a ``MarketData``.

        Each frame must be indexed by a tz-naive ``DatetimeIndex`` (one row per
        trading date) with exactly the columns open, high, low, close, volume.

        Raises:
            DataValidationError: if any frame breaks any invariant in :class:`Rule`.
                All violations across all symbols are reported together.
            TypeError / ValueError: if ``frames`` is not a non-empty mapping.
        """
        if not isinstance(frames, Mapping):
            raise TypeError(
                f"frames must be a mapping of symbol -> DataFrame, got {type(frames).__name__}"
            )
        if not frames:
            raise ValueError("frames is empty: provide at least one symbol")

        violations: list[Violation] = []
        for symbol, frame in frames.items():
            violations.extend(validate_frame(symbol, frame))
        if violations:
            raise DataValidationError(violations)

        return cls({s: canonicalize_frame(f) for s, f in frames.items()}, _key=_FROM_FRAMES)

    # --- accessors -----------------------------------------------------------

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self._calendar

    @property
    def start(self) -> pd.Timestamp:
        return self._calendar[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._calendar[-1]

    def __len__(self) -> int:
        return len(self._calendar)

    def __repr__(self) -> str:
        return (
            f"MarketData(symbols={list(self._symbols)}, "
            f"{_fmt(self.start)} to {_fmt(self.end)}, {len(self)} dates)"
        )

    def bar(self, t: pd.Timestamp | str) -> dict[str, Bar]:
        """Bars for every symbol that traded on date ``t``.

        Engine-only accessor. Raises ``KeyError`` if ``t`` is not in the calendar.
        """
        ts = _as_timestamp(t)
        bars: dict[str, Bar] = {}
        for symbol in self._symbols:
            try:
                i = self._index[symbol].get_loc(ts)
            except KeyError:
                continue
            o, h, lo, c, v = self._values[symbol][i]
            bars[symbol] = Bar(float(o), float(h), float(lo), float(c), float(v))
        if not bars:
            raise KeyError(self._not_a_trading_date(ts))
        return bars

    def view(self, cutoff: pd.Timestamp | str) -> MarketView:
        """A read-only view of every bar dated at or before ``cutoff``, and nothing after.

        This is the only way strategies see data. ``cutoff`` is inclusive: the
        bar dated ``cutoff`` is visible (decisions are made after its close).
        See :mod:`backtester.data.view` for the exact boundary semantics.

        Raises ``KeyError`` if ``cutoff`` is not a date on this calendar, so a
        view's ``now`` is always a real trading date.
        """
        from backtester.data.view import _make_view  # deferred: view.py imports this module

        ts = _as_timestamp(cutoff)
        if ts not in self._calendar:
            raise KeyError(self._not_a_trading_date(ts))
        windows: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
        for symbol in self._symbols:
            n = self._visible_rows(symbol, ts)
            if n > 0:  # a symbol with no bars yet is invisible, not empty
                windows[symbol] = (self._index[symbol][:n], self._values[symbol][:n])
        return _make_view(ts, windows)

    def _visible_rows(self, symbol: str, cutoff: pd.Timestamp) -> int:
        """How many of ``symbol``'s bars are dated at or before ``cutoff``.

        This one line is the lookahead boundary. Dates are sorted and unique
        (validated), so ``searchsorted(cutoff, side="right")`` is exactly
        ``#{d : d <= cutoff}``: the bar on ``cutoff`` is included, and no later
        bar can be. The test suite's leaky negative control overrides this
        method to add one row, and the lookahead tests must then fail.
        """
        return int(self._index[symbol].searchsorted(cutoff, side="right"))

    def frame(self, symbol: str) -> pd.DataFrame:
        """A copy of one symbol's OHLCV frame (safe to modify)."""
        if symbol not in self._frames:
            raise KeyError(f"unknown symbol {symbol!r}; available: {list(self._symbols)}")
        return self._frames[symbol].copy()

    def to_panel(self) -> pd.DataFrame:
        """A copy of the full panel, indexed by (date, symbol)."""
        panel = pd.concat(self._frames, names=["symbol", "date"])
        return panel.swaplevel("symbol", "date").sort_index()

    def _not_a_trading_date(self, ts: pd.Timestamp) -> str:
        return (
            f"{_fmt(ts)} is not a trading date in this dataset "
            f"(calendar spans {_fmt(self.start)} to {_fmt(self.end)}, {len(self)} dates)"
        )


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a *validated* frame into the internal representation.

    Only representation changes: fixed column order, float64 dtype, a
    nanosecond ``DatetimeIndex`` named "date" with no ``freq``. Values are
    untouched (validation has already guaranteed they are finite and numeric).
    """
    out = frame.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    out.index = pd.DatetimeIndex(out.index.to_numpy(), name="date").as_unit("ns")
    out.columns.name = None
    return out


def _as_timestamp(t: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tz is not None:
        raise ValueError(f"dates must be timezone-naive, got {ts}")
    return ts
