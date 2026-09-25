"""Point-in-time, read-only views of market data: the lookahead guard.

A strategy never receives ``MarketData``. At each step the engine calls
``MarketData.view(t)`` and passes the strategy the resulting ``MarketView``,
which can only return bars dated at or before ``t``.

Boundary semantics (the lookahead guarantee depends on exactly this)
--------------------------------------------------------------------
For cutoff ``T`` and a symbol whose sorted, unique dates are
``d_0 < d_1 < ... < d_{m-1}``, the visible rows are positions ``[0, n)`` where::

    n = searchsorted(dates, T, side="right") = #{i : d_i <= T}

So the bar dated ``T`` is visible (the cutoff is inclusive), and every bar dated
after ``T`` is not. Including ``T`` is correct because decisions are made after
the close of ``T`` and executed no earlier than the next bar's open.
``side="left"`` would hide ``T``'s own bar (an off-by-one in the safe direction
that throws away the latest close). Anything that admits a row with ``d_i > T``
is lookahead. The count is computed in exactly one place,
``MarketData._visible_rows``.

Other guarantees
----------------
- ``cutoff`` must be a date on the parent's calendar, so ``now`` is always a
  real trading date.
- ``symbols`` lists only symbols with at least one visible bar. A symbol that
  first trades after ``T`` is invisible, because knowing it *will* list is
  itself future information. Asking for it raises the same error as asking for
  a symbol that does not exist at all.
- Every returned Series or DataFrame is a fresh copy backed by a read-only
  array, so writing into it raises ``ValueError``. Nothing returned shares
  memory with the parent, so no caller can change what this view, a later view,
  or the engine sees.
- No method accepts a date, and no public attribute leads back to the parent.
  As with ``MarketData``'s internals, the truncated arrays live in underscore
  attributes. Python cannot make those truly private, which is why the test
  suite checks for lookahead from the outside (future-perturbation tests)
  instead of trusting encapsulation alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from backtester.data.market_data import OHLCV_COLUMNS

#: One symbol's visible data: its dates and an (n, 5) OHLCV array, both already
#: truncated to the cutoff.
Window = tuple[pd.DatetimeIndex, np.ndarray]

_VIEW_KEY = object()  # construction key: only MarketData.view may build a MarketView


class MarketView:
    """Everything a strategy may know at the close of ``now``.

    Created only by :meth:`MarketData.view`. Exposes ``now``, ``symbols``,
    ``history()`` and ``panel()``. Nothing else is public.
    """

    __slots__ = ("_now", "_windows")

    def __init__(self, now: pd.Timestamp, windows: dict[str, Window], *, _key: object = None):
        if _key is not _VIEW_KEY:
            raise TypeError(
                "MarketView is created by MarketData.view(cutoff); it cannot be built directly."
            )
        self._now = now
        self._windows = windows

    @property
    def now(self) -> pd.Timestamp:
        """The cutoff: the date of the latest bar this view can contain."""
        return self._now

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols with at least one bar at or before ``now``, in the parent's order."""
        return tuple(self._windows)

    def __repr__(self) -> str:
        return f"MarketView(now={self._now:%Y-%m-%d}, symbols={list(self._windows)})"

    def history(self, symbol: str, field: str = "close", lookback: int | None = None) -> pd.Series:
        """One field for one symbol, for every visible date (or the last ``lookback`` of them).

        Returns a date-indexed Series whose last date is ``symbol``'s most recent
        bar at or before ``now``. The Series is a copy backed by a read-only
        array: ``s.iloc[0] = x``, ``s.to_numpy()[0] = x`` and similar writes raise
        ``ValueError``.

        Raises:
            KeyError: ``symbol`` has no bar at or before ``now``.
            ValueError: unknown ``field``, or ``lookback`` asks for more bars than
                are visible. A strategy's ``warmup`` should cover its longest lookback.
        """
        if not isinstance(field, str):
            raise TypeError(f"field must be a string, got {type(field).__name__}")
        (column,) = _field_positions([field])
        index, values = self._window(symbol)
        start = self._window_start(len(index), lookback, f"{symbol} {field}")
        return pd.Series(
            _read_only_copy(values[start:, column]),
            index=_copy_index(index[start:]),
            name=field,
            copy=False,
        )

    def panel(
        self, fields: Sequence[str] = OHLCV_COLUMNS, lookback: int | None = None
    ) -> pd.DataFrame:
        """Several fields for every visible symbol, indexed by (date, symbol).

        ``lookback`` counts calendar dates: the last ``lookback`` dates on which
        any visible symbol traded. A symbol with no bar on a date simply has no
        row for it (nothing is filled in). Rows are sorted by date, then symbol,
        matching ``MarketData.to_panel()``. The frame is a copy backed by a
        read-only array, so writes into it raise ``ValueError``.

        Raises:
            TypeError: ``fields`` is a bare string (pass a list).
            ValueError: unknown or duplicated fields, or ``lookback`` asks for
                more dates than are visible.
        """
        if isinstance(fields, str):
            raise TypeError(
                f"fields must be a sequence of field names, not a string; use [{fields!r}]"
            )
        fields = list(fields)
        columns = _field_positions(fields)

        first_date = None
        if lookback is not None:
            _check_lookback(lookback)
            # The last `lookback` dates of the union are always among each symbol's own
            # last `lookback` dates, so the tails are enough (and if the tails hold fewer
            # than `lookback` dates, so does the full union).
            tails = [index[-lookback:].to_numpy() for index, _ in self._windows.values()]
            visible_dates = np.unique(np.concatenate(tails))
            start = self._window_start(len(visible_dates), lookback, "the panel's calendar")
            first_date = visible_dates[start]

        symbols, blocks, dates, codes = [], [], [], []
        for symbol in sorted(self._windows):
            index, values = self._windows[symbol]
            s = 0 if first_date is None else int(index.searchsorted(first_date, side="left"))
            if s == len(index):
                continue  # no bars inside the lookback window (e.g. stopped trading)
            codes.append(np.full(len(index) - s, len(symbols)))
            symbols.append(symbol)
            blocks.append(values[s:][:, columns])
            dates.append(index[s:].to_numpy())

        date_arr, code_arr = np.concatenate(dates), np.concatenate(codes)
        order = np.lexsort((code_arr, date_arr))  # primary key: date; secondary: symbol
        unique_dates, date_codes = np.unique(date_arr[order], return_inverse=True)
        row_index = pd.MultiIndex(  # built from codes directly: far faster than from_arrays
            levels=[pd.DatetimeIndex(unique_dates), pd.Index(symbols)],
            codes=[date_codes.ravel(), code_arr[order]],
            names=["date", "symbol"],
            verify_integrity=False,
        )
        return pd.DataFrame(
            _read_only_copy(np.concatenate(blocks)[order]),
            index=row_index,
            columns=fields,
            copy=False,
        )

    # --- internals -----------------------------------------------------------

    def _window(self, symbol: str) -> Window:
        try:
            return self._windows[symbol]
        except KeyError:
            # Same message whether the symbol lists later or never exists, so the
            # error itself cannot reveal the future.
            raise KeyError(
                f"no data for {symbol!r} at or before {self._now:%Y-%m-%d}; "
                f"visible symbols: {list(self._windows)}"
            ) from None

    def _window_start(self, n_visible: int, lookback: int | None, what: str) -> int:
        if lookback is None:
            return 0
        _check_lookback(lookback)
        if lookback > n_visible:
            raise ValueError(
                f"lookback={lookback} requested for {what}, but only {n_visible} bar(s) are "
                f"visible at {self._now:%Y-%m-%d}; a strategy's warmup should cover its "
                "longest lookback"
            )
        return n_visible - lookback


def _make_view(now: pd.Timestamp, windows: dict[str, Window]) -> MarketView:
    """Called only by ``MarketData.view``."""
    return MarketView(now, windows, _key=_VIEW_KEY)


def _check_lookback(lookback: object) -> None:
    if isinstance(lookback, bool) or not isinstance(lookback, int | np.integer):
        raise TypeError(f"lookback must be an int or None, got {lookback!r}")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")


def _field_positions(fields: list[str]) -> list[int]:
    if not fields:
        raise ValueError("at least one field is required")
    unknown = [f for f in fields if f not in OHLCV_COLUMNS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; choose from {list(OHLCV_COLUMNS)}")
    duplicated = sorted({f for f in fields if fields.count(f) > 1})
    if duplicated:
        raise ValueError(f"duplicated field(s) {duplicated}")
    return [OHLCV_COLUMNS.index(f) for f in fields]


def _read_only_copy(values: np.ndarray) -> np.ndarray:
    """A contiguous copy that shares no memory with the parent, with writes disabled."""
    out = np.array(values, dtype="float64", copy=True)
    out.flags.writeable = False
    return out


def _copy_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """A DatetimeIndex backed by a fresh array (a slice would still reference the parent)."""
    return pd.DatetimeIndex(index.to_numpy(copy=True), name="date")
