"""Load daily OHLCV data from yfinance through a local Parquet cache.

Cache layout (one pair of files per symbol, in ``cache_dir``)::

    SPY_1d.parquet     normalized OHLCV frame (date index; open/high/low/close/volume)
    SPY_1d.meta.json   provenance: source and settings, requested range, actual data
                       range, row count, download time (UTC), sha256 of the parquet

Cache policy:

- **Hit**: metadata exists, was written with the same settings, and its
  *requested* range covers the new request. The parquet must still match the
  checksum recorded at download time, or loading fails loudly.
- **Miss**: download, validate, then write. An invalid download is never cached.
- **Range extension**: a request outside the cached range re-downloads the union
  of the old and new ranges and replaces the cache. Downloads are never merged,
  because yfinance rescales the entire adjusted history after every dividend or
  split. Stitching two downloads together would silently mix two adjustment bases.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from backtester.data.market_data import (
    OHLCV_COLUMNS,
    DataValidationError,
    MarketData,
    Rule,
    Violation,
    canonicalize_frame,
    validate_frame,
)

logger = logging.getLogger(__name__)

SOURCE = "yfinance"
INTERVAL = "1d"
AUTO_ADJUST = True
CACHE_SCHEMA_VERSION = 1

# Yahoo-style tickers: SPY, BRK-B, ^GSPC, EURUSD=X, RDS.A. Uppercase only, so that
# "spy" and "SPY" can never become two cache files that collide on a
# case-insensitive filesystem (the macOS default).
_SYMBOL_RE = re.compile(r"[A-Z0-9^][A-Z0-9.=^-]*")

DateLike = date | str | pd.Timestamp

#: ``(symbol, start, end) -> raw frame``, with ``end`` INCLUSIVE.
Downloader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]


@runtime_checkable
class DataLoader(Protocol):
    """Anything that can produce validated ``MarketData`` for a date range."""

    def load(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> MarketData: ...


class DataDownloadError(RuntimeError):
    """The data source failed, or returned nothing usable."""


class CacheIntegrityError(RuntimeError):
    """A cache file is unreadable, or no longer matches its recorded checksum."""


# ---------------------------------------------------------------------------
# yfinance adapter
# ---------------------------------------------------------------------------


def yfinance_download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch raw daily bars from yfinance for ``[start, end]``, both inclusive.

    yfinance treats ``end`` as exclusive, so one day is added here. The response
    is returned as-is; :func:`normalize_yfinance_frame` handles its shape.
    """
    import yfinance as yf  # lazy: slow to import, and the test suite never needs it

    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        # yf.download logs failures instead of raising; surface its reason if it kept one.
        reason = None
        try:
            from yfinance import shared

            reason = getattr(shared, "_ERRORS", {}).get(symbol)
        except ImportError:  # pragma: no cover - internal yfinance layout changed
            pass
        detail = f": {reason}" if reason else ""
        raise DataDownloadError(
            f"yfinance returned no rows for {symbol} between {_iso(start)} and {_iso(end)}{detail}"
        )
    return raw


def normalize_yfinance_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert a raw yfinance response into the ``MarketData`` frame format.

    Shape changes only: select this symbol from yfinance's (Price, Ticker)
    column MultiIndex, lower-case the column names, keep the five OHLCV
    columns, and drop the timezone while keeping the exchange-local calendar
    date. Row order and values are left untouched so validation sees exactly
    what the source sent.
    """
    df = raw
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            if symbol in df.columns.get_level_values(level):
                df = df.xs(symbol, axis=1, level=level)
                break
        else:
            raise DataDownloadError(
                f"yfinance response has no columns for {symbol}; got {list(df.columns)}"
            )

    df = df.rename(columns=lambda c: str(c).strip().lower())
    if "adj close" in df.columns:
        raise DataDownloadError(
            f"yfinance response for {symbol} includes an 'Adj Close' column, so prices were "
            "not auto-adjusted; refusing to mix unadjusted prices into the cache"
        )
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            [
                Violation(
                    symbol,
                    Rule.COLUMNS,
                    f"yfinance response is missing required columns {missing}; "
                    f"got {list(df.columns)}",
                )
            ]
        )

    df = df.loc[:, list(OHLCV_COLUMNS)].copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df.columns.name = None
    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class YFinanceLoader:
    """``DataLoader`` backed by yfinance, with a per-symbol Parquet cache.

    Args:
        cache_dir: directory for cache files (created on first write).
        refresh: ignore any existing cache, re-download, and overwrite it.
        downloader: ``(symbol, start, end_inclusive) -> raw yfinance-shaped frame``.
            Defaults to :func:`yfinance_download`. Tests inject a fake so the
            suite never touches the network.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        refresh: bool = False,
        downloader: Downloader | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self._download = downloader or yfinance_download

    def load(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> MarketData:
        """Return validated daily bars for ``symbols`` over ``[start, end]`` (inclusive).

        ``end`` must be before today: the current session's bar may still be
        forming, and pinning a past end date keeps results reproducible.

        Raises:
            ValueError / TypeError: bad symbols or date range.
            DataDownloadError: the source failed or returned no rows.
            DataValidationError: the data breaks a ``MarketData`` invariant.
            CacheIntegrityError: a cache file is unreadable or was modified.
        """
        checked = _check_symbols(symbols)
        start_ts, end_ts = _check_range(start, end)
        frames = {s: self._load_symbol(s, start_ts, end_ts) for s in checked}
        return MarketData.from_frames(frames)

    def cache_paths(self, symbol: str) -> tuple[Path, Path]:
        """``(parquet_path, meta_path)`` for a symbol."""
        stem = f"{symbol}_{INTERVAL}"
        return self.cache_dir / f"{stem}.parquet", self.cache_dir / f"{stem}.meta.json"

    # --- internals -------------------------------------------------------------

    def _load_symbol(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        parquet_path, meta_path = self.cache_paths(symbol)
        meta = None if self.refresh else self._read_meta(symbol, meta_path, parquet_path)

        if meta is not None and _covers(meta, start, end):
            logger.info("%s: cache hit (%s to %s)", symbol, _iso(start), _iso(end))
            full = self._read_parquet(parquet_path, meta_path, meta)
        else:
            dl_start, dl_end = start, end
            if meta is not None:
                dl_start = min(start, pd.Timestamp(meta["requested_start"]))
                dl_end = max(end, pd.Timestamp(meta["requested_end"]))
                logger.warning(
                    "%s: %s to %s is outside the cached range %s to %s. Re-downloading "
                    "%s to %s and replacing the cache; adjusted prices may differ slightly "
                    "from the previous snapshot (downloaded %s).",
                    symbol,
                    _iso(start),
                    _iso(end),
                    meta["requested_start"],
                    meta["requested_end"],
                    _iso(dl_start),
                    _iso(dl_end),
                    meta.get("downloaded_at"),
                )
            else:
                logger.info("%s: cache miss, downloading %s to %s", symbol, _iso(start), _iso(end))
            full = self._fetch(symbol, dl_start, dl_end)
            self._write_cache(symbol, full, dl_start, dl_end)

        in_range = (full.index >= start) & (full.index <= end)
        sliced = full.loc[in_range]
        if sliced.empty:
            raise ValueError(
                f"no {symbol} bars between {_iso(start)} and {_iso(end)} "
                f"(available data runs {_iso(full.index[0])} to {_iso(full.index[-1])})"
            )
        return sliced

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Download, normalize, and fully validate one symbol. Never touches the cache."""
        try:
            raw = self._download(symbol, start, end)
        except DataDownloadError:
            raise
        except Exception as exc:
            raise DataDownloadError(
                f"downloading {symbol} for {_iso(start)} to {_iso(end)} failed: {exc!r}"
            ) from exc
        if raw is None or len(raw) == 0:
            raise DataDownloadError(
                f"{SOURCE} returned no rows for {symbol} between {_iso(start)} and {_iso(end)} "
                "(unknown ticker, no trading days in range, or a network/rate-limit failure)"
            )

        frame = normalize_yfinance_frame(raw, symbol)
        violations = validate_frame(symbol, frame)
        if violations:
            raise DataValidationError(violations)
        return canonicalize_frame(frame)

    def _read_meta(self, symbol: str, meta_path: Path, parquet_path: Path) -> dict[str, Any] | None:
        """Cache metadata if it describes a usable cache for this request, else ``None``."""
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(
                f"cache metadata {meta_path} is unreadable ({exc}). "
                "Delete it or reload with refresh=True."
            ) from exc
        if not isinstance(meta, dict):
            raise CacheIntegrityError(
                f"cache metadata {meta_path} is not a JSON object. "
                "Delete it or reload with refresh=True."
            )

        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "symbol": symbol,
            "source": SOURCE,
            "interval": INTERVAL,
            "auto_adjust": AUTO_ADJUST,
        }
        mismatched = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
        if mismatched:
            logger.warning(
                "%s: ignoring cache written with different settings %s (expected %s); "
                "re-downloading",
                symbol,
                mismatched,
                {k: expected[k] for k in mismatched},
            )
            return None

        missing = [k for k in ("requested_start", "requested_end", "sha256") if k not in meta]
        if missing:
            raise CacheIntegrityError(
                f"cache metadata {meta_path} is missing {missing}. "
                "Delete it or reload with refresh=True."
            )

        if not parquet_path.exists():
            logger.warning("%s: %s is missing; re-downloading", symbol, parquet_path.name)
            return None
        return meta

    def _read_parquet(
        self, parquet_path: Path, meta_path: Path, meta: dict[str, Any]
    ) -> pd.DataFrame:
        digest = _sha256(parquet_path)
        if digest != meta["sha256"]:
            raise CacheIntegrityError(
                f"{parquet_path} no longer matches the checksum recorded in {meta_path.name} "
                f"when it was downloaded ({meta.get('downloaded_at')}), so it was modified "
                "after download. Reload with refresh=True to re-download it."
            )
        return pd.read_parquet(parquet_path)

    def _write_cache(
        self,
        symbol: str,
        frame: pd.DataFrame,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path, meta_path = self.cache_paths(symbol)

        # Invalidate the metadata first. If anything below fails part-way, the next
        # run sees a clean miss, never metadata describing a different file.
        meta_path.unlink(missing_ok=True)

        tmp_parquet = parquet_path.with_name(parquet_path.name + ".tmp")
        frame.to_parquet(tmp_parquet, engine="pyarrow")
        os.replace(tmp_parquet, parquet_path)

        meta = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "symbol": symbol,
            "source": SOURCE,
            "source_version": _package_version("yfinance")
            if self._download is yfinance_download
            else None,
            "interval": INTERVAL,
            "auto_adjust": AUTO_ADJUST,
            "requested_start": _iso(requested_start),
            "requested_end": _iso(requested_end),
            "first_date": _iso(frame.index[0]),
            "last_date": _iso(frame.index[-1]),
            "n_rows": len(frame),
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "parquet_file": parquet_path.name,
            "sha256": _sha256(parquet_path),
        }
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(tmp_meta, meta_path)
        logger.info("%s: cached %d rows to %s", symbol, len(frame), parquet_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_symbols(symbols: Sequence[str]) -> list[str]:
    if isinstance(symbols, str):
        raise TypeError(f"symbols must be a list of tickers, not a string; use [{symbols!r}]")
    checked = list(symbols)
    if not checked:
        raise ValueError("symbols is empty: provide at least one ticker")
    invalid = [s for s in checked if not isinstance(s, str) or not _SYMBOL_RE.fullmatch(s)]
    if invalid:
        raise ValueError(
            f"invalid symbol(s) {invalid}: use uppercase Yahoo tickers such as "
            "'SPY', 'BRK-B', or '^GSPC'"
        )
    duplicates = sorted({s for s in checked if checked.count(s) > 1})
    if duplicates:
        raise ValueError(f"duplicate symbol(s) {duplicates}")
    return checked


def _check_range(start: DateLike, end: DateLike) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts, end_ts = _as_date(start, "start"), _as_date(end, "end")
    if start_ts > end_ts:
        raise ValueError(f"start ({_iso(start_ts)}) is after end ({_iso(end_ts)})")
    today = pd.Timestamp(date.today())
    if end_ts >= today:
        raise ValueError(
            f"end ({_iso(end_ts)}) must be before today ({_iso(today)}): today's daily bar may "
            "still be forming, and a pinned past end date keeps results reproducible"
        )
    return start_ts, end_ts


def _as_date(value: DateLike, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise ValueError(f"{name} is missing (got {value!r})")
    if ts.tz is not None:
        raise ValueError(f"{name} must be a timezone-naive date, got {value!r}")
    if ts != ts.normalize():
        raise ValueError(f"{name} must be a calendar date with no time component, got {value!r}")
    return ts.as_unit("ns")


def _covers(meta: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return (
        pd.Timestamp(meta["requested_start"]) <= start
        and pd.Timestamp(meta["requested_end"]) >= end
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")
