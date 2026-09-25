"""YFinanceLoader: cache hit/miss behaviour, meta.json, integrity, and rejection of bad data.

Every test runs offline against ``FakeDownloader``, except the one marked
``network`` (deselected by default; run with ``pytest -m network``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtester.data import (
    CacheIntegrityError,
    DataDownloadError,
    DataLoader,
    DataValidationError,
    MarketData,
    YFinanceLoader,
)
from backtester.data.loader import yfinance_download
from synthetic import BAD_ROW, FakeDownloader, InvalidCase, make_ohlcv, to_yfinance_format

START = pd.Timestamp("2020-02-03")  # a Monday
END = pd.Timestamp("2020-06-30")
FULL_YEAR = ("2020-01-01", "2020-12-31")


def _read_meta(loader: YFinanceLoader, symbol: str = "AAA") -> dict:
    _, meta_path = loader.cache_paths(symbol)
    return json.loads(meta_path.read_text())


def _expected(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The source rows for [start, end] in MarketData's canonical representation."""
    expected = frame.loc[start:end].astype("float64")
    expected.index = expected.index.as_unit("ns")
    return expected


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------


class TestCacheMiss:
    def test_downloads_once_and_writes_parquet_and_meta(self, make_loader, fake_downloader):
        loader = make_loader()

        loader.load(["AAA"], START, END)

        assert fake_downloader.calls == [("AAA", START, END)]
        parquet_path, meta_path = loader.cache_paths("AAA")
        assert parquet_path.exists()
        assert meta_path.exists()
        assert not list(loader.cache_dir.glob("*.tmp"))  # atomic writes leave no temp files

    def test_returns_the_requested_range_exactly(self, make_loader, universe):
        md = make_loader().load(["AAA"], START, END)

        assert isinstance(md, MarketData)
        pd.testing.assert_frame_equal(
            md.frame("AAA"), _expected(universe["AAA"], START, END), check_freq=False
        )

    def test_meta_json_records_provenance(self, make_loader):
        loader = make_loader()
        before = datetime.now(UTC)

        loader.load(["AAA"], "2020-02-01", END)  # a Saturday: no bar on the requested start

        meta = _read_meta(loader)
        parquet_path, _ = loader.cache_paths("AAA")
        assert set(meta) == {
            "schema_version",
            "symbol",
            "source",
            "source_version",
            "interval",
            "auto_adjust",
            "requested_start",
            "requested_end",
            "first_date",
            "last_date",
            "n_rows",
            "downloaded_at",
            "parquet_file",
            "sha256",
        }
        assert meta["schema_version"] == 1
        assert meta["symbol"] == "AAA"
        assert meta["source"] == "yfinance"
        assert meta["source_version"] is None  # a fake downloader, not the real yfinance
        assert meta["interval"] == "1d"
        assert meta["auto_adjust"] is True
        # The requested range and the actual data range are recorded separately.
        assert meta["requested_start"] == "2020-02-01"
        assert meta["requested_end"] == "2020-06-30"
        assert meta["first_date"] == "2020-02-03"
        assert meta["last_date"] == "2020-06-30"
        assert meta["n_rows"] == len(pd.read_parquet(parquet_path))
        assert meta["parquet_file"] == parquet_path.name
        assert meta["sha256"] == hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        downloaded_at = datetime.fromisoformat(meta["downloaded_at"])
        assert downloaded_at.utcoffset() == timedelta(0)
        assert before - timedelta(seconds=1) <= downloaded_at <= datetime.now(UTC)

    def test_cached_parquet_holds_the_normalized_frame(self, make_loader):
        loader = make_loader()
        md = loader.load(["AAA"], START, END)

        cached = pd.read_parquet(loader.cache_paths("AAA")[0])

        pd.testing.assert_frame_equal(cached, md.frame("AAA"), check_freq=False)


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------


class TestCacheHit:
    def test_second_load_is_served_from_disk(self, make_loader, fake_downloader):
        first = make_loader().load(["AAA"], START, END)
        second = make_loader().load(["AAA"], START, END)  # a new loader: only the files persist

        assert len(fake_downloader.calls) == 1
        pd.testing.assert_frame_equal(first.frame("AAA"), second.frame("AAA"))

    def test_subrange_is_served_from_cache_and_sliced(self, make_loader, fake_downloader):
        make_loader().load(["AAA"], START, END)

        md = make_loader().load(["AAA"], "2020-03-02", "2020-03-31")

        assert len(fake_downloader.calls) == 1
        assert md.start == pd.Timestamp("2020-03-02")
        assert md.end == pd.Timestamp("2020-03-31")

    def test_coverage_uses_the_requested_range_not_the_first_bar(
        self, make_loader, fake_downloader
    ):
        # 2020-02-01 is a Saturday, so the first bar is 2020-02-03. Judging coverage by
        # the data's first date would turn every repeat of this request into a miss.
        make_loader().load(["AAA"], "2020-02-01", END)
        make_loader().load(["AAA"], "2020-02-01", END)

        assert len(fake_downloader.calls) == 1

    def test_hit_does_not_rewrite_the_cache(self, make_loader):
        loader = make_loader()
        loader.load(["AAA"], START, END)
        parquet_path, meta_path = loader.cache_paths("AAA")
        before = (parquet_path.stat().st_mtime_ns, meta_path.read_text())

        make_loader().load(["AAA"], START, END)

        assert (parquet_path.stat().st_mtime_ns, meta_path.read_text()) == before

    def test_symbols_are_cached_independently(self, make_loader, fake_downloader):
        make_loader().load(["AAA"], START, END)

        md = make_loader().load(["AAA", "BBB"], START, END)

        assert md.symbols == ("AAA", "BBB")
        assert len(fake_downloader.calls_for("AAA")) == 1
        assert len(fake_downloader.calls_for("BBB")) == 1


# ---------------------------------------------------------------------------
# Refresh, range extension, stale caches
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_refresh_forces_a_new_download(self, make_loader, fake_downloader):
        make_loader().load(["AAA"], START, END)
        make_loader(refresh=True).load(["AAA"], START, END)

        assert len(fake_downloader.calls) == 2

    def test_range_extension_downloads_the_union_and_replaces_the_cache(
        self, make_loader, fake_downloader, caplog
    ):
        make_loader().load(["AAA"], "2020-02-03", "2020-03-31")

        with caplog.at_level(logging.WARNING, logger="backtester.data.loader"):
            md = make_loader().load(["AAA"], "2020-05-01", "2020-06-30")

        assert fake_downloader.calls[-1] == (
            "AAA",
            pd.Timestamp("2020-02-03"),
            pd.Timestamp("2020-06-30"),
        )
        assert "outside the cached range" in caplog.text
        assert md.start == pd.Timestamp("2020-05-01")  # still sliced to the request
        meta = _read_meta(make_loader())
        assert (meta["requested_start"], meta["requested_end"]) == ("2020-02-03", "2020-06-30")

        make_loader().load(["AAA"], "2020-02-03", "2020-06-30")  # now fully covered
        assert len(fake_downloader.calls) == 2

    def test_cache_written_with_other_settings_is_redownloaded(
        self, make_loader, fake_downloader, caplog
    ):
        loader = make_loader()
        loader.load(["AAA"], START, END)
        _, meta_path = loader.cache_paths("AAA")
        meta = _read_meta(loader)
        meta["auto_adjust"] = False
        meta_path.write_text(json.dumps(meta))

        with caplog.at_level(logging.WARNING, logger="backtester.data.loader"):
            make_loader().load(["AAA"], START, END)

        assert len(fake_downloader.calls) == 2
        assert "different settings" in caplog.text
        assert _read_meta(loader)["auto_adjust"] is True

    def test_deleted_parquet_is_redownloaded(self, make_loader, fake_downloader):
        loader = make_loader()
        loader.load(["AAA"], START, END)
        loader.cache_paths("AAA")[0].unlink()

        make_loader().load(["AAA"], START, END)

        assert len(fake_downloader.calls) == 2


# ---------------------------------------------------------------------------
# Cache integrity
# ---------------------------------------------------------------------------


class TestCacheIntegrity:
    def test_modified_parquet_fails_loudly(self, make_loader):
        loader = make_loader()
        loader.load(["AAA"], START, END)
        parquet_path, _ = loader.cache_paths("AAA")
        tampered = pd.read_parquet(parquet_path)
        tampered["close"] *= 1.01
        tampered.to_parquet(parquet_path)

        with pytest.raises(CacheIntegrityError, match=r"checksum.*refresh=True"):
            make_loader().load(["AAA"], START, END)

    def test_unreadable_meta_fails_loudly(self, make_loader):
        loader = make_loader()
        loader.load(["AAA"], START, END)
        loader.cache_paths("AAA")[1].write_text("{not json")

        with pytest.raises(CacheIntegrityError, match="unreadable"):
            make_loader().load(["AAA"], START, END)

    def test_refresh_recovers_from_a_tampered_cache(self, make_loader, universe):
        loader = make_loader()
        loader.load(["AAA"], START, END)
        loader.cache_paths("AAA")[1].write_text("{not json")

        md = make_loader(refresh=True).load(["AAA"], START, END)

        pd.testing.assert_frame_equal(
            md.frame("AAA"), _expected(universe["AAA"], START, END), check_freq=False
        )


# ---------------------------------------------------------------------------
# Rejecting bad data from the source
# ---------------------------------------------------------------------------


class TestRejectsBadDownloads:
    def test_invalid_download_is_rejected_and_never_cached(
        self, make_loader, fake_downloader, universe, value_case: InvalidCase
    ):
        fake_downloader.frames["AAA"] = value_case.mutate(universe["AAA"])
        loader = make_loader()

        with pytest.raises(DataValidationError) as info:
            loader.load(["AAA"], *FULL_YEAR)

        err = info.value
        assert err.rules == {value_case.rule}
        assert f"[AAA] {value_case.rule}" in str(err)
        assert value_case.fragment in str(err)
        assert universe["AAA"].index[BAD_ROW].strftime("%Y-%m-%d") in str(err)
        assert not any(p.exists() for p in loader.cache_paths("AAA"))

    def test_bad_bar_outside_the_requested_range_still_blocks_caching(
        self, make_loader, fake_downloader, universe
    ):
        # The whole download is validated before caching, not just the requested slice,
        # so a bad bar can never sit in the cache waiting for a later, wider request.
        fake_downloader.frames["AAA"] = universe["AAA"]
        make_loader().load(["AAA"], "2020-02-03", "2020-03-31")
        fake_downloader.frames["AAA"] = universe["AAA"].copy()
        fake_downloader.frames["AAA"].iloc[5, 3] = np.nan  # 2020-01-09, before both ranges

        with pytest.raises(DataValidationError, match="2020-01-09"):
            make_loader().load(["AAA"], "2020-01-01", "2020-03-31")

    def test_unknown_ticker_raises_download_error(self, make_loader):
        with pytest.raises(DataDownloadError, match="no rows for ZZZ between 2020-02-03"):
            make_loader().load(["ZZZ"], START, END)

    def test_downloader_exceptions_are_wrapped(self, make_loader):
        def failing(symbol, start, end):
            raise ConnectionError("proxy said no")

        loader = make_loader(downloader=failing)
        with pytest.raises(DataDownloadError, match="proxy said no") as info:
            loader.load(["AAA"], START, END)

        assert isinstance(info.value.__cause__, ConnectionError)
        assert not loader.cache_paths("AAA")[0].exists()

    def test_response_missing_a_column_is_rejected(self, make_loader, universe):
        def no_volume(symbol, start, end):
            return to_yfinance_format(universe[symbol], symbol).drop(columns="Volume", level=0)

        with pytest.raises(DataValidationError, match=r"missing required columns \['volume'\]"):
            make_loader(downloader=no_volume).load(["AAA"], START, END)

    def test_unadjusted_response_is_rejected(self, make_loader, universe):
        def unadjusted(symbol, start, end):
            raw = to_yfinance_format(universe[symbol], symbol, multi_level=False)
            return raw.assign(**{"Adj Close": raw["Close"] * 0.98})

        with pytest.raises(DataDownloadError, match="not auto-adjusted"):
            make_loader(downloader=unadjusted).load(["AAA"], START, END)

    def test_range_with_no_trading_days_raises(self, make_loader):
        make_loader().load(["AAA"], START, END)
        with pytest.raises(ValueError, match="no AAA bars between 2020-02-01 and 2020-02-02"):
            make_loader().load(["AAA"], "2020-02-01", "2020-02-02")  # a weekend


# ---------------------------------------------------------------------------
# yfinance response shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multi_level", [True, False], ids=["multiindex", "flat"])
@pytest.mark.parametrize("tz", [None, "America/New_York"], ids=["naive", "tz-aware"])
def test_yfinance_response_shapes_normalize_identically(cache_dir, universe, multi_level, tz):
    downloader = FakeDownloader(universe, multi_level=multi_level, tz=tz)

    md = YFinanceLoader(cache_dir, downloader=downloader).load(["AAA"], START, END)

    pd.testing.assert_frame_equal(
        md.frame("AAA"), _expected(universe["AAA"], START, END), check_freq=False
    )


def test_yfinance_adapter_passes_inclusive_end_and_auto_adjust(monkeypatch):
    import yfinance as yf

    captured: dict = {}

    def fake_download(tickers, **kwargs):
        captured.update(kwargs, tickers=tickers)
        return to_yfinance_format(make_ohlcv(n_days=5), tickers)

    monkeypatch.setattr(yf, "download", fake_download)

    yfinance_download("SPY", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-31"))

    assert captured["tickers"] == "SPY"
    assert captured["start"] == "2020-01-02"
    assert captured["end"] == "2020-02-01"  # yfinance's end is exclusive; ours is inclusive
    assert captured["auto_adjust"] is True
    assert captured["interval"] == "1d"


def test_yfinance_adapter_raises_on_an_empty_response(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda tickers, **kwargs: pd.DataFrame())

    with pytest.raises(DataDownloadError, match="no rows for SPY"):
        yfinance_download("SPY", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-31"))


# ---------------------------------------------------------------------------
# Input validation (fails before any download)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbols", "start", "end", "error", "match"),
    [
        ("AAA", START, END, TypeError, "not a string"),
        ([], START, END, ValueError, "symbols is empty"),
        (["aaa"], START, END, ValueError, "uppercase Yahoo tickers"),
        (["../AAA"], START, END, ValueError, "invalid symbol"),
        (["AAA", "AAA"], START, END, ValueError, "duplicate symbol"),
        (["AAA"], END, START, ValueError, "is after end"),
        (["AAA"], START, date.today(), ValueError, "must be before today"),
        (["AAA"], "2020-02-03 10:30", END, ValueError, "no time component"),
        (["AAA"], None, END, ValueError, "start is missing"),
    ],
    ids=[
        "bare-string",
        "empty",
        "lowercase",
        "path-chars",
        "duplicate",
        "start-after-end",
        "end-today",
        "time-component",
        "missing-start",
    ],
)
def test_bad_arguments_are_rejected_before_downloading(
    make_loader, fake_downloader, symbols, start, end, error, match
):
    with pytest.raises(error, match=match):
        make_loader().load(symbols, start, end)
    assert fake_downloader.calls == []


def test_yfinance_loader_satisfies_the_dataloader_protocol(cache_dir):
    assert isinstance(YFinanceLoader(cache_dir), DataLoader)


# ---------------------------------------------------------------------------
# Live data (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_spy_download_validates_and_caches(tmp_path):
    loader = YFinanceLoader(tmp_path)

    md = loader.load(["SPY"], "2020-01-01", "2020-12-31")

    assert len(md) == 253  # NYSE trading days in 2020
    meta = _read_meta(loader, "SPY")
    assert meta["source_version"] is not None
    assert meta["n_rows"] == 253
