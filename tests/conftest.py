"""Shared pytest fixtures. All data is synthetic and seeded; nothing touches the network.

The generators behind these fixtures live in ``tests/synthetic.py`` so tests can
also call them directly (e.g. to build a frame with a specific shape).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from backtester.data import MarketData, YFinanceLoader
from synthetic import (
    STRUCTURAL_CASES,
    SYMBOLS,
    VALUE_CASES,
    FakeDownloader,
    InvalidCase,
    make_ohlcv,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """One valid symbol: 260 business days starting 2020-01-02, seed 42."""
    return make_ohlcv(seed=42)


@pytest.fixture
def universe() -> dict[str, pd.DataFrame]:
    """Three valid symbols over the same 2020 date range, each with its own seed."""
    return {sym: make_ohlcv(seed=i, start_price=50.0 * (i + 1)) for i, sym in enumerate(SYMBOLS)}


@pytest.fixture
def market_data(universe: dict[str, pd.DataFrame]) -> MarketData:
    """``universe`` as MarketData: three symbols sharing one 260-day calendar."""
    return MarketData.from_frames(universe)


@pytest.fixture
def staggered_market_data() -> MarketData:
    """Four symbols whose calendars differ, as real multi-asset data does.

    EARLY trades all year; MID lists on 2020-04-01; LATE lists on 2020-09-01;
    GONE stops trading after 2020-07-31.
    """
    return MarketData.from_frames(
        {
            "EARLY": make_ohlcv(seed=10),
            "MID": make_ohlcv(seed=11).loc["2020-04-01":],
            "LATE": make_ohlcv(seed=12).loc["2020-09-01":],
            "GONE": make_ohlcv(seed=13).loc[:"2020-07-31"],
        }
    )


@pytest.fixture
def fake_downloader(universe: dict[str, pd.DataFrame]) -> FakeDownloader:
    """A yfinance stand-in serving ``universe``. Tests may swap in bad frames."""
    return FakeDownloader(universe)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A fresh, not-yet-created cache directory per test."""
    return tmp_path / "cache"


@pytest.fixture
def make_loader(cache_dir: Path, fake_downloader: FakeDownloader) -> Callable[..., YFinanceLoader]:
    """Factory for loaders that share one cache dir and (by default) the fake downloader.

    Building a *new* loader per load call proves cache behaviour comes from the
    files on disk, not from state held in the loader object.
    """

    def _make(**kwargs: object) -> YFinanceLoader:
        kwargs.setdefault("downloader", fake_downloader)
        return YFinanceLoader(cache_dir, **kwargs)

    return _make


@pytest.fixture(params=VALUE_CASES, ids=lambda case: case.id)
def value_case(request: pytest.FixtureRequest) -> InvalidCase:
    """Each bad-value case in turn (one broken rule per case)."""
    return request.param


@pytest.fixture(params=STRUCTURAL_CASES, ids=lambda case: case.id)
def structural_case(request: pytest.FixtureRequest) -> InvalidCase:
    """Each malformed-frame case in turn (one broken rule per case)."""
    return request.param


# ---------------------------------------------------------------------------
# Experiments (config file + pre-filled data cache, no network)
# ---------------------------------------------------------------------------

#: A small experiment: SMA(5, 20) on a year of synthetic "SPY", $100k, no costs.
BASE_CONFIG: dict = {
    "name": "test_run",
    "data": {
        "symbol": "SPY",
        "start": "2020-01-02",
        "end": "2020-12-31",
        "cache_dir": "../data/cache",
    },
    "strategy": {"fast": 5, "slow": 20},
    "backtest": {"initial_cash": 100_000, "costs": "zero"},
    "output": {"reports_dir": "../reports"},
}


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., Path]:
    """Write an experiment config under ``tmp_path/project/configs`` and return its path.

    Keyword arguments replace whole top-level entries (``strategy={"fast": 3, "slow": 9}``);
    ``drop`` removes top-level keys. Unless ``prefill=False``, the data cache is
    filled first from a fake downloader serving ``prices`` (default: a seeded
    random walk), so running the experiment never touches the network.
    """

    def _make(
        *, prices: pd.DataFrame | None = None, prefill: bool = True, drop: tuple = (), **overrides
    ) -> Path:
        import yaml

        config = {**BASE_CONFIG, **overrides}
        for key in drop:
            config.pop(key)
        folder = tmp_path / "project" / "configs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{config.get('name', 'unnamed')}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        if prefill:
            data = config["data"]
            frame = prices if prices is not None else make_ohlcv(seed=4, start_price=300.0)
            loader = YFinanceLoader(
                folder / data["cache_dir"], downloader=FakeDownloader({data["symbol"]: frame})
            )
            loader.load([data["symbol"]], data["start"], data["end"])
        return path

    return _make
