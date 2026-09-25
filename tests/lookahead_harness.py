"""Test doubles and a minimal decision loop for the lookahead tests.

``synthetic.py`` only *generates data*. This module holds everything built to
*expose leaks*:

- :class:`ProbeStrategy`: output is a continuous function of the latest visible
  close, and it records exactly what it was shown at every step.
- :class:`PanelProbe`: the same idea, read through ``panel(lookback=...)``, so a
  leak confined to ``panel()`` is caught too.
- :class:`MomentumStrategy`: a realistic long/flat strategy, the kind whose
  discrete output could pass a leak test by luck.
- :class:`LeakyMarketData`: NEGATIVE CONTROL. Its views show one bar too many.
- :func:`run_decisions`: the decision half of the future engine loop.
- :func:`assert_no_lookahead`: the future-perturbation check.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.data import MarketData, MarketView
from backtester.strategies import Strategy
from synthetic import replace_after

# ---------------------------------------------------------------------------
# Strategies built to expose leaks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """What a probe was shown in one call."""

    now: pd.Timestamp
    last_seen: dict[str, pd.Timestamp]  # last date history() returned, per visible symbol
    fingerprint: str  # sha256 over everything panel() exposed


def fingerprint(view: MarketView) -> str:
    """A hash of every visible value, date, and symbol. Any change in what is visible changes it."""
    panel = view.panel()
    dates, symbols = panel.index.levels
    date_codes, symbol_codes = panel.index.codes
    digest = hashlib.sha256()
    for part in (panel.to_numpy(), dates.to_numpy(), date_codes, symbol_codes):
        digest.update(part.tobytes())
    digest.update("|".join(symbols).encode())
    return digest.hexdigest()


class ProbeStrategy(Strategy):
    """A strategy that cannot pass a lookahead test by luck.

    Its weight for each visible symbol is ``latest close % 1.0``. That is
    continuous and deterministic, so any change to the latest close it can see,
    however small, changes its output. A moving-average crossover, by contrast,
    only changes its output when a perturbation happens to flip a crossover.

    Every call also records ``view.now``, the last date ``history()`` returned for
    each symbol, and a fingerprint of the entire visible panel. Tests compare
    these observations as well as the decisions, so a leak into *any* visible
    value is caught, not just a leak into the latest close.
    """

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def target_weights(self, view: MarketView) -> dict[str, float]:
        weights: dict[str, float] = {}
        last_seen: dict[str, pd.Timestamp] = {}
        for symbol in view.symbols:
            close = view.history(symbol, "close")
            last_seen[symbol] = close.index[-1]
            weights[symbol] = float(close.iloc[-1] % 1.0)
        self.observations.append(Observation(view.now, last_seen, fingerprint(view)))
        return weights


class PanelProbe(Strategy):
    """Continuous output read through ``panel(lookback=3)``: ``mean(last 3 closes) % 1.0``."""

    warmup = 3

    def target_weights(self, view: MarketView) -> dict[str, float]:
        closes = view.panel(["close"], lookback=3)["close"]
        symbols, codes = closes.index.levels[1], closes.index.codes[1]
        means = np.bincount(codes, weights=closes.to_numpy()) / np.bincount(codes)
        return {str(symbol): float(mean % 1.0) for symbol, mean in zip(symbols, means, strict=True)}


class MomentumStrategy(Strategy):
    """Equal-weight long every symbol whose close is above its close ``lookback`` bars ago.

    A stand-in for a real rule-based strategy. Its output is discrete (long or
    flat), so on its own it could pass a leaky test by luck. That is why the
    negative control relies on the probes.
    """

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback
        self.warmup = lookback + 1
        self.name = f"momentum({lookback})"

    def target_weights(self, view: MarketView) -> dict[str, float]:
        size = 1.0 / len(view.symbols)
        weights: dict[str, float] = {}
        for symbol in view.symbols:
            close = view.history(symbol, "close")
            if len(close) > self.lookback:  # a newly listed symbol may not have enough history
                rising = close.iloc[-1] > close.iloc[-1 - self.lookback]
                weights[symbol] = size if rising else 0.0
        return weights


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


class LeakyMarketData(MarketData):
    """NEGATIVE CONTROL. Never use outside tests.

    Reproduces the classic off-by-one bug. ``view(t)`` still reports
    ``now == t``, but every symbol shows one extra bar: the one dated after
    ``t``. It overrides only ``MarketData._visible_rows`` (the single boundary
    method), so everything else about it is the real implementation. If a
    lookahead test passes against this class, that test is not checking anything.
    """

    def _visible_rows(self, symbol: str, cutoff: pd.Timestamp) -> int:
        honest = super()._visible_rows(symbol, cutoff)
        return min(honest + 1, len(self._index[symbol]))


def make_leaky(data: MarketData) -> LeakyMarketData:
    """The same bars as ``data``, with leaky views."""
    return LeakyMarketData.from_frames({s: data.frame(s) for s in data.symbols})


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_decisions(data: MarketData, strategy: Strategy) -> pd.DataFrame:
    """The decision half of the future engine loop, and nothing more.

    For every calendar date ``t`` with at least ``strategy.warmup`` dates visible,
    record ``strategy.target_weights(data.view(t))``. There are no fills,
    portfolio or equity yet; those arrive with the engine. Returns a
    (date x symbol) frame of target weights. A symbol the strategy left out is
    0.0, per the Strategy contract.
    """
    dates, rows = [], []
    for i, t in enumerate(data.calendar):
        if i + 1 < strategy.warmup:
            continue
        view = data.view(t)
        weights = strategy.target_weights(view)
        invisible = set(weights) - set(view.symbols)
        if invisible:
            raise AssertionError(
                f"{strategy.name} returned weights for {sorted(invisible)} on "
                f"{t:%Y-%m-%d}, but they are not visible then"
            )
        dates.append(t)
        rows.append([float(weights.get(symbol, 0.0)) for symbol in data.symbols])
    index = pd.DatetimeIndex(dates, name="date")
    return pd.DataFrame(rows, index=index, columns=list(data.symbols))


def assert_no_lookahead(
    data: MarketData,
    strategy_factory: Callable[[], Strategy],
    cutoff: pd.Timestamp,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Future-perturbation check: rewriting the future must not change the past.

    Runs a fresh strategy on ``data`` and another on ``replace_after(data, cutoff,
    seed)``, then requires every decision dated at or before ``cutoff`` to be
    bit-for-bit identical. For probes, it also requires everything they
    *observed* up to ``cutoff`` to be identical. Returns both decision frames so
    callers can check where they diverge.
    """
    cutoff = pd.Timestamp(cutoff)
    original, perturbed = strategy_factory(), strategy_factory()
    base = run_decisions(data, original)
    alt = run_decisions(replace_after(data, cutoff, seed), perturbed)

    pd.testing.assert_frame_equal(
        base.loc[:cutoff],
        alt.loc[:cutoff],
        check_exact=True,
        obj=f"{original.name} decisions at or before {cutoff:%Y-%m-%d}",
    )
    if isinstance(original, ProbeStrategy) and isinstance(perturbed, ProbeStrategy):
        seen = [o for o in original.observations if o.now <= cutoff]
        seen_alt = [o for o in perturbed.observations if o.now <= cutoff]
        assert seen == seen_alt, (
            f"{original.name} was shown different data at or before {cutoff:%Y-%m-%d}"
        )
    return base, alt


def first_divergence(a: pd.DataFrame, b: pd.DataFrame) -> pd.Timestamp | None:
    """The first date on which two decision frames differ, or ``None`` if they never do."""
    differs = (a != b).any(axis=1)
    return differs.idxmax() if differs.any() else None
