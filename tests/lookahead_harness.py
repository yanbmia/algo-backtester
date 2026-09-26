"""Test doubles and the future-perturbation harness for the lookahead tests.

``synthetic.py`` only *generates data*. This module holds everything built to
*expose leaks*:

- :class:`ProbeStrategy`: output is a continuous function of the latest visible
  close, and it records exactly what it was shown at every step.
- :class:`PanelProbe`: the same idea, read through ``panel(lookback=...)``, so a
  leak confined to ``panel()`` is caught too.
- :class:`MomentumStrategy`: a realistic long/flat strategy, the kind whose
  discrete output could pass a leak test by luck.
- :class:`LeakyMarketData`: NEGATIVE CONTROL. Its views show one bar too many.
- :func:`run_backtest`: the real engine, with v1's stated execution assumptions.
- :func:`assert_no_lookahead`: the future-perturbation check, over decisions,
  equity, cash, positions, and fills.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.data import MarketData, MarketView
from backtester.engine import Backtester, NextOpenExecution, ZeroCost
from backtester.results import BacktestResult
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

    Its weight for each visible symbol is ``(latest close % 1.0) / n``, where
    ``n`` is the number of visible symbols. That is continuous and deterministic,
    so any change to the latest close it can see, however small, changes its
    output. A moving-average crossover, by contrast, only changes its output when
    a perturbation happens to flip a crossover. (Dividing by ``n`` keeps the
    weights inside the engine's long/flat, unlevered limits: each is in [0, 1)
    and they sum to less than 1.)

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
        n = len(view.symbols)
        for symbol in view.symbols:
            close = view.history(symbol, "close")
            last_seen[symbol] = close.index[-1]
            weights[symbol] = float(close.iloc[-1] % 1.0) / n
        self.observations.append(Observation(view.now, last_seen, fingerprint(view)))
        return weights


class PanelProbe(Strategy):
    """Continuous output read through ``panel(lookback=3)``: ``(mean of last 3 closes % 1) / n``."""

    warmup = 3

    def target_weights(self, view: MarketView) -> dict[str, float]:
        closes = view.panel(["close"], lookback=3)["close"]
        symbols, codes = closes.index.levels[1], closes.index.codes[1]
        means = np.bincount(codes, weights=closes.to_numpy()) / np.bincount(codes)
        n = len(symbols)
        return {str(s): float(mean % 1.0) / n for s, mean in zip(symbols, means, strict=True)}


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


def run_backtest(
    data: MarketData, strategy: Strategy, start: object = None, end: object = None
) -> BacktestResult:
    """Run the real engine under v1's stated assumptions: next-open fills, zero cost, $100k."""
    engine = Backtester(
        data, strategy, execution=NextOpenExecution(ZeroCost()), initial_cash=100_000
    )
    return engine.run(start=start, end=end)


def assert_no_lookahead(
    data: MarketData,
    strategy_factory: Callable[[], Strategy],
    cutoff: pd.Timestamp,
    seed: int = 7,
    end: object = None,
) -> tuple[BacktestResult, BacktestResult]:
    """Future-perturbation check: rewriting the future must not change the past.

    Runs the full engine with a fresh strategy on ``data``, and again on
    ``replace_after(data, cutoff, seed)``. Everything dated at or before
    ``cutoff`` must then be bit-for-bit identical: decisions, equity, cash,
    positions, and fills. For probes, everything they *observed* up to
    ``cutoff`` must match too. Returns both results so callers can check where
    they diverge.

    Decisions are checked first on purpose. A one-bar leak changes the decision
    at the cutoff itself, but that decision only trades at the next open, so
    equity and positions through the cutoff are unaffected by it. Equity alone
    therefore cannot catch that leak; see ``test_lookahead.py``.
    """
    cutoff = pd.Timestamp(cutoff)
    original, perturbed = strategy_factory(), strategy_factory()
    base = run_backtest(data, original, end=end)
    alt = run_backtest(replace_after(data, cutoff, seed), perturbed, end=end)

    label = f"{original.name} {{}} at or before {cutoff:%Y-%m-%d}"
    pd.testing.assert_frame_equal(
        base.decisions.loc[:cutoff],
        alt.decisions.loc[:cutoff],
        check_exact=True,
        obj=label.format("decisions"),
    )
    for name in ("equity", "cash"):
        pd.testing.assert_series_equal(
            getattr(base, name).loc[:cutoff],
            getattr(alt, name).loc[:cutoff],
            check_exact=True,
            obj=label.format(name),
        )
    pd.testing.assert_frame_equal(
        base.positions.loc[:cutoff],
        alt.positions.loc[:cutoff],
        check_exact=True,
        obj=label.format("positions"),
    )
    pd.testing.assert_frame_equal(
        base.fills[base.fills["date"] <= cutoff],
        alt.fills[alt.fills["date"] <= cutoff],
        check_exact=True,
        obj=label.format("fills"),
    )
    if isinstance(original, ProbeStrategy) and isinstance(perturbed, ProbeStrategy):
        seen = [o for o in original.observations if o.now <= cutoff]
        seen_alt = [o for o in perturbed.observations if o.now <= cutoff]
        assert seen == seen_alt, (
            f"{original.name} was shown different data at or before {cutoff:%Y-%m-%d}"
        )
    return base, alt


def first_divergence(
    a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series
) -> pd.Timestamp | None:
    """The first date on which two date-indexed frames or series differ, or ``None``."""
    differs = a != b
    if isinstance(differs, pd.DataFrame):
        differs = differs.any(axis=1)
    return differs.idxmax() if differs.any() else None
