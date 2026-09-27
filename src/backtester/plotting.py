"""Static report figures (PNG), built with matplotlib's object API.

Figures are created with ``matplotlib.figure.Figure`` directly, never through
``pyplot``, so no window, GUI backend, or global plotting state is involved.
The same code runs headless in CI, from the script, or inside a notebook.

Styling follows one small, validated palette:

- The strategy is always categorical slot 1 (blue) and the benchmark slot 2
  (orange). The pair passes colorblind-separation and contrast checks on the
  light surface, and every chart also names each line in a legend and at its
  end, so color is never the only cue.
- Lines are thin; gridlines are solid hairlines; text is never drawn in a
  series color.

Functions take plain pandas objects (``metrics.rebased_equity`` output, a
``BacktestResult``, a price series) and return the ``Figure`` after saving it.
They compute nothing that isn't already in ``metrics``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MaxNLocator,
    NullLocator,
    PercentFormatter,
)

from backtester.metrics import drawdown_series, exposure
from backtester.results import BacktestResult

# Light-surface tokens (reference palette, validated for these two series).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # strategy, benchmark: fixed order, never cycled

FIGSIZE = (10.0, 5.4)
DPI = 200
LINE_WIDTH = 1.6


def plot_equity(
    curves: pd.DataFrame,
    path: str | Path,
    *,
    labels: Mapping[str, str] | None = None,
    title: str,
    subtitle: str = "",
) -> Figure:
    """Equity curves on one log-scale axis: strategy first (blue), benchmark second (orange).

    ``curves`` is ``metrics.rebased_equity`` output: one column per strategy,
    all starting at the same value on the same date. Each line ends in a dot
    with its name and final value beside it.
    """
    names = _checked_columns(curves)
    label_ends = not _ends_collide(curves)
    fig, ax = _figure(title, subtitle, right_margin=0.80 if label_ends else 0.97)
    handles, texts = _lines(ax, curves, names, labels)
    ax.set_yscale("log")
    _log_ticks(ax, curves.min().min(), curves.max().max(), money=True)
    if label_ends:
        _end_labels(ax, curves, names, labels)
    _finish(ax, curves.index)
    _legend(ax, handles, texts)
    return _save(fig, path)


def plot_drawdowns(
    curves: pd.DataFrame,
    path: str | Path,
    *,
    labels: Mapping[str, str] | None = None,
    title: str,
    subtitle: str = "",
) -> Figure:
    """Each curve's distance below its running peak, with its worst drawdown in the legend."""
    fig, ax = _figure(title, subtitle)
    names = _checked_columns(curves)
    drawdowns = pd.DataFrame({name: drawdown_series(curves[name]) for name in names})
    worst = {
        name: f"{_label(labels, name)}  ·  worst {_pct(drawdowns[name].min())}" for name in names
    }
    handles, texts = _lines(ax, drawdowns, names, worst, end_dot=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(top=0.0)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    _finish(ax, curves.index)
    _legend(ax, handles, texts)
    return _save(fig, path)


def plot_positions(
    result: BacktestResult,
    prices: pd.Series,
    path: str | Path,
    *,
    start: pd.Timestamp | None = None,
    label: str | None = None,
    price_label: str = "Close",
    title: str,
    subtitle: str = "",
) -> Figure:
    """The traded symbol's price, shaded wherever the strategy holds a position.

    A position shown on date d is the one held at d's close. Shading runs from
    the first held date to the next date after the last held one, so each band
    covers the days the strategy was exposed to the price moves.
    """
    fig, ax = _figure(title, subtitle)
    first = result.equity.index[0] if start is None else pd.Timestamp(start)
    held = result.positions.loc[first:]
    window = prices.loc[held.index[0] : held.index[-1]]
    invested = (held != 0).any(axis=1).to_numpy()

    ax.plot(
        window.index,
        window.to_numpy(),
        color=TEXT_SECONDARY,
        linewidth=1.2,
        solid_joinstyle="round",
        label=price_label,
        zorder=3,
    )
    dates = held.index
    for begin, stop in _runs(invested):
        right = dates[stop] if stop < len(dates) else dates[-1]
        ax.axvspan(dates[begin], right, color=SERIES[0], alpha=0.14, linewidth=0, zorder=1)

    ax.set_yscale("log")
    _log_ticks(ax, window.min(), window.max(), money=False)
    _finish(ax, window.index)
    name = label or result.strategy_name
    handles, texts = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=SERIES[0], alpha=0.14, linewidth=0))
    texts.append(f"{name} in the market ({exposure(held):.0%} of days)")
    _legend(ax, handles, texts)
    return _save(fig, path)


# --- building blocks ----------------------------------------------------------


def _figure(title: str, subtitle: str, right_margin: float = 0.97) -> tuple[Figure, Axes]:
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0.08, 0.11, right_margin - 0.08, 0.69))
    ax.set_facecolor(SURFACE)
    fig.text(0.08, 0.93, title, fontsize=13, fontweight="semibold", color=TEXT, va="bottom")
    if subtitle:
        fig.text(0.08, 0.915, subtitle, fontsize=9.5, color=TEXT_SECONDARY, va="top")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, labelsize=9, length=0, pad=6)
    ax.tick_params(axis="x", length=3, color=BASELINE)
    return fig, ax


def _lines(
    ax: Axes,
    frame: pd.DataFrame,
    names: list[str],
    labels: Mapping[str, str] | None,
    end_dot: bool = True,
) -> tuple[list, list[str]]:
    """One line per column, colored by position; the first (the strategy) is drawn on top.

    Returns legend handles and texts in column order (strategy first), which is
    not the drawing order.
    """
    handles: dict[str, object] = {}
    for i in reversed(range(len(names))):
        name = names[i]
        handles[name] = _line(
            ax, frame[name], SERIES[i], _label(labels, name), 3 + len(names) - i, end_dot
        )
    return [handles[n] for n in names], [_label(labels, n) for n in names]


def _legend(ax: Axes, handles: list, texts: list[str]) -> None:
    """A single row of legend entries just above the plot, clear of the data."""
    ax.legend(
        handles,
        texts,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=len(handles),
        frameon=False,
        fontsize=9.5,
        labelcolor=TEXT,
        handlelength=1.8,
        columnspacing=2.0,
        borderaxespad=0.0,
        borderpad=0.0,
    )


def _line(
    ax: Axes, series: pd.Series, color: str, label: str, zorder: int = 3, end_dot: bool = True
) -> object:
    (line,) = ax.plot(
        series.index,
        series.to_numpy(),
        color=color,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        zorder=zorder,
    )
    if end_dot:
        ax.plot(
            series.index[-1:],
            series.to_numpy()[-1:],
            marker="o",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor=SURFACE,  # surface ring keeps the dot legible over lines
            markeredgewidth=1.2,
            linestyle="none",
            zorder=zorder + 0.5,
            clip_on=False,  # the last point sits on the axis edge; show the whole dot
        )
    return line


def _ends_collide(curves: pd.DataFrame) -> bool:
    """Whether the lines finish too close together (on the log axis) to label each end."""
    log = np.log(curves.to_numpy(dtype=float))
    span = log.max() - log.min()
    if span == 0:
        return len(curves.columns) > 1
    ends = np.sort((log[-1] - log.min()) / span)
    return bool(np.any(np.diff(ends) < 0.07))


def _end_labels(
    ax: Axes, curves: pd.DataFrame, names: list[str], labels: Mapping[str, str] | None
) -> None:
    """Name and final value beside each line's end (callers check ``_ends_collide`` first)."""
    last = curves.iloc[-1]
    for name in names:
        ax.annotate(
            f"{_label(labels, name)}\n${last[name]:,.0f}",
            xy=(curves.index[-1], last[name]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT,
            annotation_clip=False,
        )


def _log_ticks(ax: Axes, lo: float, hi: float, money: bool) -> None:
    """Round-number ticks on a log axis: the coarsest set giving at least four ticks."""
    lo, hi = float(lo), float(hi)
    ax.set_ylim(lo / 1.04, hi * 1.04)
    lo_axis, hi_axis = ax.get_ylim()
    ticks: list[float] = []
    for mantissas in ((1,), (1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 4, 5, 6, 8)):
        ticks = [
            m * 10.0**p
            for p in range(math.floor(math.log10(lo_axis)) - 1, math.ceil(math.log10(hi_axis)) + 1)
            for m in mantissas
            if lo_axis <= m * 10.0**p <= hi_axis
        ]
        if len(ticks) >= 4:
            break
    if len(ticks) < 4:  # less than about a doubling: round linear steps read better
        nice = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo_axis, hi_axis)
        ticks = [t for t in nice if lo_axis <= t <= hi_axis]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_money if money else _price))


def _money(value: float, _pos: int | None = None) -> str:
    if value >= 10_000:
        thousands = value / 1000
        return f"${thousands:,.0f}k" if thousands.is_integer() else f"${thousands:,.1f}k"
    return f"${value:,.0f}"


def _pct(value: float) -> str:
    return f"{value:.1%}".replace("-", "\u2212")  # a true minus sign, matching the axis


def _price(value: float, _pos: int | None = None) -> str:
    return f"${value:,.0f}" if value >= 10 else f"${value:,.2f}"


def _finish(ax: Axes, index: pd.DatetimeIndex) -> None:
    locator = AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator, show_offset=False))
    ax.set_xlim(index[0], index[-1])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs for each run of True values."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2], strict=True))


def _checked_columns(curves: pd.DataFrame) -> list[str]:
    names = [str(c) for c in curves.columns]
    if not 1 <= len(names) <= len(SERIES):
        raise ValueError(f"expected 1 to {len(SERIES)} curves, got {len(names)}")
    return names


def _label(labels: Mapping[str, str] | None, name: str) -> str:
    return (labels or {}).get(name, name)


def _save(fig: Figure, path: str | Path) -> Figure:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    return fig
