"""Report figures: they render headless, draw the right data, and keep their styling contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.image import imread

from backtester import plotting
from backtester.experiment import load_config, run_experiment
from backtester.plotting import SERIES, _ends_collide, _runs

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def experiment(make_config):
    return run_experiment(load_config(make_config()))


def curves_frame(strategy: list[float], benchmark: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=len(strategy))
    return pd.DataFrame({"strategy": strategy, "benchmark": benchmark}, index=dates)


def data_lines(fig):
    """The plotted series (not end-dots), in drawing order."""
    return [line for line in fig.axes[0].get_lines() if len(line.get_xdata()) > 1]


class TestEquity:
    def test_draws_both_curves_from_the_common_start(self, experiment, tmp_path):
        curves = experiment.curves()

        fig = plotting.plot_equity(curves, tmp_path / "eq.png", labels=experiment.labels, title="t")

        lines = {line.get_label(): line for line in data_lines(fig)}
        strategy_label, benchmark_label = experiment.labels.values()
        # The validated pair: strategy blue (slot 1), benchmark orange (slot 2).
        assert lines[strategy_label].get_color() == "#2a78d6"
        assert lines[benchmark_label].get_color() == "#eb6834"
        for line in lines.values():
            assert line.get_ydata()[0] == pytest.approx(100_000)
            assert pd.Timestamp(line.get_xdata()[0]) == experiment.window[0]
        assert fig.axes[0].get_yscale() == "log"

    def test_strategy_is_drawn_on_top_but_listed_first(self, experiment, tmp_path):
        fig = plotting.plot_equity(
            experiment.curves(), tmp_path / "eq.png", labels=experiment.labels, title="t"
        )
        strategy_line, benchmark_line = sorted(data_lines(fig), key=lambda line: -line.get_zorder())
        assert strategy_line.get_color() == SERIES[0]
        legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert legend == list(experiment.labels.values())

    def test_end_labels_only_when_the_ends_are_apart(self, tmp_path):
        apart = curves_frame([100, 150, 200], [100, 110, 120])
        together = curves_frame([100, 150, 200], [100, 150, 199])

        labelled = plotting.plot_equity(apart, tmp_path / "a.png", title="t")
        crowded = plotting.plot_equity(together, tmp_path / "b.png", title="t")

        assert [t.get_text() for t in labelled.axes[0].texts] == [
            "strategy\n$200",
            "benchmark\n$120",
        ]
        assert len(crowded.axes[0].texts) == 0
        assert crowded.axes[0].get_legend() is not None  # the legend still names both

    def test_ticks_are_round_dollar_values(self, experiment, tmp_path):
        fig = plotting.plot_equity(experiment.curves(), tmp_path / "eq.png", title="t")
        ticks = fig.axes[0].get_yticks()
        assert len(ticks) >= 2
        assert all(float(t).is_integer() for t in ticks)


class TestDrawdowns:
    def test_draws_drawdowns_of_the_same_curves(self, experiment, tmp_path):
        curves = experiment.curves()

        fig = plotting.plot_drawdowns(
            curves, tmp_path / "dd.png", labels=experiment.labels, title="t"
        )

        for line in data_lines(fig):
            assert max(line.get_ydata()) == 0.0
            assert min(line.get_ydata()) <= 0.0
        legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert all("worst −" in text or "worst 0.0%" in text for text in legend)
        assert fig.axes[0].get_ylim()[1] == 0.0


class TestPositions:
    def test_shades_each_period_in_the_market(self, experiment, tmp_path):
        start = experiment.window[0]
        held = (experiment.strategy.positions.loc[start:] != 0).any(axis=1).to_numpy()

        fig = plotting.plot_positions(
            experiment.strategy,
            experiment.data.frame("SPY")["close"],
            tmp_path / "pos.png",
            start=start,
            title="t",
        )

        spans = [p for p in fig.axes[0].patches]
        assert len(_runs(held)) > 1  # the synthetic run really goes in and out
        assert len(spans) == len(_runs(held))
        (price,) = data_lines(fig)
        assert pd.Timestamp(price.get_xdata()[0]) == start

    def test_runs_helper(self):
        mask = np.array([False, True, True, False, True, False, False, True])
        assert _runs(mask) == [(1, 3), (4, 5), (7, 8)]
        assert _runs(np.zeros(3, dtype=bool)) == []


class TestFiles:
    @pytest.mark.parametrize("kind", ["equity", "drawdown", "positions"])
    def test_pngs_are_written_at_report_resolution(self, experiment, tmp_path, kind):
        path = tmp_path / "figures" / f"{kind}.png"
        curves = experiment.curves()
        if kind == "equity":
            plotting.plot_equity(curves, path, title="t")
        elif kind == "drawdown":
            plotting.plot_drawdowns(curves, path, title="t")
        else:
            plotting.plot_positions(
                experiment.strategy, experiment.data.frame("SPY")["close"], path, title="t"
            )

        height, width, _ = imread(path).shape
        assert (width, height) == (2000, 1080)

    def test_too_many_curves_are_rejected(self, tmp_path):
        frame = pd.DataFrame(
            {c: [1.0, 2.0] for c in "abc"}, index=pd.bdate_range("2021-01-04", periods=2)
        )
        with pytest.raises(ValueError, match="expected 1 to 2 curves"):
            plotting.plot_equity(frame, tmp_path / "x.png", title="t")

    def test_plotting_never_touches_pyplot(self):
        # No GUI backend, no global state: importing the module must not import pyplot.
        code = "import sys, backtester.plotting; assert 'matplotlib.pyplot' not in sys.modules"
        subprocess.run(
            [sys.executable, "-c", code], check=True, env={"PYTHONPATH": str(SRC), "MPLBACKEND": ""}
        )


def test_end_label_collision_rule():
    assert not _ends_collide(curves_frame([100, 150, 200], [100, 110, 120]))
    assert _ends_collide(curves_frame([100, 150, 200], [100, 150, 199]))
    assert _ends_collide(curves_frame([100, 100], [100, 100]))
