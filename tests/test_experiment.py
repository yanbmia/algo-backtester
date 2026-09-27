"""The experiment layer: config validation, running both strategies, and writing outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.data import YFinanceLoader
from backtester.experiment import (
    BacktestError,
    ConfigError,
    DataError,
    format_table,
    load_config,
    run_experiment,
    write_outputs,
)
from synthetic import FakeDownloader

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_config(self, make_config):
        config = load_config(make_config())

        assert (config.name, config.symbol) == ("test_run", "SPY")
        assert (config.start, config.end) == (
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-12-31"),
        )
        assert (config.fast, config.slow, config.initial_cash, config.costs) == (
            5,
            20,
            100_000.0,
            "zero",
        )

    def test_relative_paths_resolve_from_the_config_folder_not_the_cwd(
        self, make_config, tmp_path, monkeypatch
    ):
        path = make_config()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        config = load_config(path)

        assert config.cache_dir == (tmp_path / "project" / "data" / "cache").resolve()
        assert config.reports_dir == (tmp_path / "project" / "reports").resolve()

    def test_the_repository_config_is_pinned(self):
        config = load_config(REPO / "configs" / "sma_spy.yaml")

        assert (config.symbol, config.fast, config.slow) == ("SPY", 50, 200)
        assert (config.start, config.end) == (
            pd.Timestamp("2004-01-02"),
            pd.Timestamp("2024-12-31"),
        )
        assert (config.initial_cash, config.costs) == (100_000.0, "zero")
        assert config.cache_dir == REPO / "data" / "cache"
        assert config.reports_dir == REPO / "reports"

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"drop": ("strategy",)}, "missing key 'strategy'"),
            ({"extra": 1}, "unknown key 'extra'"),
            (
                {"data": {"symbol": "SPY", "start": "2020-01-02", "end": "2020-12-31"}},
                "missing key 'data.cache_dir'",
            ),
            ({"strategy": {"fast": 5, "slow": 20, "signal": 3}}, "unknown key 'strategy.signal'"),
            ({"name": "my run!"}, "'name' must use only"),
            (
                {
                    "data": {
                        **{"symbol": "spy"},
                        "start": "2020-01-02",
                        "end": "2020-12-31",
                        "cache_dir": "c",
                    }
                },
                "uppercase Yahoo tickers",
            ),
            (
                {
                    "data": {
                        "symbol": "SPY",
                        "start": "2020-01-02",
                        "end": "2099-01-01",
                        "cache_dir": "c",
                    }
                },
                "must be before today",
            ),
            (
                {
                    "data": {
                        "symbol": "SPY",
                        "start": "2020-06-01",
                        "end": "2020-01-02",
                        "cache_dir": "c",
                    }
                },
                "is after end",
            ),
            ({"strategy": {"fast": 20, "slow": 20}}, "shorter than slow"),
            ({"strategy": {"fast": 5.5, "slow": 20}}, "fast must be an int"),
            ({"backtest": {"initial_cash": -1, "costs": "zero"}}, "positive number"),
            ({"backtest": {"initial_cash": True, "costs": "zero"}}, "positive number"),
            ({"backtest": {"initial_cash": 1000, "costs": "5bps"}}, "only 'zero' is supported"),
            ({"output": {"reports_dir": 3}}, "must be a path string"),
        ],
        ids=[
            "missing-section",
            "unknown-key",
            "missing-nested-key",
            "unknown-nested-key",
            "bad-name",
            "lowercase-symbol",
            "future-end",
            "start-after-end",
            "equal-windows",
            "float-window",
            "negative-cash",
            "bool-cash",
            "unsupported-costs",
            "non-string-path",
        ],
    )
    def test_invalid_configs_are_rejected_with_the_reason(self, make_config, overrides, fragment):
        path = make_config(prefill=False, **overrides)
        with pytest.raises(ConfigError, match=fragment):
            load_config(path)

    def test_every_problem_is_reported_at_once(self, make_config):
        path = make_config(
            prefill=False,
            strategy={"fast": 30, "slow": 20},
            backtest={"initial_cash": 0, "costs": "5bps"},
        )
        with pytest.raises(ConfigError, match=r"has 3 problem\(s\)"):
            load_config(path)

    def test_unreadable_files_are_config_errors(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot read config"):
            load_config(tmp_path / "nope.yaml")
        bad = tmp_path / "bad.yaml"
        bad.write_text("data: [unclosed")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(bad)
        bad.write_text("- just\n- a list\n")
        with pytest.raises(ConfigError, match="must contain a mapping"):
            load_config(bad)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment(make_config):
    return run_experiment(load_config(make_config()))


class TestRunExperiment:
    def test_runs_the_crossover_and_its_benchmark_on_the_same_data(self, experiment):
        strategy, benchmark = experiment.results

        assert strategy.strategy_name == "sma_crossover(SPY, 5, 20)"
        assert benchmark.strategy_name == "buy_and_hold(SPY)"
        assert strategy.equity.index.equals(benchmark.equity.index)
        assert strategy.equity.index.equals(experiment.data.calendar)
        for result in experiment.results:
            assert result.config["execution"] == "NextOpenExecution(cost_model=ZeroCost())"
            assert result.config["initial_cash"] == 100_000.0

    def test_comparison_starts_at_the_crossovers_first_decision(self, experiment):
        calendar = experiment.data.calendar
        strategy, benchmark = experiment.results

        assert strategy.decisions.index[0] == calendar[19]  # slow = 20 bars of warmup
        assert benchmark.decisions.index[0] == calendar[0]
        assert experiment.window == (calendar[19], calendar[-1])
        table = experiment.comparison()
        assert set(table["start"]) == {calendar[19]}

    def test_benchmark_is_measured_from_the_common_start_in_comparisons_only(self, experiment):
        _, benchmark = experiment.results
        calendar = experiment.data.calendar
        equity = benchmark.equity

        solo = experiment.standalone().loc[benchmark.strategy_name]
        compared = experiment.comparison().loc[benchmark.strategy_name]

        assert solo["start"] == calendar[0]
        assert solo["total_return"] == pytest.approx(equity.iloc[-1] / equity.iloc[0] - 1)
        assert compared["total_return"] == pytest.approx(
            equity.iloc[-1] / equity.loc[calendar[19]] - 1
        )
        assert compared["ending_value"] == pytest.approx(
            100_000 * equity.iloc[-1] / equity.loc[calendar[19]]
        )

    def test_curves_start_together_at_the_starting_cash(self, experiment):
        curves = experiment.curves()
        assert list(curves.columns) == [r.strategy_name for r in experiment.results]
        np.testing.assert_allclose(curves.iloc[0], [100_000, 100_000])

    def test_labels_are_readable(self, experiment):
        assert list(experiment.labels.values()) == ["SMA crossover (5/20)", "Buy and hold"]

    def test_records_where_the_data_came_from(self, experiment):
        provenance = experiment.data_provenance
        assert provenance["symbol"] == "SPY"
        assert len(provenance["sha256"]) == 64

    def test_data_problems_become_data_errors(self, make_config):
        config = load_config(make_config(prefill=False))

        def offline(symbol, start, end):
            raise ConnectionError("no network")

        loader = YFinanceLoader(config.cache_dir, downloader=offline)
        with pytest.raises(DataError, match="could not load SPY.*no network"):
            run_experiment(config, loader=loader)

    def test_too_little_data_for_the_warmup_is_a_backtest_error(self, make_config):
        config = load_config(make_config(strategy={"fast": 50, "slow": 300}))
        with pytest.raises(BacktestError, match=r"needs warmup=300 bars"):
            run_experiment(config)

    def test_uses_the_injected_loader(self, make_config, universe):
        config = load_config(make_config(prefill=False))
        loader = YFinanceLoader(
            config.cache_dir, downloader=FakeDownloader({"SPY": universe["AAA"]})
        )
        experiment = run_experiment(config, loader=loader)
        assert len(experiment.data) == len(universe["AAA"])


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class TestWriteOutputs:
    def test_writes_tables_figures_and_manifest(self, experiment):
        paths = write_outputs(experiment)
        reports = experiment.config.reports_dir

        assert set(paths) == {
            "comparison",
            "standalone",
            "manifest",
            "equity",
            "drawdown",
            "positions",
        }
        assert paths["comparison"] == reports / "test_run_comparison.csv"
        assert paths["equity"] == reports / "figures" / "test_run_equity.png"
        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_comparison_csv_round_trips(self, experiment):
        paths = write_outputs(experiment)
        table = pd.read_csv(paths["comparison"], index_col="strategy")

        assert list(table.index) == ["sma_crossover(SPY, 5, 20)", "buy_and_hold(SPY)"]
        assert set(table["start"]) == {f"{experiment.window[0]:%Y-%m-%d}"}
        np.testing.assert_allclose(
            table["total_return"], experiment.comparison()["total_return"], rtol=1e-9
        )

    def test_manifest_is_portable_and_complete(self, experiment, tmp_path):
        paths = write_outputs(experiment)
        text = paths["manifest"].read_text()
        manifest = json.loads(text)

        assert str(tmp_path) not in text  # no machine-specific paths
        assert manifest["config"]["data"]["cache_dir"] == "../data/cache"
        assert manifest["comparison_window"]["start"] == f"{experiment.window[0]:%Y-%m-%d}"
        assert manifest["data"]["cache"]["sha256"] == experiment.data_provenance["sha256"]
        assert set(manifest["strategies"]) == {r.strategy_name for r in experiment.results}
        assert {"python", "pandas", "matplotlib", "backtester"} <= set(manifest["versions"])

    def test_rerunning_rewrites_identical_files(self, make_config):
        config = load_config(make_config())
        first = {k: p.read_bytes() for k, p in write_outputs(run_experiment(config)).items()}
        second = {k: p.read_bytes() for k, p in write_outputs(run_experiment(config)).items()}
        assert first == second


def test_format_table_makes_a_display_copy(experiment):
    raw = experiment.comparison()

    shown = format_table(raw, experiment.labels)

    assert list(shown.index) == ["SMA crossover (5/20)", "Buy and hold"]
    row = shown.loc["Buy and hold"]
    assert row["start"] == f"{experiment.window[0]:%Y-%m-%d}"
    assert row["exposure"] == "100.0%"
    assert row["ending_value"].startswith("$")
    assert row["total_return"] == f"{raw.iloc[1]['total_return']:.1%}"
    assert raw["total_return"].dtype == "float64"  # the original is untouched
