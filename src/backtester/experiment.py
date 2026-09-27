"""One experiment, end to end: config -> data -> strategy and benchmark runs -> outputs.

``scripts/run_backtest.py`` and ``notebooks/01_results.ipynb`` both go through
this module, so neither holds logic of its own and everything they show is
covered by the test suite.

An experiment runs ``SMACrossover`` and its ``BuyAndHold`` benchmark on the same
data, through the same engine, with the same execution and cost assumptions.
Standalone metrics measure each strategy from its own first decision.
Comparison outputs (the comparison table and the equity and drawdown charts)
use ``metrics.comparison_window``, so the benchmark gets no head start while
the crossover is still warming up.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

import backtester
from backtester import plotting
from backtester.data import (
    CacheIntegrityError,
    DataDownloadError,
    DataLoader,
    DataValidationError,
    MarketData,
    YFinanceLoader,
)
from backtester.data.loader import _check_range, _check_symbols
from backtester.engine import Backtester, EngineError, NextOpenExecution, StrategyError, ZeroCost
from backtester.metrics import compare, comparison_window, rebased_equity, summarize
from backtester.results import BacktestResult
from backtester.strategies import BuyAndHold, SMACrossover

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

#: Every key a config may contain. Anything else is a typo and is rejected.
_SCHEMA: dict[str, tuple[str, ...]] = {
    "data": ("symbol", "start", "end", "cache_dir"),
    "strategy": ("fast", "slow"),
    "backtest": ("initial_cash", "costs"),
    "output": ("reports_dir",),
}

#: The only cost setting v1 supports. The config must state it explicitly.
SUPPORTED_COSTS = ("zero",)


class ConfigError(ValueError):
    """The config file is missing, unreadable, or invalid. Lists every problem found."""


class ExperimentError(RuntimeError):
    """A run failed for a reason outside the code: bad data or an impossible setup."""


class DataError(ExperimentError):
    """The market data could not be loaded or failed validation."""


class BacktestError(ExperimentError):
    """The engine rejected the run (e.g. too little data for the strategy's warmup)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """A validated experiment config. Paths are resolved from the config file's folder."""

    path: Path
    name: str
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    fast: int
    slow: int
    initial_cash: float
    costs: str
    cache_dir: Path
    reports_dir: Path

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly settings, with paths relative to the config file (no machine paths)."""
        here = self.path.parent
        return {
            "config_file": self.path.name,
            "name": self.name,
            "data": {
                "symbol": self.symbol,
                "start": self.start.strftime("%Y-%m-%d"),
                "end": self.end.strftime("%Y-%m-%d"),
                "cache_dir": os.path.relpath(self.cache_dir, here),
            },
            "strategy": {"fast": self.fast, "slow": self.slow},
            "backtest": {"initial_cash": self.initial_cash, "costs": self.costs},
            "output": {"reports_dir": os.path.relpath(self.reports_dir, here)},
        }


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate a YAML experiment config.

    Checks run up front so a bad config fails before any download or backtest.
    Unknown or missing keys, wrong types, an end date that is not in the past,
    an invalid ticker, crossover windows the strategy would reject, and
    unsupported cost settings all raise ``ConfigError``, listing every problem.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror or exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping of settings, got {type(raw).__name__}")

    problems: list[str] = []
    expected_top = {"name", *_SCHEMA}
    problems += [f"unknown key '{k}'" for k in raw if k not in expected_top]
    problems += [f"missing key '{k}'" for k in sorted(expected_top) if k not in raw]

    sections: dict[str, dict[str, Any]] = {}
    for section, keys in _SCHEMA.items():
        value = raw.get(section)
        if section in raw and not isinstance(value, dict):
            problems.append(f"'{section}' must be a mapping")
            value = {}
        value = value or {}
        problems += [f"unknown key '{section}.{k}'" for k in value if k not in keys]
        problems += [f"missing key '{section}.{k}'" for k in keys if k not in value]
        sections[section] = value

    name = raw.get("name")
    if "name" in raw and not (isinstance(name, str) and _NAME_RE.fullmatch(name)):
        problems.append(f"'name' must use only letters, digits, '_' and '-', got {name!r}")

    data, strategy, backtest, output = (sections[s] for s in _SCHEMA)
    symbol = data.get("symbol")
    if "symbol" in data:
        try:
            _check_symbols([symbol])
        except (TypeError, ValueError) as exc:
            problems.append(f"data.symbol: {exc}")

    start = end = None
    if "start" in data and "end" in data:
        try:
            start, end = _check_range(_as_date(data["start"]), _as_date(data["end"]))
        except (TypeError, ValueError) as exc:
            problems.append(f"data.start/end: {exc}")

    fast, slow = strategy.get("fast"), strategy.get("slow")
    if "fast" in strategy and "slow" in strategy:
        try:
            SMACrossover("CHECK", fast, slow)  # the strategy's own validation
        except (TypeError, ValueError) as exc:
            problems.append(f"strategy: {exc}")

    cash = backtest.get("initial_cash")
    if "initial_cash" in backtest and not (
        isinstance(cash, int | float)
        and not isinstance(cash, bool)
        and math.isfinite(cash)
        and cash > 0
    ):
        problems.append(f"backtest.initial_cash must be a positive number, got {cash!r}")

    costs = backtest.get("costs")
    if "costs" in backtest and costs not in SUPPORTED_COSTS:
        problems.append(
            f"backtest.costs: only {', '.join(repr(c) for c in SUPPORTED_COSTS)} is "
            f"supported in v1, got {costs!r}"
        )

    for section, key in (("data", "cache_dir"), ("output", "reports_dir")):
        value = sections[section].get(key)
        if key in sections[section] and not (isinstance(value, str) and value):
            problems.append(f"{section}.{key} must be a path string, got {value!r}")

    if problems:
        raise ConfigError(f"{path} has {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))

    here = path.resolve().parent
    return ExperimentConfig(
        path=path.resolve(),
        name=name,
        symbol=symbol,
        start=start,
        end=end,
        fast=fast,
        slow=slow,
        initial_cash=float(cash),
        costs=costs,
        cache_dir=(here / data["cache_dir"]).resolve(),
        reports_dir=(here / output["reports_dir"]).resolve(),
    )


def _as_date(value: object) -> object:
    # YAML reads 2024-12-31 as a date; quoted values arrive as strings. Both are fine.
    if isinstance(value, date | str):
        return value
    raise TypeError(f"dates must look like 2024-12-31, got {value!r}")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """The outcome of one run: the data, the strategy's result, and the benchmark's."""

    config: ExperimentConfig
    data: MarketData
    strategy: BacktestResult
    benchmark: BacktestResult
    data_provenance: Mapping[str, Any] | None = None

    @property
    def results(self) -> tuple[BacktestResult, BacktestResult]:
        """Strategy first, then benchmark (charts color them in this order)."""
        return self.strategy, self.benchmark

    @property
    def labels(self) -> dict[str, str]:
        """Human-readable names for charts and tables, keyed by ``strategy_name``."""
        return {
            self.strategy.strategy_name: f"SMA crossover ({self.config.fast}/{self.config.slow})",
            self.benchmark.strategy_name: "Buy and hold",
        }

    @property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The common comparison window (see ``metrics.comparison_window``)."""
        return comparison_window(self.results)

    def comparison(self, rf_annual: float = 0.0) -> pd.DataFrame:
        """Strategy vs. benchmark over the common window, ending values in dollars."""
        return compare(self.results, rf_annual, base=self.config.initial_cash)

    def standalone(self, rf_annual: float = 0.0) -> pd.DataFrame:
        """Each strategy over its own window, from its own first decision."""
        return pd.DataFrame([summarize(r, rf_annual) for r in self.results])

    def curves(self) -> pd.DataFrame:
        """Both equity curves on the common window, each starting at ``initial_cash``."""
        return rebased_equity(self.results, base=self.config.initial_cash)


def run_experiment(
    config: ExperimentConfig, *, loader: DataLoader | None = None, refresh: bool = False
) -> Experiment:
    """Load the data once and run the crossover and its benchmark on it.

    Both runs use ``NextOpenExecution(ZeroCost())``, the config's stated
    execution and cost assumption, and the same starting cash. ``loader``
    defaults to a ``YFinanceLoader`` on the config's cache directory (tests
    pass one with a fake downloader).

    Raises:
        DataError: the data could not be loaded or failed validation.
        BacktestError: the engine rejected the run (e.g. the date range is too
            short for the slow window's warmup).
    """
    loader = loader if loader is not None else YFinanceLoader(config.cache_dir, refresh=refresh)
    try:
        data = loader.load([config.symbol], config.start, config.end)
    except (DataDownloadError, DataValidationError, CacheIntegrityError, ValueError) as exc:
        raise DataError(
            f"could not load {config.symbol} from {config.start:%Y-%m-%d} to "
            f"{config.end:%Y-%m-%d}: {exc}"
        ) from exc

    execution = NextOpenExecution(ZeroCost())  # the only costs setting v1 accepts
    runs = []
    for strategy in (
        SMACrossover(config.symbol, config.fast, config.slow),
        BuyAndHold(config.symbol),
    ):
        engine = Backtester(data, strategy, execution=execution, initial_cash=config.initial_cash)
        try:
            runs.append(engine.run())
        except (StrategyError, EngineError, ValueError) as exc:
            raise BacktestError(f"{strategy.name}: {exc}") from exc

    provenance = None
    if isinstance(loader, YFinanceLoader):
        _, meta_path = loader.cache_paths(config.symbol)
        if meta_path.exists():
            provenance = json.loads(meta_path.read_text())
    return Experiment(config, data, runs[0], runs[1], provenance)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

_PERCENT = ("total_return", "cagr", "annualized_vol", "max_drawdown", "exposure")
_DATES = ("start", "end", "max_drawdown_peak", "max_drawdown_trough")
_DOLLARS = ("ending_value", "final_equity")


def format_table(table: pd.DataFrame, labels: Mapping[str, str] | None = None) -> pd.DataFrame:
    """A display copy of a ``compare``/``summarize`` table: percentages, dates, dollars."""
    out = pd.DataFrame(index=table.index, columns=table.columns, dtype=object)
    for column in table.columns:
        values = table[column]
        if column in _PERCENT:
            out[column] = [f"{v:.1%}" for v in values]
        elif column in _DATES:
            out[column] = ["" if pd.isna(v) else f"{pd.Timestamp(v):%Y-%m-%d}" for v in values]
        elif column in _DOLLARS:
            out[column] = [f"${v:,.0f}" for v in values]
        elif column == "sharpe":
            out[column] = ["n/a" if pd.isna(v) else f"{v:.2f}" for v in values]
        else:
            out[column] = [str(v) for v in values]
    if labels:
        out.index = [labels.get(name, name) for name in out.index]
    return out


def write_outputs(experiment: Experiment) -> dict[str, Path]:
    """Write the tables, figures, and a run manifest. Returns the paths written.

    Output is deterministic: rerunning the same config on the same cached data
    rewrites byte-identical files, so a clean ``git diff`` means nothing changed.
    """
    config = experiment.config
    reports = config.reports_dir
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison": reports / f"{config.name}_comparison.csv",
        "standalone": reports / f"{config.name}_standalone.csv",
        "manifest": reports / f"{config.name}_run.json",
        "equity": figures / f"{config.name}_equity.png",
        "drawdown": figures / f"{config.name}_drawdown.png",
        "positions": figures / f"{config.name}_positions.png",
    }

    _to_csv(experiment.comparison(), paths["comparison"])
    _to_csv(experiment.standalone(), paths["standalone"])
    paths["manifest"].write_text(json.dumps(_manifest(experiment), indent=2) + "\n")

    start, end = experiment.window
    symbol = config.symbol
    window = f"{start:%b %Y} to {end:%b %Y}"
    curves = experiment.curves()
    plotting.plot_equity(
        curves,
        paths["equity"],
        labels=experiment.labels,
        title=f"{experiment.labels[experiment.strategy.strategy_name]} vs. buy and hold, {symbol}",
        subtitle=f"Portfolio value, both starting at ${config.initial_cash:,.0f} on "
        f"{start:%b} {start.day}, {start:%Y} · daily, log scale · no costs",
    )
    plotting.plot_drawdowns(
        curves,
        paths["drawdown"],
        labels=experiment.labels,
        title=f"Drawdown from previous peak, {symbol}",
        subtitle=f"Same comparison window, {window}",
    )
    plotting.plot_positions(
        experiment.strategy,
        experiment.data.frame(symbol)["close"],
        paths["positions"],
        start=start,
        label=experiment.labels[experiment.strategy.strategy_name],
        price_label=f"{symbol} close",
        title=f"When the crossover is in the market, {symbol}",
        subtitle=f"{symbol} daily close; shaded while the strategy holds a position · {window}",
    )
    return paths


def _to_csv(table: pd.DataFrame, path: Path) -> None:
    out = table.copy()
    for column in _DATES:
        if column in out:
            out[column] = ["" if pd.isna(v) else f"{pd.Timestamp(v):%Y-%m-%d}" for v in out[column]]
    out.to_csv(path, index_label="strategy", float_format="%.10g", lineterminator="\n")


def _manifest(experiment: Experiment) -> dict[str, Any]:
    start, end = experiment.window
    data = experiment.data
    return {
        "experiment": experiment.config.name,
        "config": experiment.config.as_dict(),
        "data": {
            "symbol": experiment.config.symbol,
            "first_bar": f"{data.start:%Y-%m-%d}",
            "last_bar": f"{data.end:%Y-%m-%d}",
            "bars": len(data),
            "cache": dict(experiment.data_provenance) if experiment.data_provenance else None,
        },
        "comparison_window": {"start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"},
        "strategies": {
            r.strategy_name: {
                "first_decision": r.config["first_decision"],
                "fills": len(r.fills),
                "execution": r.config["execution"],
                "timing": r.config["timing"],
                "share_sizing": r.config["share_sizing"],
            }
            for r in experiment.results
        },
        "versions": {
            "python": platform.python_version(),
            "backtester": backtester.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
