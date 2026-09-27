"""Run one experiment from a config file and write its tables and figures.

    python scripts/run_backtest.py configs/sma_spy.yaml
    python scripts/run_backtest.py configs/sma_spy.yaml --refresh   # re-download the data

Runs the SMA crossover and its buy-and-hold benchmark on the same data through
the same engine, then writes:

- reports/<name>_comparison.csv   both strategies over their common window
- reports/<name>_standalone.csv   each strategy over its own window
- reports/<name>_run.json         settings, data provenance, library versions
- reports/figures/<name>_{equity,drawdown,positions}.png

Exit codes: 0 on success, 2 for a config problem, 1 for a data or backtest
problem. Each failure prints a one-line reason, not a stack trace. The logic
lives in ``backtester.experiment``; this file only handles the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtester.experiment import (
    BacktestError,
    ConfigError,
    DataError,
    format_table,
    load_config,
    run_experiment,
    write_outputs,
)

EXIT_OK, EXIT_RUN_ERROR, EXIT_CONFIG_ERROR = 0, 1, 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the SMA crossover against buy-and-hold for one config."
    )
    parser.add_argument("config", type=Path, help="experiment config, e.g. configs/sma_spy.yaml")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download market data even if it is cached"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(
        f"Running {config.name}: {config.symbol} {config.start:%Y-%m-%d} to {config.end:%Y-%m-%d}"
    )
    try:
        experiment = run_experiment(config, refresh=args.refresh)
    except DataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_RUN_ERROR
    except BacktestError as exc:
        print(f"backtest error: {exc}", file=sys.stderr)
        return EXIT_RUN_ERROR

    paths = write_outputs(experiment)
    start, end = experiment.window
    print(f"\nComparison window: {start:%Y-%m-%d} to {end:%Y-%m-%d}\n")
    columns = [
        "total_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "exposure",
        "n_fills",
        "ending_value",
    ]
    table = format_table(experiment.comparison(), experiment.labels)[columns]
    print(table.T.to_string())
    print("\nWrote:")
    for path in paths.values():
        print(f"  {_display(path)}")
    return EXIT_OK


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
