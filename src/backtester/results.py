"""The output of one backtest run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

FILL_COLUMNS: tuple[str, ...] = ("date", "symbol", "qty", "price", "cost")


@dataclass(frozen=True)
class BacktestResult:
    """Everything a run produced, indexed by the run's trading dates.

    Attributes:
        strategy_name: ``strategy.name`` at run time.
        equity: portfolio value at each close (cash + shares x close).
        returns: simple return of ``equity`` from the previous close. The first
            date's return is measured against the starting cash.
        cash: cash at each close.
        positions: shares held at each close (date x symbol; 0.0 when flat).
        decisions: target weights decided at each close from the first decision
            onward (date x symbol; 0.0 for symbols the strategy left out). A
            decision on date t is filled at the open of the next date, so the
            final decision of a run is never executed.
        fills: one row per fill, with columns date, symbol, qty (shares, signed),
            price, and cost.
        config: the run's settings (read-only mapping).

    The frames are ordinary pandas objects. Treat them as read-only.
    """

    strategy_name: str
    equity: pd.Series
    returns: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    decisions: pd.DataFrame
    fills: pd.DataFrame
    config: Mapping[str, Any]

    def __repr__(self) -> str:
        if self.equity.empty:  # pragma: no cover - the engine never produces an empty run
            return f"BacktestResult(strategy={self.strategy_name!r}, empty)"
        return (
            f"BacktestResult(strategy={self.strategy_name!r}, "
            f"{self.equity.index[0]:%Y-%m-%d} to {self.equity.index[-1]:%Y-%m-%d}, "
            f"{len(self.equity)} bars, {len(self.fills)} fills, "
            f"final equity {self.equity.iloc[-1]:,.2f})"
        )
