"""The backtest engine: one bar-by-bar pass over the trading calendar.

At each date ``t`` in the run, in this order:

1. **Execute.** If a decision is pending from the previous close, size orders at
   ``t``'s open and fill them there.
2. **Mark.** Value the portfolio at ``t``'s close and record equity, cash, and
   positions.
3. **Decide.** Once the strategy's warmup is satisfied, pass it
   ``data.view(t)``, validate the target weights it returns, and hold them as
   pending for ``t + 1``.

A decision made from ``t``'s close can therefore only ever trade at the next
session's open. The strategy only ever receives a ``MarketView``. The engine
reads ``data.bar(t)``, and only for the current date.
"""

from __future__ import annotations

import math
import reprlib
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from backtester.data import Bar, MarketData
from backtester.engine.execution import ExecutionModel
from backtester.engine.portfolio import Portfolio, weight_problems
from backtester.results import FILL_COLUMNS, BacktestResult
from backtester.strategies import Strategy

#: After a rebalance, cash may be negative by at most this fraction of portfolio
#: value (float noise). Anything more means the account was levered.
CASH_TOLERANCE: float = 1e-8


class StrategyError(RuntimeError):
    """A strategy failed or misbehaved during a run. Names the strategy and the date."""

    def __init__(self, message: str, *, strategy_name: str, date: pd.Timestamp) -> None:
        super().__init__(message)
        self.strategy_name = strategy_name
        self.date = date


class InvalidWeightsError(StrategyError):
    """A strategy returned target weights that break v1's long/flat, unlevered rules."""

    def __init__(
        self,
        *,
        strategy_name: str,
        date: pd.Timestamp,
        weights: object,
        problems: list[str],
    ) -> None:
        self.weights = weights
        self.problems = problems
        super().__init__(
            f"{strategy_name} returned invalid target weights on {date:%Y-%m-%d}: "
            f"{'; '.join(problems)}. Output was {reprlib.repr(weights)}",
            strategy_name=strategy_name,
            date=date,
        )


class EngineError(RuntimeError):
    """An accounting invariant broke during a run (e.g. the account went negative)."""


class Backtester:
    """Runs one strategy over one dataset.

    Args:
        data: the full dataset. Only the engine holds it.
        strategy: sees nothing but ``data.view(t)``.
        execution: how orders fill, and at what cost. Required, so the cost
            assumption is always stated at the call site, e.g.
            ``NextOpenExecution(ZeroCost())``.
        initial_cash: starting cash (finite, > 0).
    """

    def __init__(
        self,
        data: MarketData,
        strategy: Strategy,
        *,
        execution: ExecutionModel,
        initial_cash: float = 100_000.0,
    ) -> None:
        if not isinstance(data, MarketData):
            raise TypeError(f"data must be MarketData, got {type(data).__name__}")
        if not isinstance(strategy, Strategy):
            raise TypeError(f"strategy must be a Strategy, got {type(strategy).__name__}")
        if not isinstance(execution, ExecutionModel):
            raise TypeError(
                "execution must implement reference_prices(bars) and execute(orders, bars), "
                f"got {type(execution).__name__}"
            )
        initial_cash = float(initial_cash)
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError(f"initial_cash must be finite and > 0, got {initial_cash}")

        self._data = data
        self._strategy = strategy
        self._execution = execution
        self._initial_cash = initial_cash
        self._warmup = _checked_warmup(strategy)

    def run(self, start: object = None, end: object = None) -> BacktestResult:
        """Run over the calendar dates in ``[start, end]`` (inclusive; default: all data).

        History before ``start`` counts toward warmup, so a strategy can decide on
        the first run date if enough earlier bars exist.

        Raises:
            ValueError: empty window, warmup that leaves no decision dates, or a
                symbol that stops trading partway through the window.
            StrategyError / InvalidWeightsError: the strategy failed or returned
                bad weights. Names the strategy and date.
            EngineError: an accounting invariant broke (e.g. negative cash).
        """
        data, strategy = self._data, self._strategy
        run_dates = self._run_dates(start, end)
        self._check_calendars(run_dates)
        first_decision = self._first_decision_date(run_dates)

        portfolio = Portfolio(self._initial_cash)
        pending: dict[str, float] | None = None
        equity, cash, positions = [], [], []
        decision_dates, decision_rows, fills = [], [], []

        for t in run_dates:
            bars = data.bar(t)

            # 1. Execute the previous close's decision at this bar's open.
            if pending is not None:
                self._execute(portfolio, pending, bars, t, fills)
                pending = None

            # 2. Mark to market at this bar's close.
            closes = {symbol: bar.close for symbol, bar in bars.items()}
            held = portfolio.positions
            equity.append(portfolio.value(closes))
            cash.append(portfolio.cash)
            positions.append([held.get(symbol, 0.0) for symbol in data.symbols])

            # 3. Decide, once warmed up. Filled at the next bar's open.
            if t >= first_decision:
                view = data.view(t)
                if t == first_decision:
                    self._call(strategy.fit, view, t)
                pending = self._decide(view, t)
                decision_dates.append(t)
                decision_rows.append([pending.get(symbol, 0.0) for symbol in data.symbols])

        return self._result(
            run_dates, equity, cash, positions, decision_dates, decision_rows, fills, first_decision
        )

    # --- steps ---------------------------------------------------------------

    def _execute(
        self,
        portfolio: Portfolio,
        targets: dict[str, float],
        bars: dict[str, Bar],
        t: pd.Timestamp,
        fills_log: list[tuple],
    ) -> None:
        prices = self._execution.reference_prices(bars)
        orders = portfolio.orders_for(targets, prices)
        for fill in self._execution.execute(orders, bars):
            portfolio.apply(fill)
            fills_log.append((t, fill.symbol, fill.quantity, fill.price, fill.cost))

        value = portfolio.value(prices)
        shorts = {s: q for s, q in portfolio.positions.items() if q < 0}
        if shorts:
            raise EngineError(f"after fills on {t:%Y-%m-%d} the account is short {shorts}")
        if portfolio.cash < -CASH_TOLERANCE * max(value, 1.0):
            raise EngineError(
                f"after fills on {t:%Y-%m-%d} cash is {portfolio.cash:.6f}: the fills cost more "
                "than the account held. v1 sizes orders before costs, so a fully invested "
                "target with a non-zero cost model overdraws the account."
            )

    def _decide(self, view: Any, t: pd.Timestamp) -> dict[str, float]:
        weights = self._call(self._strategy.target_weights, view, t)
        problems = weight_problems(weights, view.symbols)
        if problems:
            raise InvalidWeightsError(
                strategy_name=self._strategy.name, date=t, weights=weights, problems=problems
            )
        return {symbol: float(w) for symbol, w in weights.items()}

    def _call(self, method: Any, view: Any, t: pd.Timestamp) -> Any:
        """Call a strategy method, re-raising any failure with the strategy, date, and bar count."""
        name = self._strategy.name
        try:
            return method(view)
        except Exception as exc:
            bars = int(self._data.calendar.get_loc(t)) + 1
            raise StrategyError(
                f"{name}.{method.__name__} failed on {t:%Y-%m-%d} "
                f"({bars} bars visible, warmup={self._warmup}): {exc}",
                strategy_name=name,
                date=t,
            ) from exc

    # --- setup checks ----------------------------------------------------------

    def _run_dates(self, start: object, end: object) -> pd.DatetimeIndex:
        calendar = self._data.calendar
        lo = calendar[0] if start is None else _as_date(start, "start")
        hi = calendar[-1] if end is None else _as_date(end, "end")
        if lo > hi:
            raise ValueError(f"start ({lo:%Y-%m-%d}) is after end ({hi:%Y-%m-%d})")
        dates = calendar[(calendar >= lo) & (calendar <= hi)]
        if dates.empty:
            raise ValueError(
                f"no trading dates between {lo:%Y-%m-%d} and {hi:%Y-%m-%d} "
                f"(data covers {calendar[0]:%Y-%m-%d} to {calendar[-1]:%Y-%m-%d})"
            )
        return dates

    def _check_calendars(self, run_dates: pd.DatetimeIndex) -> None:
        """v1: once a symbol starts trading, it must trade on every remaining run date.

        Listings partway through a run are fine: the symbol is invisible until its
        first bar. Gaps (halts, delistings, mismatched exchange calendars) would
        leave a position with no price to trade or value at, and are rejected
        rather than papered over.
        """
        end = run_dates[-1]
        for symbol in self._data.symbols:
            dates = self._data.frame(symbol).index
            if dates[0] > end:
                continue  # first trades after the run: never visible during it
            missing = run_dates[run_dates >= dates[0]].difference(dates)
            if len(missing):
                raise ValueError(
                    f"{symbol} has no bar on {len(missing)} run date(s) after it started "
                    f"trading (first: {missing[0]:%Y-%m-%d}). The v1 engine requires every "
                    "symbol to trade on every date from its first bar to the end of the run; "
                    "delistings, halts and mismatched calendars are not supported yet. "
                    "Shorten the run with end=... or drop the symbol."
                )

    def _first_decision_date(self, run_dates: pd.DatetimeIndex) -> pd.Timestamp:
        """First run date with at least ``warmup`` bars visible (history before start counts)."""
        visible = self._data.calendar.get_indexer(run_dates) + 1
        ready = np.flatnonzero(visible >= max(self._warmup, 1))
        if not len(ready):
            raise ValueError(
                f"{self._strategy.name} needs warmup={self._warmup} bars before its first "
                f"decision, but only {visible[-1]} bars exist through {run_dates[-1]:%Y-%m-%d}; "
                "the run would never trade"
            )
        return run_dates[ready[0]]

    # --- output --------------------------------------------------------------

    def _result(
        self,
        run_dates: pd.DatetimeIndex,
        equity: list[float],
        cash: list[float],
        positions: list[list[float]],
        decision_dates: list[pd.Timestamp],
        decision_rows: list[list[float]],
        fills: list[tuple],
        first_decision: pd.Timestamp,
    ) -> BacktestResult:
        index = pd.DatetimeIndex(run_dates, name="date")
        symbols = pd.Index(list(self._data.symbols), name="symbol")

        equity_arr = np.asarray(equity, dtype="float64")
        previous = np.concatenate([[self._initial_cash], equity_arr[:-1]])
        fills_frame = pd.DataFrame(fills, columns=list(FILL_COLUMNS)).astype(
            {
                "date": "datetime64[ns]",
                "symbol": object,
                "qty": "float64",
                "price": "float64",
                "cost": "float64",
            }
        )

        config = {
            "strategy": self._strategy.name,
            "strategy_class": type(self._strategy).__name__,
            "warmup": self._warmup,
            "initial_cash": self._initial_cash,
            "execution": repr(self._execution),
            "symbols": list(self._data.symbols),
            "start": run_dates[0].strftime("%Y-%m-%d"),
            "end": run_dates[-1].strftime("%Y-%m-%d"),
            "first_decision": first_decision.strftime("%Y-%m-%d"),
            "timing": "decide at close t; size and fill at open t+1",
            "share_sizing": "fractional",
        }
        return BacktestResult(
            strategy_name=self._strategy.name,
            equity=pd.Series(equity_arr, index=index, name="equity"),
            returns=pd.Series(equity_arr / previous - 1.0, index=index, name="returns"),
            cash=pd.Series(cash, index=index, name="cash", dtype="float64"),
            positions=pd.DataFrame(positions, index=index, columns=symbols, dtype="float64"),
            decisions=pd.DataFrame(
                decision_rows,
                index=pd.DatetimeIndex(decision_dates, name="date"),
                columns=symbols,
                dtype="float64",
            ),
            fills=fills_frame,
            config=MappingProxyType(config),
        )


def _checked_warmup(strategy: Strategy) -> int:
    warmup = strategy.warmup
    if isinstance(warmup, bool) or not isinstance(warmup, int | np.integer):
        raise TypeError(f"{strategy.name}.warmup must be an int, got {warmup!r}")
    if warmup < 0:
        raise ValueError(f"{strategy.name}.warmup must be >= 0, got {warmup}")
    return int(warmup)


def _as_date(value: object, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT or ts.tz is not None:
        raise ValueError(f"{name} must be a timezone-naive date, got {value!r}")
    return ts
