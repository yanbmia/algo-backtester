"""Performance metrics: pure functions of equity, returns, and positions.

Nothing here imports the engine. Every function takes plain pandas objects, so
the same code scores a single-symbol backtest, a multi-asset portfolio, or a
series built by hand in a test.

Conventions (stated once, used everywhere)
------------------------------------------
- **Returns** are simple period returns, ``r_t = E_t / E_{t-1} - 1``.
- **Time** is counted in periods (trading days), not calendar days. A series
  of ``n`` equity values spans ``n - 1`` periods, which is
  ``(n - 1) / periods_per_year`` years. 252 trading days per year by default.
- **CAGR** compounds geometrically: ``(E_end / E_start) ** (1 / years) - 1``.
- **Volatility** is the sample standard deviation (``ddof=1``) of period
  returns, annualized by ``sqrt(periods_per_year)``.
- **Sharpe** is the mean per-period *excess* return divided by its sample
  standard deviation, annualized by ``sqrt(periods_per_year)``. The annual
  risk-free rate is converted to a per-period rate geometrically:
  ``(1 + rf_annual) ** (1 / periods_per_year) - 1``. The square-root scaling
  assumes returns are roughly independent from one period to the next, which
  is the usual convention and an approximation.
- **Drawdown** is measured from the running peak and reported as a negative
  fraction: -0.25 means 25% below the highest value reached so far.

Invalid input raises ``ValueError``. The one metric that can be undefined on
valid input is Sharpe with zero volatility (e.g. a strategy that never
trades), which returns ``NaN`` rather than a misleading number.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtester.results import BacktestResult

TRADING_DAYS_PER_YEAR = 252


def total_return(equity: pd.Series) -> float:
    """Growth from the first to the last value: ``E_end / E_start - 1``."""
    values = _equity_values(equity, minimum=1)
    return float(values[-1] / values[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate over ``(len(equity) - 1) / periods_per_year`` years."""
    values = _equity_values(equity, minimum=2)
    years = (len(values) - 1) / _positive(periods_per_year, "periods_per_year")
    growth = values[-1] / values[0]
    if growth < 0:
        raise ValueError(f"CAGR is undefined for negative final equity ({values[-1]})")
    return float(growth ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    """Sample standard deviation (ddof=1) of period returns, times ``sqrt(periods_per_year)``."""
    values = _return_values(returns)
    return float(
        np.std(values, ddof=1) * math.sqrt(_positive(periods_per_year, "periods_per_year"))
    )


def sharpe(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of period returns against an annual risk-free rate.

    ``mean(r - rf_period) / std(r - rf_period, ddof=1) * sqrt(periods_per_year)``,
    where ``rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1``. Returns
    ``NaN`` when the excess returns have zero volatility (Sharpe is undefined).
    """
    values = _return_values(returns)
    periods = _positive(periods_per_year, "periods_per_year")
    if not math.isfinite(rf_annual) or rf_annual <= -1:
        raise ValueError(f"rf_annual must be finite and > -1, got {rf_annual}")
    rf_period = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = values - rf_period
    std = np.std(excess, ddof=1)
    if std == 0:
        return math.nan
    return float(np.mean(excess) / std * math.sqrt(periods))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional distance below the running peak at each date (0 at a new high, else < 0)."""
    _equity_values(equity, minimum=1)
    drawdown = equity / equity.cummax() - 1.0
    return drawdown.rename("drawdown")


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """The deepest drawdown as ``(depth, peak_date, trough_date)``.

    ``depth`` is negative (e.g. -0.25). ``peak_date`` is the *last* date the
    running peak was set before the trough, which is where the decline started
    (if equity sat flat at its high, the drawdown began when it left that high).
    If equity never falls below a previous high, returns ``(0.0, NaT, NaT)``:
    there is no drawdown to date.
    """
    drawdown = drawdown_series(equity)
    depth = float(drawdown.min())
    if depth == 0.0:
        return 0.0, pd.NaT, pd.NaT
    trough = drawdown.idxmin()
    before = equity.loc[:trough]
    peak = before[before == before.max()].index[-1]
    return depth, peak, trough


def exposure(positions: pd.DataFrame) -> float:
    """Fraction of dates with any non-zero position."""
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        raise ValueError("positions must be a non-empty DataFrame (date x symbol)")
    return float((positions != 0).any(axis=1).mean())


def summarize(
    result: BacktestResult,
    rf_annual: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """One row of headline metrics for a backtest, named after the strategy.

    Everything is measured over the **evaluation window**: from the close of
    the first decision date to the end of the run. Warmup days, when the
    strategy could not yet act, are excluded, so they don't drag down returns
    or Sharpe. The window's start and end are part of the row, so strategies
    evaluated over different windows are easy to spot. Rows from several
    results combine with ``pd.DataFrame([summarize(a), summarize(b)])``.
    """
    start = result.decisions.index[0]
    equity = result.equity.loc[start:]
    returns = equity.pct_change().iloc[1:]
    depth, peak, trough = max_drawdown(equity)
    enough = len(returns) >= 2
    return pd.Series(
        {
            "start": equity.index[0],
            "end": equity.index[-1],
            "periods": len(returns),
            "total_return": total_return(equity),
            "cagr": cagr(equity, periods_per_year) if len(equity) >= 2 else math.nan,
            "annualized_vol": annualized_vol(returns, periods_per_year) if enough else math.nan,
            "sharpe": sharpe(returns, rf_annual, periods_per_year) if enough else math.nan,
            "max_drawdown": depth,
            "max_drawdown_peak": peak,
            "max_drawdown_trough": trough,
            "exposure": exposure(result.positions.loc[start:]),
            "n_fills": len(result.fills),
            "final_equity": float(equity.iloc[-1]),
        },
        name=result.strategy_name,
    )


# --- input checks ------------------------------------------------------------


def _equity_values(equity: pd.Series, minimum: int) -> np.ndarray:
    if not isinstance(equity, pd.Series):
        raise TypeError(f"equity must be a pandas Series, got {type(equity).__name__}")
    values = equity.to_numpy(dtype="float64")
    if len(values) < minimum:
        raise ValueError(f"equity needs at least {minimum} value(s), got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError("equity contains NaN or infinite values")
    if values[0] <= 0:
        raise ValueError(f"equity must start positive, got {values[0]}")
    return values


def _return_values(returns: pd.Series) -> np.ndarray:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pandas Series, got {type(returns).__name__}")
    values = returns.to_numpy(dtype="float64")
    if len(values) < 2:
        raise ValueError(f"need at least 2 returns for a standard deviation, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError("returns contain NaN or infinite values")
    return values


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return float(value)
