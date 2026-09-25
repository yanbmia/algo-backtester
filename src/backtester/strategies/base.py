"""The Strategy interface.

A strategy turns what it can see (a point-in-time ``MarketView``) into target
portfolio weights. It never receives ``MarketData``, never places orders, and
never sees the portfolio. Deciding *what* to hold is the strategy's job;
deciding *how* to get there (sizing, fills, costs) belongs to the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backtester.data.view import MarketView


class Strategy(ABC):
    """Base class for every strategy.

    Attributes:
        name: identifier used in results. Defaults to the subclass's name; set it
            per instance to include parameters, e.g. ``"sma_crossover(50, 200)"``.
        warmup: bars of history needed before the first decision. The engine
            will not call ``target_weights`` until at least this many bars are
            visible, so ``view.history(..., lookback=warmup)`` is always safe.
    """

    name: str = "Strategy"
    warmup: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

    def fit(self, view: MarketView) -> None:  # noqa: B027 - intentionally a no-op hook
        """Optional training hook. The default does nothing, which suits rule-based strategies.

        Model-based strategies override this to train on ``view``. Because
        training data arrives as a ``MarketView`` rather than raw data, fitting
        inherits the same lookahead guard as trading decisions.
        """

    @abstractmethod
    def target_weights(self, view: MarketView) -> dict[str, float]:
        """Desired portfolio weights, decided after the close of ``view.now``.

        The engine executes them no earlier than the next bar's open. Keys are
        symbols from ``view.symbols``. Any visible symbol left out has a target
        weight of 0 (flat). Values are fractions of portfolio equity, so 1.0
        means fully invested in that symbol.
        """
