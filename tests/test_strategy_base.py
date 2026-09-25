"""The Strategy base class contract."""

from __future__ import annotations

import pytest

from backtester.strategies import Strategy


class Flat(Strategy):
    def target_weights(self, view):
        return {}


class Named(Strategy):
    name = "custom-name"
    warmup = 20

    def target_weights(self, view):
        return {symbol: 1.0 / len(view.symbols) for symbol in view.symbols}


def test_strategy_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Strategy()


def test_subclass_must_implement_target_weights():
    class Incomplete(Strategy):
        pass

    with pytest.raises(TypeError, match="target_weights"):
        Incomplete()


def test_defaults():
    strategy = Flat()
    assert strategy.name == "Flat"
    assert strategy.warmup == 0


def test_name_and_warmup_can_be_set_on_the_class():
    assert (Named.name, Named.warmup) == ("custom-name", 20)


def test_name_can_be_set_per_instance():
    strategy = Flat()
    strategy.name = "flat(v2)"
    assert strategy.name == "flat(v2)"
    assert Flat().name == "Flat"


def test_subclass_of_a_named_strategy_gets_its_own_default_name():
    class Child(Named):
        pass

    assert Child.name == "Child"


def test_fit_is_a_no_op_by_default(market_data):
    assert Flat().fit(market_data.view(market_data.end)) is None


def test_target_weights_receives_a_view(market_data):
    weights = Named().target_weights(market_data.view(market_data.start))
    assert weights == {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
