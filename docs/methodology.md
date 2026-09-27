# Methodology: preventing lookahead bias

This document explains how the backtester keeps future information out of past decisions, and how the test suite checks that it does. Every mechanism and test is named by file so it can be verified directly. The short version is in the README's [Rigor](../README.md#rigor-preventing-lookahead-bias) section.

## 1. Why lookahead bias matters

A backtest replays history and asks what a strategy would have done. The answer only means something if each simulated decision uses information that existed when the decision was made. Lookahead bias is any leak of later information into an earlier decision, and I designed against it instead of relying on being careful, for three reasons:

- **It flatters.** Knowing tomorrow is the best trading signal there is, so a leak almost always improves results. The error goes in the direction you were hoping for, so it's easy to accept.
- **It is silent.** Nothing crashes. A leaky backtest produces a plausible equity curve, just a better one than the strategy deserves.
- **It is small in code.** The usual causes are one-token mistakes: `<=` where `<` was needed, a forgotten `.shift(1)`, deciding and filling on the same close, normalizing with full-sample statistics, or letting a strategy know which tickers will exist later.

On daily bars, even a one-bar leak is severe: a strategy that can see tomorrow's close can be long on up days and flat on down days. So v1 makes the bug hard to write in the first place, then tests for it from the outside anyway.

## 2. The timing convention

Everything in v1 follows one convention, documented in `src/backtester/engine/backtester.py` and recorded as `timing` in `reports/sma_spy_run.json`. At each trading date *t*, `Backtester.run` does three things in this order:

1. **Execute.** If a decision is pending from the previous close, size orders at *t*'s open and fill them there.
2. **Mark.** Value the portfolio at *t*'s close and record equity, cash, and positions.
3. **Decide.** Once the strategy's warmup is met, call `strategy.target_weights(data.view(t))`, validate the weights, and hold them as pending for *t*+1.

Bar *t* is visible in step 3 because the decision is made after *t*'s close. Nothing it decides can take effect before *t*+1's open. The last decision of a run is recorded but never executed (`test_the_final_decision_is_recorded_but_never_executed` in `tests/test_backtester.py`).

## 3. The structural guard, part 1: data truncation

`MarketData` (`src/backtester/data/market_data.py`) holds the full, validated dataset, and only the engine holds it. A strategy only ever receives a `MarketView` (`src/backtester/data/view.py`), created by `MarketData.view(t)`. `test_the_strategy_only_ever_receives_a_view` in `tests/test_backtester.py` checks this for every decision call in a run.

- **One boundary.** `MarketData._visible_rows(symbol, cutoff)` returns `searchsorted(dates, cutoff, side="right")`, which is the number of bars dated at or before the cutoff. This is the only place that count is computed. `side="left"` would hide *t*'s own bar (an off-by-one in the safe direction that throws away the latest close); anything that admits a bar dated after *t* is lookahead. `TestTruncation` in `tests/test_view.py` pins the inclusive cutoff (`test_cutoff_is_inclusive`, `test_first_date_shows_exactly_one_bar`).
- **A small surface with no date arguments.** A view exposes exactly `now`, `symbols`, `history(symbol, field, lookback)`, and `panel(fields, lookback)` (`test_public_api_is_exactly_now_symbols_history_panel`). None of them accepts a date, so the public API has no way to ask for the future. `lookback` can only reach backward, and asking for more bars than are visible raises `ValueError`.
- **Only `MarketData` can build a view.** The constructor requires a private key (`test_only_marketdata_can_create_a_view`), so a strategy cannot construct its own view with a later cutoff.
- **Read-only copies that share no memory.** `history()` and `panel()` return fresh copies backed by read-only arrays, so element writes raise `ValueError`, and nothing returned shares memory with the parent (`TestReadOnly` in `tests/test_lookahead.py`). Some pandas operations rebind an object to new arrays instead of writing into it (pandas 3's `s += 1`, and `df["col"] = ...` in any version); those cannot be blocked on a plain pandas object, so `test_rebinding_the_returned_object_cannot_reach_the_view_or_parent` proves they cannot change the view, later views, or the parent.
- **Future listings are invisible.** A symbol whose first bar comes after *t* is not in `view.symbols`, and asking for it raises the same `KeyError` as a ticker that doesn't exist, so even the error message cannot reveal that it will list (`test_unlisted_and_nonexistent_symbols_are_indistinguishable` in `tests/test_view.py`, `test_a_symbol_that_lists_after_the_cutoff_is_invisible` in `tests/test_lookahead.py`).
- **Training goes through the same door.** `Strategy.fit(view)` (`src/backtester/strategies/base.py`) is called once, at the first decision date, with a view. A future model-based strategy therefore trains on data truncated by the same mechanism.

## 4. The structural guard, part 2: next-bar execution

Truncation alone does not stop the most common daily-bar leak: computing a signal from *t*'s close and filling at *t*'s close. The data access is technically fine (*t*'s close is visible at *t*), but the trade is impossible, because you cannot observe a closing price and still trade at it. v1 closes that gap in the engine, independently of any strategy:

- `Backtester.run` holds each decision as pending and executes it at the start of the next bar.
- `NextOpenExecution` (`src/backtester/engine/execution.py`) sizes orders at the same open it fills them at, like a market-on-open order.
- The execution model has no default. `Backtester(..., execution=...)` must be given one (`test_execution_must_be_passed_explicitly`), so the fill and cost assumption is stated at every call site.

The tests are in `TestTiming` in `tests/test_backtester.py`:

- `test_a_decision_at_t_fills_at_the_next_open_never_the_close` uses `make_gapped_ohlcv` bars where open = 1000 + *i* and close = 500 + *i*, so a fill at the wrong price is off by hundreds rather than cents. A buy decided on row 5 must fill at row 6's open (1006), not row 5's close (505) or open (1005), and equity through row 5 must be untouched.
- `test_every_fill_happens_at_its_own_dates_open_one_bar_after_a_decision` checks every fill of an active multi-symbol strategy.
- `test_weights_right_after_each_fill_match_the_previous_decision` checks that the portfolio at each open matches the decision from the close before.

The two parts protect different things. Truncation controls what a strategy can see; next-bar execution controls when its decision can take effect. Either one alone leaves a hole.

## 5. What the structure cannot guarantee

Python has no enforced privacy, so this design makes leakage hard to do by accident but cannot make it impossible on purpose. A concrete example: the arrays inside a `MarketView` are NumPy slices of the parent's arrays. The view holds no reference to the `MarketData` object itself (`test_view_holds_no_reference_to_its_parent` in `tests/test_view.py`), but a slice's `.base` attribute leads straight back to the full, untruncated data. A strategy that reached into `view._windows` and followed `.base` would see the entire future, and nothing in the language would stop it. Walking the call stack up to the engine's frame, which holds the full dataset, would work too.

So I don't rely on encapsulation. The check that matters is a test that doesn't care how a strategy reads data, only whether its outputs depend on data it shouldn't have.

## 6. Testing from the outside: future perturbation

The property being tested: if nothing the engine produces through date *T* depends on data after *T*, then replacing everything after *T* must leave everything through *T* unchanged.

`assert_no_lookahead` in `tests/lookahead_harness.py` checks it in four steps:

1. Run the real engine (`NextOpenExecution(ZeroCost())`, $100,000) with a fresh strategy on the data *D*.
2. Build *D′* = `replace_after(D, T, seed)` (`tests/synthetic.py`). Every bar after *T* becomes a seeded random walk with valid OHLC, starting 50% above the last pre-cutoff close with triple the volatility. Dates, and every bar through *T*, are kept. `test_perturbation_rewrites_only_the_future` confirms that every post-cutoff value differs and nothing before the cutoff changes.
3. Run the engine again on *D′* with another fresh strategy.
4. Require decisions, equity, cash, positions, and fills dated at or before *T* to be bit-for-bit identical (`check_exact=True`). For `ProbeStrategy`, everything it observed through *T* must match as well.

**Why this catches leakage regardless of mechanism.** The test treats the strategy and engine as a black box. Whether a leak comes from a private attribute, `.base`, a module-level cache, or frame inspection, if any path lets data after *T* influence anything at or before *T*, the two runs differ and the test fails. It never has to anticipate how the leak works.

**Why probes.** A strategy with discrete output can pass by luck: an SMA crossover might not flip at *T* even if it could see *T*+1. The harness therefore includes strategies built to be sensitive:

- `ProbeStrategy` sets each weight to (latest close mod 1) / *n*, a continuous function, so any change to the latest visible close changes its output. At every step it also records `view.now`, the last date `history()` returned for each symbol, and a SHA-256 fingerprint of the entire visible panel, so a leak into *any* visible value is caught, not just the latest close.
- `PanelProbe` does the same through `panel(lookback=3)`, so a leak confined to `panel()` is caught.
- `MomentumStrategy` is a realistic discrete strategy, and the real `SMACrossover` (with 5/20 windows so it trades inside the synthetic runs) and `BuyAndHold` are included too. `test_the_crossover_really_trades_in_these_scenarios` guards against the crossover passing vacuously by never trading.

**Coverage.** `TestFuturePerturbation::test_nothing_at_or_before_the_cutoff_depends_on_the_future` in `tests/test_lookahead.py` runs those 5 strategies × 3 cutoffs (30%, 50%, and 80% of the way through the run) × 2 datasets: three symbols on a shared calendar, and four symbols with staggered calendars (one lists mid-run, one lists after the run ends, one stops trading), for 30 cases.

**Sensitivity.** Invariance only means something if the setup could have detected a change. `test_perturbation_shows_up_on_the_very_next_bar` checks that the probe's decisions, positions, and equity first diverge exactly at *T*+1.

**Why decisions and not just equity.** A one-bar leak changes the decision made at *T*, but that decision trades at *T*+1's open, and *T*+1 is perturbed anyway. The leaky run's equity therefore first diverges at *T*+1, exactly where an honest run's does. `test_equity_alone_cannot_see_a_one_bar_leak` demonstrates this, which is why `assert_no_lookahead` compares decisions first.

**Limits of the test.**

- It detects dependence on the data the engine was given. A strategy that read the future from somewhere else, such as the Parquet cache on disk or the network, would see the same future in both runs and pass. v1's strategies are short and read nothing but their view, which can be confirmed by reading `src/backtester/strategies/`.
- It only checks the strategies it runs. A new strategy gets checked by adding one line to `STRATEGIES` in `tests/test_lookahead.py`.

## 7. The negative control

The harness is code too, and it can be wrong: it could compare a frame with itself, slice the wrong dates, or perturb nothing, and every lookahead test would still pass. So the suite includes a deliberately broken dataset and requires the checks to catch it.

`LeakyMarketData` in `tests/lookahead_harness.py` subclasses `MarketData` and overrides only `_visible_rows`, returning one extra row: the classic off-by-one. Everything else is the real implementation. Its views report the correct `now` and the correct symbols, and only their contents are wrong (`test_the_leak_is_subtle`). It's the smallest realistic leak, and the hardest to spot by looking at results. `replace_after` rebuilds data with `type(data).from_frames`, so the perturbed data keeps the leak.

The control tests, in `tests/test_lookahead.py`:

- `TestNegativeControl::test_perturbation_check_fails_against_a_leaky_view`: `assert_no_lookahead` must raise an `AssertionError` about decisions, for both probes at all three cutoffs.
- `TestNegativeControl::test_leak_changes_the_decision_at_the_cutoff_itself`: with the leak, the first changed decision moves from *T*+1 to *T*. So the test finds where the leak is, not just that something changed.
- `TestBoundary::test_boundary_check_fails_against_a_leaky_view`: the boundary check in the next section fails on leaky data as well.

## 8. Other checks

- **Boundary at every step.** `TestBoundary::test_probe_sees_exactly_up_to_now_at_every_step` confirms that at every date of a run, the last date the probe saw for each symbol equals that symbol's latest bar at or before `now`, recomputed independently from `data.frame()`. `test_latest_visible_bar_is_the_bar_the_engine_sees_on_that_date` checks that a view's last bar matches `data.bar(t)` on all five fields.
- **Live data (opt-in).** `test_live_spy_backtest_ignores_the_future` runs the same perturbation check, including the leaky control, on SPY daily bars for 2020 downloaded from Yahoo. It is marked `network`, deselected by default, and not run in CI; run it with `pytest -m network`.

## 9. Other ways a backtest can flatter itself

Lookahead is the main risk, but not the only one. v1 also guards against these:

- **Mismatched windows.** Buy-and-hold can decide on day one, while the crossover needs 200 bars of warmup. Comparing their full runs would credit buy-and-hold with the warmup months. `metrics.comparison_window` starts the comparison at the latest first-decision date, and `compare` and `rebased_equity` use that window for the comparison table and the equity and drawdown charts (`TestCompare::test_benchmark_gets_no_head_start` in `tests/test_metrics.py`). One nuance remains: the window starts at a close, when buy-and-hold is already invested and the crossover's first trade is still pending for the next open, so the benchmark gets one extra overnight move. The standalone table (`reports/sma_spy_standalone.csv`) measures each strategy from its own first decision instead.
- **Tuned parameters.** 50/200 was fixed a priori in `configs/sma_spy.yaml`, not selected on this sample.
- **Hidden costs.** The cost assumption is explicit wherever a backtest is built: the execution model is a required argument, the config accepts only `costs: zero`, and the equity chart's subtitle says "no costs".
- **Numbers that drift.** The date range is pinned, the data cache carries a SHA-256 and download timestamp, `reports/sma_spy_run.json` records the exact cache and library versions behind the committed results, and `write_outputs` in `src/backtester/experiment.py` is deterministic, so a rerun on the same cache produces byte-identical files.

## 10. What this does not cover

These tests establish that decisions depend only on the past. They say nothing about whether a strategy works out of sample, whether its results survive costs, or whether a Sharpe difference is statistically meaningful. Those are listed, with their extension points, under [Scope and limitations](../README.md#scope-and-limitations) and [Roadmap](../README.md#roadmap) in the README.
