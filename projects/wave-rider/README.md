# Wave Rider

Adaptive cross-asset trend strategy research project.

The goal is not to replicate an all-weather portfolio. The goal is to build a
robust system that can survive rocky markets, reduce drawdowns, and outperform
simple passive benchmarks on a risk-adjusted basis.

## Current Status

`wave_rider` now has a proper research foundation:
- a correct mark-to-market backtest engine
- explicit cash handling
- turnover and transaction cost tracking
- signal logging
- benchmark comparison support
- unit tests for core portfolio mechanics

The current strategy logic now has two research profiles:
- broader cross-asset research universe
- longer trend windows
- relative strength ranking
- inverse-volatility sizing
- moderate bucket diversification cap
- a 2% no-trade band to reduce small churn
- `IEF` as the default parking sleeve for unused capital
- an aggressive profile with no defense overlay
- a defensive profile with breadth-based de-risking
- cash as a valid output when we disable the parking sleeve

That means the project is now ready to run proper backtests, but it is still
not a validated strategy yet.

## V2 Direction

Wave Rider v2 will be closer to a simplified cross-asset trend allocator than a
static balanced portfolio.

Core ideas:
- absolute trend filter
- relative strength ranking
- volatility-aware sizing
- cash as a valid allocation
- diversification across distinct macro buckets
- defensive de-risking when market breadth and trend weaken
- preset-based universe selection for research vs later implementation

Wave Rider is explicitly not trying to be an all-weather portfolio.
It is allowed to rotate, concentrate, and hold cash when trends are weak.

## Working Docs

- `RESEARCH_GOAL.md`: what success looks like and what we are not trying to do
- `STRATEGY_SPEC.md`: proposed v2 design, universe, signals, and validation plan
- `TODO.md`: implementation and research checklist
- `DECISIONS.md`: major design choices and why we made them

## Immediate Priorities

1. Run the first full baseline backtest on real data.
2. Run the validation sweep and inspect robustness versus nearby variants.
3. Use those results to refine the universe and simplify weak overlays.

## Universe Presets

The default preset is `us_research_core` because it is the safest research
universe for Yahoo Finance history. A `uk_ucits_candidate` preset is also in
the config for later portability work.

Switch presets in `config.py` by changing:

```python
UNIVERSE_PRESET = "us_research_core"
```

## Running The Backtest

From this folder:

```bash
conda activate allweather
python main.py
```

This is the recommended path today because `wave_rider` has already been
verified inside the existing `allweather` environment.

If you later want a dedicated environment, you can create one with:

```bash
conda env create -f environment.yml
conda activate wave_rider
python main.py
```

There is no strong need to split the environments yet because the dependency
stack is still effectively the same.

This will:
- download or reuse cached price history
- run the aggressive and defensive Wave Rider profiles
- compare it with the configured benchmarks
- save the latest outputs under `results/`
- archive an immutable snapshot under `results/runs/`

Expected output files:
- `results/backtest_history.csv`
- `results/signal_log.csv`
- `results/signal_log_aggressive.csv`
- `results/signal_log_defensive.csv`
- `results/stats.csv`
- `results/stats_table.txt`
- `results/backtest_overview.png`
- `results/strategy_state_aggressive.png`
- `results/strategy_state_defensive.png`

Treat any output as research evidence, not proof that the final strategy has
merit.

## Running Validation

Once the baseline run completes, run the first validation sweep:

```bash
python validation.py
```

This will save:
- per-variant histories and signal logs under `results/validation/`
- `validation_summary.csv`
- `benchmark_summary.csv`
- `walkforward_summary.csv`
- `validation_comparison.png`
- `benchmark_comparison.png`
- `walkforward_summary.png`

It will also archive the summary outputs under `results/runs/`.

## Result Storage

Running `main.py` or `validation.py` will overwrite the latest files in
`results/` and `results/validation/`.

That is intentional.

The latest files are for convenience. Every run is also archived under
`results/runs/<timestamp>_*` with run metadata so we keep a durable research
trail instead of losing old evidence.

This is closer to how professionals organize research:
- one convenient "latest" view
- immutable archived runs
- metadata describing the parameters and environment
- comparisons based on saved artifacts, not memory

## When To Start Backtesting

You should start backtesting now.

The codebase is ready for the first real research cycle because it now has:
- a runnable baseline strategy
- benchmark comparisons
- saved outputs
- a sensitivity validation runner
- tests around the core mechanics

The right sequence is:
1. Run `python main.py`
2. Read `results/stats.csv` and `results/stats_table.txt`
3. Run `python validation.py`
4. Compare the baseline with nearby variants before changing the strategy

## What To Inspect First

On the first real run, focus on:
- whether Wave Rider beats `Global_Equity` and `60_40_Proxy` on Calmar and drawdown
- whether CAGR is acceptable after costs
- how much average cash it holds
- whether the validation variants tell a consistent story or a fragile one
- whether the walk-forward windows tell a stable story across different regimes

Current baseline intent:
- `Wave Rider Aggressive`: weekly rebalance, longer trend windows, moderate bucket cap, 30% max position cap, 2% no-trade band, `IEF` parking sleeve
- `Wave Rider Defensive`: same core logic with breadth-based de-risking on top of the new baseline
- neither profile is trying to maintain all-weather-style balance across asset classes

The most useful visuals are:
- `backtest_overview.png` for equity curves and drawdowns
- `strategy_state_aggressive.png` and `strategy_state_defensive.png` for profile state
- `validation_comparison.png` for variant robustness at a glance
