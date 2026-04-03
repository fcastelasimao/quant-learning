# Wave Rider v2 Strategy Spec

## Summary

Wave Rider v2 should be a cross-asset trend-following allocator with a cash
fallback, not a static diversified basket.

The system should be allowed to concentrate in the strongest distinct trends,
de-risk when breadth deteriorates, and do very little when the opportunity set
is poor.

## Design Principles

- prefer robustness over cleverness
- allow cash as a first-class position
- diversify across macro buckets, not across ticker count
- add one risk layer at a time and validate each one
- avoid hidden leverage in v2

## Implemented Baseline

The current codebase now implements the first baseline version of this spec:
- preset-based research universe selection
- absolute trend filter using longer medium-term total return
- blended relative strength ranking
- inverse-volatility sizing with annualised units
- grouped bucket diversification cap
- explicit residual cash

Current promoted baseline:
- 168-day absolute trend filter
- blended 84/168/252-day ranking
- 84-day annualised volatility
- weekly rebalance
- moderate bucket cap
- 30% max position cap
- 2% no-trade band
- `IEF` parking sleeve for residual capital
- no defense overlay by default

Interpretation:
- this is a trend allocator first
- it is not trying to maintain a permanent balanced mix of equities, bonds,
  and real assets

## Proposed Universe

We should expand from 5 ETFs to a smaller set of distinct macro exposures.
Rough target: 10 to 14 liquid ETFs.

Suggested buckets:
- global equities
- US large cap
- US growth or tech
- developed ex-US equities
- emerging markets
- long-duration bonds
- intermediate bonds
- inflation-linked bonds
- gold
- broad commodities
- energy
- short-duration Treasuries or cash proxy

Example candidates to research:
- `VWRL.L` or equivalent global equity proxy
- `SPY`
- `QQQ`
- `VEA` or a UCITS equivalent
- `VWO`
- `TLT`
- `IEF`
- `TIP`
- `GLD`
- `DBC` or `GSG`
- `XLE`
- `SHY` or a UCITS short-duration substitute

Final selection should prefer:
- high liquidity
- low fees
- distinct economic roles
- enough history for testing

Current implementation:
- default preset: `us_research_core`
- alternate preset: `uk_ucits_candidate`
- the UK preset is a candidate mapping, not yet fully validated

## Benchmarks

Wave Rider should be judged against simple alternatives:
- buy and hold global equity
- 60/40 proxy
- equal-weight diversified basket
- simple cross-asset trend baseline without overlays
- current v1 prototype

## Signal Stack

### 1. Absolute Trend Filter

Only allow risk assets that pass a trend threshold.

Candidate definitions:
- price above 200-day moving average
- 6 to 12 month total return greater than 0
- blended medium-term trend score

Version 1 of v2 should pick one simple definition and stick to it.

Current baseline:
- uses a 168-day total return filter

### 2. Relative Strength

Among assets that pass the absolute trend filter, rank them by cross-sectional
strength.

Candidate ranking inputs:
- 3 month return
- 6 month return
- blended 3/6/12 month return
- Sharpe-like momentum score

Current baseline:
- uses blended 84/168/252-day total return

### 3. Volatility-Aware Sizing

Size positions so unstable assets receive less capital.

Requirements:
- annualised volatility units
- position caps
- explicit residual cash
- no fake precision from overly short lookbacks

Current baseline:
- uses 84-day annualised volatility

### 4. Correlation Or Similarity Penalty

Avoid ending up with three versions of the same trade.

Candidate methods:
- bucket caps
- pairwise correlation penalty
- rank reduction for highly similar assets

For v2, bucket caps may be the simplest reliable option.

Current baseline:
- uses grouped buckets with a two-per-bucket cap

Current promoted baseline:
- uses a moderate grouped cap to allow clustering without going fully loose

### 5. Defense Mode

Reduce gross exposure when market internals worsen.

Candidate triggers:
- weak equity trend breadth
- rising cross-asset volatility
- too few assets passing the trend filter
- poor dispersion of trend leadership

Defense mode should reduce exposure, not add complex market timing.

Current baseline:
- disables defense mode by default because the first validation pass did not
  show enough evidence that breadth scaling was helping
- the current parking-sleeve version also leaves the defensive profile behind
  the aggressive profile on both CAGR and max drawdown

## Allocation Logic

Initial v2 allocation proposal:
- rank valid assets
- select top `N` distinct candidates
- weight by inverse volatility
- cap individual weights
- map unused capital to a parking sleeve or cash
- optionally scale total exposure down in defense mode

Important:
- gross exposure does not need to equal 100%
- cash is a valid output
- a low-risk parking sleeve can improve CAGR if it does not damage drawdowns
- if no strong trends exist, the portfolio should mostly stand aside

## Rebalancing

Suggested starting point:
- evaluate signals daily
- rebalance weekly

This keeps the system responsive without turning it into a churn-heavy
short-horizon strategy.

## Backtest Requirements

The backtest engine must support:
- mark-to-market position updates
- explicit cash balance
- turnover and transaction costs
- portfolio history
- signal logging
- benchmark comparison
- deterministic testing

Current status:
- implemented in the repo as the new engine foundation
- validated with unit tests for compounding, cash handling, and costs
- ready for the v2 strategy logic to be layered on top

## Metrics

Primary metrics:
- CAGR
- max drawdown
- Sharpe
- Sortino
- Calmar
- ulcer index

Operational metrics:
- turnover
- average cash allocation
- average number of holdings
- worst monthly loss
- percentage of time invested

## Validation Process

Validation should happen in this order:

1. Build a simple baseline v2 and compare it with benchmarks.
2. Add volatility scaling and test whether it improves risk-adjusted results.
3. Add defense mode and test whether it improves drawdown control.
4. Add a diversification penalty and test whether it adds value.
5. Run sensitivity analysis across nearby parameters.
6. Run walk-forward or held-out validation.

No new layer should stay unless it improves robustness, not just one backtest.

We have completed step 1 in code structure, but not yet in empirical evidence.

Current validation workflow in the repo:
- `main.py` runs the baseline strategy versus benchmarks
- `validation.py` runs a first sensitivity sweep around the baseline
- `validation.py` also saves walk-forward window summaries for the promoted
  baseline

Current sweep emphasis:
- parking sleeve choice
- no-trade band size
- transaction-cost sensitivity
- whether defense mode adds enough value to deserve complexity

Current sweep includes:
- promoted aggressive baseline
- defensive profile
- legacy cash fallback comparison
- `SHY`, `IEF`, and `TIP` parking alternatives
- no-trade band comparisons
- three holdings
- longer trend windows
- a simpler trend baseline with weaker diversification constraints

## Things To Avoid In v2

- machine learning
- hidden leverage
- shorting as a core feature
- options overlays
- too many knobs
- too many overlapping ETFs
