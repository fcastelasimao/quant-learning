# Wave Rider TODO

## Phase 1: Foundations

- [x] Replace the current portfolio simulator with a correct backtest engine
- [x] Add explicit cash handling
- [x] Add transaction costs and turnover tracking
- [x] Add benchmark series support
- [x] Add unit tests for compounding, rebalancing, and drawdown logic

## Phase 2: Universe Design

- [x] Draft preset universes by macro bucket
- [ ] Confirm ticker history length and data quality
- [ ] Decide on UK/US ticker preference and portability
- [x] Add a short-duration bond or cash proxy
- [ ] Remove redundant exposures if the universe becomes too correlated

## Phase 3: Baseline Strategy

- [x] Implement an absolute trend filter
- [x] Implement relative strength ranking
- [x] Implement inverse-volatility sizing with annualised units
- [x] Add residual cash allocation
- [x] Cap weights and holding count
- [x] Promote the longer-trend version to the default baseline
- [x] Promote monthly rebalance and a looser bucket cap to the default baseline
- [x] Promote weekly rebalance and a tighter bucket cap to the default baseline
- [x] Promote weekly rebalance with a moderate bucket cap to the default baseline
- [x] Add aggressive and defensive profile comparison
- [x] Add a no-trade band to reduce small rebalances
- [x] Add parking-sleeve support for unused capital
- [x] Promote the `IEF` parking sleeve plus 2% no-trade band to the baseline
- [x] Promote a 30% max position cap after the CAGR-focused validation pass

## Phase 4: Risk Overlays

- [x] Add a simple defense mode based on breadth or participation
- [x] Add a bucket or correlation penalty
- [ ] Test weekly vs monthly rebalancing
- [x] Remove the ad hoc emergency cut from the baseline
- [x] Remove defense mode from the default baseline until it proves useful

## Phase 5: Validation

- [x] Compare against global equity buy and hold
- [x] Compare against a 60/40 proxy
- [x] Compare against an equal-weight diversified basket
- [x] Compare against a simple trend baseline with no overlays
- [x] Add a first sensitivity sweep on lookback windows and caps
- [x] Add walk-forward validation reporting
- [x] Review walk-forward results after the next run

## Decision Rules

- Keep a feature only if it improves robustness, not just historical return.
- Prefer simpler rules when two variants perform similarly.
- Treat lower drawdown with acceptable return as a win.

## Current Next Step

- [x] Run the first full backtest on cached price data and inspect the outputs
- [x] Run the validation sweep and review which variants are robust
- [x] Inspect whether the promoted baseline improves upside capture without losing too much downside control
- [x] Decide whether the strategy benefits from clustered trend exposure more than strict diversification
- [x] Review rebalance-speed and cost sensitivity around the new weekly baseline
- [x] Review whether moderate clustering is the sweet spot between tight and loose diversification
- [x] Review whether the defensive profile is strong enough to keep as a first-class option
- [x] Add core research plots for performance, drawdown, and validation comparisons
- [x] Pin dependency versions and add an environment definition
- [ ] Decide whether to retire or redesign the defensive profile now that it lags the aggressive profile
- [ ] Test a broader opportunity set if we want to lift CAGR further without relying only on parking sleeves
- [ ] Add no-trade-band sensitivity around the new 2% default
