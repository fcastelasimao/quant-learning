# Final Pattern Discovery Plan

## Summary

Analyze TQQQ and SQQQ separately on the full 2013-2026 trade data.

Goal: discover whether pre-trade variables can explain PnL, identify losing trades, or flag a useful subset of bad trades before entry. The output should be organized as a research pack that is easy to review later.

## Output Organization

Create a clean run folder, timestamped or versioned:

`personal_projects/projects/TQQQ_SQQQ_analysis/full_history_research/YYYYMMDD_HHMM_feature_scan/`

Inside it:

- `00_data_quality/`
  - schema audit
  - missingness report
  - leakage audit
  - feature inclusion/exclusion table

- `01_single_variable/`
  - correlation tables
  - bucket tables
  - RSI-style plots for each selected feature
  - nonlinear fit plots

- `02_interactions/`
  - RSI x ATR heatmaps
  - RSI x BBP heatmaps
  - ATR x volume heatmaps
  - regime x feature tables

- `03_regime_analysis/`
  - regime summary tables
  - feature diagnostics by regime
  - regime-conditioned plots

- `04_candidate_rules/`
  - candidate loser-filter table
  - skip/rescale impact tables
  - rule comparison plots

- `05_validation/`
  - IS/OOS validation tables
  - walk-forward checks
  - final selected-rule performance

- `06_reports/`
  - main research memo
  - updated `FINDINGS.md` excerpt
  - plot index/manifest

Also create a `manifest.csv` listing every output file, symbol, feature, plot type, and short description.

## Step 1: Data And Feature Audit

- Read the six CSV files in `full_history_canonical/trades_backtest/`.
- Build separate TQQQ and SQQQ datasets.
- Keep only pre-trade variables.
- Exclude leakage fields:
  - exit fields
  - PnL fields
  - `capital_after`
  - `capital_end`
  - `cumulative_profit`
  - post-entry values

Feature transformations:

- `atr_pct = atr / avg_order_price`
- MA distances: `avg_order_price / MA20 - 1`, etc.
- optional rolling percentiles for RSI, ATR, BBP, and volume
- keep `regime_entry` only if confirmed pre-trade/causal

## Step 2: Single-Variable Discovery

For every valid pre-trade variable, per symbol:

- Pearson correlation with `pnl_pct`
- Spearman correlation with `pnl_pct`
- correlation with `abs(pnl_pct)`
- winner vs loser distribution comparison
- loser rate by bucket
- severe-loss rate by bucket
- mean/median PnL by bucket
- total PnL contribution by bucket

Produce RSI-style plots for:

- `RSI_entry`
- `atr_pct`
- `BBP_entry`
- `volume_ratio`
- `bars_since_last_stop`
- `hour_of_entry`
- top MA-derived variables

Each plot should include:

- scatter vs `pnl_pct`
- binned mean/median PnL
- binned loser rate
- correlation values
- clipped outlier count

## Step 3: Nonlinear Fits

Yes, add polynomial/nonlinear fits, but use them as exploratory diagnostics, not evidence by themselves.

For selected continuous variables:

- degree-2 polynomial fit
- degree-3 polynomial fit only if degree-2 is clearly insufficient
- LOWESS or spline smooth as a safer visual check
- report incremental explanatory value versus linear fit

Use these for:

- `RSI_entry`
- `atr_pct`
- `BBP_entry`
- `volume_ratio`
- top-ranked continuous features

Important caution: high-degree polynomial fits can invent structure. Keep them low-degree and require bucket tables/validation to support any visual pattern.

## Step 4: Interaction And Regime Discovery

Run interaction diagnostics because useful patterns may be conditional.

Required heatmaps:

- RSI x ATR
- RSI x BBP
- ATR x volume ratio
- regime x RSI bucket
- regime x ATR bucket
- hour x ATR bucket
- bars-since-last-stop x ATR bucket

Each cell reports:

- trade count
- loser rate
- severe-loss rate
- mean PnL
- median PnL
- total PnL contribution

Regime analysis using `regime_entry`:

- analyze separately from HMM
- report missing regime rows separately
- test whether RSI/ATR effects differ by regime
- do not use `regime_entry` predictively unless it is confirmed causal

## Step 5: Candidate Loser Rules

Translate discovered patterns into candidate rules.

Examples of rule shapes:

- high ATR + specific RSI zone
- poor regime + weak BBP
- low bars-since-last-stop + high ATR
- specific hour + unfavorable volatility bucket

For each rule:

- percent of trades flagged
- precision: flagged trades that are losers
- recall: total losers caught
- skipped-loser PnL avoided
- skipped-winner PnL sacrificed
- net PnL impact
- effect on CAGR, Sharpe, Sortino, Calmar, Ulcer Index, and max drawdown

Compare each rule against a random filter with the same trigger rate.

## Step 6: Dedicated Validation Phase

The IS/OOS split is applied here, after discovery.

Do not force every exploratory plot to be split upfront. Instead:

1. Discover patterns on the in-sample period.
2. Freeze candidate variables, thresholds, and rules.
3. Evaluate those frozen rules on out-of-sample data.

Default split:

- IS: 2013-2020
- OOS: 2021-2026

Also add rolling or expanding walk-forward checks for the strongest rules.

Validation reports:

- IS vs OOS precision
- IS vs OOS recall
- IS vs OOS net PnL impact
- IS vs OOS drawdown impact
- IS vs OOS Calmar and Ulcer improvement
- threshold stability
- whether the rule still works by symbol

## Step 7: Traditional Metrics

For baseline and validated candidate filters, per symbol:

- CAGR
- total return
- volatility
- Sharpe
- Sortino
- Calmar
- Ulcer Index
- max drawdown
- drawdown duration
- VaR/CVaR
- skew/kurtosis
- win rate
- profit factor
- expectancy

## Final Deliverables

- Organized research folder with plot/data manifest
- Full-history EDA report
- Variable ranking table
- Nonlinear fit summary
- Interaction heatmaps
- Regime-conditioned diagnostics
- Candidate rule table
- IS/OOS validation report
- Updated `PROJECT_PLAN.md`
- Updated `FINDINGS.md`

## Critical Defaults

- Analyze TQQQ and SQQQ separately.
- Primary target: `pnl_pct < 0`.
- Secondary target: severe losers.
- Use normalized ATR as primary ATR variable.
- Use polynomial fits only as exploratory support.
- Validate only frozen candidate rules in the dedicated validation phase.

---

## Related documents

- `FINDINGS.md` — dated lab notebook; load-bearing data semantics live here.
- `FEATURE_DICTIONARY.md` — naming conventions for generated features and rules.
- `CLAUDE.md` — collaboration directives and operational facts for future sessions.
- `archive/` — everything from the paused RSI leverage-overlay project: `LEGACY_RSI_OVERLAY_PLAN.md`, `LEGACY_FINDINGS.md`, `STEP*_PLAN.md`, and the legacy `.py` scripts.
