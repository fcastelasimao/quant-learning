# Tax × drift-threshold × lot-selector sweep (D.18)

**Status:** complete. Verdict: **propose new production policy** (drift-based
rebalancing beats monthly under US tax). Script: `research/tax_threshold_sweep.py`.
Tests: `tests/test_tax_threshold_sweep.py`. Decision gate: handoff **F.26**.

## Question

Does drift-based rebalancing improve the strategy's risk-adjusted return once
realistic US tax is modelled — and if so, is the gain achievable on Alpaca
(FIFO), or only with lot selection we can't actually execute there?

## Method

`run_tax_aware_backtest` (D.16) over the full grid:

- **rebalance policy:** `monthly_unconditional` (baseline);
  `drift_relative` 10/15/20/25/30/40%; `drift_absolute` 1/2/3/5pp
- **tax regime:** `us`, `none`
- **lot selector:** `fifo`, `tax_optimal`

Each configuration scored on **Calmar** over the three OOS windows
(2018 / 2020 / 2022 → backtest end). Prices: FMP `adj_close` (total-return);
central dividend store; transaction cost 0.1%.

### Pre-registered kill criterion

> Under the US regime, if the best drift policy beats the monthly baseline on
> Calmar by **≥ 5%** on at least **2 of the 3** OOS windows (matched on lot
> selector), **propose a new production policy**. Otherwise the tax model stays
> research-only and `monthly_unconditional` remains production.

## Result — US regime, FIFO (Alpaca-achievable)

Calmar by OOS window (run 2026-06-03, `6asset_tip_gsg_rpavg`):

| Policy | 2018 | 2020 | 2022 |
|---|---|---|---|
| **monthly_unconditional** (baseline) | 0.295 | 0.299 | 0.250 |
| drift_relative_10pct | 0.325 | 0.333 | 0.279 |
| drift_relative_25pct | 0.348 | 0.360 | 0.305 |
| drift_relative_40pct | 0.375 | 0.383 | 0.322 |
| drift_absolute_2pp | 0.329 | 0.333 | 0.285 |
| drift_absolute_3pp | 0.354 | 0.356 | 0.296 |
| **drift_absolute_5pp** | **0.378** | **0.399** | **0.327** |

**Every** drift policy clears the ≥5% bar on **all three** windows. The best
FIFO candidate, `drift_absolute_5pp`, improves Calmar by roughly **+28% / +33%
/ +30%** over monthly. 17 policy×selector configs passed in total (10 FIFO, 7
tax_optimal).

### Why

Monthly unconditional rebalancing realizes taxable capital gains every month
regardless of need. Drift policies trade far less often, deferring realization
and compounding the deferred tax. The benefit shows up under `us` and largely
vanishes under `none` — confirming it is a **tax-deferral** effect, not a
market-timing artefact. Critically, the win is present under **FIFO**, so it is
achievable on Alpaca today and does **not** depend on the (unexecutable)
`tax_optimal` selector.

## Decision

Kill criterion **passed** → `verdict.json: "propose_new_production_policy"`.
This does **not** auto-flip production. The handoff **F.26** gate governs the
rollout: update `strategies.json` with a `rebalance_policy` block, point
`live/rebalance.py` at the drift trigger (it currently enforces only the 31-day
cadence), update `../rebalance_frequency/findings.md` with the new
tax-aware verdict, and require one paper-month of agreement before flipping the
live policy.

**Recommended candidate for F.26:** `drift_absolute(0.05)` (5pp) under FIFO —
strongest Calmar across all three windows and trivial to express as a live
drift gate.

## Caveats

- Single sweep on FMP `adj_close`. Confirm against yfinance total-return before
  the F.26 flip (they were near-identical in the 2026-05-09 rerun, per CLAUDE.md).
- Monthly tax granularity (see `findings_tax_model.md`).
- `tax_optimal` rows are research counterfactuals; do not cite them as
  Alpaca-achievable (`findings_alpaca_lot_selection.md`).

## Artifacts

`results/tax_threshold_sweep/<ts>_<strategy>/`:
`threshold_sweep_summary.csv` (one row per policy×regime×selector×window),
`verdict.json` (kill-criterion evaluation), `run_config.json` (inputs +
price provenance).
