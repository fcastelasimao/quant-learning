# Research Investigations

Each subfolder is a self-contained investigation: runner scripts, findings doc with verdict, and results. Shared plotting/analysis helpers live in `_shared/`.

## Summary

| Investigation | Verdict | Key Result |
|---|---|---|
| [universe_selection](universe_selection/) | **CLOSED** | 6-asset confirmed optimal (50k subsets + 8-asset variants) |
| [optimiser_comparison](optimiser_comparison/) | **CLOSED** | DE fails OOS; SLSQP risk parity dominates |
| [rolling_rp](rolling_rp/) | **CLOSED** | Converges to static weights on all splits |
| [rebalance_frequency](rebalance_frequency/) | **REOPENED** | Failed pre-tax; superseded by tax_drift_trigger |
| [momentum_overlay](momentum_overlay/) | **CLOSED** | Re-entry timing not learnable (126 combos) |
| [bond_leverage](bond_leverage/) | **CLOSED** | Destroys Calmar in rising-rate regime |
| [allw_benchmark](allw_benchmark/) | **PRODUCTION** | DIY beats ALLW on Calmar (2.797 vs 1.961) |
| [data_source_validation](data_source_validation/) | **PRODUCTION** | yfinance = FMP adj_close; price-return understates |
| [tax_drift_trigger](tax_drift_trigger/) | **PRODUCTION (gated)** | Drift beats monthly under US tax; F.26 pending |
| [rsi_leverage_overlay](rsi_leverage_overlay/) | **ACTIVE RESEARCH** | SPY+GLD strongest; walk-forward gate pending |
| [production_validation](production_validation/) | **PRODUCTION** | Bundle builder for strategy marimo |
| [shadow_comparison](shadow_comparison/) | **TODO** | Live vs simulated reconciliation (Section G) |

## Verdicts

- **CLOSED** — tested, didn't work, no further action
- **REOPENED** — previously closed, new evidence warrants revisiting
- **ACTIVE RESEARCH** — positive results but not yet promoted to production
- **PRODUCTION** — findings changed the production system
- **PRODUCTION (gated)** — ready to promote, pending human review
- **TODO** — planned but not yet started

## Shared helpers (`_shared/`)

| Module | Purpose |
|---|---|
| `export.py` | Timestamped result directory creation, CSV/Excel export, master log |
| `strategy_plotting.py` | Dark-theme matplotlib figures for strategy comparison marimo |
| `leverage_plotting.py` | Matplotlib figures for leverage comparison marimo |
| `leverage_analysis.py` | Analysis helpers for leverage plots |
| `sensitivity.py` | Sensitivity analysis helpers |
| `validation.py` | Walk-forward + Pareto frontier helpers |

## How to reproduce any investigation

Each `findings.md` has a **Run** section with the exact command. All scripts import from `engine/`, so run from the project root (`projects/all-weather/`):

```bash
conda run -n allweather python3 research/<investigation>/<script>.py
```

## Demoted strategy registry entries

`strategies_archive.json` holds experimental entries removed from the production `strategies.json`.
