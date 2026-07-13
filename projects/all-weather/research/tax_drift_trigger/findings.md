---
verdict: production (gated)
summary: "Drift beats monthly under US tax on all 3 OOS windows; best FIFO candidate drift_absolute(0.05); promotion gated by F.26 human review"
promoted: "engine/backtest.py::RebalancePolicy, engine/tax_backtest.py, engine/tax.py, engine/lot_ledger.py"
---

# Tax Model & Drift-Trigger Rebalancing

This investigation built the US federal tax model, lot ledger, drift-trigger rebalance policies, and the tax-aware backtest engine. It then swept drift thresholds under realistic tax modelling and found that drift beats monthly — reopening the closed `rebalance_frequency` investigation.

## Key result (D.18 verdict: PASSED)

Under US tax, **every drift policy beat monthly on Calmar on all 3 OOS windows**. Best FIFO (Alpaca-achievable): `drift_absolute_5pp` Calmar 0.378/0.399/0.327 vs monthly 0.295/0.299/0.250 (approx +28/33/30%). Effect is tax-deferral (largely vanishes under `none` regime).

## Detailed findings

- `findings_tax_model.md` — US federal tax schedule, per-asset classification
- `findings_drift_trigger.md` — drift-trigger policy design and rationale
- `findings_threshold_sweep.md` — sweep methodology, kill criterion, full results
- `findings_alpaca_lot_selection.md` — why tax_optimal selector is research-only on Alpaca

## Promotion status

**Gated by F.26 (human gate).** Before flipping live:
1. Confirm sweep on yfinance total-return matches FMP (near-identical per 2026-05-09 rerun)
2. Add `rebalance_policy` block to `strategies.json`
3. Add drift gate to `live/rebalance.py` alongside 31-day cadence
4. One paper-month of agreement before flipping live policy

## Scripts

- `tax_threshold_sweep.py` — D.18 sweep: drift threshold x tax regime x lot selector
- `rebalance_thresholds.py` — drift-threshold policy sweep helpers (richer hybrid mode + diagnostics)

## Run
```bash
conda run -n allweather python3 research/tax_drift_trigger/tax_threshold_sweep.py
```
