---
verdict: active-research
summary: "RSI ETF leverage overlay: SPY+GLD default strongest; walk-forward validation gate pending before production"
promoted: null
---

# RSI Leverage Overlay

Research-only investigation into applying an RSI-based leverage overlay to individual ETFs in the portfolio. When an ETF's RSI-14 drops below an entry threshold, temporarily increase its allocation by a leverage percentage; exit when RSI recovers above the exit threshold.

## Key results

- **Default rule:** ETF's own RSI-14, entry <30, exit >50, +20%, one-day execution lag, one ETF at a time
- **GLD and SPY** are the strongest single-ETF overlays
- **GSG default** is rejected
- **Mixed-pair (SPY+GLD)** OOS validation completed

## Next gate

Walk-forward / train-test validation on top of the mixed-pair OOS run before promoting any threshold to production. This is an open research direction.

## Reviewed bundles

- Single-ETF: `results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg`
- Mixed SPY+GLD: `results/mixed_leverage/2026-05-15_16-27-59_6asset_tip_gsg_rpavg`
- Mixed-pair OOS: `results/mixed_leverage_oos_validation/2026-05-15_17-54-04_6asset_tip_gsg_rpavg`

## Scripts

- `build_leverage_comparison_report.py` — single-ETF overlay report builder
- `build_mixed_leverage_report.py` — mixed-pair overlay report builder
- `validate_leverage_oos.py` — OOS validation for single-ETF overlays
- `validate_mixed_leverage_oos.py` — OOS for mixed-pair overlays

## Run
```bash
conda run -n allweather python3 research/rsi_leverage_overlay/build_leverage_comparison_report.py
```
