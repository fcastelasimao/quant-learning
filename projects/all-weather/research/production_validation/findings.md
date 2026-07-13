---
verdict: production
summary: "Bundle builder for strategy comparison marimo — generates all CSV/JSON artifacts notebooks consume"
promoted: "notebooks/strategy_comparison.py reads these bundles"
---

# Production Validation Bundle

Generates the timestamped artifact bundles that the marimo notebooks consume. Includes strategy comparison (DIY vs benchmarks), tax addendum (regime comparison, rebalance events, tax summary), and leverage overlay candidates.

## Artifacts per bundle

- `manifest.json`, `price_provenance.json`
- `daily_series.csv`, `monthly_returns.csv`, `summary_metrics.csv`
- `calendar_year_metrics.csv`, `rolling_metrics.csv`, `drawdown_events.csv`
- `stress_period_metrics.csv`, `risk_contribution.csv`, `turnover_costs.csv`
- Tax addendum: `rebalance_events.csv`, `tax_summary.csv`, `tax_monthly_series.csv`, `tax_regime_comparison.csv`

## Scripts

- `production_validation.py` — top-level bundle runner (CLI entry point)
- `build_strategy_comparison_report.py` — core comparison builder (yfinance + FMP paths)

## Run
```bash
conda run -n allweather python3 research/production_validation/production_validation.py
```
