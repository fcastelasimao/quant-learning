# All Weather Strategy Bank Pack

## Methodology Summary

The strategy is a long-only ETF allocation inspired by all-weather/risk-parity
portfolio construction. The production portfolio uses six liquid ETFs:
`SPY`, `QQQ`, `TLT`, `TIP`, `GLD`/`GLDM`, and `GSG`.

Weights are derived from risk-parity research and then fixed for production.
The live portfolio rebalances monthly. The production implementation avoids
leverage, shorting, options, futures, discretionary signals, and return forecasts.

## Data Lineage

- Primary production data basis: yfinance total-return adjusted close.
- Cross-check basis: local FMP SQLite adjusted close.
- Diagnostic basis: unadjusted close/price return, used to show distribution
  omission effects.
- Each generated backtest run writes `run_config.json`; runs after the
  provenance patch also write `price_provenance.json`.
- The customer-facing report bundle writes `manifest.json` and
  `price_provenance.json`.

## Model-Risk Controls

- Strategy is deliberately simple: fixed ETF universe, fixed target weights,
  monthly rebalance, no fitted return model.
- Closed investigations are kept under `failed_strategies/` with reproducible
  scripts and conclusions.
- Rejected enhancements include leverage, weekly rebalancing, momentum overlay,
  8-asset expansion, and return-optimized differential evolution.
- Public claims must be traceable through `docs/claim_register.md`.

## Metrics Required For Review

The canonical validation bundle must include:

- CAGR, total return, volatility, Sharpe, Sortino, Calmar, and Ulcer Index.
- Monthly and daily max drawdown.
- Drawdown duration and recovery statistics.
- Beta, downside beta, up capture, and down capture versus SPY.
- Stress windows: COVID crash, COVID full shock, 2022 rate shock, ALLW overlap,
  and full available history.
- Turnover and estimated transaction-cost drag.
- Risk contribution by asset.

## Operations Controls

The live rebalancer must satisfy these controls before real capital:

- Preview mode never submits orders.
- Execute mode refuses non-month-end trading unless `--force` is explicit.
- Execute mode refuses market-closed trading.
- Open orders in target symbols block execution.
- Duplicate execution for the same account/date/strategy is blocked unless
  `--allow-duplicate-run` is explicit.
- Individual and total order size guardrails are enforced.
- Sells execute first; buys are computed after account refresh.
- Rejected, canceled, expired, or timed-out orders fail the run.
- Post-trade drift is verified and failure is reported as an execution error.

## Reproducibility

Use the pinned environment file:

```bash
conda env create -f environment.yml
conda run -n allweather python -m pytest
```

The default pytest configuration excludes network-dependent integration tests.
Run integration tests only when live vendor access is available:

```bash
conda run -n allweather python -m pytest -m integration
```

## Residual Launch Gates

- FCA/compliance review before public marketing.
- Actual live trading performance before making live-performance claims.
- Currency, tax, and suitability work if selling outside a USD educational context.
