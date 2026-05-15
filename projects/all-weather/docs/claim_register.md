# Claim Register

Every public-facing claim about the strategy should be backed by an artifact,
data basis, date range, and metric definition. Claims not listed here should not
be used in customer or employer materials.

| Claim | Status | Evidence artifact | Data basis | Date range | Notes |
|---|---|---|---|---|---|
| Production strategy is a long-only six-ETF risk-balanced allocation. | Approved | `strategies.json` | Strategy registry | Current | Public claim is structural, not performance-based. |
| Production weights are SPY 13.4%, QQQ 10.3%, TLT 17.5%, TIP 34.8%, GLD/GLDM 14.2%, GSG 9.8%. | Approved | `strategies.json` | Strategy registry | Current | GLD is backtest ETF; GLDM is live gold ETF substitution. |
| Strategy is designed for capital preservation rather than return maximization. | Approved | `docs/customer_pack.md` | Methodology | Current | Must be phrased as design intent, not guaranteed outcome. |
| Recent OOS stress-window Calmar is around 0.45 to 0.50. | Approved with caveat | `results/rp_rerun_2026-05-10_10-25-19/rp_oos_summary.csv` and per-split `results/2026-05-10_*_yfinance_total_return_rpavg_*oos/stats.csv` | yfinance total return | 2018/2020/2022 starts to 2026-05-10 | Windows overlap; call them stress windows, not independent samples. |
| Daily max drawdown in the current reruns is around -17.7%. | Approved with caveat | `results/rp_rerun_2026-05-10_10-25-19/rp_oos_summary.csv` and per-split `results/2026-05-10_*_yfinance_total_return_rpavg_*oos/stats.csv` | yfinance total return | 2018/2020/2022 starts to 2026-05-10 | Use exact run artifact when quoting. |
| FMP adjusted close confirms the yfinance total-return conclusion. | Approved with caveat | `results/rp_rerun_2026-05-10_10-25-19/rp_oos_summary.csv` and per-split `results/2026-05-10_*_fmp_adj_close_rpavg_*oos/stats.csv` | FMP adjusted close | 2018/2020/2022 starts to 2026-05-10 | Keep phrasing to "materially consistent," not identical in all future data pulls. |
| Default RSI ETF overlay research shows GLD and SPY as the strongest one-ETF overlays at 20% leverage. | Internal research only | `results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg/overlay_summary.csv` | yfinance total return | 2006-07-21 to 2026-05-08 | Default rule only: RSI-14 entry <30, exit >50, +20%, one ETF at a time. GLD: 7.27% CAGR, 0.324 Calmar, -22.45% Max DD. SPY: 7.64% CAGR, 0.323 Calmar, -23.62% Max DD. Not production. |
| Best in-sample RSI overlay grid row is GLD entry 22 / exit 46 / 50% overlay. | Internal research only | `results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg/leverage_summary.csv` and `threshold_grid.csv` | yfinance total return | 2006-07-21 to 2026-05-08 | In-sample grid result: 8.07% CAGR, 0.429 Calmar, -18.83% Max DD. Treat as hypothesis because the grid has 6,912 tests and needs OOS/walk-forward validation. |
| ALLW comparison supports the product story. | Marketing-supporting only | `results/production_validation/2026-05-10_10-01-45_6asset_tip_gsg_rpavg/summary_metrics.csv` and `research.compare_allw` outputs | yfinance total return | ALLW launch on 2025-03-06 to 2026-05-08 | Short history only; do not use as full-cycle validation. |
| Paper trading started in 2026. | Internal only until audited | `logs/performance_tracking_*.csv` | Alpaca paper account | 2026 onward | Do not use as customer proof until reconciled and reviewed. |
| The strategy is live-trading ready. | Not approved | N/A | N/A | N/A | Requires passing live acceptance checks and a clean release tag. |
| The strategy has live results. | Not approved | N/A | N/A | N/A | Requires actual audited live account history. |
| The strategy is suitable for all investors. | Prohibited | N/A | N/A | N/A | Never use. Suitability is investor-specific. |
| The strategy guarantees lower drawdowns or positive returns. | Prohibited | N/A | N/A | N/A | Never use. |
