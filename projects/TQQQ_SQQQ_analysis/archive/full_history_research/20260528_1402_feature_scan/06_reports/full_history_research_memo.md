# Full-History Feature Scan

## Scope

- Source files: `/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/TQQQ_SQQQ_analysis/full_history_canonical/trades_backtest`
- Rows: 4,273
- Symbols: TQQQ, SQQQ
- TQQQ and SQQQ are analyzed separately.
- Primary target: `pnl_pct < 0`.
- Severe-loss target: `pnl_pct <= -1%`.

## Key Review Order

1. `00_data_quality/feature_inclusion_exclusion.csv`
2. `01_single_variable/single_variable_correlations.csv`
3. `01_single_variable/single_variable_bucket_summary.csv`
4. `02_interactions/` heatmaps and tables
5. `03_regime_analysis/regime_summary.csv`
6. `04_candidate_rules/candidate_rule_skip_impact.csv`
7. `05_validation/is_oos_rule_validation.csv`

## Top Feature Scores

### TQQQ
| feature | spearman_pnl | pearson_pnl | loser_rate_spread_q5 | score |
| --- | --- | --- | --- | --- |
| atr_pct | 0.0502 | 0.1711 | 0.2239 | 0.2104 |
| atr_pct_roll_pctile_252 | 0.0357 | 0.1289 | 0.2232 | 0.1874 |
| atr | 0.0400 | 0.0785 | 0.1377 | 0.1222 |
| MA20_D5 | -0.0637 | -0.1183 | 0.1055 | 0.1059 |
| MA100_D1 | -0.0707 | -0.1188 | 0.0888 | 0.1016 |
| volume_cur | 0.0304 | 0.0762 | 0.1407 | 0.1007 |
| MA20_D3 | -0.0506 | -0.1040 | 0.0897 | 0.0985 |
| dist_to_MA100 | -0.0500 | -0.1066 | 0.1199 | 0.0954 |

### SQQQ
| feature | spearman_pnl | pearson_pnl | loser_rate_spread_q5 | score |
| --- | --- | --- | --- | --- |
| atr_pct | -0.0862 | 0.1222 | 0.1943 | 0.2367 |
| atr_pct_roll_pctile_252 | -0.0562 | 0.1379 | 0.1855 | 0.1926 |
| dist_to_MA20 | -0.0326 | 0.1296 | 0.1244 | 0.1163 |
| dist_to_MA100 | -0.0195 | 0.0425 | 0.0740 | 0.1046 |
| MA50_D1 | -0.0141 | 0.0315 | 0.0896 | 0.0973 |
| volume_cur | -0.0426 | 0.0037 | 0.1114 | 0.0951 |
| MA50_D5 | -0.0233 | 0.0151 | 0.0779 | 0.0908 |
| MA100_D1 | -0.0072 | 0.0524 | 0.0651 | 0.0857 |

## Candidate Rule Validation

| symbol | rule_name | trigger_rate | precision_loser_rate_flagged | recall_losers_caught | net_pnl_pct_impact | delta_calmar | delta_ulcer_index |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SQQQ | rsi_x_atr_cell_3_1_high_loser_rate | 0.0110 | 0.8750 | 0.0182 | 6.3485 | 0.2371 | -0.0364 |
| SQQQ | volume_cur_bucket_3_high_loser_rate | 0.0000 |  | 0.0000 | -0.0000 | 0.0000 | 0.0000 |
| SQQQ | volume_cur_bucket_0_high_loser_rate | 0.0000 |  | 0.0000 | -0.0000 | 0.0000 | 0.0000 |
| SQQQ | rsi_x_bbp_cell_4_2_high_loser_rate | 0.0206 | 0.6667 | 0.0260 | -7.3537 | -0.2416 | -0.0704 |
| SQQQ | atr_pct_bucket_0_high_loser_rate | 0.0371 | 0.5556 | 0.0391 | -10.0661 | -0.3591 | 0.0111 |
| SQQQ | atr_x_volume_cell_2_0_high_loser_rate | 0.0247 | 0.3889 | 0.0182 | -25.4089 | -0.4127 | 0.1769 |
| SQQQ | atr_pct_bucket_1_high_loser_rate | 0.1195 | 0.6552 | 0.1484 | -27.4716 | -1.5381 | 0.3896 |
| SQQQ | dist_to_MA100_bucket_1_high_loser_rate | 0.1813 | 0.5909 | 0.2031 | -44.7739 | -0.6934 | -0.2703 |
| SQQQ | MA50_D1_bucket_1_high_loser_rate | 0.1648 | 0.5583 | 0.1745 | -76.0594 | -0.1975 | 0.1878 |
| SQQQ | atr_pct_roll_pctile_252_bucket_0_high_loser_rate | 0.2610 | 0.6105 | 0.3021 | -79.8703 | -1.8016 | -0.4292 |
| SQQQ | dist_to_MA20_bucket_0_high_loser_rate | 0.1745 | 0.5827 | 0.1927 | -88.4692 | -0.1796 | -0.3425 |
| SQQQ | MA50_D5_bucket_3_high_loser_rate | 0.1676 | 0.5328 | 0.1693 | -100.5380 | -0.9922 | 0.2455 |
| SQQQ | atr_pct_roll_pctile_252_bucket_1_high_loser_rate | 0.2033 | 0.5338 | 0.2057 | -107.4139 | -5.7532 | 0.5542 |
| SQQQ | MA100_D1_bucket_0_high_loser_rate | 0.2527 | 0.5598 | 0.2682 | -110.0352 | -1.3710 | 0.0863 |
| SQQQ | MA100_D1_bucket_1_high_loser_rate | 0.1745 | 0.5512 | 0.1823 | -118.0206 | -4.4735 | 0.2463 |
| SQQQ | dist_to_MA20_bucket_2_high_loser_rate | 0.1635 | 0.4790 | 0.1484 | -125.3451 | -4.1290 | 0.5494 |
| SQQQ | regime_sideways_lowvol_high_loser_rate | 0.3228 | 0.5191 | 0.3177 | -161.2987 | -6.0875 | 1.4278 |
| SQQQ | MA50_D1_bucket_4_high_loser_rate | 0.2720 | 0.5000 | 0.2578 | -229.2131 | -8.1185 | 0.0982 |
| SQQQ | MA50_D5_bucket_4_high_loser_rate | 0.2624 | 0.4921 | 0.2448 | -242.7771 | -7.6362 | 0.7626 |
| SQQQ | dist_to_MA100_bucket_4_high_loser_rate | 0.2843 | 0.5072 | 0.2734 | -267.8611 | -8.7325 | 1.2647 |

## Notes

- Polynomial fits are exploratory only. Treat a pattern as credible only if bucket tables and OOS validation agree.
- `regime_entry` is included as a source pre-trade label, but its exact construction should be confirmed before live predictive use.
- Random same-trigger-rate comparisons are included for candidate rules to expose rules that only look good because they skip many trades.