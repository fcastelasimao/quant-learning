# Step 1 — EDA Report


## Headline

- **TQQQ**: slope=-0.00455, p=0.397, R²=0.0004 — NOT statistically significant (p≥0.05).
- **SQQQ**: slope=-0.00008, p=0.992, R²=0.0000 — NOT statistically significant (p≥0.05).

## Data availability

### TQQQ
  - Rows in window 2020-01-03 → 2026-05-08: **1,595**
  - Rows with `adj_close IS NULL`: **0** (expected 0)

### SQQQ
  - Rows in window 2020-01-03 → 2026-05-08: **1,595**
  - Rows with `adj_close IS NULL`: **0** (expected 0)

### ^IRX
  - Rows in window 2020-01-03 → 2026-05-08: **1,595**

## Canonical trade logs

### TQQQ
  - Rows: **1,627**
  - Source run: `2026-05-11 16:23:17`
  - Span: `2020-01-02 13:30:00` → `2026-05-08 15:45:00`
  - Columns: 53

### SQQQ
  - Rows: **1,572**
  - Source run: `2026-05-05 15:18:00`
  - Span: `2020-01-03 09:45:00` → `2026-05-01 15:45:00`
  - Columns: 53

## Validation results

### Nulls in critical columns

**TQQQ**
  - `RSI_entry`: 0 null(s)
  - `decision_price`: 0 null(s)
  - `avg_order_price`: 0 null(s)
  - `exit_decision_price`: 0 null(s)
  - `exit_avg_order_price`: 0 null(s)
  - `pnl_pct`: 0 null(s)
  - `entry_time`: 0 null(s)
  - `exit_time`: 0 null(s)

**SQQQ**
  - `RSI_entry`: 0 null(s)
  - `decision_price`: 0 null(s)
  - `avg_order_price`: 0 null(s)
  - `exit_decision_price`: 0 null(s)
  - `exit_avg_order_price`: 0 null(s)
  - `pnl_pct`: 0 null(s)
  - `entry_time`: 0 null(s)
  - `exit_time`: 0 null(s)

### Timestamp validity

**TQQQ**
  - `exit_time < entry_time`: **0**
  - `entry_time` outside NYSE hours or weekend: **0**
  - `entry_time` minute not in {0,15,30,45}: **0**
  - `exit_time` outside NYSE hours or weekend: **0**
  - `exit_time` minute not in {0,15,30,45}: **0**

**SQQQ**
  - `exit_time < entry_time`: **0**
  - `entry_time` outside NYSE hours or weekend: **0**
  - `entry_time` minute not in {0,15,30,45}: **0**
  - `exit_time` outside NYSE hours or weekend: **0**
  - `exit_time` minute not in {0,15,30,45}: **0**

### pnl_pct math check

**TQQQ** (formula: exit_avg/avg − 1 vs stored pnl_pct)
  - Rows with |diff| > 0.001: **1627**
  - diff max=13.550043, mean=1.786341, p99=6.427459
  - First 5 rows with |diff| > 0.001:
```
   avg_order_price  exit_avg_order_price   pnl_pct  recomputed
0        11.170582             11.258577  0.784898    0.007877
1        11.114454             11.200549  0.771798    0.007746
2        11.210603             11.055965 -1.382185   -0.013794
3        11.083039             11.171030  0.791095    0.007939
4        11.160578             11.309091  1.327881    0.013307
```
  - **NOTE:** pnl_pct mean≈1.8044 vs recomputed mean≈0.0180 — pnl_pct appears to be in percentage form (×100). Re-checking with diff = |recomputed×100 − pnl_pct|: 1627 rows > 0.001; max=0.006303, mean=0.001785

**SQQQ** (formula: exit_avg/avg − 1 vs stored pnl_pct)
  - Rows with |diff| > 0.001: **1572**
  - diff max=17.769449, mean=2.412359, p99=8.846295
  - First 5 rows with |diff| > 0.001:
```
   avg_order_price  exit_avg_order_price   pnl_pct  recomputed
0     68002.734375          69930.197125  2.832990    0.028344
1     68059.012500          66696.852642 -2.002791   -0.020014
2     64094.531250          64924.763430  1.293949    0.012953
3     63583.338281          64548.094906  1.515955    0.015173
4     63078.398438          63350.440961  0.429967    0.004313
```
  - **NOTE:** pnl_pct mean≈2.4367 vs recomputed mean≈0.0244 — pnl_pct appears to be in percentage form (×100). Re-checking with diff = |recomputed×100 − pnl_pct|: 311 rows > 0.001; max=0.001401, mean=0.000892

### Outliers (reported, not removed)

**NOTE:** pnl_pct is in percentage form (×100). The threshold |pnl_pct| > 0.50 catches returns > 0.5%, not > 50%. Virtually all TQQQ/SQQQ trades exceed 0.5% so this count is expected to be large. For >50% outliers, the equivalent threshold would be >50.0.

**TQQQ**
  - |pnl_pct| > 0.50: **1595**
```
           entry_time           exit_time   pnl_pct
0 2020-01-02 13:30:00 2020-01-03 09:45:00  0.784898
1 2020-01-03 10:15:00 2020-01-03 12:45:00  0.771798
2 2020-01-03 13:15:00 2020-01-03 15:45:00 -1.382185
3 2020-01-06 10:30:00 2020-01-06 15:30:00  0.791095
4 2020-01-07 09:45:00 2020-01-07 11:30:00  1.327881
```
  - Duration > 5 calendar days: **0**

**SQQQ**
  - |pnl_pct| > 0.50: **1567**
```
           entry_time           exit_time   pnl_pct
0 2020-01-03 09:45:00 2020-01-06 09:45:00  2.832990
1 2020-01-06 10:45:00 2020-01-08 11:45:00 -2.002791
2 2020-01-09 11:15:00 2020-01-10 09:30:00  1.293949
3 2020-01-10 11:15:00 2020-01-13 09:45:00  1.515955
5 2020-01-17 11:00:00 2020-01-22 09:30:00 -1.471243
```
  - Duration > 5 calendar days: **7**
```
              entry_time           exit_time  _dur_days
19   2020-02-13 13:00:00 2020-02-18 13:30:00   5.020833
202  2020-12-02 10:45:00 2020-12-07 11:00:00   5.010417
863  2023-06-30 15:00:00 2023-07-06 13:00:00   5.916667
998  2024-02-14 11:00:00 2024-02-20 15:00:00   6.166667
1000 2024-02-23 10:15:00 2024-02-29 09:30:00   5.968750
```

## Trade structure

### Self-overlap (within symbol)

**TQQQ** (1627 trades sorted by entry_time)
  - Consecutive-pair overlaps (entry[i+1] < exit[i]): **0**

**SQQQ** (1572 trades sorted by entry_time)
  - Consecutive-pair overlaps (entry[i+1] < exit[i]): **0**

### Cross-symbol overlap (TQQQ ∩ SQQQ)

  - Overlapping pairs: **1,425**
  - Fraction of TQQQ trades overlapping ≥1 SQQQ trade: **76.3%**
  - Total overlapping wall-time: **16,526.8 hours**
  - Max simultaneous open positions (both symbols): **2**

### Disagreements vs other runs

**TQQQ**
  - Non-canonical rows with matching (entry_time, exit_time): checked
  - Disagreements (any field differs > 1e-6): **406**
  - Max |diff_avg|: 0.58114250
  - Max |diff_exit_avg|: 1.44407705
  - Max |diff_RSI|: 4.99326309
  - Runs that disagreed (44): ['2026-04-29 13:51:41', '2026-04-29 13:54:55', '2026-04-29 14:03:55', '2026-04-29 14:53:51', '2026-05-10 14:21:32', '2026-05-10 14:26:16', '2026-05-10 14:30:29', '2026-05-10 14:33:44', '2026-05-10 14:35:02', '2026-05-10 14:40:43'] …

**SQQQ**
  - Non-canonical rows with matching (entry_time, exit_time): checked
  - Disagreements (any field differs > 1e-6): **141**
  - Max |diff_avg|: 406.45312500
  - Max |diff_exit_avg|: 135.06495536
  - Max |diff_RSI|: 13.15403491
  - Runs that disagreed (46): ['2026-04-29 13:48:39', '2026-04-29 13:51:41', '2026-04-29 13:54:55', '2026-04-29 14:03:55', '2026-04-29 14:53:51', '2026-04-29 15:21:12', '2026-04-29 15:24:19', '2026-04-29 15:43:54', '2026-05-01 10:39:28', '2026-05-01 19:57:36'] …

## Distributions

### TQQQ

**RSI_entry**
  mean=55.63, median=55.92, std=9.93, min=35.06, max=71.95, IQR=16.69

**pnl_pct**
  mean=0.24577, median=0.66176, std=2.14839, win_rate=54.5%, skew=0.722, excess_kurt=2.011

**Trade duration (calendar days)**
  median=0.20, p90=2.76, max=4.11

**exit_reason value_counts**
  - TRAIL_STOP: 1626
  - FINAL_LIQUIDATION: 1

**regime_entry value_counts**
  - all NaN

**hour_of_entry value_counts**
  - hour 9: 152
  - hour 10: 442
  - hour 11: 192
  - hour 12: 233
  - hour 13: 231
  - hour 14: 198
  - hour 15: 179

### SQQQ

**RSI_entry**
  mean=51.99, median=51.68, std=9.67, min=35.00, max=71.99, IQR=14.46

**pnl_pct**
  mean=0.35829, median=0.63523, std=2.97999, win_rate=50.6%, skew=0.619, excess_kurt=2.293

**Trade duration (calendar days)**
  median=0.74, p90=2.79, max=7.03

**exit_reason value_counts**
  - TRAIL_STOP: 1571
  - FINAL_LIQUIDATION: 1

**regime_entry value_counts**
  - chop_highvol: 513
  - bear: 421
  - sideways_lowvol: 379
  - bull: 259

**hour_of_entry value_counts**
  - hour 9: 64
  - hour 10: 352
  - hour 11: 280
  - hour 12: 214
  - hour 13: 198
  - hour 14: 222
  - hour 15: 242

## Edge prior — pnl_pct vs RSI_entry

**TQQQ**
  - slope=-0.004548, intercept=0.498773, p=0.3969, R²=0.0004
  - Bins with data: 8/8

**SQQQ**
  - slope=-0.000081, intercept=0.362522, p=0.9916, R²=0.0000
  - Bins with data: 8/8

See `prior_pnl_vs_rsi.png` for the scatter/regression/binned-mean plot.

**Binned mean pnl_pct by RSI bin (5-pt bins)**

| symbol | rsi_bin | n | mean_pnl_pct | median_pnl_pct | win_rate | sem |
|--------|---------|---|-------------|---------------|----------|-----|
| SQQQ | [35,40) | 214 | 0.42339 | -0.59740 | 48.1% | 0.15788 |
| SQQQ | [40,45) | 188 | 0.54015 | 0.69101 | 51.1% | 0.16390 |
| SQQQ | [45,50) | 300 | 0.21046 | 0.64224 | 50.7% | 0.16185 |
| SQQQ | [50,55) | 281 | 0.38234 | 0.89662 | 52.7% | 0.17521 |
| SQQQ | [55,60) | 228 | 0.21678 | -0.76320 | 49.6% | 0.21402 |
| SQQQ | [60,65) | 178 | 0.25604 | -1.05489 | 47.8% | 0.27207 |
| SQQQ | [65,70) | 134 | 0.78220 | 1.37362 | 56.7% | 0.30477 |
| SQQQ | [70,75) | 49 | 0.01389 | -1.02861 | 46.9% | 0.54435 |
| TQQQ | [35,40) | 113 | 0.09605 | 0.48355 | 51.3% | 0.18477 |
| TQQQ | [40,45) | 139 | 0.38572 | 0.83910 | 56.8% | 0.16870 |
| TQQQ | [45,50) | 279 | 0.23782 | 0.75837 | 54.1% | 0.15358 |
| TQQQ | [50,55) | 251 | 0.34973 | 0.89049 | 57.4% | 0.15281 |
| TQQQ | [55,60) | 238 | 0.43552 | 0.84800 | 58.8% | 0.14124 |
| TQQQ | [60,65) | 248 | 0.03278 | -0.58807 | 48.8% | 0.11976 |
| TQQQ | [65,70) | 230 | 0.25299 | 0.56904 | 53.9% | 0.12858 |
| TQQQ | [70,75) | 129 | 0.08752 | 0.42621 | 54.3% | 0.14063 |

## Conclusions

- **TQQQ: no statistically significant RSI edge detected** (slope=-0.00455, p=0.397, R²=0.0004). Proceed to the threshold sweep cautiously — any effect found may be noise.
- **SQQQ: no statistically significant RSI edge detected** (slope=-0.00008, p=0.992, R²=0.0000). Proceed to the threshold sweep cautiously — any effect found may be noise.

- Data integrity: see Validation results section for any violations found.
- Cross-symbol overlap: see Trade structure section. Step 2 combined-portfolio run must handle simultaneous TQQQ+SQQQ positions.
- Disagreement diagnostic: see disagreements_TQQQ.csv and disagreements_SQQQ.csv for any price/RSI inconsistencies across runs.
