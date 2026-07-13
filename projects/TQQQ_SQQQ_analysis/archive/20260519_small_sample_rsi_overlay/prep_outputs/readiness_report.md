# New Trade Data Readiness Report
## Status
- Loaded **4,273** trades from **6** Excel files.
- `RSI_entry` is missing for **4,273 / 4,273** trades, so RSI-gated sleeve tests remain blocked.
- Non-RSI preparation is complete: dates, schema normalization, trade summaries, and quality checks are available.

## Source Summary

| symbol   | source_file                                                       |   n_trades | entry_start         | exit_end            |   starting_capital |   ending_capital |   mean_pnl_pct |   win_rate |
|:---------|:------------------------------------------------------------------|-----------:|:--------------------|:--------------------|-------------------:|-----------------:|---------------:|-----------:|
| SQQQ     | SQQQ_backtest_intraday_2013-01-01_2015-12-31_20260526_084542.xlsx |        588 | 2013-01-03 14:15:00 | 2015-12-31 15:45:00 |          5000      | 204446           |       0.678215 |   0.426871 |
| SQQQ     | SQQQ_backtest_intraday_2016-01-01_2020-12-31_20260526_085954.xlsx |        614 | 2016-01-04 14:45:00 | 2020-12-03 15:00:00 |          5000      |      3.50493e+07 |       0.779791 |   0.456026 |
| SQQQ     | SQQQ_backtest_intraday_2021-01-01_2026-05-22_20260526_093210.xlsx |        728 | 2021-03-10 12:00:00 | 2026-05-12 14:45:00 |            37.6273 |      2.07079e+08 |       1.01106  |   0.472527 |
| TQQQ     | TQQQ_backtest_intraday_2013-01-01_2015-12-31_20260526_084542.xlsx |        701 | 2013-01-03 10:45:00 | 2015-12-31 14:15:00 |          5000      | 201585           |       0.440711 |   0.526391 |
| TQQQ     | TQQQ_backtest_intraday_2016-01-01_2020-12-31_20260526_085954.xlsx |        787 | 2016-01-04 13:45:00 | 2020-12-31 15:45:00 |          5000      |      9.12507e+07 |       1.03224  |   0.585769 |
| TQQQ     | TQQQ_backtest_intraday_2021-01-01_2026-05-22_20260526_093210.xlsx |        855 | 2021-01-04 14:45:00 | 2026-05-22 15:45:00 |         10000      |      4.40141e+08 |       1.04781  |   0.562573 |

## Quality Checks

| check                            | severity                | status   |   failing_count | detail                                                                                            |
|:---------------------------------|:------------------------|:---------|----------------:|:--------------------------------------------------------------------------------------------------|
| RSI_entry availability           | BLOCKER_FOR_RSI_OVERLAY | FAIL     |            4273 | Expected to fail for current files; sleeve gates cannot be evaluated until RSI_entry is supplied. |
| entry_time non-null              | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| exit_time non-null               | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| avg_order_price non-null         | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| exit_avg_order_price non-null    | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| capital_before non-null          | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| capital_end non-null             | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| pnl non-null                     | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| pnl_pct non-null                 | ERROR                   | PASS     |               0 | Required for canonical trade replay and summary analysis.                                         |
| exit_time >= entry_time          | ERROR                   | PASS     |               0 | Negative holding periods indicate date parsing or source-data issues.                             |
| weekday timestamps               | WARN                    | PASS     |               0 | Weekend timestamps can reflect parsing errors or non-market artifacts.                            |
| 15-minute bar alignment          | WARN                    | PASS     |               0 | Existing strategy works on 15-minute bars; off-grid timestamps need review.                       |
| pnl_pct matches exit/entry fills | WARN                    | PASS     |               0 | Tolerance is 10 bps in percentage-return units. Max diff=0.080003.                                |
| duplicate trade keys             | WARN                    | PASS     |               0 | Duplicate keys may be real repeated trades, but should be inspected before canonical use.         |
| self-overlap within symbol       | WARN                    | PASS     |               0 | A non-zero value requires capital allocation logic that supports concurrent same-symbol trades.   |

## Cross-symbol Overlap

- Overlapping TQQQ/SQQQ trade pairs: **1,287**
- TQQQ trades with at least one overlap: **1,134**
- SQQQ trades with at least one overlap: **1,071**
- Total overlap time: **12,839.2 hours**

## Waiting On RSI

Once RSI columns arrive, the remaining work is mechanical:
1. Join or append `RSI_entry` to `canonical_trades_no_rsi.csv` by symbol and timestamp.
2. Re-run this prep script and confirm the RSI availability check passes.
3. Re-run the existing sleeve sweeps on the longer 2013-2026 history.
4. Add SQQQ standalone and combined-portfolio allocation tests.
