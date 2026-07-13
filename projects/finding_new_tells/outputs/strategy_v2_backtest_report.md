# V2 Backtest Report

This report compares train and validation results only. The frozen 2022-01-01 onward test window is untouched.

```text
split                 name    cagr  sharpe  sortino  calmar  maxdd_pct  maxdd_duration_days  current_dd_duration_days  exposure_pct  trade_count  turnover  vs_tqqq_bh_excess_cagr  vs_qqq_bh_excess_cagr
train  v1_current_strategy  0.0627  0.6491   0.2417  0.4408   -14.2336                  685                       107       11.6314         98.0       NaN                     NaN                    NaN
train v2_weighted_strategy  0.1341  0.5333   0.2975  0.2291   -58.5502                 1283                         0       25.2266        198.0    0.0997                 -0.3999                -0.0411
train         qqq_buy_hold  0.1752  1.0607   1.0082  1.0709   -16.3598                  166                         7           NaN          NaN       NaN                     NaN                    NaN
train        tqqq_buy_hold  0.5341  1.1112   0.9838  1.1550   -46.2406                  297                         6           NaN          NaN       NaN                     NaN                    NaN
train        tqqq_fixed_25  0.1395  1.1112   0.9838  1.0554   -13.2214                  182                         6           NaN          NaN       NaN                     NaN                    NaN
train        tqqq_fixed_50  0.2783  1.1112   0.9838  1.1041   -25.2101                  189                         6           NaN          NaN       NaN                     NaN                    NaN
  val  v1_current_strategy -0.0230 -0.0826  -0.0260 -0.0646   -35.6410                  271                       221        8.8294         38.0       NaN                     NaN                    NaN
  val v2_weighted_strategy  0.0999  0.4517   0.2654  0.1463   -68.2652                  471                       470       29.8611         92.0    0.0913                 -0.5380                -0.1591
  val         qqq_buy_hold  0.2590  1.0655   0.9649  0.9069   -28.5594                  158                        27           NaN          NaN       NaN                     NaN                    NaN
  val        tqqq_buy_hold  0.6379  1.0685   0.9652  0.9192   -69.3933                  275                        26           NaN          NaN       NaN                     NaN                    NaN
  val        tqqq_fixed_25  0.1842  1.0685   0.9652  0.7950   -23.1741                  139                        26           NaN          NaN       NaN                     NaN                    NaN
  val        tqqq_fixed_50  0.3608  1.0685   0.9652  0.8575   -42.0700                  141                        26           NaN          NaN       NaN                     NaN                    NaN
```
