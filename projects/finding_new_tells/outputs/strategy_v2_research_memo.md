# ETF Strategy Research Memo

## Hypothesis

    A smaller set of train-selected metric votes, with harmful vote directions inverted, should provide a cleaner TQQQ timing signal than the current equal-weight softmax ensemble.

## Motivation

The current v1 strategy uses many votes with equal influence plus regime machinery. The visual and IC work suggests some metrics have useful information, but direction and stability vary a lot. V2 tests whether a smaller evidence-gated signal set can improve timing discipline before adding more model complexity.

## Method

    - Signal selection, direction, and weights used train only.
    - Validation is used only for evaluation/context.
    - Frozen test window remains unused: 2022-01-01 onward.
- Decisions are based on TQQQ next-open tradable forward returns.
- Selected exposure rule on train only: `{'mode': 'binary', 'threshold': 0.0, 'medium_threshold': 0.0}`.
- HSMM regime gating is disabled for v2.
- Fill rule: close[t] signal fills at open[t+1].
- Costs: `6.0` bps on each position change.

## Metric Decisions

Kept or inverted:

```text
              metric decision   weight  tqqq_edge_bps_train  tqqq_edge_bps_val
qqq_sma50_200_regime     keep 1.000000           169.622144         139.074846
          qqq_bb_z20     keep 1.000000           368.842638         155.171836
qqq_williams_vix_fix     keep 1.000000           253.982705         102.774992
            qqq_rsi2     keep 1.000000           137.259854          44.728661
         tnx_20d_chg     keep 0.568974            56.897444         -35.520589
 qqq_spy_ratio_slope     keep 0.055821             5.582133         -26.121064
          qqq_rv_60d   invert 1.000000          -112.364037         -33.144141
      qqq_yz_vol_20d   invert 1.000000          -110.414150          13.110390
  vix_term_structure   invert 1.000000          -232.189108          62.766399
        qqq_rv_ratio   invert 0.993222           -99.322207          36.799492
       tqqq_volume_z   invert 0.977668           -97.766752         346.806458
       qqq_20d_slope   invert 0.814636           -81.463584           1.516770
   hyg_lqd_ratio_chg   invert 0.697827           -69.782698         -46.010117
          qqq_rv_20d   invert 0.477425           -47.742548          36.411122
  tqqq_path_residual   invert 0.459309           -45.930906        -141.853489
   tqqq_vol_drag_est   invert 0.329702           -32.970153          -2.948852
```

Dropped:

```text
qqq_mom_term_structure, fomc_drift
```

## Results

| Split | CAGR | Sharpe | MaxDD | DD Duration | Exposure | Turnover | Trades | vs TQQQ B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 13.41% | 0.53 | -58.6% | 1283 | 25.2% | 0.0997 | 198 | -39.99% |
| Val | 9.99% | 0.45 | -68.3% | 471 | 29.9% | 0.0913 | 92 | -53.80% |

Comparison:

```text
split                 name      cagr  sharpe  sortino  calmar  maxdd_pct  maxdd_duration_days  current_dd_duration_days  exposure_pct  trade_count  turnover  vs_tqqq_bh_excess_cagr  vs_qqq_bh_excess_cagr
train  v1_current_strategy  0.062739  0.6491   0.2417  0.4408   -14.2336                  685                       107     11.631420         98.0       NaN                     NaN                    NaN
train v2_weighted_strategy  0.134129  0.5333   0.2975  0.2291   -58.5502                 1283                         0     25.226586        198.0  0.099698               -0.399935              -0.041072
train         qqq_buy_hold  0.175201  1.0607   1.0082  1.0709   -16.3598                  166                         7           NaN          NaN       NaN                     NaN                    NaN
train        tqqq_buy_hold  0.534064  1.1112   0.9838  1.1550   -46.2406                  297                         6           NaN          NaN       NaN                     NaN                    NaN
train        tqqq_fixed_25  0.139533  1.1112   0.9838  1.0554   -13.2214                  182                         6           NaN          NaN       NaN                     NaN                    NaN
train        tqqq_fixed_50  0.278349  1.1112   0.9838  1.1041   -25.2101                  189                         6           NaN          NaN       NaN                     NaN                    NaN
  val  v1_current_strategy -0.023025 -0.0826  -0.0260 -0.0646   -35.6410                  271                       221      8.829365         38.0       NaN                     NaN                    NaN
  val v2_weighted_strategy  0.099868  0.4517   0.2654  0.1463   -68.2652                  471                       470     29.861111         92.0  0.091270               -0.537997              -0.159140
  val         qqq_buy_hold  0.259008  1.0655   0.9649  0.9069   -28.5594                  158                        27           NaN          NaN       NaN                     NaN                    NaN
  val        tqqq_buy_hold  0.637865  1.0685   0.9652  0.9192   -69.3933                  275                        26           NaN          NaN       NaN                     NaN                    NaN
  val        tqqq_fixed_25  0.184228  1.0685   0.9652  0.7950   -23.1741                  139                        26           NaN          NaN       NaN                     NaN                    NaN
  val        tqqq_fixed_50  0.360769  1.0685   0.9652  0.8575   -42.0700                  141                        26           NaN          NaN       NaN                     NaN                    NaN
```

## Benchmark Tables

Train:

```text
                 cagr  sharpe  sortino  calmar  maxdd_pct  maxdd_duration_days  current_dd_duration_days
name                                                                                                    
strategy_v2  0.134129  0.5333   0.2975  0.2291   -58.5502                 1283                         0
qqq_bah      0.175201  1.0607   1.0082  1.0709   -16.3598                  166                         7
tqqq_bah     0.534064  1.1112   0.9838  1.1550   -46.2406                  297                         6
tqqq_25      0.139533  1.1112   0.9838  1.0554   -13.2214                  182                         6
tqqq_50      0.278349  1.1112   0.9838  1.1041   -25.2101                  189                         6
```

Validation:

```text
                 cagr  sharpe  sortino  calmar  maxdd_pct  maxdd_duration_days  current_dd_duration_days
name                                                                                                    
strategy_v2  0.099868  0.4517   0.2654  0.1463   -68.2652                  471                       470
qqq_bah      0.259008  1.0655   0.9649  0.9069   -28.5594                  158                        27
tqqq_bah     0.637865  1.0685   0.9652  0.9192   -69.3933                  275                        26
tqqq_25      0.184228  1.0685   0.9652  0.7950   -23.1741                  139                        26
tqqq_50      0.360769  1.0685   0.9652  0.8575   -42.0700                  141                        26
```

## Failure Modes

- The selected metric set is still partly broad and may be a proxy for beta reduction.
- Validation CAGR remains weak, so this is not an investable candidate.
- Threshold choice is deliberately simple; better validation requires a clearer signal, not just more tuning.
- Several inverted metrics are plausible warnings that the current vote definitions need finance review.
- The frozen test set was not used, so there is still no final OOS claim.

## Decision

Decision: Revise.

Next action: narrow the signal set further, starting with the metrics whose train/validation direction and finance intuition agree, then retest with the same split discipline.
