# Failed Strategies

Each folder contains the code for a closed investigation. The experiments are reproducible but are clearly **not** part of the production strategy.

| Investigation | Conclusion | Folder |
|---|---|---|
| **Differential Evolution** | All 26 trials fail OOS — SLSQP risk parity strictly dominates return-based optimisation on this universe | `differential_evolution/` |
| **SPY Momentum Overlay** | 126 parameter combos: +1.3% on 2/3 OOS splits, -5.3% on the hardest split — re-entry timing unsolvable | `momentum_overlay/` |
| **Rolling RP** | Quarterly recomputation converges to the same weights as static RP on all 3 OOS windows | `rolling_rp/` |
| **Weekly Rebalancing** | No improvement over monthly after transaction costs on 3 OOS splits | `weekly_rebalance/` |
| **8-Asset Universe** | SPY+QQQ+IJR+TLT+IEF+GLD+CPER+DBA variants: 6-asset production beats on all Calmar windows | `eight_asset_universe/` |
| **Bond Leverage (1.0×–2.5×)** | Every 0.25× step adds ~3% deeper drawdown; 2022 Calmar collapses from 0.355 to 0.079 at 2× | `bond_leverage/` |
| **100-ETF Universe Scan** | 50k random subsets of 50 ETFs; 6-asset universe confirmed optimal | `universe_scan/` |

## How to reproduce

Each folder has its own README with the exact command to run. All scripts import from the parent `engine/` package, so run from the `projects/all-weather/` root:

```bash
conda run -n allweather python3 -m failed_strategies.rolling_rp.run_rolling_rp
# or directly:
conda run -n allweather python3 failed_strategies/rolling_rp/run_rolling_rp.py
```

## Strategies archive

`strategies_archive.json` holds the demoted/experimental strategy registry entries removed from `strategies.json` (8-asset manual and 8-asset CPER variants).
