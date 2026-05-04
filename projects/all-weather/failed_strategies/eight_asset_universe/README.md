# 8-Asset Universe

**Status:** Closed — 6-asset production wins on all Calmar windows.

**Question:** Do two 8-asset candidates (scan winner + TIP variant) beat the 6-asset production portfolio?

**Experiments:**
- Experiment A: SPY, QQQ, IJR, TLT, IEF, GLD, CPER, DBA (scan winner, no TIP)
- Experiment B: SPY, QQQ, IJR, TLT, TIP, GLD, CPER, DBA (keeps TIP, drops IEF)

**Result:** RP weights averaged across 3 OOS splits (2018, 2020, 2022). Both 8-asset variants underperform `6asset_tip_gsg_rpavg` on Calmar across all windows. Adding IJR/IEF/CPER/DBA introduces diversification noise that RP cannot exploit without history of inflation regimes.

**Run:**
```bash
conda run -n allweather python3 failed_strategies/eight_asset_universe/run_8asset_experiments.py
```
