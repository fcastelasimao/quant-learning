# Bond Leverage

**Status:** Closed — destroys Calmar in rising-rate regime.

**Question:** Does leveraging bond positions (TLT, TIP) improve Calmar by letting bonds contribute equal return as well as equal risk?

**Result:** Leverage sweep from 1.0× to 2.5× in 0.25× steps across 3 OOS splits. Every increment adds ~3% deeper drawdown. In the 2022 rate shock, 2× leverage causes Calmar to collapse from 0.355 → 0.079. The strategy is highly sensitive to the rate regime; bond leverage is only viable when rates are falling.

**Run:**
```bash
conda run -n allweather python3 failed_strategies/bond_leverage/run_leverage_experiment.py
```
