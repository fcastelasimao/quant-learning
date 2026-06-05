---
verdict: closed
summary: "Bond leverage 1.0x-2.5x destroys Calmar in rising-rate regime (2022: 0.355 -> 0.079 at 2x)"
promoted: null
---

# Bond Leverage

**Question:** Does leveraging bond positions (TLT, TIP) improve Calmar by letting bonds contribute equal return as well as equal risk?

**Result:** Leverage sweep from 1.0x to 2.5x in 0.25x steps across 3 OOS splits. Every increment adds ~3% deeper drawdown. In the 2022 rate shock, 2x leverage causes Calmar to collapse from 0.355 to 0.079. The strategy is highly sensitive to the rate regime; bond leverage is only viable when rates are falling.

**Run:**
```bash
conda run -n allweather python3 research/bond_leverage/run_leverage_experiment.py
```
