---
verdict: closed
summary: "All 26 DE trials fail OOS — SLSQP risk parity strictly dominates return-based optimisation"
promoted: null
---

# Optimiser Comparison (Differential Evolution)

**Question:** Can differential evolution (a population-based global optimiser) find allocations that generalise better OOS than risk parity or Calmar-random search?

**Result:** 26 experiments across multiple universe configurations, IS windows, and hyperparameters. All fail OOS. Root cause: the IS period 2006-2020 is a single falling-rates regime. DE finds TLT-heavy weights that produce excellent IS Calmar but collapse in the 2022 rate shock. This is structural — not fixable by tuning DE parameters or changing the IS/OOS split. Risk parity is structurally regime-agnostic because it uses only covariance, not returns.

**Run:**
```bash
conda run -n allweather python3 research/optimiser_comparison/run_experiment.py --dry-run
```

**Files:**
- `optimiser_de.py` — DE optimiser implementation (archived from engine/optimiser.py)
- `run_experiment.py` — batch experiment runner (IS optimise -> walk-forward -> OOS evaluate pipeline)
