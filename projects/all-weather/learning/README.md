# Learning: Guided Engine Rewrite

> **For the full session-by-session walkthrough — including theory, exact verification numbers, and common pitfalls — see [CURRICULUM.md](CURRICULUM.md).**

Six sessions to internalise the production engine. Each session produces a standalone Python module that you write yourself, then diff against the production `engine/` code to confirm your understanding.

| Session | Topic | File to write | Cross-check against |
|---|---|---|---|
| 1 | Data + log returns | `01_data.py` | `engine/data.py` |
| 2 | Covariance + risk contributions | `02_risk.py` | `engine/optimiser.py` (cov block) |
| 3 | SLSQP risk parity | `03_rp.py` | `engine/optimiser.compute_risk_parity_weights` |
| 4 | Monthly rebalance simulation | `04_backtest.py` | `engine/backtest.run_backtest` |
| 5 | Performance statistics | `05_stats.py` | `engine/stats.py` |
| 6 | IS/OOS discipline + 3-window RP averaging | `06_validation.py` | `strategies.json` weights (must match to 4 d.p.) |

## Workflow per session

1. Read the relevant production module.
2. Close it.
3. Write your own version from scratch in this folder.
4. Run it and verify output numerically matches the production version.
5. Diff the two files; for every difference decide: is yours cleaner (candidate to upstream) or wrong (fix yours)?

The diff in session 6 is the deliverable that demonstrates you own the code.
