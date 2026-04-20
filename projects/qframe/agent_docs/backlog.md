# qframe — Active Backlog

*Single source of truth for open work. Referenced from `CLAUDE.md`. Every session reads this when picking the next task.*

Review cadence: every 2 weeks, move one Backlog item to Active and update status.

Status values: **Active** | **Backlog** | **Blocked** | **Done** | **Retired**

---

## Active

*(currently empty — the 2026-04-19 hot path is complete: impl_82 retired, six new guards active, 182 tests passing, mean_reversion SKIP/ERROR taxonomy fixed 2026-04-20.)*

---

## Backlog — prioritised

### #12 — Sealed-hold-out enforcement
- **Source:** prior plan §7.3 #1
- **Effort:** 2 hr
- **Why:** `HOLDOUT_START=2024-06-01` is currently a convention, not a rule. Any function in `factor_harness/` that receives a date past this should raise `NotImplementedError` unless passed `unseal=True`.
- **Unblocks:** honest final-stage validation once a new candidate is found.

### #13 — `agent_docs/factor-graveyard.md`
- **Source:** prior plan §7.3 #8
- **Effort:** 30 min
- **Why:** impl_82 was retired but nothing stops someone (or the LLM) re-proposing it. Append-only ledger of retired factors, 100 words per entry, explains why each was cut.

### #14 — `scripts/cross_market_check.py --impl-id N` CLI
- **Source:** prior plan §6 task E / §7.3
- **Effort:** 2 hr
- **Why:** today step 6.D ("validate a passing factor on crypto") is a manual notebook-edit workflow. A ~60-line script turns it into one command.

### #15 — Factor-attribution dashboard
- **Source:** prior plan §7.3 #2
- **Effort:** 1 day
- **Why:** when a factor passes, we need per-sector / per-regime / per-decile breakdowns to trust it. Currently only aggregate IC is shown.

### #16 — `scripts/rebuild_sp500_close.py`
- **Source:** prior plan §7.3 #3
- **Effort:** 2 hr
- **Why:** `sp500_close.parquet` is 12 MB in the repo with no regeneration script. Data-lineage gap.

### #17 — Live-trade simulator
- **Source:** prior plan §7.3 #4
- **Effort:** 1 day
- **Why:** next-day open fills + 10 bps slippage + volume cap. Exposes Gate-3 candidates to realistic frictions before paper trading.

### #18 — Dirty-worktree guard in `loop.py`
- **Source:** prior plan §7.3 #7
- **Effort:** 30 min
- **Why:** `git_hash` is logged, but if the worktree is dirty the hash is meaningless. Refuse to start `loop.py` on unclean git state.

### #19 — Regime-robustness gate
- **Source:** prior plan §7.3 #6
- **Effort:** 2 hr
- **Why:** promote regime analysis from advisory to a hard gate: reject factors positive in <3 of 5 regimes.

### #20 — Literature factor library (MAJOR)
- **Source:** prior plan §7.1 #2a
- **Effort:** 1 weekend
- **Why:** seeds the next research phase. YAML of 50 factors from Hou-Xue-Zhang, FF5 + momentum, QMJ, value/momentum everywhere, BAB. Each is peer-reviewed; our job is measuring how much IC survives on our data.

### #21 — Feature primitives + `gplearn` symbolic regression (MAJOR)
- **Source:** prior plan §7.1 #2b
- **Effort:** 1 month
- **Why:** replaces LLM generation with a search-based factor mill. 30 primitives (rolling mean/std/rank/z, ratio, log, diff, shift, residual, ewma, Hurst). Generation=100, 15 generations, walk-forward OOS IC fitness.

### #22 — Survivorship-bias fix (C3: PIT constituents)
- **Source:** prior plan §7.3 #5
- **Effort:** 1 week
- **Why:** not urgent until a validated factor exists, but blocks global deployment.

### #23 — Collapse duplicate `research-log.md`
- **Source:** prior plan §7.2
- **Effort:** 15 min
- **Why:** two files (root + `agent_docs/`) drift. Keep the `agent_docs/` one, leave a pointer file in root.

### #24 — Move gate/regime/KB docs into `agent_docs/`
- **Source:** prior plan §7.2
- **Effort:** 30 min
- **Why:** cosmetic — root directory has 4 overlapping config-ish files. Keep only `README.md`, `CLAUDE.md`, and the active plan at root.

### #25 — Extract `GateChain` class from `loop.py`
- **Source:** prior plan §7.2
- **Effort:** 1 day
- **Why:** `loop.py` is ~900 lines doing synth+impl+backtest+correlation+ensemble+BHY+novelty. New gates are hard to add without touching `run_iteration`.

### #26 — `config/pipeline.yaml`
- **Source:** prior plan §7.2
- **Effort:** 2 hr
- **Why:** centralise `HOLDOUT_START`, `PRE_GATE_START/END`, thresholds, cost params. Enables a second "paranoid" configuration for sensitivity analysis without code edits.

### #27 — Reduce mean_reversion duplicate rate
- **Source:** crystalline-questing-star plan (Task 4)
- **Effort:** 30 min
- **Why:** synthesis agent keeps proposing near-duplicates within `mean_reversion` (5 of 10 attempts in 2026-04-20 run blocked by novelty filter). Add full history + domain round-robin on ≥3 consecutive DUPLICATEs.

### #28 — Backlog review cadence
- **Source:** crystalline-questing-star plan (Task 3b)
- **Effort:** 15 min biweekly
- **Why:** user opens this file every 2 weeks, moves one Backlog row to Active. Prevents the backlog from becoming a write-only graveyard.

### #29 — Plan-file housekeeping
- **Source:** crystalline-questing-star plan (Task 3b)
- **Effort:** 15 min
- **Why:** anything in `~/.claude/plans/` older than 30 days whose tasks are all Done or migrated here can be deleted.

---

## Done

- **2026-04-19** — impl_82 retired; six new guards active; crypto replication added; 137→177 tests (prior plan tasks 1–10).
- **2026-04-20** — Pipeline run with new guards (prior plan task 11).
- **2026-04-20** — SKIP/ERROR verdict taxonomy fix; 3 LLM anti-patterns; backlog consolidation (crystalline-questing-star plan tasks 1–3).
