# Knowledge Base Schema

*SQLite database at `knowledge_base/qframe.db`. All backtest results are logged here.*

---

## Tables

### hypotheses

One row per factor idea, before implementation.

```sql
CREATE TABLE hypotheses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name      TEXT,                   -- short snake_case identifier (e.g. 'momentum_12_1')
    description      TEXT NOT NULL,          -- plain English: what the factor is
    rationale        TEXT,                   -- why it should work (economic mechanism)
    mechanism_score  INTEGER DEFAULT 3,      -- 1-5: how well-understood is the causal story?
    status           TEXT DEFAULT 'backlog', -- backlog | active | passed | failed | retired
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`factor_name` was added 2026-04-13 via migration-safe `ALTER TABLE ADD COLUMN`.

### implementations

One row per code version of a hypothesis.

```sql
CREATE TABLE implementations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    code          TEXT,     -- factor computation code (Python function body)
    git_hash      TEXT,     -- git commit hash for reproducibility
    notes         TEXT,     -- e.g. 'factor_type=momentum|attempt=2'
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### backtest_results

One row per walk-forward run.

```sql
CREATE TABLE backtest_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    implementation_id INTEGER REFERENCES implementations(id),

    -- Core metrics (OOS period)
    ic                REAL,   -- mean cross-sectional rank IC (gross, Spearman)
    icir              REAL,   -- rolling ICIR at end of OOS (IC / std over 63d window)
    net_ic            REAL,   -- IC after deducting trading + borrow + funding costs
    sharpe            REAL,   -- annualised Sharpe of daily IC series (IC Sharpe, not P&L Sharpe)
    max_drawdown      REAL,   -- maximum peak-to-trough drawdown of cumulative IC
    turnover          REAL,   -- mean daily one-way turnover (fraction of portfolio)

    -- IC decay
    decay_halflife    REAL,   -- estimated IC half-life in days (exponential decay fit)
    ic_horizon_1      REAL,   -- mean IC at 1-day forward horizon
    ic_horizon_5      REAL,   -- mean IC at 5-day forward horizon
    ic_horizon_21     REAL,   -- mean IC at 21-day forward horizon (monthly)
    ic_horizon_63     REAL,   -- mean IC at 63-day forward horizon (quarterly)
    ic_decay_json     TEXT,   -- full 63-point curve as JSON {horizon: mean_ic}; NULL for old results

    -- Slow ICIR (non-overlapping windows — the honest metric for slow signals)
    slow_icir_21      REAL,   -- ICIR on non-overlapping 21-day windows
    slow_icir_63      REAL,   -- ICIR on non-overlapping 63-day windows; NULL for old results

    -- Context
    regime            TEXT,   -- 'all' | 'state_1' | 'state_2' | etc.
    universe          TEXT,   -- 'sp500_survivorship_biased' | 'russell3000' | 'crypto_top50'
    oos_start         TEXT,   -- OOS period start date (YYYY-MM-DD)
    oos_end           TEXT,   -- OOS period end date (YYYY-MM-DD)
    cost_bps          REAL,   -- round-trip spread cost assumed (basis points)
    gate_level        INTEGER,-- which gate does this result relate to?
    passed_gate       INTEGER,-- 1 if gate passed, 0 if failed, NULL if not evaluated

    -- Metadata
    mlflow_run_id     TEXT,   -- link to MLflow run for full artifacts
    notes             TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`slow_icir_21`, `slow_icir_63`, `ic_decay_json` were added 2026-04-13 via migration-safe `ALTER TABLE ADD COLUMN`. Earlier results have NULL for these columns.

### factor_correlations

Pairwise factor rank correlations. Populated by `loop.run_correlation_analysis()`.

```sql
CREATE TABLE factor_correlations (
    factor_a    TEXT NOT NULL,
    factor_b    TEXT NOT NULL,
    correlation REAL,         -- Spearman rank correlation of cross-sectional signals
    period      TEXT,         -- date range string, e.g. '2018-2024'
    universe    TEXT,         -- universe identifier
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (factor_a, factor_b, period, universe)
);
```

UPSERT on conflict: `run_correlation_analysis()` updates the value and timestamp if the pair already exists.

---

## Mechanism Score Guide

| Score | Meaning | Example |
|---|---|---|
| 1 | Pure data mining — no theory | Random signal found in screen |
| 2 | Weak theory — one possible explanation | Some behavioural bias story |
| 3 | Moderate — plausible mechanism, not tested | Momentum from underreaction |
| 4 | Strong — mechanism documented in literature | Value from risk compensation (Fama-French) |
| 5 | Causal — natural experiment or strong quasi-causal evidence | Post-earnings announcement drift |

Factors with `mechanism_score` < 3 require higher IC evidence (ICIR > 1.0) before advancing to Gate 3. Factors with `mechanism_score` ≥ 4 can advance with ICIR > 0.7.

---

## Gate Thresholds (quick reference)

| Gate | Metric | Weak | Pass |
|------|--------|------|------|
| Factor IC | Mean OOS IC | ≥ 0.015 | ≥ 0.030 |
| Signal consistency | ICIR (rolling 63d) | ≥ 0.15 | ≥ 0.40 |
| Slow signal check | slow_icir_63 | ≥ 0.10 | ≥ 0.25 |
| Cost efficiency | net_ic | Positive | Positive |

See `gate-thresholds.md` for full gate definitions including Sharpe, drawdown, and cross-market requirements.

---

## Python Interface

```python
from qframe.knowledge_base.db import KnowledgeBase

kb = KnowledgeBase('knowledge_base/qframe.db')
kb.init_schema()  # run once; migration-safe on repeat calls

# --- Hypotheses ---
hyp_id = kb.add_hypothesis(
    description="12-1 momentum factor (Jegadeesh-Titman)",
    rationale="Underreaction to information causes price continuation over 3-12 months",
    mechanism_score=4,
    status="active",
    factor_name="momentum_12_1",
)

# --- Implementations ---
impl_id = kb.add_implementation(
    hypothesis_id=hyp_id,
    code="def factor(prices): return prices.shift(21) / prices.shift(252) - 1",
    git_hash="abc123",
    notes="factor_type=momentum",
)

# --- Results ---
result_id = kb.log_result(impl_id, {
    "ic": 0.0157,
    "icir": 0.166,
    "net_ic": 0.0156,
    "sharpe": 1.00,
    "turnover": 0.041,
    "slow_icir_21": 0.14,
    "slow_icir_63": 0.12,
    "regime": "all",
    "universe": "sp500_survivorship_biased",
    "oos_start": "2018-01-01",
    "gate_level": 1,
    "passed_gate": 0,
})

# --- Queries ---
all_results  = kb.get_all_results()           # full JOIN, ordered by IC DESC
correlations = kb.get_factor_correlations()   # all logged factor pairs
results      = kb.get_results(limit=10)       # most recent 10 results
```

---

## Initialisation

The database is created automatically when `KnowledgeBase(path)` is first called. Run `init_schema()` once per session (idempotent):

```bash
python3 -c "
from qframe.knowledge_base.db import KnowledgeBase
kb = KnowledgeBase('knowledge_base/qframe.db')
kb.init_schema()
"
```

---

## Multiple Testing Module

After every batch of results, run BHY correction from `src/qframe/factor_harness/multiple_testing.py`:

```python
from qframe.factor_harness.multiple_testing import correct_ic_pvalues, print_correction_summary

corrected = correct_ic_pvalues(kb.get_all_results(), alpha=0.05, n_oos_days=1760)
print_correction_summary(corrected)
# Returns DataFrame: factor_name, ic, t_stat, p_raw, bhy_significant, hlz_sig
```

**t-stat formulas:**
- Fast signal (h=1): `t = (sharpe / √252) × √N_OOS` ≈ `sharpe × 2.64` for 7-year OOS
- Slow signal (h=63): `t = slow_icir_63 × √(N_OOS/63)` ≈ `slow_icir_63 × 5.28`

**Current thresholds (m=14 positive-IC factors, α=0.05):**
- BHY: t ≥ 3.07  — **recommended**
- HLZ: t ≥ 2.30  — lower bar
- Bonferroni: t ≥ 2.69  — too conservative for correlated tests

---

## Current state (2026-04-13, updated)

```
hypotheses:          62 rows
implementations:     61 rows
backtest_results:    30 rows (results 1–28 on 48-stock cache; results 29–30 on 449-stock cache)
factor_correlations: 66 rows (populated 2026-04-13 after 30th result)

Universe: 449 stocks (sp500_close.parquet, overwritten from 48-stock cache)
BHY-significant factors: 0
HLZ-significant factors: 1 (impl_1, 12-1 momentum, t=2.64)
```
