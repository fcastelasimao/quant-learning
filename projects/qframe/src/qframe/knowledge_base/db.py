"""
Knowledge base interface — SQLite backend.

Every backtest result MUST be logged here before the session ends (CLAUDE.md).

Usage:
    from qframe.knowledge_base.db import KnowledgeBase

    kb = KnowledgeBase()  # uses default path: knowledge_base/qframe.db
    kb.init_schema()      # run once to create tables

    hyp_id = kb.add_hypothesis(
        description="12-1 momentum factor",
        rationale="Underreaction to information (Jegadeesh & Titman 1993)",
        mechanism_score=4,
    )
    impl_id = kb.add_implementation(hypothesis_id=hyp_id, code="...", git_hash="abc123")
    kb.log_result(impl_id, metrics_dict)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Default DB path relative to the repo root
DEFAULT_DB_PATH = Path("knowledge_base/qframe.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name      TEXT,
    description      TEXT NOT NULL,
    rationale        TEXT,
    mechanism_score  INTEGER DEFAULT 3,
    status           TEXT DEFAULT 'backlog',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS implementations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id  INTEGER REFERENCES hypotheses(id),
    code           TEXT,
    git_hash       TEXT,
    notes          TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    implementation_id INTEGER REFERENCES implementations(id),
    ic                REAL,
    icir              REAL,
    net_ic            REAL,
    sharpe            REAL,
    max_drawdown      REAL,
    turnover          REAL,
    decay_halflife    REAL,
    slow_icir_21      REAL,
    slow_icir_63      REAL,
    ic_decay_json     TEXT,
    ic_horizon_1      REAL,
    ic_horizon_5      REAL,
    ic_horizon_21     REAL,
    ic_horizon_63     REAL,
    regime            TEXT,
    universe          TEXT,
    oos_start         TEXT,
    oos_end           TEXT,
    cost_bps          REAL,
    gate_level        INTEGER,
    passed_gate       INTEGER,
    mlflow_run_id     TEXT,
    notes             TEXT,
    signal_cache_json TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS factor_correlations (
    factor_a    TEXT NOT NULL,
    factor_b    TEXT NOT NULL,
    correlation REAL,
    period      TEXT,
    universe    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (factor_a, factor_b, period, universe)
);

CREATE TABLE IF NOT EXISTS regime_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id    INTEGER NOT NULL REFERENCES hypotheses(id),
    n_states         INTEGER NOT NULL,
    best_state       INTEGER,
    lift             REAL,
    best_state_ic    REAL,
    unconditional_ic REAL,
    go_verdict       INTEGER,
    by_state_json    TEXT,
    logged_at        TEXT DEFAULT (datetime('now'))
);
"""

# Columns that exist in backtest_results (excluding id and created_at)
_RESULT_COLUMNS = {
    "implementation_id", "ic", "icir", "net_ic", "sharpe", "max_drawdown",
    "turnover", "decay_halflife", "slow_icir_21", "slow_icir_63",
    "ic_decay_json", "ic_horizon_1", "ic_horizon_5", "ic_horizon_21",
    "ic_horizon_63", "regime", "universe", "oos_start", "oos_end", "cost_bps",
    "gate_level", "passed_gate", "mlflow_run_id", "notes", "signal_cache_json",
}


class KnowledgeBase:
    """
    SQLite-backed knowledge base for hypotheses, implementations, and results.

    Args:
        db_path: Path to the SQLite database file.
                 Created automatically if it does not exist.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create all tables if they don't already exist. Also migrates older DBs."""
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # Migrations: add new columns to existing DBs without destroying data
            hyp_cols = {row[1] for row in conn.execute("PRAGMA table_info(hypotheses)")}
            if "factor_name" not in hyp_cols:
                conn.execute("ALTER TABLE hypotheses ADD COLUMN factor_name TEXT")

            res_cols = {row[1] for row in conn.execute("PRAGMA table_info(backtest_results)")}
            for col, typ in [
                ("slow_icir_21", "REAL"),
                ("slow_icir_63", "REAL"),
                ("ic_decay_json", "TEXT"),
                ("signal_cache_json", "TEXT"),
            ]:
                if col not in res_cols:
                    conn.execute(f"ALTER TABLE backtest_results ADD COLUMN {col} {typ}")
            # Migrate: ensure regime_results table exists on older DBs
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if "regime_results" not in tables:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS regime_results ("
                    "  id               INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  hypothesis_id    INTEGER NOT NULL REFERENCES hypotheses(id),"
                    "  n_states         INTEGER NOT NULL,"
                    "  best_state       INTEGER,"
                    "  lift             REAL,"
                    "  best_state_ic    REAL,"
                    "  unconditional_ic REAL,"
                    "  go_verdict       INTEGER,"
                    "  by_state_json    TEXT,"
                    "  logged_at        TEXT DEFAULT (datetime('now'))"
                    ")"
                )
        print(f"Knowledge base initialised at {self.db_path}")

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------

    def add_hypothesis(
        self,
        description: str,
        rationale: str | None = None,
        mechanism_score: int = 3,
        status: str = "backlog",
        factor_name: str | None = None,
    ) -> int:
        """
        Insert a new hypothesis. Returns the new row id.

        Args:
            description:      Plain English description of the factor.
            rationale:        Economic mechanism (why should this work?).
            mechanism_score:  1-5 scale (1=data mining, 5=causal evidence).
            status:           'backlog' | 'active' | 'passed' | 'failed' | 'retired'
            factor_name:      Short snake_case identifier for the factor.

        Returns:
            int: hypothesis id
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO hypotheses (factor_name, description, rationale, mechanism_score, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (factor_name, description, rationale, mechanism_score, status),
            )
            return cur.lastrowid

    def update_hypothesis_status(self, hypothesis_id: int, status: str) -> None:
        """Update the status of a hypothesis ('backlog','active','passed','failed','retired')."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (status, hypothesis_id),
            )

    def get_hypothesis(self, hypothesis_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Implementations
    # ------------------------------------------------------------------

    def add_implementation(
        self,
        hypothesis_id: int,
        code: str | None = None,
        git_hash: str | None = None,
        notes: str | None = None,
    ) -> int:
        """
        Record an implementation of a hypothesis. Returns the new row id.

        Args:
            hypothesis_id: FK to hypotheses table.
            code:          Factor computation code (or git file path).
            git_hash:      Git commit hash for reproducibility.
            notes:         Implementation decisions / edge cases.

        Returns:
            int: implementation id
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO implementations (hypothesis_id, code, git_hash, notes) "
                "VALUES (?, ?, ?, ?)",
                (hypothesis_id, code, git_hash, notes),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Backtest results
    # ------------------------------------------------------------------

    def log_result(
        self,
        implementation_id: int,
        metrics: dict[str, Any],
    ) -> int:
        """
        Log a backtest result. Non-optional — must be called before session ends.

        Args:
            implementation_id: FK to implementations table.
            metrics:           Dict with keys from _RESULT_COLUMNS.
                               Unknown keys are silently ignored.

        Returns:
            int: backtest_results row id
        """
        # Filter to known columns only
        filtered = {k: v for k, v in metrics.items() if k in _RESULT_COLUMNS}
        filtered["implementation_id"] = implementation_id

        cols = ", ".join(filtered.keys())
        placeholders = ", ".join(["?"] * len(filtered))
        values = list(filtered.values())

        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO backtest_results ({cols}) VALUES ({placeholders})",
                values,
            )
            return cur.lastrowid

    def get_results(
        self,
        implementation_id: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Retrieve backtest results, optionally filtered by implementation.

        Args:
            implementation_id: filter to a specific implementation (optional).
            limit:             maximum rows to return.

        Returns:
            List of dicts, most recent first.
        """
        with self._connect() as conn:
            if implementation_id is not None:
                rows = conn.execute(
                    "SELECT * FROM backtest_results WHERE implementation_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (implementation_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Factor correlations
    # ------------------------------------------------------------------

    def get_all_results(self, include_ensembles: bool = False) -> list[dict]:
        """
        Return all backtest results joined with hypothesis and implementation info.
        Useful for correlation analysis and notebook visualisation.

        Args:
            include_ensembles: if True, include ensemble/combined-strategy entries
                (factor_name prefixes: 'phase25_', 'ensemble_', 'combined_').
                Default False — these meta-results pollute IC charts and multiple-
                testing correction tables.
        """
        ensemble_filter = "" if include_ensembles else """
            AND (
                h.factor_name IS NULL
                OR (
                    h.factor_name NOT LIKE 'phase25_%'
                    AND h.factor_name NOT LIKE 'ensemble_%'
                    AND h.factor_name NOT LIKE 'combined_%'
                )
            )
        """
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT r.*, h.factor_name, h.description, h.mechanism_score,
                       h.id as hypothesis_id,
                       i.code, i.notes as impl_notes
                FROM backtest_results r
                JOIN implementations i ON r.implementation_id = i.id
                JOIN hypotheses h ON i.hypothesis_id = h.id
                WHERE 1=1
                {ensemble_filter}
                ORDER BY r.ic DESC NULLS LAST
            """).fetchall()
        return [dict(r) for r in rows]

    def get_results_by_domain(
        self,
        domain: str,
        limit: int = 5,
        order_by: str = "ic",
        min_ic: float = 0.0,
    ) -> list[dict]:
        """
        Return top backtest results for a given factor domain (factor_type).

        Used by the SynthesisAgent for dynamic seeding — showing the model
        what has already worked in this domain to guide exploration.

        Args:
            domain:    Factor type string, e.g. 'momentum', 'mean_reversion'.
            limit:     Maximum number of results to return.
            order_by:  Column to sort by descending (default 'ic').
            min_ic:    Minimum IC filter (default 0.0 — only positive-IC factors).

        Returns:
            List of dicts with factor_name, description, ic, icir fields.
        """
        safe_cols = {"ic", "icir", "net_ic", "sharpe", "created_at"}
        col = order_by if order_by in safe_cols else "ic"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT r.ic, r.icir, h.factor_name, h.description, i.notes
                FROM backtest_results r
                JOIN implementations i ON r.implementation_id = i.id
                JOIN hypotheses h      ON i.hypothesis_id = h.id
                WHERE (i.notes LIKE ? OR i.notes LIKE ?)
                  AND r.ic > ?
                ORDER BY r.{col} DESC
                LIMIT ?
                """,
                (f"%{domain}%", f"%factor_type={domain}%", min_ic, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_correlation(self, factor_a: str, factor_b: str) -> float | None:
        """
        Return the logged Spearman correlation between two factors, or None if not found.
        Checks both orderings (a,b) and (b,a).
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT correlation FROM factor_correlations
                WHERE (factor_a = ? AND factor_b = ?)
                   OR (factor_a = ? AND factor_b = ?)
                LIMIT 1
                """,
                (factor_a, factor_b, factor_b, factor_a),
            ).fetchone()
        return float(row["correlation"]) if row else None

    def get_factor_correlations(self) -> list[dict]:
        """Return all logged factor correlations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM factor_correlations ORDER BY ABS(correlation) DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def log_factor_correlation(
        self,
        factor_a: str,
        factor_b: str,
        correlation: float,
        period: str,
        universe: str,
    ) -> None:
        """
        Upsert a factor-pair correlation into the knowledge base.

        Args:
            factor_a:    Name of factor A.
            factor_b:    Name of factor B.
            correlation: Pearson / rank correlation value.
            period:      Date range string, e.g. '2010-2024'.
            universe:    Universe identifier, e.g. 'sp500'.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO factor_correlations
                    (factor_a, factor_b, correlation, period, universe)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(factor_a, factor_b, period, universe)
                DO UPDATE SET correlation = excluded.correlation,
                              created_at  = CURRENT_TIMESTAMP
                """,
                (factor_a, factor_b, correlation, period, universe),
            )

    # ------------------------------------------------------------------
    # BHY-significant factors
    # ------------------------------------------------------------------

    def get_bhy_significant(self, alpha: float = 0.05) -> list[dict]:
        """
        Return all backtest results that survive BHY multiple-testing correction,
        ordered by IC descending.

        This avoids storing a `bhy_significant` flag in the DB (which would go
        stale whenever new factors are added) and instead recomputes the correction
        fresh every call — correct behaviour, since BHY depends on the full set.

        Args:
            alpha: FDR level for BHY correction (default 0.05).

        Returns:
            List of dicts with all backtest_results fields plus factor_name,
            factor_domain, description, code, t_stat, bhy_significant columns.
        """
        from qframe.factor_harness.multiple_testing import correct_ic_pvalues

        all_results = self.get_all_results()
        if not all_results:
            return []

        # Exclude meta/ensemble entries — their code is not executable Python and
        # they should not be treated as standalone factors in BHY correction.
        _ENSEMBLE_PREFIXES = ("phase25_", "ensemble_", "combined_")
        all_results = [
            r for r in all_results
            if not any(
                (r.get("factor_name") or "").startswith(p) for p in _ENSEMBLE_PREFIXES
            )
        ]
        if not all_results:
            return []

        corrected = correct_ic_pvalues(all_results, alpha=alpha)

        # Deduplicate by factor_name: keep the highest-IC row per name so that
        # factors run on multiple universes (e.g. 48-stock + 449-stock) are not
        # double-counted in the BHY correction.
        corrected = (
            corrected
            .sort_values("ic", ascending=False)
            .drop_duplicates(subset=["factor_name"])
        )

        bhy = corrected[corrected["bhy_significant"] == 1].copy()

        # Merge t_stat back into original dicts and return
        t_stat_map = dict(zip(corrected["factor_name"], corrected["t_stat"]))
        result_list = []
        seen_names: set[str] = set()
        for row in all_results:
            name = row.get("factor_name") or ""
            if name in seen_names:
                continue
            if name not in bhy["factor_name"].values:
                continue
            row_copy = dict(row)
            row_copy["t_stat"] = t_stat_map.get(name, 0.0)
            result_list.append(row_copy)
            seen_names.add(name)

        # Sort by IC descending
        result_list.sort(key=lambda r: r.get("ic", 0) or 0, reverse=True)
        return result_list

    def get_implementation(self, hypothesis_id: int) -> dict | None:
        """
        Return the most recent implementation for a given hypothesis id.

        Args:
            hypothesis_id: id from the hypotheses table.

        Returns:
            Dict with implementation fields, or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM implementations WHERE hypothesis_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (hypothesis_id,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Regime results
    # ------------------------------------------------------------------

    def log_regime_result(
        self,
        hypothesis_id: int,
        n_states: int,
        best_state: int | None,
        lift: float | None,
        best_state_ic: float | None,
        unconditional_ic: float | None,
        go_verdict: int,
        by_state_json: str | None = None,
    ) -> int:
        """
        Log a Phase 2 regime IC decomposition result.

        Args:
            hypothesis_id:    FK to hypotheses table.
            n_states:         Number of HSMM states used.
            best_state:       Index of the state with highest IC.
            lift:             best_state_ic / unconditional_ic.
            best_state_ic:    IC in the best regime state.
            unconditional_ic: Overall (unconditional) IC.
            go_verdict:       1 = GO (lift ≥ 1.5), 0 = NO-GO.
            by_state_json:    JSON string of the by_state DataFrame.

        Returns:
            int: new row id in regime_results.
        """
        with self._connect() as conn:
            # Delete any existing row for this hypothesis + n_states combination so
            # re-running phase2 notebook doesn't accumulate duplicate rows.
            conn.execute(
                "DELETE FROM regime_results WHERE hypothesis_id = ? AND n_states = ?",
                (hypothesis_id, n_states),
            )
            cur = conn.execute(
                "INSERT INTO regime_results "
                "(hypothesis_id, n_states, best_state, lift, best_state_ic, "
                " unconditional_ic, go_verdict, by_state_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (hypothesis_id, n_states, best_state, lift,
                 best_state_ic, unconditional_ic, go_verdict, by_state_json),
            )
            return cur.lastrowid

    def get_regime_results(self) -> list[dict]:
        """
        Return all logged regime analysis results, joined with hypothesis info.

        Returns:
            List of dicts ordered by lift DESC.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rr.*, h.factor_name, h.description
                FROM regime_results rr
                JOIN hypotheses h ON rr.hypothesis_id = h.id
                -- Keep only the latest row per (hypothesis_id, n_states) pair.
                -- Uses a self-join to filter out older duplicates.
                WHERE rr.id = (
                    SELECT MAX(r2.id)
                    FROM regime_results r2
                    WHERE r2.hypothesis_id = rr.hypothesis_id
                      AND r2.n_states      = rr.n_states
                )
                ORDER BY rr.lift DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
