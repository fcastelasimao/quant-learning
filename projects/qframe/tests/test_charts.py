"""
Unit tests for src/qframe/viz/charts.py

Tests the two new chart functions:
  - plot_ic_by_period  (Chart 14 — temporal stability)
  - plot_multiple_testing  (Chart 15 — BHY / HLZ / Bonferroni significance)

Uses a temporary SQLite knowledge base and synthetic price data so tests
run in isolation without touching the real knowledge_base/qframe.db.

All tests use a non-interactive matplotlib backend (Agg) so no window
is opened during CI runs.
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qframe.viz.charts import plot_ic_by_period, plot_multiple_testing


# ---------------------------------------------------------------------------
# Helpers — synthetic data and temporary KB
# ---------------------------------------------------------------------------

def _make_prices(n_dates: int = 800, n_stocks: int = 40, seed: int = 42) -> pd.DataFrame:
    """Synthetic price matrix spanning ~3 years from 2018."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.012, (n_dates, n_stocks)),
        index=dates, columns=tickers,
    )
    return (1 + returns).cumprod()


# Simple valid factor: 21-day momentum
_FACTOR_CODE = "return prices.pct_change(21)"


def _make_temp_kb(
    *,
    factor_code: str = _FACTOR_CODE,
    add_result: bool = True,
    ic: float = 0.018,
    sharpe: float = 1.1,
    slow_icir_63: float | None = None,
) -> str:
    """
    Create a temporary SQLite knowledge base with one hypothesis, one
    implementation and (optionally) one backtest result.

    Returns the path to the temporary DB file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = tmp.name

    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_name TEXT,
            description TEXT NOT NULL,
            rationale TEXT,
            mechanism_score INTEGER DEFAULT 3,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE implementations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id INTEGER,
            code TEXT,
            git_hash TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            implementation_id INTEGER,
            ic REAL,
            icir REAL,
            net_ic REAL,
            sharpe REAL,
            max_drawdown REAL,
            turnover REAL,
            decay_halflife REAL,
            ic_horizon_1 REAL,
            ic_horizon_5 REAL,
            ic_horizon_21 REAL,
            ic_horizon_63 REAL,
            ic_decay_json TEXT,
            slow_icir_21 REAL,
            slow_icir_63 REAL,
            regime TEXT,
            universe TEXT,
            oos_start TEXT,
            oos_end TEXT,
            cost_bps REAL,
            gate_level INTEGER,
            passed_gate INTEGER,
            mlflow_run_id TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE factor_correlations (
            factor_a TEXT NOT NULL,
            factor_b TEXT NOT NULL,
            correlation REAL,
            period TEXT,
            universe TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (factor_a, factor_b, period, universe)
        );
    """)

    conn.execute(
        "INSERT INTO hypotheses (factor_name, description) VALUES (?, ?)",
        ("test_momentum", "21-day momentum test factor"),
    )
    conn.execute(
        "INSERT INTO implementations (hypothesis_id, code, notes) VALUES (1, ?, ?)",
        (factor_code, "factor_type=momentum|attempt=1"),
    )

    if add_result:
        conn.execute(
            """INSERT INTO backtest_results
               (implementation_id, ic, icir, net_ic, sharpe, turnover,
                ic_horizon_1, ic_horizon_5, ic_horizon_21, ic_horizon_63,
                slow_icir_63, regime, universe, oos_start, gate_level, passed_gate)
               VALUES (1, ?, 0.12, ?, ?, 0.04, ?, ?, ?, ?, ?, 'all', 'test', '2018-01-01', 1, 0)""",
            (ic, ic * 0.98, sharpe,
             ic * 1.0, ic * 0.9, ic * 0.7, ic * 0.5,
             slow_icir_63),
        )

    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prices_df():
    return _make_prices()


@pytest.fixture(scope="module")
def prices_parquet(tmp_path_factory, prices_df):
    """Write synthetic prices to a temporary parquet file."""
    p = tmp_path_factory.mktemp("prices") / "sp500_close.parquet"
    prices_df.to_parquet(str(p))
    return str(p)


@pytest.fixture
def kb_with_result(tmp_path):
    path = _make_temp_kb()
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def kb_empty(tmp_path):
    path = _make_temp_kb(add_result=False)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def kb_multiple_factors(tmp_path):
    """KB with several factors spanning a range of t-stats."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = tmp.name

    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_name TEXT, description TEXT NOT NULL,
            mechanism_score INTEGER DEFAULT 3,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE implementations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id INTEGER, code TEXT, notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            implementation_id INTEGER,
            ic REAL, icir REAL, net_ic REAL, sharpe REAL,
            max_drawdown REAL, turnover REAL, decay_halflife REAL,
            ic_horizon_1 REAL, ic_horizon_5 REAL, ic_horizon_21 REAL, ic_horizon_63 REAL,
            ic_decay_json TEXT, slow_icir_21 REAL, slow_icir_63 REAL,
            regime TEXT, universe TEXT, oos_start TEXT, oos_end TEXT,
            cost_bps REAL, gate_level INTEGER, passed_gate INTEGER,
            mlflow_run_id TEXT, notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE factor_correlations (
            factor_a TEXT, factor_b TEXT, correlation REAL,
            period TEXT, universe TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (factor_a, factor_b, period, universe)
        );
    """)

    # Insert 5 factors with varying IC / Sharpe
    factors = [
        ("factor_a", "Factor A — strong",   0.030, 1.80, None),
        ("factor_b", "Factor B — medium",   0.020, 1.10, None),
        ("factor_c", "Factor C — weak",     0.015, 0.70, None),
        ("factor_d", "Factor D — slow",     0.010, 0.40, 0.22),   # has slow_icir_63
        ("factor_e", "Factor E — negative", -0.005, -0.30, None),
    ]
    for fname, desc, ic, sharpe, slow63 in factors:
        conn.execute(
            "INSERT INTO hypotheses (factor_name, description) VALUES (?, ?)",
            (fname, desc),
        )
        hyp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO implementations (hypothesis_id, code, notes) VALUES (?, ?, ?)",
            (hyp_id, _FACTOR_CODE, f"factor_type=momentum"),
        )
        impl_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO backtest_results
               (implementation_id, ic, icir, net_ic, sharpe, turnover,
                ic_horizon_1, slow_icir_63, regime, universe, oos_start)
               VALUES (?, ?, 0.10, ?, ?, 0.05, ?, ?, 'all', 'test', '2018-01-01')""",
            (impl_id, ic, ic * 0.99, sharpe, ic, slow63),
        )

    conn.commit()
    conn.close()
    yield path
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# plot_multiple_testing tests
# ---------------------------------------------------------------------------

class TestPlotMultipleTesting:
    """Tests for Chart 15 — significance chart."""

    def test_returns_figure(self, kb_with_result):
        fig = plot_multiple_testing(kb_with_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_kb_returns_figure(self, kb_empty):
        """KB with no positive-IC factors should still return a Figure."""
        fig = plot_multiple_testing(kb_empty)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_factors_renders(self, kb_multiple_factors):
        fig = plot_multiple_testing(kb_multiple_factors)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self, kb_with_result):
        fig = plot_multiple_testing(kb_with_result)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_only_positive_ic_factors_shown(self, kb_multiple_factors):
        """Factor E has negative IC and should NOT appear in the chart."""
        fig = plot_multiple_testing(kb_multiple_factors)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # 'factor_e' or its description should not appear
        for lbl in labels:
            assert "factor_e" not in lbl.lower() and "negative" not in lbl.lower(), (
                f"Negative-IC factor appeared in chart: {lbl}"
            )
        plt.close(fig)

    def test_t_stat_formula_with_sharpe(self):
        """t = sharpe/√252 × √N matches manually computed value."""
        sharpe = 1.1
        n_oos = 1762
        expected_t = (sharpe / math.sqrt(252)) * math.sqrt(n_oos)
        path = _make_temp_kb(ic=0.02, sharpe=sharpe, slow_icir_63=None)
        try:
            fig = plot_multiple_testing(path, n_oos_days=n_oos)
            ax = fig.axes[0]
            # Read the annotated t-value text
            texts = [t.get_text() for t in ax.texts]
            # At least one text should contain something close to expected_t
            t_texts = [t for t in texts if t.startswith("t=")]
            assert len(t_texts) >= 1
            parsed_t = float(t_texts[0].replace("t=", ""))
            assert abs(parsed_t - expected_t) < 0.01, (
                f"t-stat annotation {parsed_t:.3f} ≠ expected {expected_t:.3f}"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_slow_t_stat_preferred_when_available(self):
        """When slow_icir_63 is available, slow formula should be used instead of fast."""
        slow63 = 0.22
        n_oos = 1762
        # The chart uses floor(n_oos / 63) for n_windows, matching compute_slow_t_stat
        n_windows = math.floor(n_oos / 63)
        expected_t_slow = slow63 * math.sqrt(n_windows)
        # Fast formula would give a much higher t-stat with sharpe=3.0
        sharpe_high = 3.0
        expected_t_fast = (sharpe_high / math.sqrt(252)) * math.sqrt(n_oos)

        path = _make_temp_kb(ic=0.01, sharpe=sharpe_high, slow_icir_63=slow63)
        try:
            fig = plot_multiple_testing(path, n_oos_days=n_oos)
            ax = fig.axes[0]
            t_texts = [t.get_text() for t in ax.texts if t.get_text().startswith("t=")]
            assert len(t_texts) >= 1
            parsed_t = float(t_texts[0].replace("t=", ""))
            # Annotations are formatted to 2dp, so allow ±0.005 rounding tolerance
            assert abs(parsed_t - expected_t_slow) < 0.01, (
                f"Expected slow t={expected_t_slow:.3f} (floor-based), got {parsed_t:.3f}"
            )
            assert abs(parsed_t - expected_t_fast) > 0.5, (
                "Fast t-stat was used instead of slow t-stat"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_bhy_threshold_in_title(self, kb_multiple_factors):
        """The figure title should mention the BHY threshold value."""
        fig = plot_multiple_testing(kb_multiple_factors)
        title = fig.axes[0].get_title()
        assert "BHY" in title, f"'BHY' not found in title: {title!r}"
        assert "t≥" in title, f"'t≥' not found in title: {title!r}"
        plt.close(fig)

    def test_never_calls_plt_show(self, kb_with_result, monkeypatch):
        """Chart functions must not call plt.show() — they return a Figure."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        fig = plot_multiple_testing(kb_with_result)
        assert called == [], "plot_multiple_testing called plt.show() — it must not"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_ic_by_period tests
# ---------------------------------------------------------------------------

class TestSlowSignalDetection:
    """
    The slow-signal criterion is |IC@63d| > |IC@1d|, NOT IC@63d > IC@1d.

    A factor with IC going −0.01 → −0.04 is a slow short signal and must be
    flagged as slow just as reliably as one going +0.01 → +0.04.
    """

    def _make_kb_with_decay(
        self,
        ic1: float,
        ic63: float,
        *,
        factor_name: str = "decay_test",
    ) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        path = tmp.name
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_name TEXT, description TEXT NOT NULL,
                mechanism_score INTEGER DEFAULT 3,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE implementations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER, code TEXT, notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                implementation_id INTEGER,
                ic REAL, icir REAL, net_ic REAL, sharpe REAL,
                max_drawdown REAL, turnover REAL, decay_halflife REAL,
                ic_horizon_1 REAL, ic_horizon_5 REAL, ic_horizon_21 REAL,
                ic_horizon_63 REAL, ic_decay_json TEXT,
                slow_icir_21 REAL, slow_icir_63 REAL,
                regime TEXT, universe TEXT, oos_start TEXT, oos_end TEXT,
                cost_bps REAL, gate_level INTEGER, passed_gate INTEGER,
                mlflow_run_id TEXT, notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE factor_correlations (
                factor_a TEXT, factor_b TEXT, correlation REAL,
                period TEXT, universe TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (factor_a, factor_b, period, universe)
            );
        """)
        conn.execute(
            "INSERT INTO hypotheses (factor_name, description) VALUES (?, ?)",
            (factor_name, f"Decay test factor {factor_name}"),
        )
        conn.execute(
            "INSERT INTO implementations (hypothesis_id, code, notes) VALUES (1, ?, ?)",
            (_FACTOR_CODE, "factor_type=momentum"),
        )
        conn.execute(
            """INSERT INTO backtest_results
               (implementation_id, ic, icir, net_ic, sharpe, turnover,
                ic_horizon_1, ic_horizon_5, ic_horizon_21, ic_horizon_63,
                regime, universe, oos_start)
               VALUES (1, ?, 0.10, ?, 0.80, 0.05, ?, ?, ?, ?, 'all', 'test', '2018-01-01')""",
            (ic1, ic1 * 0.99, ic1, ic1 * 0.8, ic1 * 0.6, ic63),
        )
        conn.commit()
        conn.close()
        return path

    def test_heatmap_flags_negative_slow_signal(self):
        """IC: −0.01 at h=1, −0.04 at h=63 → should be flagged as slow (orange outline)."""
        path = self._make_kb_with_decay(ic1=-0.01, ic63=-0.04)
        try:
            import matplotlib
            matplotlib.use("Agg")
            from qframe.viz.charts import plot_ic_decay_heatmap
            fig = plot_ic_decay_heatmap(path)
            ax = fig.axes[0]
            # An orange Rectangle patch should have been added
            orange_patches = [
                p for p in ax.patches
                if hasattr(p, "get_edgecolor") and
                np.allclose(p.get_edgecolor()[:3],
                            matplotlib.colors.to_rgb("darkorange"), atol=0.05)
            ]
            assert len(orange_patches) >= 1, (
                "Negative slow signal (IC: −0.01→−0.04) was not flagged with orange outline"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_heatmap_does_not_flag_positive_decay(self):
        """IC: +0.04 at h=1, +0.01 at h=63 → signal decays; must NOT be flagged."""
        path = self._make_kb_with_decay(ic1=0.04, ic63=0.01)
        try:
            import matplotlib
            matplotlib.use("Agg")
            from qframe.viz.charts import plot_ic_decay_heatmap
            fig = plot_ic_decay_heatmap(path)
            ax = fig.axes[0]
            orange_patches = [
                p for p in ax.patches
                if hasattr(p, "get_edgecolor") and
                np.allclose(p.get_edgecolor()[:3],
                            matplotlib.colors.to_rgb("darkorange"), atol=0.05)
            ]
            assert len(orange_patches) == 0, (
                "Decaying-IC factor (+0.04→+0.01) was incorrectly flagged as slow"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_heatmap_flags_positive_slow_signal(self):
        """IC: +0.01 at h=1, +0.04 at h=63 → positive slow signal; should be flagged."""
        path = self._make_kb_with_decay(ic1=0.01, ic63=0.04)
        try:
            import matplotlib
            matplotlib.use("Agg")
            from qframe.viz.charts import plot_ic_decay_heatmap
            fig = plot_ic_decay_heatmap(path)
            ax = fig.axes[0]
            orange_patches = [
                p for p in ax.patches
                if hasattr(p, "get_edgecolor") and
                np.allclose(p.get_edgecolor()[:3],
                            matplotlib.colors.to_rgb("darkorange"), atol=0.05)
            ]
            assert len(orange_patches) >= 1, (
                "Positive slow signal (IC: +0.01→+0.04) was not flagged"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_heatmap_does_not_flag_cross_sign(self):
        """
        IC@1d = −0.01, IC@63d = +0.04 — opposite signs.

        Option A: must NOT be flagged as a slow signal. A sign flip between
        horizons means two different economic effects are present (e.g. short-term
        reversal + long-term drift), not a clean slow/mean-reversion signal.
        """
        path = self._make_kb_with_decay(ic1=-0.01, ic63=0.04)
        try:
            import matplotlib
            matplotlib.use("Agg")
            from qframe.viz.charts import plot_ic_decay_heatmap
            fig = plot_ic_decay_heatmap(path)
            ax = fig.axes[0]
            orange_patches = [
                p for p in ax.patches
                if hasattr(p, "get_edgecolor") and
                np.allclose(p.get_edgecolor()[:3],
                            matplotlib.colors.to_rgb("darkorange"), atol=0.05)
            ]
            assert len(orange_patches) == 0, (
                "Cross-sign IC (−0.01 → +0.04) was incorrectly flagged as slow (Option A violation)"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_decay_curve_labels_negative_slow_as_slow(self, prices_parquet):
        """plot_ic_decay_curves must label a negative slow factor with [SLOW]."""
        path = self._make_kb_with_decay(ic1=-0.01, ic63=-0.04)
        try:
            import json
            import sqlite3 as _sq
            # Inject a dense ic_decay_json so the chart uses it
            decay = {str(h): -0.01 - 0.0005 * h for h in range(1, 64)}
            conn = _sq.connect(path)
            conn.execute(
                "UPDATE backtest_results SET ic_decay_json = ? WHERE id = 1",
                (json.dumps(decay),),
            )
            conn.commit()
            conn.close()

            from qframe.viz.charts import plot_ic_decay_curves
            fig = plot_ic_decay_curves(path, top_n=5)
            ax = fig.axes[0]
            legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
            slow_labels = [t for t in legend_texts if "[SLOW]" in t]
            assert len(slow_labels) >= 1, (
                f"Negative slow factor not labelled [SLOW]. Legend: {legend_texts}"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)


class TestPlotICByPeriod:
    """Tests for Chart 14 — temporal stability bar chart."""

    def test_returns_figure(self, kb_with_result, prices_parquet):
        fig = plot_ic_by_period(
            kb_with_result, impl_id=1, prices_path=prices_parquet,
            oos_start="2018-01-01",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self, kb_with_result, prices_parquet):
        fig = plot_ic_by_period(kb_with_result, impl_id=1, prices_path=prices_parquet)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_error_figure_when_no_impl_id(self, kb_with_result, prices_parquet):
        """Neither impl_id nor factor_name supplied → error figure, not exception."""
        fig = plot_ic_by_period(
            kb_with_result, prices_path=prices_parquet,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_error_figure_when_missing_prices(self, kb_with_result):
        """Non-existent prices path → error figure (not an exception)."""
        fig = plot_ic_by_period(
            kb_with_result, impl_id=1,
            prices_path="/nonexistent/path/prices.parquet",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_error_figure_when_impl_id_not_found(self, kb_with_result, prices_parquet):
        """Unknown impl_id → error figure, not exception."""
        fig = plot_ic_by_period(
            kb_with_result, impl_id=9999, prices_path=prices_parquet,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_factor_name_lookup(self, prices_parquet):
        """Can look up factor by factor_name instead of impl_id."""
        path = _make_temp_kb()
        try:
            fig = plot_ic_by_period(
                path, factor_name="test_momentum", prices_path=prices_parquet,
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_bars_drawn(self, kb_with_result, prices_parquet):
        """The figure should contain at least one bar (BarContainer)."""
        fig = plot_ic_by_period(
            kb_with_result, impl_id=1, prices_path=prices_parquet,
            oos_start="2018-01-01",
        )
        ax = fig.axes[0]
        bar_containers = [c for c in ax.containers
                          if hasattr(c, "__class__") and
                          "Bar" in type(c).__name__]
        assert len(bar_containers) >= 1, "Expected at least one bar container in axes"
        plt.close(fig)

    def test_x_axis_labels_are_year_ranges(self, kb_with_result, prices_parquet):
        """X-tick labels should look like year ranges (e.g. '2018–2020')."""
        fig = plot_ic_by_period(
            kb_with_result, impl_id=1, prices_path=prices_parquet,
            oos_start="2018-01-01", period_years=2.0,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert len(labels) >= 1
        for lbl in labels:
            # Should contain a 4-digit year somewhere
            digits = [c for c in lbl if c.isdigit()]
            assert len(digits) >= 4, f"Label '{lbl}' doesn't look like a year range"
        plt.close(fig)

    def test_factor_code_with_markdown_fences(self, prices_parquet):
        """Factor code stored with ```python fences should still execute correctly."""
        fenced_code = "```python\nreturn prices.pct_change(21)\n```"
        path = _make_temp_kb(factor_code=fenced_code)
        try:
            fig = plot_ic_by_period(path, impl_id=1, prices_path=prices_parquet)
            assert isinstance(fig, plt.Figure)
            # A successful render has the factor name in the title and shows bars.
            # An error render has "failed" or "execution error" in the title.
            title = fig.axes[0].get_title().lower()
            assert "failed" not in title and "execution" not in title, (
                f"Unexpected error title for fenced code: {title!r}"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_broken_factor_code_returns_error_figure(self, prices_parquet):
        """Factor code that raises an exception → error figure, not a crash."""
        path = _make_temp_kb(factor_code="raise ValueError('deliberate test error')")
        try:
            fig = plot_ic_by_period(path, impl_id=1, prices_path=prices_parquet)
            assert isinstance(fig, plt.Figure)
            # Title should indicate failure
            title = fig.axes[0].get_title().lower()
            assert "error" in title or "failed" in title, (
                f"Expected error title, got: {title!r}"
            )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_oos_start_respected(self, prices_parquet):
        """With oos_start beyond all available dates, return an informative figure."""
        path = _make_temp_kb()
        try:
            fig = plot_ic_by_period(
                path, impl_id=1, prices_path=prices_parquet,
                oos_start="2099-01-01",
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_never_calls_plt_show(self, kb_with_result, prices_parquet, monkeypatch):
        """Chart functions must not call plt.show()."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        fig = plot_ic_by_period(
            kb_with_result, impl_id=1, prices_path=prices_parquet,
        )
        assert called == [], "plot_ic_by_period called plt.show() — it must not"
        plt.close(fig)

    def test_different_horizons_produce_different_titles(self, kb_with_result, prices_parquet):
        """Changing horizon should change the figure title."""
        fig1 = plot_ic_by_period(kb_with_result, impl_id=1,
                                  prices_path=prices_parquet, horizon=1)
        fig5 = plot_ic_by_period(kb_with_result, impl_id=1,
                                  prices_path=prices_parquet, horizon=5)
        title1 = fig1.axes[0].get_title()
        title5 = fig5.axes[0].get_title()
        assert title1 != title5, "Different horizons should produce different titles"
        plt.close(fig1)
        plt.close(fig5)
