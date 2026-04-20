"""
Unit tests for the Phase 1 pipeline.

Tests executor and models without making any LLM calls.
LLM agents are tested via the demo notebook (integration test).
"""
import numpy as np
import pandas as pd
import pytest

from qframe.pipeline.executor import (
    extract_function,
    make_factor_fn,
    run_factor_with_timeout,
    validate_factor_output,
)
from qframe.pipeline.agents.implementation import ImplementationAgent
from qframe.pipeline.models import (
    HypothesisSpec,
    IterationResult,
    ResearchSpec,
    VERDICT_PASS,
    VERDICT_FAIL,
    VERDICT_SKIP,
    VERDICT_ERROR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    tickers = [f"S{i}" for i in range(20)]
    ret = pd.DataFrame(rng.normal(0.0005, 0.02, (100, 20)), index=dates, columns=tickers)
    return (1 + ret).cumprod()


@pytest.fixture
def hypothesis():
    return HypothesisSpec(
        name="test_momentum",
        description="21-day momentum factor",
        rationale="Underreaction",
        mechanism_score=4,
        factor_type="momentum",
    )


# ---------------------------------------------------------------------------
# executor: extract_function
# ---------------------------------------------------------------------------

class TestExtractFunction:
    def test_strips_markdown_fence(self):
        code = "```python\ndef factor(prices):\n    return prices\n```"
        cleaned = extract_function(code)
        assert "```" not in cleaned
        assert "def factor" in cleaned

    def test_passthrough_clean_code(self):
        code = "def factor(prices):\n    return prices"
        assert extract_function(code) == code


# ---------------------------------------------------------------------------
# executor: make_factor_fn
# ---------------------------------------------------------------------------

class TestMakeFactorFn:
    def test_returns_callable(self):
        code = "def factor(prices):\n    return prices.pct_change(21)"
        fn = make_factor_fn(code)
        assert callable(fn)

    def test_raises_if_no_factor_defined(self):
        code = "x = 1 + 1"
        with pytest.raises(ValueError, match="factor"):
            make_factor_fn(code)

    def test_raises_on_syntax_error(self):
        code = "def factor(prices)\n    return prices"
        with pytest.raises(SyntaxError):
            make_factor_fn(code)

    def test_executes_correctly(self, prices):
        code = "def factor(prices):\n    return -prices.pct_change(21)"
        fn = make_factor_fn(code)
        result = fn(prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == prices.shape

    def test_strips_markdown_before_exec(self, prices):
        code = "```python\ndef factor(prices):\n    return prices.pct_change(5)\n```"
        fn = make_factor_fn(code)
        result = fn(prices)
        assert result.shape == prices.shape


# ---------------------------------------------------------------------------
# executor: run_factor_with_timeout
# ---------------------------------------------------------------------------

class TestRunFactorWithTimeout:
    def test_runs_fast_factor(self, prices):
        fn = make_factor_fn("def factor(prices):\n    return prices.pct_change(21)")
        result = run_factor_with_timeout(fn, prices, timeout=10)
        assert result.shape == prices.shape

    def test_raises_on_timeout(self, prices):
        code = "def factor(prices):\n    import time; time.sleep(999); return prices"
        fn = make_factor_fn(code)
        with pytest.raises(TimeoutError):
            run_factor_with_timeout(fn, prices, timeout=1)

    def test_propagates_runtime_error(self, prices):
        code = "def factor(prices):\n    raise ValueError('bad input')"
        fn = make_factor_fn(code)
        with pytest.raises(ValueError, match="bad input"):
            run_factor_with_timeout(fn, prices)


# ---------------------------------------------------------------------------
# executor: validate_factor_output
# ---------------------------------------------------------------------------

class TestValidateFactorOutput:
    def test_passes_valid_output(self, prices):
        factor_df = prices.pct_change(21)
        validate_factor_output(factor_df, prices)  # should not raise

    def test_raises_on_wrong_type(self, prices):
        with pytest.raises(ValueError, match="pd.DataFrame"):
            validate_factor_output(prices.values, prices)

    def test_raises_on_shape_mismatch(self, prices):
        bad = prices.iloc[:, :5]
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_factor_output(bad, prices)

    def test_raises_on_all_nan(self, prices):
        all_nan = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        with pytest.raises(ValueError, match="NaN"):
            validate_factor_output(all_nan, prices)

    def test_raises_on_high_nan_fraction(self, prices):
        """A factor with 60% NaN (> 50% threshold) should be rejected."""
        factor_df = prices.pct_change(21).copy()
        # Set 60% of cells to NaN
        rng = np.random.default_rng(99)
        mask = rng.random(prices.shape) < 0.60
        factor_df[mask] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_factor_output(factor_df, prices)

    def test_passes_factor_with_acceptable_nan_fraction(self, prices):
        """A factor with 30% NaN should pass the NaN threshold check."""
        factor_df = prices.pct_change(21).copy()
        # Introduce only 30% NaN
        rng = np.random.default_rng(7)
        mask = rng.random(prices.shape) < 0.30
        factor_df[mask] = np.nan
        validate_factor_output(factor_df, prices)  # should not raise

    def test_raises_on_constant_factor(self, prices):
        """A factor with zero cross-sectional variance on all dates should be rejected."""
        constant = pd.DataFrame(
            1.0,
            index=prices.index,
            columns=prices.columns,
        )
        with pytest.raises(ValueError, match="cross-sectional variance"):
            validate_factor_output(constant, prices)

    def test_raises_on_infinite_values(self, prices):
        """A factor containing inf should be rejected."""
        factor_df = prices.pct_change(21).copy()
        factor_df.iloc[5, 3] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            validate_factor_output(factor_df, prices)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class TestResearchSpec:
    def test_defaults(self):
        spec = ResearchSpec(factor_domain="momentum")
        assert spec.oos_start == "2018-01-01"
        assert len(spec.constraints) > 0


class TestIterationResult:
    def test_error_result(self, hypothesis):
        result = IterationResult(
            hypothesis=hypothesis,
            code="def factor(p): return p",
            wf_result=None,
            analysis="failed",
            verdict=VERDICT_ERROR,
            kb_hypothesis_id=1,
            kb_implementation_id=1,
            kb_result_id=None,
            error="some error",
        )
        assert result.verdict == VERDICT_ERROR
        assert result.error == "some error"

    def test_print_summary_with_empty_error_does_not_raise(self, hypothesis, capsys):
        """print_summary should not crash when error is an empty string.

        An empty string is falsy, so the error block is skipped entirely —
        no "(empty error)" is printed, but no IndexError either.
        """
        result = IterationResult(
            hypothesis=hypothesis,
            code="def factor(p): return p",
            wf_result=None,
            analysis="execution failed",
            verdict=VERDICT_ERROR,
            kb_hypothesis_id=1,
            kb_implementation_id=1,
            kb_result_id=None,
            error="",          # empty string — falsy, block skipped
        )
        result.print_summary()   # must not raise IndexError
        captured = capsys.readouterr()
        # error="" is falsy → error section not printed at all
        assert "Error:" not in captured.out

    def test_print_summary_with_whitespace_only_error(self, hypothesis, capsys):
        """print_summary should print '(empty error)' for whitespace-only strings.

        A whitespace-only string like '   \\n   ' is truthy (non-empty), so the
        error block IS entered — but after filtering out blank lines, the list is
        empty, and the fallback '(empty error)' label is used instead of crashing
        with IndexError.
        """
        result = IterationResult(
            hypothesis=hypothesis,
            code="def factor(p): return p",
            wf_result=None,
            analysis="execution failed",
            verdict=VERDICT_ERROR,
            kb_hypothesis_id=1,
            kb_implementation_id=1,
            kb_result_id=None,
            error="   \n   ",   # truthy but all-whitespace → empty after strip
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "(empty error)" in captured.out

    def test_print_summary_shows_last_error_line(self, hypothesis, capsys):
        """print_summary should display the last non-empty line of the traceback."""
        result = IterationResult(
            hypothesis=hypothesis,
            code="def factor(p): return p",
            wf_result=None,
            analysis="execution failed",
            verdict=VERDICT_ERROR,
            kb_hypothesis_id=1,
            kb_implementation_id=1,
            kb_result_id=None,
            error="Traceback (most recent call last):\n  File ...\nValueError: bad factor",
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "ValueError: bad factor" in captured.out


# ---------------------------------------------------------------------------
# implementation agent compatibility
# ---------------------------------------------------------------------------

class TestImplementationAgentCompatibility:
    """
    Tests for _call_ollama backward-compatibility logic.

    The allweather env may not have `ollama` installed, so these tests inject a
    mock module directly into `qframe.pipeline.agents.implementation.ollama`
    before patching its `generate` attribute.  `raising=False` on setattr
    allows creating the attribute on the mock module even when it is absent.
    """

    @pytest.fixture(autouse=True)
    def _inject_mock_ollama(self, monkeypatch):
        """Ensure a non-None ollama module is in place regardless of installation."""
        import types
        import qframe.pipeline.agents.implementation as impl_mod

        if impl_mod.ollama is None:
            mock_mod = types.ModuleType("ollama")
            monkeypatch.setattr(impl_mod, "ollama", mock_mod)

    def test_call_ollama_retries_without_timeout_on_legacy_sdk(self, monkeypatch):
        import qframe.pipeline.agents.implementation as impl_mod

        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            if "timeout" in kwargs:
                raise TypeError("Client.generate() got an unexpected keyword argument 'timeout'")
            return {"response": "def factor(prices):\n    return prices"}

        monkeypatch.setattr(impl_mod.ollama, "generate", fake_generate, raising=False)

        agent = ImplementationAgent(timeout=120)
        out = agent._call_ollama("prompt", temperature=0.2)
        assert "def factor" in out["response"]
        assert len(calls) == 2
        assert "timeout" in calls[0]
        assert "timeout" not in calls[1]

    def test_call_ollama_preserves_other_type_errors(self, monkeypatch):
        import qframe.pipeline.agents.implementation as impl_mod

        def fake_generate(**kwargs):
            raise TypeError("some other type error")

        monkeypatch.setattr(impl_mod.ollama, "generate", fake_generate, raising=False)

        agent = ImplementationAgent(timeout=120)
        with pytest.raises(TypeError, match="some other type error"):
            agent._call_ollama("prompt", temperature=0.2)


# ---------------------------------------------------------------------------
# Look-ahead bias guard regression tests (§1.4 of 2026-04-19 plan)
# ---------------------------------------------------------------------------

class TestCheckLookaheadBias:
    """Regression tests for check_lookahead_bias in executor.py."""

    @pytest.fixture
    def long_prices(self):
        """400 business-day price panel — long enough for warm-up + boundary check."""
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2019-01-01", periods=400)
        tickers = [f"S{i}" for i in range(20)]
        ret = pd.DataFrame(rng.normal(0.0005, 0.02, (400, 20)), index=dates, columns=tickers)
        return (1 + ret).cumprod()

    def test_catches_shift_minus_1(self, long_prices):
        """A factor using shift(-1) (look-ahead) should raise ValueError."""
        from qframe.pipeline.executor import check_lookahead_bias, make_factor_fn

        code = "def factor(prices):\n    return prices.pct_change(5).shift(-1)"
        fn = make_factor_fn(code)
        full = fn(long_prices)
        with pytest.raises(ValueError, match="look-ahead bias"):
            check_lookahead_bias(fn, long_prices, full, name="shift_minus_1")

    def test_passes_clean_rolling_factor(self, long_prices):
        """A clean rolling-mean factor should not raise."""
        from qframe.pipeline.executor import check_lookahead_bias, make_factor_fn

        code = "def factor(prices):\n    return prices.rolling(20).mean()"
        fn = make_factor_fn(code)
        full = fn(long_prices)
        # Should complete without raising
        check_lookahead_bias(fn, long_prices, full, name="clean_rolling")


# ---------------------------------------------------------------------------
# Signal novelty filter regression tests (§1.4 of 2026-04-19 plan)
# ---------------------------------------------------------------------------

class TestSignalNovelty:
    """Regression tests for _check_signal_novelty in loop.py."""

    @pytest.fixture
    def kb_path(self, tmp_path):
        """Temporary KB with schema initialised, ready for use."""
        from qframe.knowledge_base.db import KnowledgeBase
        path = str(tmp_path / "test_novelty.db")
        KnowledgeBase(path).init_schema()   # create all tables before the test runs
        return path

    def _make_signal(self, seed: int = 42, periods: int = 300, tickers: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2018-01-02", periods=periods)
        cols = [f"S{i}" for i in range(tickers)]
        data = rng.standard_normal((periods, tickers))
        return pd.DataFrame(data, index=dates, columns=cols)

    def test_rejects_highly_correlated_signal(self, kb_path):
        """A signal almost identical to a cached factor should be flagged as duplicate."""
        from qframe.knowledge_base.db import KnowledgeBase
        from qframe.pipeline.loop import PipelineLoop

        kb = KnowledgeBase(kb_path)
        signal = self._make_signal(seed=1)

        # Cache the signal in KB as a positive-IC factor
        hyp_id = kb.add_hypothesis(
            factor_name="cached_factor",
            description="test",
            rationale="test",
            mechanism_score=3,
        )
        impl_id = kb.add_implementation(hyp_id, code="def factor(p): pass", notes="")
        kb.log_result(impl_id, {
            "ic": 0.05, "icir": 0.3, "sharpe": 1.5, "max_drawdown": -0.1,
            "turnover": 0.02, "oos_start": "2018-01-02", "oos_end": "2023-12-31",
            "gate_level": 1, "passed_gate": 1,
            "signal_cache_json": signal.to_json(orient="split", date_format="iso"),
        })

        # Build a nearly-identical signal (same + tiny noise)
        rng = np.random.default_rng(99)
        new_signal = signal + rng.normal(0, 1e-6, signal.shape)

        # Build a minimal PipelineLoop just to access _check_signal_novelty
        loop = PipelineLoop.__new__(PipelineLoop)
        loop.kb = kb
        loop.oos_start = "2018-01-02"

        is_dup, max_corr, similar_name = loop._check_signal_novelty(new_signal)
        assert is_dup is True, f"Expected duplicate, got max_corr={max_corr:.4f}"
        assert max_corr > 0.99
        assert similar_name == "cached_factor"

    def test_aligns_different_date_ranges(self, kb_path):
        """Novelty check must align signals on date intersection, not raw array position."""
        from qframe.knowledge_base.db import KnowledgeBase
        from qframe.pipeline.loop import PipelineLoop

        kb = KnowledgeBase(kb_path)
        # Cached signal: 2018–2022 (shorter range)
        old_signal = self._make_signal(seed=5, periods=250)

        hyp_id = kb.add_hypothesis(
            factor_name="short_range_factor",
            description="test",
            rationale="test",
            mechanism_score=3,
        )
        impl_id = kb.add_implementation(hyp_id, code="def factor(p): pass", notes="")
        kb.log_result(impl_id, {
            "ic": 0.04, "icir": 0.25, "sharpe": 1.2, "max_drawdown": -0.12,
            "turnover": 0.02, "oos_start": "2018-01-02", "oos_end": "2022-12-31",
            "gate_level": 1, "passed_gate": 1,
            "signal_cache_json": old_signal.to_json(orient="split", date_format="iso"),
        })

        # New signal: 2018–2024 (longer range), completely uncorrelated
        rng = np.random.default_rng(999)
        new_dates = pd.bdate_range("2018-01-02", periods=500)
        new_signal = pd.DataFrame(
            rng.standard_normal((500, 20)),
            index=new_dates,
            columns=[f"S{i}" for i in range(20)],
        )

        loop = PipelineLoop.__new__(PipelineLoop)
        loop.kb = kb
        loop.oos_start = "2018-01-02"

        # With the fixed alignment logic, correlation is computed on date intersection.
        # An uncorrelated signal should NOT be flagged as duplicate.
        is_dup, max_corr, _ = loop._check_signal_novelty(new_signal)
        assert is_dup is False, (
            f"Uncorrelated signal over different date range should not be a duplicate. "
            f"max_corr={max_corr:.4f}"
        )


# ---------------------------------------------------------------------------
# SKIP verdict — regression tests for guard-rejection taxonomy (2026-04-20)
#
# Background: prior to 2026-04-20, the DUPLICATE (novelty filter) and
# PRE-GATE FAILED branches in loop.py used VERDICT_ERROR, making healthy
# guard rejections indistinguishable from real crashes in the summary.
# These tests pin the taxonomy: guard rejections = SKIP, real crashes = ERROR.
# ---------------------------------------------------------------------------

class TestSkipVerdict:
    def test_verdict_skip_constant_is_defined(self):
        """VERDICT_SKIP must be exported from qframe.pipeline.models with value 'SKIP'."""
        from qframe.pipeline.models import VERDICT_SKIP as V
        assert V == "SKIP"

    def test_loop_duplicate_branch_uses_skip_verdict(self):
        """The DUPLICATE rejection path in loop.py must return verdict=VERDICT_SKIP."""
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "src" / "qframe" / "pipeline" / "loop.py"
        text = src.read_text()
        # Locate the DUPLICATE block and check it uses VERDICT_SKIP (not VERDICT_ERROR).
        dup_idx = text.index("DUPLICATE: signal correlation")
        # Grab the next ~30 lines after the message — enough to cover the return statement.
        window = text[dup_idx : dup_idx + 2000]
        assert "verdict=VERDICT_SKIP" in window, (
            "DUPLICATE branch in loop.py must use VERDICT_SKIP, not VERDICT_ERROR. "
            "This was the 2026-04-20 taxonomy fix."
        )

    def test_loop_pregate_branch_uses_skip_verdict(self):
        """The PRE-GATE FAILED rejection path in loop.py must return verdict=VERDICT_SKIP."""
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "src" / "qframe" / "pipeline" / "loop.py"
        text = src.read_text()
        pg_idx = text.index("PRE-GATE FAILED:")
        window = text[pg_idx : pg_idx + 2000]
        assert "verdict=VERDICT_SKIP" in window, (
            "PRE-GATE FAILED branch in loop.py must use VERDICT_SKIP, not VERDICT_ERROR. "
            "This was the 2026-04-20 taxonomy fix."
        )

    def test_iteration_result_accepts_skip_verdict(self, hypothesis):
        """IterationResult must accept VERDICT_SKIP as a valid verdict."""
        r = IterationResult(
            hypothesis=hypothesis,
            code="def factor(p): return p",
            wf_result=None,
            analysis="DUPLICATE: …",
            verdict=VERDICT_SKIP,
            kb_hypothesis_id=1,
            kb_implementation_id=1,
            kb_result_id=None,
            error="DUPLICATE: …",
        )
        assert r.verdict == "SKIP"
        assert r.verdict != VERDICT_ERROR


# ---------------------------------------------------------------------------
# #12 — Sealed hold-out enforcement (2026-04-20)
#
# WalkForwardValidator must raise RuntimeError when price data extends past
# HOLDOUT_START unless allow_holdout=True or QFRAME_UNSEAL_HOLDOUT=1 is set.
# ---------------------------------------------------------------------------

class TestHoldoutEnforcement:
    """Tests for the sealed hold-out guard in WalkForwardValidator."""

    @pytest.fixture
    def prices_before_holdout(self):
        """Prices that end well before HOLDOUT_START=2024-06-01."""
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2018-01-01", "2024-05-30")
        tickers = [f"S{i}" for i in range(10)]
        ret = pd.DataFrame(rng.normal(0, 0.01, (len(dates), 10)), index=dates, columns=tickers)
        return (1 + ret).cumprod()

    @pytest.fixture
    def prices_past_holdout(self):
        """Prices that extend past HOLDOUT_START=2024-06-01."""
        rng = np.random.default_rng(8)
        dates = pd.bdate_range("2018-01-01", "2024-12-31")
        tickers = [f"S{i}" for i in range(10)]
        ret = pd.DataFrame(rng.normal(0, 0.01, (len(dates), 10)), index=dates, columns=tickers)
        return (1 + ret).cumprod()

    def _trivial_factor(self, prices):
        return prices.pct_change(5)

    def test_raises_when_prices_include_holdout_by_default(self, prices_past_holdout):
        """Default allow_holdout=False must raise when data extends past HOLDOUT_START."""
        from qframe.factor_harness.walkforward import WalkForwardValidator
        validator = WalkForwardValidator(
            factor_fn=self._trivial_factor,
            oos_start="2018-01-01",
        )
        with pytest.raises(RuntimeError, match="hold-out"):
            validator.run(prices_past_holdout)

    def test_no_raise_when_data_ends_before_holdout(self, prices_before_holdout):
        """Validator must not raise when data ends before HOLDOUT_START."""
        from qframe.factor_harness.walkforward import WalkForwardValidator
        validator = WalkForwardValidator(
            factor_fn=self._trivial_factor,
            oos_start="2018-01-01",
        )
        result = validator.run(prices_before_holdout)  # must not raise
        assert result is not None

    def test_allow_holdout_true_bypasses_guard(self, prices_past_holdout):
        """Passing allow_holdout=True must allow holdout data through."""
        from qframe.factor_harness.walkforward import WalkForwardValidator
        validator = WalkForwardValidator(
            factor_fn=self._trivial_factor,
            oos_start="2018-01-01",
            allow_holdout=True,
        )
        result = validator.run(prices_past_holdout)  # must not raise
        assert result is not None

    def test_env_var_bypasses_guard(self, prices_past_holdout, monkeypatch):
        """QFRAME_UNSEAL_HOLDOUT=1 env var must bypass the guard."""
        monkeypatch.setenv("QFRAME_UNSEAL_HOLDOUT", "1")
        from qframe.factor_harness import walkforward as wf_mod
        import importlib
        # Re-instantiate validator after env var is set — __init__ reads the env var
        validator = wf_mod.WalkForwardValidator(
            factor_fn=self._trivial_factor,
            oos_start="2018-01-01",
        )
        result = validator.run(prices_past_holdout)  # must not raise
        assert result is not None

    def test_run_py_load_prices_truncates_at_holdout(self, monkeypatch, tmp_path):
        """load_prices() must truncate data at HOLDOUT_START - 1 day by default."""
        import pandas as pd
        from qframe.factor_harness.walkforward import HOLDOUT_START

        # Build a fake parquet with data extending past HOLDOUT_START
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2018-01-01", "2024-12-31")
        tickers = [f"S{i}" for i in range(5)]
        ret = pd.DataFrame(rng.normal(0, 0.01, (len(dates), 5)), index=dates, columns=tickers)
        fake_prices = (1 + ret).cumprod()
        fake_cache = tmp_path / "sp500_close.parquet"
        fake_prices.to_parquet(fake_cache)

        # Patch the module-level _PRICE_CACHE to point at our fake file
        import qframe.pipeline.run as run_mod
        monkeypatch.setattr(run_mod, "_PRICE_CACHE", fake_cache)
        monkeypatch.delenv("QFRAME_UNSEAL_HOLDOUT", raising=False)

        prices = run_mod.load_prices()
        cutoff = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
        assert prices.index[-1] <= cutoff, (
            f"load_prices() must truncate at or before {cutoff.date()}, "
            f"but last date is {prices.index[-1].date()}"
        )


# ---------------------------------------------------------------------------
# #27 — Domain round-robin after 3 consecutive SKIPs (2026-04-20)
#
# run_n() must pivot to the next domain after _SKIP_ROTATION_THRESHOLD
# consecutive SKIP verdicts to prevent getting stuck in a saturated domain.
# ---------------------------------------------------------------------------

class TestDomainRotation:
    """Tests for the domain round-robin logic in loop._next_domain and run_n."""

    def test_next_domain_cycles_through_list(self):
        from qframe.pipeline.loop import _next_domain, _DOMAIN_ORDER
        for i, domain in enumerate(_DOMAIN_ORDER):
            expected = _DOMAIN_ORDER[(i + 1) % len(_DOMAIN_ORDER)]
            assert _next_domain(domain) == expected

    def test_next_domain_unknown_returns_first(self):
        from qframe.pipeline.loop import _next_domain, _DOMAIN_ORDER
        assert _next_domain("unknown_domain") == _DOMAIN_ORDER[0]

    def test_next_domain_wraps_around(self):
        from qframe.pipeline.loop import _next_domain, _DOMAIN_ORDER
        last = _DOMAIN_ORDER[-1]
        assert _next_domain(last) == _DOMAIN_ORDER[0]

    def test_run_n_rotates_domain_after_threshold_skips(self, monkeypatch):
        """After _SKIP_ROTATION_THRESHOLD consecutive SKIPs, run_n must rotate domain."""
        from unittest.mock import MagicMock, patch
        from qframe.pipeline.loop import PipelineLoop, _SKIP_ROTATION_THRESHOLD, _DOMAIN_ORDER
        from qframe.pipeline.models import ResearchSpec, VERDICT_SKIP, VERDICT_PASS

        # Build a minimal PipelineLoop without real KB / LLM
        loop = PipelineLoop.__new__(PipelineLoop)

        # Track which domains were used
        domains_used = []

        def mock_run_iteration(spec):
            domains_used.append(spec.factor_domain)
            result = MagicMock()
            result.verdict = VERDICT_SKIP
            result.wf_result = None
            result.hypothesis = MagicMock()
            result.hypothesis.name = "test_factor"
            return result

        loop.run_iteration = mock_run_iteration
        # Stub out KB calls used in run_n
        loop.kb = MagicMock()
        loop.kb.get_results.return_value = []
        loop._update_research_log = MagicMock()

        starting_domain = "mean_reversion"
        spec = ResearchSpec(factor_domain=starting_domain)

        # Run enough iterations to trigger at least one rotation
        n = _SKIP_ROTATION_THRESHOLD + 1
        loop.run_n(spec, n=n)

        # The first _SKIP_ROTATION_THRESHOLD iters should use the starting domain
        assert all(d == starting_domain for d in domains_used[:_SKIP_ROTATION_THRESHOLD])
        # The iteration after the threshold should use the rotated domain
        expected_rotated = _DOMAIN_ORDER[(_DOMAIN_ORDER.index(starting_domain) + 1) % len(_DOMAIN_ORDER)]
        assert domains_used[_SKIP_ROTATION_THRESHOLD] == expected_rotated

    def test_consecutive_skip_counter_resets_on_non_skip(self, monkeypatch):
        """A non-SKIP verdict must reset the consecutive-skip counter, preventing rotation."""
        from unittest.mock import MagicMock
        from qframe.pipeline.loop import PipelineLoop, _SKIP_ROTATION_THRESHOLD
        from qframe.pipeline.models import ResearchSpec, VERDICT_SKIP, VERDICT_FAIL

        loop = PipelineLoop.__new__(PipelineLoop)
        domains_used = []
        call_count = [0]

        def mock_run_iteration(spec):
            domains_used.append(spec.factor_domain)
            call_count[0] += 1
            result = MagicMock()
            # Alternate: 2 SKIPs, then a FAIL (resets), then 2 more SKIPs — never reaches threshold
            if call_count[0] in (1, 2):
                result.verdict = VERDICT_SKIP
            else:
                result.verdict = VERDICT_FAIL
            result.wf_result = None
            result.hypothesis = MagicMock()
            result.hypothesis.name = "test_factor"
            return result

        loop.run_iteration = mock_run_iteration
        loop.kb = MagicMock()
        loop.kb.get_results.return_value = []
        loop._update_research_log = MagicMock()

        starting_domain = "mean_reversion"
        spec = ResearchSpec(factor_domain=starting_domain)
        loop.run_n(spec, n=_SKIP_ROTATION_THRESHOLD + 2)

        # No rotation should have occurred — all iterations use the starting domain
        assert all(d == starting_domain for d in domains_used)
