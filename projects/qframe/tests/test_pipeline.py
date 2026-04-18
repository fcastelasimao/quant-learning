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
    def test_call_ollama_retries_without_timeout_on_legacy_sdk(self, monkeypatch):
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            if "timeout" in kwargs:
                raise TypeError("Client.generate() got an unexpected keyword argument 'timeout'")
            return {"response": "def factor(prices):\n    return prices"}

        monkeypatch.setattr("qframe.pipeline.agents.implementation.ollama.generate", fake_generate)

        agent = ImplementationAgent(timeout=120)
        out = agent._call_ollama("prompt", temperature=0.2)
        assert "def factor" in out["response"]
        assert len(calls) == 2
        assert "timeout" in calls[0]
        assert "timeout" not in calls[1]

    def test_call_ollama_preserves_other_type_errors(self, monkeypatch):
        def fake_generate(**kwargs):
            raise TypeError("some other type error")

        monkeypatch.setattr("qframe.pipeline.agents.implementation.ollama.generate", fake_generate)

        agent = ImplementationAgent(timeout=120)
        with pytest.raises(TypeError, match="some other type error"):
            agent._call_ollama("prompt", temperature=0.2)
