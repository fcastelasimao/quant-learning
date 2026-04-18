"""
Phase 1 Agentic Pipeline — main orchestration loop.

One iteration = Synthesis → Implementation → Validation → Analysis → KB write.
The loop is synchronous and runs one factor at a time. You (the human director)
review results between iterations and decide whether to continue, pivot, or stop.

Usage (from notebook or CLI):
    from qframe.pipeline.loop import PipelineLoop, ResearchSpec
    from qframe.data.loader import load_returns

    prices = ...  # (dates x tickers) close prices

    loop = PipelineLoop(prices=prices)
    result = loop.run_iteration(ResearchSpec(factor_domain="momentum"))
    result.print_summary()
"""
from __future__ import annotations

import datetime
import re
import subprocess
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qframe.factor_harness import DEFAULT_OOS_START
from qframe.factor_harness.costs import CostParams, DEFAULT_COST_PARAMS
from qframe.factor_harness.walkforward import WalkForwardValidator
from qframe.knowledge_base.db import KnowledgeBase
from qframe.pipeline.agents.analysis import AnalysisAgent
from qframe.pipeline.agents.implementation import ImplementationAgent
from qframe.pipeline.agents.synthesis import SynthesisAgent
from qframe.pipeline.executor import (
    make_factor_fn,
    run_factor_with_timeout,
    validate_factor_output,
)
from qframe.pipeline.models import (
    HypothesisSpec,
    IterationResult,
    ResearchSpec,
    VERDICT_ERROR,
)

_DEFAULT_KB = Path("knowledge_base/qframe.db")


def _get_git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _kb_context(kb: KnowledgeBase) -> list[dict]:
    """Return all non-retired hypotheses for the synthesis prompt."""
    with kb._connect() as conn:
        rows = conn.execute(
            "SELECT factor_name, description FROM hypotheses WHERE status != 'retired'"
        ).fetchall()
    return [{"name": r[0] or "unknown", "description": r[1]} for r in rows]


class PipelineLoop:
    """
    Agentic research loop — one factor per iteration.

    Args:
        prices:       (dates x tickers) adjusted close prices, sorted ascending.
        kb_path:      Path to SQLite knowledge base.
        oos_start:    Walk-forward OOS start date.
        cost_params:  Transaction cost model parameters.
        exec_timeout: Seconds before aborting a factor computation.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        kb_path: str | Path = _DEFAULT_KB,
        oos_start: str = DEFAULT_OOS_START,
        cost_params: CostParams = DEFAULT_COST_PARAMS,
        exec_timeout: int = 120,
    ):
        self.prices = prices.sort_index()
        self.oos_start = oos_start
        self.cost_params = cost_params
        self.exec_timeout = exec_timeout

        self.kb = KnowledgeBase(db_path=kb_path)
        self.kb.init_schema()

        self._synthesis = SynthesisAgent(kb=self.kb)
        self._implementation = ImplementationAgent()
        self._analysis = AnalysisAgent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_iteration(self, spec: ResearchSpec) -> IterationResult:
        """
        Run one full pipeline iteration.

        Steps:
            1. Synthesis  — Gemini generates a hypothesis
            2. Implementation — Ollama generates factor code
            3. Validation — walk-forward harness runs the factor
            4. Analysis — Gemini interprets the result
            5. KB write — everything logged to SQLite

        Args:
            spec: ResearchSpec from the human director.

        Returns:
            IterationResult with all outputs and KB ids.
        """
        print(f"[1/5] Synthesis — generating hypothesis for domain: {spec.factor_domain}")
        hypothesis = self._synthesis.generate(
            spec, existing_hypotheses=_kb_context(self.kb)
        )
        print(f"      → {hypothesis.name}: {hypothesis.description}")

        # Log hypothesis immediately so it's in the KB even if later steps fail
        hyp_id = self.kb.add_hypothesis(
            description=hypothesis.description,
            rationale=hypothesis.rationale,
            mechanism_score=hypothesis.mechanism_score,
            status="active",
            factor_name=hypothesis.name,
        )

        print(f"[2/5] Implementation — generating code with Ollama")
        code = self._implementation.generate(hypothesis)

        impl_id = self.kb.add_implementation(
            hypothesis_id=hyp_id,
            code=code,
            git_hash=_get_git_hash(),
            notes=f"factor_type={hypothesis.factor_type}",
        )

        print(f"[3/5] Validation — running walk-forward backtest")
        wf_result, exec_error = self._run_validation(hypothesis, code)

        if exec_error:
            # Show only the last line of the traceback for clean output
            last_line = [l for l in exec_error.strip().splitlines() if l.strip()][-1]
            print(f"      ✗ Execution failed: {last_line}")
            self.kb.update_hypothesis_status(hyp_id, "failed")
            analysis = f"Factor code failed to execute: {last_line}"
            return IterationResult(
                hypothesis=hypothesis,
                code=code,
                wf_result=None,
                analysis=analysis,
                verdict=VERDICT_ERROR,
                kb_hypothesis_id=hyp_id,
                kb_implementation_id=impl_id,
                kb_result_id=None,
                error=exec_error,
            )

        print(f"[4/5] Analysis — interpreting results with Gemini")
        metrics = wf_result.summary()
        analysis, verdict = self._analysis.interpret(hypothesis, metrics)
        print(f"      → Verdict: {verdict}")

        print(f"[5/5] Logging result to knowledge base")
        # Cache OOS ranked signal as JSON to avoid re-executing code in correlation analysis
        signal_cache = None
        try:
            oos_ranked = wf_result.weights.loc[self.oos_start:].rank(axis=1, pct=True)
            signal_cache = oos_ranked.to_json(orient="split", date_format="iso")
        except Exception:
            pass  # caching is best-effort; never block logging
        result_id = self.kb.log_result(
            impl_id,
            {
                **metrics,
                "regime": "all",
                "universe": "sp500_survivorship_biased",
                "gate_level": 1,
                "passed_gate": 1 if verdict == "PASS" else 0,
                "notes": analysis,
                "signal_cache_json": signal_cache,
            },
        )
        status = "passed" if verdict == "PASS" else "failed"
        self.kb.update_hypothesis_status(hyp_id, status)

        print(f"      → KB ids: hypothesis={hyp_id}, impl={impl_id}, result={result_id}")

        return IterationResult(
            hypothesis=hypothesis,
            code=code,
            wf_result=wf_result,
            analysis=analysis,
            verdict=verdict,
            kb_hypothesis_id=hyp_id,
            kb_implementation_id=impl_id,
            kb_result_id=result_id,
        )

    def run_n(self, spec: ResearchSpec, n: int) -> list[IterationResult]:
        """
        Run n iterations sequentially, each building on the KB from the last.
        Every 5 completed backtests, automatically runs ensemble + correlation analysis.

        Args:
            spec: ResearchSpec — same domain across all iterations.
            n:    number of iterations.

        Returns:
            List of IterationResult, one per iteration.
        """
        results = []
        for i in range(n):
            print(f"\n{'─'*60}")
            print(f"Iteration {i+1}/{n}")
            print(f"{'─'*60}")
            result = self.run_iteration(spec)
            result.print_summary()
            results.append(result)

            # After every 5 iterations, run correlation + ensemble analysis
            total_results = len(self.kb.get_results(limit=1000))
            if total_results > 0 and total_results % 5 == 0:
                print(f"\n{'═'*60}")
                print(f"Auto-analysis checkpoint at {total_results} total results")
                print(f"{'═'*60}")
                self.run_correlation_analysis()
                self.run_ensemble_check(top_n=3)

        # Auto-update research-log.md with current KB stats
        self._update_research_log(results, spec.factor_domain)

        return results

    def run_correlation_analysis(self) -> None:
        """
        Compute pairwise rank correlations between all factor signals with backtest results.
        Logs results to the factor_correlations table in the KB.

        Uses the full OOS period (self.oos_start onwards) for correlation computation.
        Loads signal from the KB cache (signal_cache_json) when available, only re-runs
        factor code for factors that have no cached signal — making repeat calls near-instant.
        """
        import json as _json

        print("[Correlation] Loading factor signals...")
        all_results = self.kb.get_all_results()

        # Only process factors with valid IC > 0 and stored code
        valid = [r for r in all_results if r.get("ic") and r["ic"] > 0 and r.get("code")]
        if len(valid) < 2:
            print("[Correlation] Need at least 2 positive-IC factors — skipping")
            return

        # Compute factor signal for each valid result, using cache when available
        factor_signals: dict[str, pd.DataFrame] = {}
        for r in valid:
            name = r.get("factor_name") or f"impl_{r['implementation_id']}"
            cache_json = r.get("signal_cache_json")
            if cache_json:
                try:
                    cached = pd.read_json(cache_json, orient="split")
                    cached.index = pd.to_datetime(cached.index)
                    oos_sig = cached.loc[self.oos_start:]
                    factor_signals[name] = oos_sig
                    continue
                except Exception:
                    pass  # cache corrupt — fall through to re-execution
            try:
                fn = make_factor_fn(r["code"])
                sig = run_factor_with_timeout(fn, self.prices, timeout=self.exec_timeout)
                # Use OOS period only; rank cross-sectionally each day
                oos_sig = sig.loc[self.oos_start:].rank(axis=1, pct=True)
                factor_signals[name] = oos_sig
            except Exception as e:
                print(f"[Correlation] Skipping {name}: {e}")

        names = list(factor_signals.keys())
        n = len(names)
        if n < 2:
            print("[Correlation] Fewer than 2 factors computed — skipping")
            return

        print(f"[Correlation] Computing {n*(n-1)//2} pairwise correlations...")
        period = f"{self.oos_start[:4]}-2024"

        for i in range(n):
            for j in range(i + 1, n):
                a_name, b_name = names[i], names[j]
                a_sig = factor_signals[a_name].values.flatten()
                b_sig = factor_signals[b_name].values.flatten()
                mask = np.isfinite(a_sig) & np.isfinite(b_sig)
                if mask.sum() < 100:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(a_sig[mask], b_sig[mask])
                self.kb.log_factor_correlation(
                    factor_a=a_name, factor_b=b_name,
                    correlation=float(corr),
                    period=period, universe="sp500_survivorship_biased",
                )
                print(f"  {a_name} ↔ {b_name}: ρ = {corr:+.3f}")

    def run_ensemble_check(self, top_n: int = 3, corr_penalty: float = 0.3) -> None:
        """
        Correlation-aware ensemble of top factors: combines the best N factors
        selected by a diversification-adjusted score (IC - corr_penalty × mean_correlation)
        rather than IC alone. Highly correlated duplicates of the best factor are
        deprioritised in favour of uncorrelated signals.

        IC weights are also adjusted: w_i = IC_i / (1 + |mean_corr_i|) so correlated
        factors contribute less to the combined signal.

        Args:
            top_n:        maximum number of factors to combine.
            corr_penalty: how strongly to penalise pairwise correlation (λ in the plan).
                          0.0 = pure IC ranking (old behaviour); 0.3 = default.
        """
        all_results = self.kb.get_all_results()
        candidates = sorted(
            [r for r in all_results if r.get("ic") and r["ic"] > 0 and r.get("code")],
            key=lambda r: r["ic"], reverse=True
        )

        if len(candidates) < 2:
            print(f"[Ensemble] Need at least 2 positive-IC factors — skipping")
            return

        # Greedy correlation-aware selection
        # Start with highest-IC factor; at each step pick the candidate that maximises
        # IC - corr_penalty × mean(|correlation with already-selected factors|)
        def _get_name(r: dict) -> str:
            return r.get("factor_name") or f"impl_{r['implementation_id']}"

        selected: list[dict] = [candidates[0]]
        for candidate in candidates[1:]:
            if len(selected) >= top_n:
                break
            c_name = _get_name(candidate)
            corrs = [
                self.kb.get_correlation(c_name, _get_name(s))
                for s in selected
            ]
            valid_corrs = [c for c in corrs if c is not None]
            mean_corr = float(np.mean(np.abs(valid_corrs))) if valid_corrs else 0.0
            score = candidate["ic"] - corr_penalty * mean_corr
            if score > 0:
                selected.append(candidate)

        names = [_get_name(r) for r in selected]
        ics = [r["ic"] for r in selected]

        # Correlation-adjusted IC weights: w_i ∝ IC_i / (1 + mean |corr_i|)
        adj_ics = []
        for r in selected:
            name = _get_name(r)
            other_names = [_get_name(s) for s in selected if _get_name(s) != name]
            corrs = [self.kb.get_correlation(name, o) for o in other_names]
            valid_corrs = [c for c in corrs if c is not None]
            mean_corr = float(np.mean(np.abs(valid_corrs))) if valid_corrs else 0.0
            adj_ics.append(r["ic"] / (1 + mean_corr))

        total_adj = sum(adj_ics)
        weights = [w / total_adj for w in adj_ics]
        print(f"\n[Ensemble] Combining {len(selected)} factors (corr-aware): {names}")

        # Compute weighted sum of factor signals
        combined: pd.DataFrame | None = None
        for r, w in zip(valid, weights):
            try:
                fn = make_factor_fn(r["code"])
                sig = run_factor_with_timeout(fn, self.prices, timeout=self.exec_timeout)
                sig_ranked = sig.rank(axis=1, pct=True).subtract(0.5)  # centre at 0
                combined = sig_ranked * w if combined is None else combined + sig_ranked * w
            except Exception as e:
                print(f"[Ensemble] Error computing {r.get('factor_name')}: {e}")
                return

        if combined is None:
            return

        # Backtest the ensemble signal
        cached = combined
        def _ensemble_fn(p: pd.DataFrame) -> pd.DataFrame:
            return cached

        try:
            validator = WalkForwardValidator(
                factor_fn=_ensemble_fn,
                oos_start=self.oos_start,
                cost_params=self.cost_params,
                min_stocks=20,
            )
            wf_result = validator.run(self.prices)
            metrics = wf_result.summary()
            ensemble_name = f"ensemble_top{len(valid)}"
            hyp_id = self.kb.add_hypothesis(
                description=f"IC-weighted ensemble of top {len(valid)} factors: {', '.join(names)}",
                rationale="Signal diversification — combining uncorrelated positive-IC factors reduces noise and improves ICIR",
                mechanism_score=4,
                status="active",
                factor_name=ensemble_name,
            )
            impl_id = self.kb.add_implementation(
                hypothesis_id=hyp_id,
                code=f"# Ensemble of: {names}\n# IC weights: {dict(zip(names, weights))}",
                notes=f"ensemble|components={names}",
            )
            result_id = self.kb.log_result(impl_id, {
                **metrics,
                "regime": "all",
                "universe": "sp500_survivorship_biased",
                "gate_level": 1,
                "passed_gate": 1 if metrics.get("icir", 0) > 0.15 else 0,
                "notes": f"Auto-ensemble of top {len(valid)} factors by IC weight",
            })
            self.kb.update_hypothesis_status(hyp_id, "passed" if metrics.get("icir", 0) > 0.15 else "failed")

            ic = metrics.get("ic", float("nan"))
            icir = metrics.get("icir", float("nan"))
            slow63 = metrics.get("slow_icir_63", float("nan"))
            print(f"[Ensemble] {ensemble_name}: IC={ic:+.4f}  ICIR={icir:+.3f}  SlowICIR63={slow63:+.3f}  result_id={result_id}")
        except Exception as e:
            print(f"[Ensemble] Backtest failed: {e}")

    # ------------------------------------------------------------------
    # Auto-documentation
    # ------------------------------------------------------------------

    _LOG_PATH = Path("research-log.md")

    def _update_research_log(
        self, results: list, domain: str
    ) -> None:
        """
        Automatically update the 'Current Status' block and prepend a session
        entry to research-log.md after every run_n() call.

        Rewrites two sections in-place:
          1. The opening '## Current Status' block — fresh KB stats.
          2. Inserts a new '## Session: YYYY-MM-DD' entry directly after
             the Current Status block.

        Args:
            results:  IterationResult list from run_n().
            domain:   Factor domain that was just run.
        """
        log_path = self._LOG_PATH
        if not log_path.exists():
            return  # nothing to update if file doesn't exist

        # --- Gather fresh KB stats ---
        with self.kb._connect() as conn:
            n_hypotheses = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
            n_results    = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0]
            n_corr       = conn.execute("SELECT COUNT(*) FROM factor_correlations").fetchone()[0]
            passed       = conn.execute(
                "SELECT h.factor_name, r.ic, r.icir, r.slow_icir_63 "
                "FROM backtest_results r "
                "JOIN implementations i ON r.implementation_id=i.id "
                "JOIN hypotheses h ON i.hypothesis_id=h.id "
                "WHERE r.passed_gate=1 ORDER BY r.ic DESC LIMIT 5"
            ).fetchall()
            top5 = conn.execute(
                "SELECT h.factor_name, r.ic, r.icir, r.slow_icir_63 "
                "FROM backtest_results r "
                "JOIN implementations i ON r.implementation_id=i.id "
                "JOIN hypotheses h ON i.hypothesis_id=h.id "
                "WHERE r.ic IS NOT NULL ORDER BY r.ic DESC LIMIT 5"
            ).fetchall()

        # --- Build new Current Status block ---
        passed_lines = "\n".join(
            f"  - `{r[0] or 'unnamed'}`: IC={r[1]:.4f}, ICIR={r[2]:.4f}"
            + (f", slow_icir_63={r[3]:.4f}" if r[3] else "")
            for r in passed
        ) or "  - None yet"

        top5_lines = "\n".join(
            f"  | {r[0] or 'unnamed'} | {r[1]:.4f} | {r[2]:.4f} |"
            + (f" {r[3]:.4f} |" if r[3] else " — |")
            for r in top5
        )

        new_status = (
            f"## Current Status\n\n"
            f"**Phase:** 1 — Agentic Pipeline (running)\n\n"
            f"**Gate status:**\n"
            f"- Gate 0 (infrastructure smoke test): ✅ PASSED\n"
            f"- Gate 1 (factor library): 🔄 IN PROGRESS\n"
            f"- Gate 2+ (HSMM regime detection): ⬜ NOT STARTED\n\n"
            f"**Knowledge base:** {n_hypotheses} hypotheses, "
            f"{n_results} backtest results, {n_corr} factor correlations.\n\n"
            f"**Passed-gate factors:**\n{passed_lines}\n\n"
            f"**Top-5 by IC (OOS):**\n"
            f"  | Factor | IC | ICIR | slow_icir_63 |\n"
            f"  |--------|-----|------|----------|\n"
            f"{top5_lines}"
        )

        # --- Build new session entry ---
        today = datetime.date.today().isoformat()
        verdicts = [r.verdict for r in results]
        n_pass  = verdicts.count("PASS")
        n_fail  = verdicts.count("FAIL")
        n_error = verdicts.count("ERROR")

        session_lines = [
            f"## Session: {today} (auto — domain={domain})\n",
            f"**Done:** ran {len(results)} iteration(s): "
            f"{n_pass} PASS / {n_fail} FAIL / {n_error} ERROR\n",
        ]
        for r in results:
            if r.wf_result is not None:
                m = r.wf_result.summary()
                session_lines.append(
                    f"- `{r.hypothesis.name}`: IC={m.get('ic', float('nan')):.4f} "
                    f"ICIR={m.get('icir', float('nan')):.4f} → **{r.verdict}**"
                )
            else:
                session_lines.append(
                    f"- `{r.hypothesis.name}` → **{r.verdict}** (execution error)"
                )

        session_lines.append("")  # blank line after entry
        session_entry = "\n".join(session_lines)

        # --- Rewrite file ---
        text = log_path.read_text()

        # Replace existing Current Status block
        status_pattern = re.compile(
            r"## Current Status\n.*?(?=\n---\n|\n## Session:|\Z)",
            re.DOTALL,
        )
        if status_pattern.search(text):
            text = status_pattern.sub(new_status, text, count=1)
        else:
            text = new_status + "\n\n---\n\n" + text

        # Insert new session entry after the first '---' separator
        first_sep = text.find("\n---\n")
        if first_sep != -1:
            insert_at = first_sep + len("\n---\n")
            text = text[:insert_at] + "\n" + session_entry + "\n" + text[insert_at:]

        log_path.write_text(text)
        print(f"[Log] research-log.md updated ({n_results} results, {n_corr} correlations)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Errors that are worth attempting an automatic one-shot fix
    _FIXABLE_ERRORS = (
        "Grouper for",
        "bad operand type for unary ~",
        "Cannot convert non-finite values",
        "ndarray' object has no attribute",
        "factor contains infinite values",
        "Shape of passed values",
        "IntCastingNaNError",
        "Operands are not aligned",
        "'float' object cannot be interpreted as an integer",
    )

    def _run_validation(
        self, hypothesis: HypothesisSpec, code: str
    ) -> tuple:
        """
        Compile and run the factor code. Returns (WalkForwardResult | None, error_str | None).
        On a fixable error, attempts one automatic code fix via the implementation agent.
        """
        for attempt in range(2):  # attempt 0 = original, attempt 1 = auto-fixed
            try:
                factor_fn = make_factor_fn(code)

                # Compute factor values with timeout guard
                factor_df = run_factor_with_timeout(
                    factor_fn, self.prices, timeout=self.exec_timeout
                )
                validate_factor_output(factor_df, self.prices, name=hypothesis.name)

                # Walk-forward validation
                cached_factor = factor_df

                def _factor_fn_cached(p: pd.DataFrame) -> pd.DataFrame:
                    return cached_factor

                validator = WalkForwardValidator(
                    factor_fn=_factor_fn_cached,
                    oos_start=self.oos_start,
                    cost_params=self.cost_params,
                    min_stocks=20,
                )
                wf_result = validator.run(self.prices)
                return wf_result, None

            except Exception:
                tb = traceback.format_exc(limit=5)
                last_line = [ln for ln in tb.strip().splitlines() if ln.strip()][-1]

                # On first attempt, check if the error is auto-fixable
                if attempt == 0 and any(pat in tb for pat in self._FIXABLE_ERRORS):
                    print(f"      ↻ Auto-fix attempt: {last_line}")
                    try:
                        code = self._implementation.fix(code, last_line)
                    except Exception:
                        return None, tb  # fix itself failed — bail

                    continue  # retry with fixed code

                return None, tb
