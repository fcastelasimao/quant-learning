"""
Analysis agent — interprets backtest results using Gemini 2.5 Flash
and returns a structured verdict with reasoning.
"""
from __future__ import annotations

import re

from qframe.pipeline.agents._llm import generate
from qframe.pipeline.models import (
    HypothesisSpec,
    VERDICT_PASS,
    VERDICT_WEAK,
    VERDICT_FAIL,
)

# Gate thresholds — keep in sync with gate-thresholds.md
_IC_WEAK = 0.020
_IC_PASS = 0.030
_ICIR_WEAK = 0.30
_ICIR_PASS = 0.50

_PROMPT_TEMPLATE = """\
You are a quantitative finance analyst reviewing a walk-forward backtest result.
Be concise and critical. Overfitting is the default explanation for good results.

Factor: {name}
Description: {description}
Rationale: {rationale}
Mechanism score: {mechanism_score}/5

Out-of-sample results ({oos_start} → {oos_end}):
  Mean IC:            {ic:.4f}   (weak threshold: >{ic_weak}, pass: >{ic_pass})
  ICIR (rolling 63d): {icir:.3f}  (weak: >{icir_weak}, pass: >{icir_pass})
  Net IC:             {net_ic:.4f}
  Sharpe (IC series): {sharpe:.3f}
  Turnover:           {turnover:.1%}/day
  IC half-life:       {halflife:.1f} days
  IC decay:           1d={ic_1:.4f}  5d={ic_5:.4f}  21d={ic_21:.4f}  63d={ic_63:.4f}

Reply with exactly three lines:
VERDICT: <PASS|WEAK|FAIL>
REASONING: <2-3 sentences — what do the numbers say? is the mechanism consistent with the decay shape?>
NEXT: <one sentence — what should the pipeline explore next given this result?>\
"""


def _parse_verdict(text: str) -> tuple[str, str, str]:
    """
    Parse VERDICT / REASONING / NEXT from the analysis response.
    Returns (verdict, reasoning, next_suggestion).
    """
    verdict = VERDICT_FAIL
    reasoning = text
    next_suggestion = ""

    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            raw = line.split(":", 1)[1].strip().upper()
            if "PASS" in raw:
                verdict = VERDICT_PASS
            elif "WEAK" in raw:
                verdict = VERDICT_WEAK
            else:
                verdict = VERDICT_FAIL
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.startswith("NEXT:"):
            next_suggestion = line.split(":", 1)[1].strip()

    return verdict, reasoning, next_suggestion


class AnalysisAgent:
    """
    Calls Gemini 2.5 Flash to interpret a backtest result and produce a verdict.

    Args:
        model: Gemini model string.
    """

    def interpret(
        self,
        hypothesis: HypothesisSpec,
        metrics: dict,
    ) -> tuple[str, str]:
        """
        Interpret backtest metrics and return (full_analysis_text, verdict).

        Args:
            hypothesis: the factor that was tested.
            metrics:    dict from WalkForwardResult.summary().

        Returns:
            (analysis_text, verdict) where verdict is PASS / WEAK / FAIL.
        """
        prompt = _PROMPT_TEMPLATE.format(
            name=hypothesis.name,
            description=hypothesis.description,
            rationale=hypothesis.rationale,
            mechanism_score=hypothesis.mechanism_score,
            oos_start=metrics.get("oos_start", "?"),
            oos_end=metrics.get("oos_end", "?"),
            ic=metrics.get("ic", 0),
            icir=metrics.get("icir", 0),
            net_ic=metrics.get("net_ic", 0),
            sharpe=metrics.get("sharpe", 0),
            turnover=metrics.get("turnover", 0),
            halflife=metrics.get("decay_halflife", 0),
            ic_1=metrics.get("ic_horizon_1", 0),
            ic_5=metrics.get("ic_horizon_5", 0),
            ic_21=metrics.get("ic_horizon_21", 0),
            ic_63=metrics.get("ic_horizon_63", 0),
            ic_weak=_IC_WEAK,
            ic_pass=_IC_PASS,
            icir_weak=_ICIR_WEAK,
            icir_pass=_ICIR_PASS,
        )

        raw = generate(prompt)
        verdict, reasoning, next_suggestion = _parse_verdict(raw)

        analysis = f"{reasoning}"
        if next_suggestion:
            analysis += f"\n\nNext: {next_suggestion}"

        return analysis, verdict
