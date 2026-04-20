"""
Synthesis agent — generates factor hypotheses using Gemini 2.5 Flash.

Reads the current knowledge base to avoid duplicating existing factors,
then produces a new hypothesis in structured JSON.

Literature library
------------------
When ``factor_library.yaml`` is present in the repo root, the agent reads
``pending`` factors for the requested domain and injects them into the prompt
as high-priority candidates. This prevents the LLM from re-inventing signals
that have established academic evidence — it should implement those first and
only explore freely after the library is exhausted.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]  # graceful degradation if PyYAML absent

from qframe.pipeline.agents._llm import generate
from qframe.pipeline.models import HypothesisSpec, ResearchSpec

_FACTOR_LIBRARY_PATH = Path(__file__).parent.parent.parent.parent.parent / "factor_library.yaml"


def _load_pending_library_factors(domain: str) -> list[dict]:
    """
    Load pending factors from factor_library.yaml for the given domain.

    Returns a list of dicts with keys: id, name, formula_notes.
    Returns an empty list if the file is absent, PyYAML is unavailable,
    or no pending factors exist for this domain.
    """
    if _yaml is None or not _FACTOR_LIBRARY_PATH.exists():
        return []
    try:
        with _FACTOR_LIBRARY_PATH.open() as fh:
            factors = _yaml.safe_load(fh) or []
        return [
            f for f in factors
            if isinstance(f, dict)
            and f.get("domain") == domain
            and f.get("status", "pending") == "pending"
            and f.get("prices_only", True)  # skip factors needing fundamentals
        ]
    except Exception:
        return []


def _format_library_candidates(library_factors: list[dict]) -> str:
    """Format pending library factors for the synthesis prompt."""
    if not library_factors:
        return "  (none — all literature factors for this domain have been attempted)"
    lines = []
    for f in library_factors[:5]:  # cap at 5 to keep prompt concise
        name = f.get("name", f.get("id", "unknown"))
        notes = f.get("formula_notes", "").strip().replace("\n", " ")
        citation = f.get("citation", "")
        lines.append(f"  - [{f.get('id')}] {name}")
        if citation:
            lines.append(f"    Citation: {citation}")
        if notes:
            lines.append(f"    Formula: {notes[:200]}{'…' if len(notes)>200 else ''}")
    return "\n".join(lines)

_SYSTEM_PROMPT = """\
You are a quantitative finance researcher specialising in systematic equity factors.
Your job is to generate novel, theoretically grounded factor hypotheses.
You output ONLY valid JSON — no prose, no markdown, no explanation outside the JSON object.\
"""

# Canonical literature seeds per domain — used to ground synthesis in known alpha sources.
# These are either directly implementable from prices or conceptually related.
# Reference: Harvey, Liu & Zhu (2016) "… and the Cross-Section of Expected Returns"
_LITERATURE_SEEDS: dict[str, list[str]] = {
    "momentum": [
        "12-1 month price momentum (Jegadeesh & Titman 1993) — skip most recent month",
        "52-week high ratio: price / 52-week high (George & Hwang 2004) — anchoring effect",
        "Residual momentum: momentum orthogonalised to rolling volatility (Blitz et al. 2011)",
        "Industry momentum: cross-sectional momentum within sector (Moskowitz & Grinblatt 1999)",
        "Short-term reversal: 1-month return with sign flip (Jegadeesh 1990)",
    ],
    "mean_reversion": [
        "5-year reversal: negative of 60-month return (De Bondt & Thaler 1985)",
        "Williams %R: (N-day high close − current close) / (N-day high − N-day low of close)",
        "Bollinger Band position: z-score of price within rolling Bollinger Bands (±2σ)",
        "RSI proxy: ratio of avg up-days return to avg down-days return over 14 days",
        "Distance from long-run mean: (price − 252-day MA) / 252-day MA",
    ],
    "volatility": [
        "Realised volatility: 21-day rolling std of log returns (Ang et al. 2006 idiosyncratic vol)",
        "MAX effect: maximum daily return over past 21 days (Bali, Cakici & Whitelaw 2011)",
        "Downside volatility: semi-deviation of negative returns only",
        "Volatility of volatility: rolling std of 21-day rolling vol (vol regime stability)",
        "GARCH proxy: difference between short and long-window volatility estimates",
    ],
    "quality": [
        "Rolling Sharpe ratio: 252-day rolling mean(log_returns) / std(log_returns) (cheap trend-quality proxy)",
        "Drawdown duration: days elapsed since last 252-day rolling price high (persistence of uptrend)",
        "Low-beta anomaly: negative of rolling 252-day beta to equal-weight market (Frazzini & Pedersen 2014)",
        "Return skewness: 63-day rolling skewness of daily log returns (negative skew = fragile trend)",
        "Recovery ratio: (current price − rolling 252-day min) / (rolling 252-day max − rolling 252-day min)",
    ],
    "value": [
        "Distance from 52-week high: 1 − (price / 52-week-high close) — anchoring/reversion",
        "Price-to-N-day-high: current price / N-day rolling max close (range position)",
        "Long-run mean reversion: z-score over 5-year (1260-day) rolling window",
        "52-week low proximity: (price − 52-week low) / (52-week high − 52-week low)",
    ],
}

_HYPOTHESIS_TEMPLATE = """\
Research domain: {factor_domain}
Universe: {universe_description}
Constraints: {constraints}
{domain_rotation_hint}
=== LITERATURE LIBRARY — IMPLEMENT THESE FIRST (peer-reviewed, not yet attempted) ===
These factors have documented academic evidence and have NOT yet been run through
the harness. IMPLEMENT ONE OF THESE before exploring freely. If the list is empty,
proceed to the literature seeds below.
{library_candidates}

=== LITERATURE SEEDS FOR THIS DOMAIN (strong starting points from academic research) ===
{literature_seeds}

=== TOP-PERFORMING FACTORS ALREADY FOUND IN THIS DOMAIN (build variations of these) ===
{dynamic_seeds}

Existing factors already in the knowledge base (do NOT duplicate these):
{existing_factors}

Most recently attempted factor archetypes (AVOID these archetypes for variety):
{recent_archetypes}

=== HIGH-CORRELATION AVOID LIST (top-IC factors already in KB) ===
DO NOT propose a factor whose mathematical computation would produce signals with
Spearman correlation > 0.5 against ANY of the following signals:
{high_ic_avoid_list}

=== SATURATED ARCHETYPES — DO NOT IMPLEMENT THESE (already have 2+ versions each) ===
- 12-1 month price momentum (Jegadeesh-Titman) and any variant thereof
- Risk-adjusted or volatility-normalised momentum
- EWMA momentum of any lookback
- 52-week high proximity / distance from N-day high or low
- Price z-score to rolling moving average (MA distance)
- Calmar ratio (return/max-drawdown) — BHY-SIGNIFICANT, DO NOT REPEAT
- Return consistency (fraction of positive days) — already tested

Generate ONE new cross-sectional factor hypothesis that:
1. Has a clear economic mechanism (not pure data mining)
2. Can be computed from daily adjusted CLOSE PRICES ONLY — no Open, High, Low, or Volume available
3. Is meaningfully different in BOTH description AND computation from all existing factors above
4. Must NOT be a simple variation of "consecutive days", "run length", or "streak counting"
   unless none of the recent archetypes are streak-based
5. PREFERENCE: directly implement or closely adapt one of the literature seeds above
   if it has not yet been tried — these have documented evidence of working

=== APPROVED FACTOR ARCHETYPES (pick from whichever is under-explored) ===
- Price momentum (cross-sectional rank, EWMA, risk-adjusted)
- Mean-reversion (short-term reversal, RSI-like, z-score to MA)
- Volatility (rolling std, realized variance, vol-of-vol, GARCH proxy)
- Dispersion / skewness (daily return skewness, kurtosis)
- Trend quality (R-squared of trend, Hurst exponent proxy)
- Autocorrelation (return predictability, momentum persistence)
- Drawdown-based (max drawdown, recovery ratio, calmar-like)
- Cross-sectional correlation (beta to market, idiosyncratic vol)
- Regime signal (rolling Sharpe, information ratio proxy)
- Price level patterns (distance from 52-week high, 52-week range position)

Output a single JSON object with exactly these keys:
{{
  "name": "<short_snake_case_identifier>",
  "description": "<one sentence, plain English>",
  "rationale": "<economic mechanism — why should this predict future returns?>",
  "mechanism_score": <integer 1-5>,
  "factor_type": "<momentum|mean_reversion|quality|volatility|value|macro|microstructure>"
}}

mechanism_score guide:
1 = pure data mining, no theory
2 = weak theory, one possible explanation
3 = plausible mechanism, not well-tested
4 = strong mechanism, documented in literature
5 = causal evidence (natural experiment or quasi-causal)\
"""


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def _format_existing_factors(existing: list[dict]) -> str:
    if not existing:
        return "  (none yet — this is the first iteration)"
    lines = []
    for h in existing:
        name = h.get("name", "unknown")
        desc = h.get("description", "")
        lines.append(f"  - [{name}] {desc}")
    return "\n".join(lines)


def _format_recent_archetypes(existing: list[dict], n: int = 5) -> str:
    """Return names + descriptions of the last N factors to avoid repetition."""
    if not existing:
        return "  (none)"
    recent = existing[-n:]
    lines = [f"  - [{h.get('name', 'unknown')}] {h.get('description', '')}" for h in recent]
    return "\n".join(lines)


def _format_dynamic_seeds(domain_results: list[dict]) -> str:
    """Format top-performing KB factors as dynamic seeds for the synthesis prompt."""
    if not domain_results:
        return "  (none yet — explore the literature seeds above)"
    lines = []
    for r in domain_results:
        name = r.get("factor_name") or "unknown"
        desc = r.get("description", "")
        ic   = r.get("ic")
        ic_str = f"  IC={ic:.4f}" if ic is not None else ""
        lines.append(f"  - [{name}]{ic_str} {desc}")
    return "\n".join(lines)


def _format_high_ic_avoid_list(all_results: list[dict], top_k: int = 10) -> str:
    """
    Format the top-K highest-IC factors as an explicit correlation avoid-list.

    The LLM is instructed not to propose a signal whose *mathematical computation*
    would produce signals correlated > 0.5 with any of these.  This is stronger
    than the existing name-based dedup: it blocks math-equivalent rewrites too.
    """
    # Sort by IC descending, deduplicate by factor_name
    seen: set[str] = set()
    top: list[dict] = []
    for r in sorted(all_results, key=lambda r: r.get("ic") or 0.0, reverse=True):
        name = r.get("factor_name") or "unknown"
        if name in seen or name.startswith(("phase25_", "ensemble_", "combined_")):
            continue
        seen.add(name)
        top.append(r)
        if len(top) >= top_k:
            break

    if not top:
        return "  (none yet)"

    lines = []
    for r in top:
        name = r.get("factor_name") or "unknown"
        desc = r.get("description", "")
        ic   = r.get("ic")
        ic_str = f" IC={ic:.4f}" if ic is not None else ""
        lines.append(f"  - [{name}]{ic_str} — {desc}")
    return "\n".join(lines)


def _domain_rotation_hint(all_results: list[dict]) -> str:
    """
    Suggest which domain to explore based on how many factors have been tried per domain.
    Returns a short nudge string for the prompt.
    """
    from collections import Counter
    counts: Counter = Counter()
    for r in all_results:
        notes = r.get("impl_notes") or ""
        for domain in ("momentum", "mean_reversion", "volatility", "quality", "value",
                       "microstructure", "macro"):
            if domain in notes:
                counts[domain] += 1
                break

    if not counts:
        return ""

    # Find under-explored domains
    all_domains = ["momentum", "mean_reversion", "volatility", "quality", "value"]
    min_count = min(counts.get(d, 0) for d in all_domains)
    under = [d for d in all_domains if counts.get(d, 0) <= min_count]
    if not under:
        return ""
    return (
        f"\n=== DOMAIN ROTATION HINT ===\n"
        f"Under-explored domains (tried {min_count}× or less): {', '.join(under)}.\n"
        f"PREFER a factor from one of these domains to improve portfolio diversification.\n"
    )


class SynthesisAgent:
    """
    Generates a new factor hypothesis using the configured LLM provider.
    Provider controlled by LLM_PROVIDER in .env (default: groq).
    """

    def __init__(self, kb=None):
        """
        Args:
            kb: optional KnowledgeBase instance for dynamic seeding.
                If None, only literature seeds are used.
        """
        self._kb = kb

    def generate(
        self,
        spec: ResearchSpec,
        existing_hypotheses: list[dict] | None = None,
    ) -> HypothesisSpec:
        """
        Generate one new factor hypothesis.

        Args:
            spec:                 ResearchSpec from the human director.
            existing_hypotheses:  List of dicts from kb.get_all_hypotheses()
                                  — used to avoid duplication.

        Returns:
            HypothesisSpec ready for the Implementation agent.
        """
        existing = existing_hypotheses or []

        seeds = _LITERATURE_SEEDS.get(spec.factor_domain, [])
        seeds_text = "\n".join(f"  - {s}" for s in seeds) if seeds else "  (no specific seeds for this domain — explore freely)"

        # Literature library — peer-reviewed factors not yet attempted
        library_factors = _load_pending_library_factors(spec.factor_domain)
        library_candidates_text = _format_library_candidates(library_factors)

        # Dynamic seeds: top-performing factors from KB for this domain
        # These reinforce exploration around proven archetypes while keeping
        # literature seeds as the theoretical floor.
        dynamic_results: list[dict] = []
        all_results: list[dict] = []
        if self._kb is not None:
            try:
                dynamic_results = self._kb.get_results_by_domain(
                    domain=spec.factor_domain, limit=3, min_ic=0.02
                )
            except Exception:
                pass  # KB unavailable — fall back to literature seeds only
            try:
                all_results = self._kb.get_all_results()
            except Exception:
                pass

        dynamic_seeds_text = _format_dynamic_seeds(dynamic_results)
        high_ic_avoid_text = _format_high_ic_avoid_list(all_results, top_k=10)
        rotation_hint = _domain_rotation_hint(all_results)

        prompt = _HYPOTHESIS_TEMPLATE.format(
            factor_domain=spec.factor_domain,
            universe_description=spec.universe_description,
            constraints="\n".join(f"  - {c}" for c in spec.constraints),
            domain_rotation_hint=rotation_hint,
            library_candidates=library_candidates_text,
            literature_seeds=seeds_text,
            dynamic_seeds=dynamic_seeds_text,
            existing_factors=_format_existing_factors(existing),
            recent_archetypes=_format_recent_archetypes(existing, n=5),
            high_ic_avoid_list=high_ic_avoid_text,
        )

        raw = generate(f"{_SYSTEM_PROMPT}\n\n{prompt}")
        try:
            data = _parse_json(raw)
        except (json.JSONDecodeError, KeyError):
            # One retry — truncated responses are common near token limits
            raw = generate(f"{_SYSTEM_PROMPT}\n\n{prompt}")
            data = _parse_json(raw)  # propagate if this also fails

        score = int(data["mechanism_score"])
        if not (1 <= score <= 5):
            score = max(1, min(5, score))  # clamp rather than crash

        return HypothesisSpec(
            name=data["name"],
            description=data["description"],
            rationale=data["rationale"],
            mechanism_score=score,
            factor_type=data["factor_type"],
        )
