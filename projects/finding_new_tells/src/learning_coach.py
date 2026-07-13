"""Prediction-first market intuition coach helpers.

The coach is deliberately separate from the Marimo notebook so the learning
loop can be tested like the rest of the research code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from metrics import REGISTRY
from signal_diagnostics import research_split_panel, tradable_open_forward_returns

PredictionDirection = Literal["bullish", "neutral", "bearish"]


@dataclass(frozen=True)
class MarketCase:
    """One hidden-outcome practice case."""

    split: str
    case_date: pd.Timestamp
    horizon: int
    lookback: int
    history: pd.DataFrame
    metric_snapshot: pd.DataFrame
    outcome: dict[str, object]


@dataclass(frozen=True)
class Prediction:
    """User forecast captured before revealing the outcome."""

    direction: PredictionDirection
    confidence: int
    likely_regime: str
    mechanism: str
    expected_family: str
    prove_wrong: str


def _validate_required_columns(panel: pd.DataFrame) -> None:
    required = ["TQQQ_open", "TQQQ_close", "QQQ_open", "QQQ_close"]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing required coach columns: {missing}")


def _normalize_split(split: str) -> str:
    normalized = split.replace("_", "+").lower()
    if normalized in {"train", "val", "train+val", "trainval"}:
        return "train+val" if normalized == "trainval" else normalized
    raise ValueError("Coach cases only support split='train', 'val', or 'train+val'.")


def eligible_case_dates(
    panel: pd.DataFrame,
    *,
    split: str = "train+val",
    horizon: int = 5,
    lookback: int = 126,
) -> pd.DatetimeIndex:
    """Return dates with enough known history and hidden forward outcome.

    Forward returns are computed on the research split itself, matching
    `metric_forward_profile`; this prevents validation-end cases from peeking
    into the frozen 2022+ test window.
    """
    _validate_required_columns(panel)
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if lookback < 2:
        raise ValueError("lookback must be >= 2.")

    panel_split = research_split_panel(panel, _normalize_split(split))
    if panel_split.empty:
        return pd.DatetimeIndex([])

    tqqq_fwd = tradable_open_forward_returns(
        panel_split, symbol="TQQQ", horizons=(horizon,)
    )[horizon]
    qqq_fwd = tradable_open_forward_returns(
        panel_split, symbol="QQQ", horizons=(horizon,)
    )[horizon]

    eligible: list[pd.Timestamp] = []
    for i, date in enumerate(panel_split.index):
        if i < lookback - 1:
            continue
        window = panel_split.iloc[i - lookback + 1:i + 1]
        if window[["TQQQ_close", "QQQ_close"]].isna().any().any():
            continue
        if pd.isna(tqqq_fwd.loc[date]) or pd.isna(qqq_fwd.loc[date]):
            continue
        eligible.append(pd.Timestamp(date))

    return pd.DatetimeIndex(eligible)


def sample_case_date(
    panel: pd.DataFrame,
    *,
    split: str = "train+val",
    horizon: int = 5,
    lookback: int = 126,
    random_state: int | None = None,
) -> pd.Timestamp:
    """Sample one eligible practice date deterministically when seeded."""
    dates = eligible_case_dates(
        panel, split=split, horizon=horizon, lookback=lookback
    )
    if dates.empty:
        raise ValueError(
            f"No eligible case dates for split={split!r}, horizon={horizon}, "
            f"lookback={lookback}."
        )
    rng = np.random.default_rng(random_state)
    return pd.Timestamp(dates[int(rng.integers(0, len(dates)))])


def metric_snapshot(
    panel: pd.DataFrame,
    case_date: pd.Timestamp,
    *,
    families: Iterable[str] | None = None,
    include_watch: bool = True,
) -> pd.DataFrame:
    """Compute current metric values/votes as of the case date."""
    case_date = pd.Timestamp(case_date)
    family_filter = set(families) if families is not None else None
    rows: list[dict[str, object]] = []

    for name, metric in REGISTRY.items():
        if family_filter is not None and metric.family not in family_filter:
            continue
        if metric.status != "voting" and not include_watch:
            continue
        try:
            values = metric.compute(panel)
            votes = metric.vote(values).reindex(panel.index).fillna(0).astype(int)
        except Exception as exc:
            rows.append({
                "metric": name,
                "family": metric.family,
                "status": metric.status,
                "value": np.nan,
                "vote": 0,
                "strength": np.nan,
                "note": f"compute failed: {exc}",
            })
            continue

        history = values.loc[:case_date].dropna()
        value = values.reindex([case_date]).iloc[0]
        strength = _expanding_robust_z(history, value)
        vote = int(votes.reindex([case_date]).iloc[0]) if case_date in votes.index else 0
        rows.append({
            "metric": name,
            "family": metric.family,
            "status": metric.status,
            "value": float(value) if pd.notna(value) else np.nan,
            "vote": vote,
            "strength": strength,
            "note": "",
        })

    if not rows:
        return pd.DataFrame(
            columns=["metric", "family", "status", "value", "vote", "strength", "note"]
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["family", "metric"], kind="stable"
    ).reset_index(drop=True)


def build_case_packet(
    panel: pd.DataFrame,
    *,
    split: str = "train+val",
    horizon: int = 5,
    lookback: int = 126,
    case_date: pd.Timestamp | str | None = None,
    families: Iterable[str] | None = None,
    include_watch: bool = True,
    random_state: int | None = None,
) -> MarketCase:
    """Build one case with historical context and hidden forward outcome."""
    _validate_required_columns(panel)
    split = _normalize_split(split)
    if case_date is None:
        case_date = sample_case_date(
            panel,
            split=split,
            horizon=horizon,
            lookback=lookback,
            random_state=random_state,
        )
    case_date = pd.Timestamp(case_date)

    eligible = eligible_case_dates(
        panel, split=split, horizon=horizon, lookback=lookback
    )
    if case_date not in eligible:
        raise ValueError(
            f"{case_date.date()} is not eligible for split={split!r}, "
            f"horizon={horizon}, lookback={lookback}."
        )

    panel_split = research_split_panel(panel, split)
    pos = panel_split.index.get_loc(case_date)
    history = panel_split.iloc[pos - lookback + 1:pos + 1][
        ["TQQQ_open", "TQQQ_close", "QQQ_open", "QQQ_close"]
    ].copy()

    tqqq_fwd = tradable_open_forward_returns(
        panel_split, symbol="TQQQ", horizons=(horizon,)
    )[horizon]
    qqq_fwd = tradable_open_forward_returns(
        panel_split, symbol="QQQ", horizons=(horizon,)
    )[horizon]
    tqqq_return = float(tqqq_fwd.loc[case_date])
    qqq_return = float(qqq_fwd.loc[case_date])
    outcome = {
        "tqqq_forward_return": tqqq_return,
        "qqq_forward_return": qqq_return,
        "actual_direction": classify_direction(tqqq_return),
        "entry_date": _future_index(panel_split.index, case_date, 1),
        "exit_date": _future_index(panel_split.index, case_date, horizon + 1),
        "target_kind": "tradable_open",
    }

    return MarketCase(
        split=split,
        case_date=case_date,
        horizon=int(horizon),
        lookback=int(lookback),
        history=history,
        metric_snapshot=metric_snapshot(
            panel,
            case_date,
            families=families,
            include_watch=include_watch,
        ),
        outcome=outcome,
    )


def classify_direction(
    forward_return: float,
    *,
    neutral_band: float = 0.005,
) -> PredictionDirection:
    """Classify a forward return into bullish/neutral/bearish."""
    if forward_return > neutral_band:
        return "bullish"
    if forward_return < -neutral_band:
        return "bearish"
    return "neutral"


def score_prediction(
    prediction: Prediction | dict[str, object],
    actual_return: float,
    *,
    neutral_band: float = 0.005,
) -> dict[str, object]:
    """Score a forecast with deterministic, low-pressure feedback."""
    direction = (
        prediction.direction
        if isinstance(prediction, Prediction)
        else str(prediction["direction"])
    )
    confidence = int(
        prediction.confidence
        if isinstance(prediction, Prediction)
        else prediction.get("confidence", 3)
    )
    confidence = int(np.clip(confidence, 1, 5))
    actual = classify_direction(actual_return, neutral_band=neutral_band)

    if direction == actual:
        direction_score = 1.0
    elif direction == "neutral" or actual == "neutral":
        direction_score = 0.5
    else:
        direction_score = 0.0

    if direction_score == 1.0:
        note = (
            "Correct direction, and the confidence matched the read."
            if confidence >= 4
            else "Correct direction; confidence was deliberately cautious."
            if confidence <= 2
            else "Correct direction with moderate confidence."
        )
    elif direction_score == 0.5:
        note = "Partly right: one side called neutral while the move had only mild direction."
    else:
        note = (
            "High-confidence miss: slow down and inspect what the setup hid."
            if confidence >= 4
            else "Wrong direction, but low confidence limited the calibration damage."
            if confidence <= 2
            else "Missed direction with moderate confidence."
        )

    return {
        "predicted_direction": direction,
        "actual_direction": actual,
        "direction_score": direction_score,
        "confidence": confidence,
        "confidence_note": note,
    }


def postmortem(case: MarketCase, prediction: Prediction | dict[str, object]) -> dict[str, object]:
    """Explain the revealed case in compact, practice-oriented language."""
    score = score_prediction(prediction, float(case.outcome["tqqq_forward_return"]))
    actual_direction = score["actual_direction"]
    actual_vote = {"bullish": 1, "neutral": 0, "bearish": -1}[str(actual_direction)]

    snapshot = case.metric_snapshot.copy()
    snapshot["abs_strength"] = snapshot["strength"].abs()
    if actual_vote == 0:
        agreeing = snapshot.loc[snapshot["vote"] == 0]
        disagreeing = snapshot.loc[snapshot["vote"] != 0]
    else:
        agreeing = snapshot.loc[snapshot["vote"] == actual_vote]
        disagreeing = snapshot.loc[snapshot["vote"] == -actual_vote]

    agreeing = agreeing.sort_values("abs_strength", ascending=False).head(8)
    disagreeing = disagreeing.sort_values("abs_strength", ascending=False).head(8)

    pattern = classify_case_pattern(case)
    critique = _critique_text(case, score, pattern, agreeing, disagreeing)
    return {
        **score,
        "pattern": pattern,
        "critique": critique,
        "agreeing_metrics": agreeing.drop(columns=["abs_strength"], errors="ignore"),
        "disagreeing_metrics": disagreeing.drop(columns=["abs_strength"], errors="ignore"),
    }


def classify_case_pattern(case: MarketCase) -> str:
    """Classify the case into a simple market-learning bucket."""
    qqq = case.history["QQQ_close"]
    prior_20d = qqq.iloc[-1] / qqq.iloc[max(0, len(qqq) - 21)] - 1
    actual = float(case.outcome["tqqq_forward_return"])
    if abs(actual) < 0.005:
        return "noisy / unclear"
    if abs(actual) > 0.08:
        return "volatility shock"
    if np.sign(prior_20d) == np.sign(actual):
        return "trend continuation"
    return "mean reversion"


def journal_row(
    case: MarketCase,
    prediction: Prediction,
    revealed: dict[str, object],
) -> pd.DataFrame:
    """Create one exportable journal row for the current revealed case."""
    return pd.DataFrame([{
        "case_date": case.case_date.date().isoformat(),
        "split": case.split,
        "horizon": case.horizon,
        "prediction": prediction.direction,
        "confidence": prediction.confidence,
        "actual": revealed["actual_direction"],
        "score": revealed["direction_score"],
        "tqqq_fwd_%": round(float(case.outcome["tqqq_forward_return"]) * 100, 3),
        "qqq_fwd_%": round(float(case.outcome["qqq_forward_return"]) * 100, 3),
        "pattern": revealed["pattern"],
        "expected_family": prediction.expected_family,
        "likely_regime": prediction.likely_regime,
        "mechanism": prediction.mechanism,
        "prove_wrong": prediction.prove_wrong,
    }])


def _expanding_robust_z(history: pd.Series, value: object) -> float:
    if pd.isna(value) or history.empty:
        return np.nan
    median = history.median()
    q1 = history.quantile(0.25)
    q3 = history.quantile(0.75)
    iqr = q3 - q1
    if iqr and not np.isnan(iqr):
        scale = iqr / 1.349
    else:
        std = history.std()
        scale = std if std and not np.isnan(std) else np.nan
    if pd.isna(scale) or scale <= 1e-12:
        return 0.0
    return float(np.clip((float(value) - median) / scale, -5, 5))


def _future_index(index: pd.DatetimeIndex, date: pd.Timestamp, offset: int) -> pd.Timestamp:
    pos = index.get_loc(date)
    return pd.Timestamp(index[pos + offset])


def _critique_text(
    case: MarketCase,
    score: dict[str, object],
    pattern: str,
    agreeing: pd.DataFrame,
    disagreeing: pd.DataFrame,
) -> str:
    tqqq_pct = float(case.outcome["tqqq_forward_return"]) * 100
    qqq_pct = float(case.outcome["qqq_forward_return"]) * 100
    agree_names = ", ".join(agreeing["metric"].head(3).tolist()) or "none"
    disagree_names = ", ".join(disagreeing["metric"].head(3).tolist()) or "none"
    return (
        f"Outcome over {case.horizon} trading days was {score['actual_direction']} "
        f"for TQQQ ({tqqq_pct:+.2f}%) and QQQ ({qqq_pct:+.2f}%). "
        f"The simple pattern label is {pattern}. "
        f"Metrics most aligned with the outcome: {agree_names}. "
        f"Main dissenting metrics: {disagree_names}. "
        f"{score['confidence_note']}"
    )
