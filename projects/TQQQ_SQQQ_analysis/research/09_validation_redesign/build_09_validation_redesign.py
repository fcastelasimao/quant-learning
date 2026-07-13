"""09_validation_redesign: define WF + embargoed holdout, demonstrate on focus rule.

New convention (replaces the previous single 2020-12-31 IS/OOS cutoff):

  TRAIN/DISCOVERY:   walk-forward expanding window, train on [2015..Y],
                     evaluate on Y+1, for Y in {2017..2024}. 8 eval years.
                     Used for rule discovery and per-year stability.

  RESEARCH OOS:      2021-2025 (still pre-embargo). Used as a unified OOS
                     reporting window for rule comparison.

  EMBARGOED HOLDOUT: 2026 partial year. NO model or rule is allowed to "see"
                     this slice during research. Final numbers reported on
                     this slice are the only truly out-of-sample test.

  REPORTING:         For each rule, report
                       - IS_metric  (2015-2020)
                       - WF_metric_median (across eval years 2018..2025)
                       - Research_OOS_metric (2021-2025)
                       - Embargo_metric (2026)

This script writes a small `validation_splits.csv` that the other research
directions can re-import, and a demonstration applying the convention to the
SQQQ focus rule (`rsi_x_atr_cell_3_1`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT

IS_END = 2020
EMBARGO_YEAR = 2026  # partial; treat as embargoed
RESEARCH_OOS_START = 2021
RESEARCH_OOS_END = 2025


def split_label(year: int) -> str:
    if year <= IS_END:
        return "IS"
    if year >= EMBARGO_YEAR:
        return "EMBARGO"
    if RESEARCH_OOS_START <= year <= RESEARCH_OOS_END:
        return "RESEARCH_OOS"
    return "UNKNOWN"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Document the splits
    rows = [
        {"split": "IS", "year_min": 2015, "year_max": IS_END,
         "purpose": "training / rule discovery"},
        {"split": "RESEARCH_OOS", "year_min": RESEARCH_OOS_START, "year_max": RESEARCH_OOS_END,
         "purpose": "research-time OOS evaluation; rules can iterate against this"},
        {"split": "EMBARGO", "year_min": EMBARGO_YEAR, "year_max": EMBARGO_YEAR,
         "purpose": "embargoed holdout; partial year; NO research access during discovery"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "validation_splits.csv", index=False)

    # Apply to the SQQQ focus rule as a demonstration
    df = pd.read_csv(CANON / "TRADES_SQQQ_full_history.csv", parse_dates=["entry_time"])
    df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    df["split"] = df["year"].apply(split_label)
    df["flagged"] = ((df["RSI_entry"].between(56.4, 59.85)) &
                     (df["atr_pct"].between(0.39, 0.47))).fillna(False)

    out = []
    for split in ("IS", "RESEARCH_OOS", "EMBARGO"):
        sub = df[df["split"] == split]
        flagged = sub[sub["flagged"]]
        if sub.empty:
            continue
        out.append({
            "split": split,
            "year_range": f"{sub['year'].min()}-{sub['year'].max()}",
            "n_total": len(sub),
            "n_flagged": len(flagged),
            "trigger_rate": float(sub["flagged"].mean()),
            "precision_loser": float(flagged["is_loser"].mean()) if len(flagged) else np.nan,
            "precision_severe": float(flagged["is_severe_loss"].mean()) if len(flagged) else np.nan,
            "mean_flagged_pnl_pct": float(flagged["pnl_pct"].mean()) if len(flagged) else np.nan,
            "net_pnl_impact_if_skipped": float(-flagged["pnl_pct"].sum()),
            "baseline_loser_rate": float(sub["is_loser"].mean()),
        })
    pd.DataFrame(out).to_csv(OUT / "focus_rule_under_new_splits.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
