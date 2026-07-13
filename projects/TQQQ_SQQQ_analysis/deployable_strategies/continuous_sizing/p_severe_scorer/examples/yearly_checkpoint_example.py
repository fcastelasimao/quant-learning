"""Minimal yearly-checkpoint integration example.

Run from the repository root:

    ~/opt/anaconda3/envs/quant/bin/python \
      TQQQ_SQQQ_analysis/deployable_strategies/continuous_sizing/p_severe_scorer/examples/yearly_checkpoint_example.py

This example uses the repository's corrected strict-prior enriched trades as
stand-ins for another backtest's completed history.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT.parent))

from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer import (  # noqa: E402
    score_trades_with_models_by_symbol,
    train_models_from_history,
)


def main() -> None:
    tqqq = pd.read_csv(ROOT / "research/06_context_enrichment/enriched_trades_TQQQ.csv", parse_dates=["entry_time"])
    sqqq = pd.read_csv(ROOT / "research/06_context_enrichment/enriched_trades_SQQQ.csv", parse_dates=["entry_time"])
    completed_history = pd.concat([tqqq, sqqq], ignore_index=True)
    completed_history = completed_history[completed_history["entry_time"].dt.year <= 2025].copy()

    models = train_models_from_history(completed_history, predict_year=2026)

    candidates = pd.read_csv(Path(__file__).with_name("candidate_trades_example.csv"), parse_dates=["entry_time"])
    scored = score_trades_with_models_by_symbol(candidates, models)
    scored["position_size_for_100k_notional"] = 100_000 * scored["size_multiplier"]

    print(scored[[
        "symbol",
        "entry_time",
        "p_severe",
        "size_multiplier",
        "position_size_for_100k_notional",
        "model_train_end_year",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
