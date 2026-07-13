"""Show the raw-daily-bars → compute_required_features integration path.

Run from the repository root:

    ~/opt/anaconda3/envs/quant/bin/python \
      TQQQ_SQQQ_analysis/deployable_strategies/continuous_sizing/p_severe_scorer/examples/raw_daily_context_example.py

This builds synthetic daily bars (enough history for the rolling windows),
creates one candidate trade with strategy-internal fields but NO daily context
columns, then calls compute_required_features(trade, daily_context=raw_bars)
to show the context join in action.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT.parent))

from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer import compute_required_features  # noqa: E402
from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer.constants import DAILY_CONTEXT  # noqa: E402


def _synthetic_ohlc(n: int, base: float, vol: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    px = base * np.exp(np.cumsum(rng.normal(0, vol, n)))
    high = px * (1 + rng.uniform(0, 0.01, n))
    low = px * (1 - rng.uniform(0, 0.01, n))
    openp = px * (1 + rng.normal(0, vol / 2, n))
    return pd.DataFrame({"date": dates, "open": openp, "high": high, "low": low, "close": px})


def main() -> None:
    n = 300
    daily_bars = {
        "QQQ": _synthetic_ohlc(n, 370, 0.01, seed=1),
        "SPY": _synthetic_ohlc(n, 450, 0.008, seed=2),
        "^VIX": _synthetic_ohlc(n, 18, 0.03, seed=3),
        "^VIX3M": _synthetic_ohlc(n, 20, 0.02, seed=4),
        "HYG": _synthetic_ohlc(n, 75, 0.005, seed=5),
        "LQD": _synthetic_ohlc(n, 108, 0.003, seed=6),
        "^TNX": _synthetic_ohlc(n, 4.2, 0.015, seed=7),
        "^IRX": _synthetic_ohlc(n, 5.0, 0.01, seed=8),
    }

    candidate = pd.DataFrame({
        "symbol": ["TQQQ"],
        "entry_time": [pd.Timestamp("2024-03-01 10:30:00")],
        "decision_price": [55.0],
        "atr": [0.90],
        "volume_ratio": [1.8],
        "MA20": [54.5],
        "MA50": [53.0],
        "MA100": [52.0],
        "MA20_D5": [0.001],
        "MA50_D5": [-0.002],
        "MA100_D1": [0.0003],
        "RSI_entry": [52.0],
        "BBP_entry": [0.55],
        "bars_since_last_stop": [12],
        "regime_entry": ["chop_highvol"],
    })

    enriched = compute_required_features(candidate, daily_context=daily_bars)

    print("Candidate trade BEFORE context join:")
    print(f"  columns: {sorted(candidate.columns.tolist())}\n")

    print("Candidate trade AFTER compute_required_features(trade, daily_context=raw_bars):")
    ctx_cols = [c for c in DAILY_CONTEXT if c in enriched.columns]
    print(f"  daily context columns joined: {len(ctx_cols)}")
    for col in ctx_cols:
        val = enriched.iloc[0][col]
        print(f"    {col}: {val:.6f}" if pd.notna(val) else f"    {col}: NaN")


if __name__ == "__main__":
    main()
