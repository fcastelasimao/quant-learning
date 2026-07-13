"""alpha_forfeit_frac — the scheduler's alpha-forfeiture input: g(h).

"If we delay entry by h minutes, what fraction of the trade's eventual edge do we give up?"
Fitted once, on real history conditioned on the P&L-bearing trades (hold >= 1 day — S12 found
hold < 2.4h trades are net LOSERS, so the earlier unconditioned S12-0a table was measuring the
wrong population at its longer-delay rows). See
`research/scheduler/01_alpha_decay/findings_01_alpha_decay.md` for the fit.

    g(h) = 1 - exp(-(h / tau)^k)          # stretched exponential

The stretch exponent k > 1 gives the flat first hour + sharp 2-4h knee the empirical curve shows;
the earlier single exponential (k=1) could not capture both and overcharged the 15-120 min range
(the scheduler's operating band) 2.5-3.5x, biasing the chosen horizon short (audit fix 2026-07-09).

Pure function, no IO — `(tau, k)` are baked, cited constants per symbol (same pattern as P02's
chase-drag curve in `predict.py`).
"""

from __future__ import annotations

import numpy as np

# Fitted (tau [min], k), unweighted-median curve, hold >= 1 day subset (findings_01_alpha_decay.md).
# 90% bootstrap CIs are noted in the research findings, not carried at runtime.
_FIT = {"TQQQ": (256.9, 2.480), "SQQQ": (251.7, 2.723)}
_FIT_PNL_WEIGHTED = {"TQQQ": (256.2, 1.636), "SQQQ": (236.5, 1.233)}


def alpha_forfeit_frac(h_min: float, symbol: str, *, pnl_weighted: bool = False) -> float:
    """Fraction of the eventual per-trade edge forfeited by delaying entry `h_min` minutes.

    Symbols outside {TQQQ, SQQQ} fall back to TQQQ's fit (documented — calibrated on TQQQ/SQQQ
    only). `pnl_weighted=True` uses the P&L-weighted fit instead of the unweighted-median one; the
    tau are close but the weighted k is lower (noisier, wide CI), so the two curves diverge more
    at short delays than the single-exponential fits did.
    """
    table = _FIT_PNL_WEIGHTED if pnl_weighted else _FIT
    tau, k = table.get(symbol, table["TQQQ"])
    return float(1.0 - np.exp(-((max(h_min, 0.0) / tau) ** k)))
