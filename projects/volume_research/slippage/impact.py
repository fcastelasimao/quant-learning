"""Market impact — the square-root law — and the capacity it implies.

    I = Y · σ · (Q/V)^β        (β = 1/2 anchor; Almgren 2005 found temp. impact ≈ 3/5)

`I` is the price impact of trading `Q` against market volume `V` over a matching horizon,
with `σ` the volatility over that horizon and `Y` an O(1) constant. **Unlike spread and
delay, `Y` is NOT identifiable from our OHLC data** — we never traded, so there is no
counterfactual to fit. `Y` is adopted from the literature as a prior (Almgren 2005 /
Bouchaud, Y ≈ 0.3–1) and the capacity result is reported as a band over that range, not a
point. These functions are unit-consistent: pass `σ`, `I_budget` in bps and get bps / a
`Q` in the same units as `V`.

Pure functions (no IO). Part of the slippage library alongside `spread` and `delay`.
"""

from __future__ import annotations

import numpy as np


def impact_bps(Q, V, sigma_bps, *, Y: float = 0.5, beta: float = 0.5):
    """Price impact in bps of an order of size `Q` against volume `V`.

    `Q` and `V` in the same units (shares or $); `sigma_bps` is volatility over the
    matching horizon, in bps. Returns impact in bps. Vectorises over `Q`, `Y`.
    """
    return Y * sigma_bps * np.power(np.asarray(Q, float) / V, beta)


def capacity(I_budget_bps, V, sigma_bps, *, Y: float = 0.5, beta: float = 0.5):
    """Largest `Q` whose impact stays within `I_budget_bps` — invert the law.

        Q_max = V · ( I_budget / (Y·σ) )^(1/β)

    Same units out as `V`. Vectorises over `I_budget_bps`, `Y`.
    """
    return V * np.power(np.asarray(I_budget_bps, float) / (Y * sigma_bps), 1.0 / beta)


def participation_for_impact(I_budget_bps, sigma_bps, *, Y: float = 0.5, beta: float = 0.5):
    """The participation rate Q/V that produces exactly `I_budget_bps` of impact.
    Volume-independent — a useful sanity lens (e.g. '10 bps ↔ ~X% of volume')."""
    return np.power(np.asarray(I_budget_bps, float) / (Y * sigma_bps), 1.0 / beta)


# --------------------------------------------------------------------------- Almgren (C01)
# Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity Market Impact" — fitted on
# 29,509 real Citigroup institutional parent orders. Coefficients are the paper's own estimates
# (its Section 4.3): gamma=0.314+/-0.041, eta=0.142+/-0.006, alpha=1 (permanent linear — the only
# arbitrage-free choice), beta=3/5 (temporary, rejects the sqrt-law's beta=1/2 at 95% confidence).
#
# `almgren_temporary` is the one threaded into this library's cost pipeline
# (`impact_model="almgren"`). `almgren_permanent` exists ONLY to reproduce the paper's *realized*
# cost J = I/2 + temporary in the golden test below — S12 found the permanent term's
# single-stock share-float mechanism (Theta/V) is the wrong one for an arbitrage-pinned,
# elastic-supply ETF, so it is NOT used anywhere in the TQQQ/SQQQ pipeline.
ALMGREN_GAMMA = 0.314
ALMGREN_ETA = 0.142
ALMGREN_ETA_SE = 0.006    # the paper's own standard error on eta (t=23) — used as a mini-band
ALMGREN_ALPHA = 1.0
ALMGREN_BETA = 0.6
ALMGREN_DELTA = 0.25


def almgren_permanent(X, V, theta, sigma_bps, *, gamma: float = ALMGREN_GAMMA, delta: float = ALMGREN_DELTA):
    """Almgren et al. (2005) permanent impact, bps: I = gamma * sigma * (X/V) * (Theta/V)^delta.

    `X` = signed order size (shares), `V` = ADV (shares), `theta` = shares outstanding. For the
    golden test only — not used in this project's own capacity pipeline (see module note above).
    """
    X, V, theta = np.asarray(X, float), np.asarray(V, float), np.asarray(theta, float)
    return gamma * sigma_bps * (X / V) * np.power(theta / V, delta)


def almgren_temporary(participation, sigma_bps, *, eta: float = ALMGREN_ETA, beta: float = ALMGREN_BETA):
    """Almgren et al. (2005) temporary impact, bps: K = eta * sigma * sign(p) * |p|^beta.

    `participation` = X/(V*T) (Almgren's normalized trade rate — the same quantity as this
    library's `participation` elsewhere: order size over volume traded during the fill window).
    Signed — vectorises over `participation`.
    """
    p = np.asarray(participation, float)
    return eta * sigma_bps * np.sign(p) * np.power(np.abs(p), beta)
