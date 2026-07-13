"""High–low / OHLC bid-ask spread estimators.

Two estimators, both backing a proportional spread `S` out of OHLC bars — no quotes needed:

  - **Corwin–Schultz (2012)** — uses high/low of two consecutive periods. Their ranges mix
    volatility (scales with interval length) and the bid-ask spread (does not); that asymmetry
    identifies `S`. Ref: CS (2012), *Journal of Finance* 67(2).
  - **EDGE (Ardia, Guidotti & Kroencke, 2024)** — uses all four OHLC prices and the previous
    close, GMM-combining two moment conditions for minimum variance. Asymptotically unbiased
    under infrequent/discrete trading (where CS is biased and noisy). Ref: JFE 161, 103916;
    reference implementation https://github.com/eguidotti/bidask (ported here, numpy-only).

`S` is the proportional *full* spread in both (e.g. 0.0010 = 10 bps round-trip); `half_spread_bps`
maps either to a one-way half-spread in bps. All functions are pure (no IO).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# k = 3 - 2*sqrt(2), the constant in the alpha expression.
_K = 3.0 - 2.0 * np.sqrt(2.0)


def _overnight_shift(h_t, l_t, h_t1, l_t1):
    """Per-pair overnight adjustment (Corwin–Schultz, Appendix).

    If period i+1's whole range sits above period i (gap up), shift it down so its
    low meets period i's high; if it sits below (gap down), shift it up to meet i's
    low. Removes the overnight return from the two-period high/low, which would
    otherwise inflate the volatility term `gamma`.

    Returns adjusted (h_t1, l_t1).

    Why (and is it "realistic"?): it isn't a price model — it's a bias-removal step under CS's
    diffusion assumption. An overnight gap inflates the two-period range `gamma = log(H2/L2)²`,
    which CS would misread as volatility and *overprice* the spread. Making period i+1 contiguous
    strips the gap return from `gamma` while leaving `beta` (the per-period ranges) untouched.
    Note: our intraday path calls CS with `overnight_adjust=False` (we pair only within-session
    bars), so this never runs on the resolution we actually use; it only matters for daily bars.
    EDGE sidesteps the issue entirely by consuming the previous close directly.
    """
    gap_up = l_t1 > h_t
    gap_dn = h_t1 < l_t
    shift = np.zeros_like(h_t1)
    shift = np.where(gap_up, l_t1 - h_t, shift)   # positive: subtract -> low meets H_t
    shift = np.where(gap_dn, h_t1 - l_t, shift)   # negative: subtract -> high meets L_t
    return h_t1 - shift, l_t1 - shift


def corwin_schultz(
    high: pd.Series,
    low: pd.Series,
    *,
    clamp_negative: bool = True,
    overnight_adjust: bool = True,
) -> pd.Series:
    """Proportional spread `S` per consecutive pair of periods.

    Parameters
    ----------
    high, low : pd.Series
        Aligned high and low prices, one row per period, in chronological order.
    clamp_negative : bool
        Set negative per-pair estimates to 0 (Corwin–Schultz's recommended
        treatment). Turn OFF for an unbiased mean (e.g. in tests/calibration).
    overnight_adjust : bool
        Apply the overnight-gap correction. Use for daily bars; leave off for
        within-session intraday bars (no overnight gap).

    Returns
    -------
    pd.Series
        Proportional spread `S` (e.g. 0.0010 = 10 bps round-trip), indexed to the
        *second* period of each pair (`high.index[1:]`).
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)

    h_t, l_t = h[:-1], l[:-1]
    h_t1, l_t1 = h[1:], l[1:]

    if overnight_adjust:
        h_t1, l_t1 = _overnight_shift(h_t, l_t, h_t1, l_t1)

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(h_t / l_t) ** 2 + np.log(h_t1 / l_t1) ** 2
        h2 = np.maximum(h_t, h_t1)
        l2 = np.minimum(l_t, l_t1)
        gamma = np.log(h2 / l2) ** 2

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    if clamp_negative:
        s = np.where(s < 0.0, 0.0, s)

    return pd.Series(s, index=high.index[1:], name="S")


def corwin_schultz_intraday(
    df: pd.DataFrame,
    *,
    clamp_negative: bool = True,
) -> pd.Series:
    """Apply the estimator within each trading session only.

    Pairs consecutive intraday bars but never across the overnight boundary, so
    the last bar of one day is not paired with the first of the next. `df` must be
    indexed by a DatetimeIndex and contain 'high' and 'low' columns.

    Returns the per-pair `S` series, indexed by the second bar's datetime.
    """
    parts = []
    for _, g in df.groupby(df.index.normalize()):
        if len(g) < 2:
            continue
        parts.append(
            corwin_schultz(
                g["high"], g["low"],
                clamp_negative=clamp_negative,
                overnight_adjust=False,
            )
        )
    if not parts:
        return pd.Series(dtype=float, name="S")
    return pd.concat(parts)


def half_spread_bps(s) -> "pd.Series | float":
    """Convert proportional spread `S` to a half-spread in basis points.

    A market order crosses half the spread per side, so the per-trade cost floor
    is `S/2`. Times 1e4 to express in bps.
    """
    return s / 2.0 * 1e4


def edge(open, high, low, close, *, sign: bool = False) -> float:
    """EDGE proportional spread `S` for one sample (Ardia, Guidotti & Kroencke 2024).

    Estimates the root-mean-square effective spread over the sample from OHLC prices,
    using the previous close to form cross-bar moment conditions and GMM-combining two
    estimators for minimum variance. Asymptotically unbiased under infrequent/discrete
    trading, where Corwin–Schultz is downward/negatively biased.

    Faithful numpy port of the reference `bidask.edge` (https://github.com/eguidotti/bidask).

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC prices, same length, chronological. Needs ≥3 observations.
    sign : bool
        If True, keep the sign of the squared-spread estimate (a signed `S`, for an unbiased
        mean over many windows). If False (default), return the non-negative `sqrt(|s²|)`.

    Returns
    -------
    float
        Proportional full spread `S` (e.g. 0.0010 = 10 bps round-trip), or `nan` if
        undefined (fewer than 3 obs, or no price variation to identify the spread).
    """
    o = np.log(np.asarray(open, dtype=float))
    h = np.log(np.asarray(high, dtype=float))
    l = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    nobs = len(o)
    if len(h) != nobs or len(l) != nobs or len(c) != nobs:
        raise ValueError("open, high, low, close must have the same length")
    if nobs < 3:
        return float("nan")

    m = (h + l) / 2.0

    # lag by one period; align current bars to t = 1..n-1
    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

    r1, r2, r3, r4, r5 = m - o, o - m1, m - c1, c1 - m1, o - c1

    # indicators: was there price variation this bar / at the open / at the prior close?
    tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan, ((h != l) | (l != c1)))
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
    po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, o != l)
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pt = np.nanmean(tau)
        po = np.nanmean(po1) + np.nanmean(po2)
        pc = np.nanmean(pc1) + np.nanmean(pc2)
        if np.nansum(tau) < 2 or po == 0 or pc == 0:
            return float("nan")

        # de-mean the returns that carry the spread signal, scaled by trade probability
        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau

        # two moment vectors → two spread estimators
        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4
        e1, e2 = np.nanmean(x1), np.nanmean(x2)
        v1 = np.nanmean(x1 ** 2) - e1 ** 2
        v2 = np.nanmean(x2 ** 2) - e2 ** 2

    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    s = np.sqrt(np.abs(s2))
    if sign:
        s *= np.sign(s2)
    return float(s)


def edge_intraday(df: pd.DataFrame, *, clamp_negative: bool = True) -> pd.Series:
    """Apply EDGE within each trading session — one `S` per day.

    Mirrors `corwin_schultz_intraday`: groups by session (never across the overnight
    boundary) and runs `edge()` on each day's bars (needs ≥3 per session). `df` must be a
    DatetimeIndex with 'open', 'high', 'low', 'close' columns.

    `clamp_negative=True` returns the non-negative per-session estimate; `False` returns the
    **signed** estimate (for an unbiased aggregate mean — clamp the aggregate, not each session,
    exactly as CS is used in `calibrate`). Returns one `S` per session, indexed by session date.
    """
    out: dict = {}
    for day, g in df.groupby(df.index.normalize()):
        if len(g) < 3:
            continue
        s = edge(g["open"].to_numpy(), g["high"].to_numpy(),
                 g["low"].to_numpy(), g["close"].to_numpy(), sign=not clamp_negative)
        if not np.isnan(s):
            out[day] = s
    return pd.Series(out, name="S") if out else pd.Series(dtype=float, name="S")


