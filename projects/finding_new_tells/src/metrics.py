"""Metric registry and all pre-registered metrics.

Each Metric has:
  compute(panel) -> pd.Series   — no lookahead at index t
  vote(series)   -> pd.Series of int in {-1, 0, +1}
  status         — "voting" or "watch" (watch metrics plot but don't vote)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Core dataclass + registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    name: str
    family: str
    compute: Callable[[pd.DataFrame], pd.Series]
    vote: Callable[[pd.Series], pd.Series]
    status: Literal["voting", "watch"] = "voting"


REGISTRY: dict[str, Metric] = {}


def register(m: Metric) -> Metric:
    REGISTRY[m.name] = m
    return m


# ---------------------------------------------------------------------------
# Vote helpers
# ---------------------------------------------------------------------------

def _pctile_vote(
    s: pd.Series,
    lo: float = 20.0,
    hi: float = 80.0,
    win: int = 252,
    sign: int = 1,
) -> pd.Series:
    """Rolling-percentile vote. sign=1 → high values bullish; sign=-1 → high values bearish."""
    min_p = max(win // 4, 20)
    q_lo = s.rolling(win, min_periods=min_p).quantile(lo / 100)
    q_hi = s.rolling(win, min_periods=min_p).quantile(hi / 100)
    v = pd.Series(0, index=s.index, dtype=int)
    v[s >= q_hi] = sign * 1
    v[s <= q_lo] = sign * -1
    v[s.isna()] = 0
    return v


def _binary_vote(s: pd.Series, positive_is_bullish: bool = True) -> pd.Series:
    """Map sign of series → {+1, -1}, 0 if NaN."""
    sign = 1 if positive_is_bullish else -1
    v = pd.Series(0, index=s.index, dtype=int)
    v[s > 0] = sign
    v[s < 0] = -sign
    v[s.isna()] = 0
    return v


def _threshold_vote(
    s: pd.Series,
    lo: float,
    hi: float,
    lo_vote: int = 1,
    hi_vote: int = -1,
) -> pd.Series:
    """Fixed-threshold vote."""
    v = pd.Series(0, index=s.index, dtype=int)
    v[s <= lo] = lo_vote
    v[s >= hi] = hi_vote
    v[s.isna()] = 0
    return v


def _watch_vote(s: pd.Series) -> pd.Series:
    """Neutral votes for metrics kept as diagnostics before promotion."""
    return pd.Series(0, index=s.index, dtype=int)


# ---------------------------------------------------------------------------
# Utility compute helpers
# ---------------------------------------------------------------------------

def _log_ret(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _yang_zhang_vol(panel: pd.DataFrame, sym: str, window: int = 20) -> pd.Series:
    """Yang-Zhang OHLC volatility estimator (annualized)."""
    o = np.log(panel[f"{sym}_open"] / panel[f"{sym}_close"].shift(1))
    c = np.log(panel[f"{sym}_close"] / panel[f"{sym}_open"])
    h = np.log(panel[f"{sym}_high"] / panel[f"{sym}_open"])
    lo = np.log(panel[f"{sym}_low"] / panel[f"{sym}_open"])
    n = window
    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    sigma_oc2 = o.rolling(n).var()
    sigma_cc2 = c.rolling(n).var()
    sigma_rs2 = (h * (h - c) + lo * (lo - c)).rolling(n).mean()
    return np.sqrt(252 * (sigma_oc2 + k * sigma_cc2 + (1 - k) * sigma_rs2)).rename(
        f"{sym}_yz_vol_{window}d"
    )


def _slope_20d(close: pd.Series) -> pd.Series:
    """OLS slope of log price over rolling 20-day window (annualized)."""
    log_p = np.log(close)
    x = np.arange(20)

    def _reg(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        slope, *_ = linregress(x, y)
        return slope * 252  # annualize

    return log_p.rolling(20).apply(_reg, raw=True)


def _normalized_hist_entropy(x: np.ndarray, *, bins: int = 8) -> float:
    """Normalized Shannon entropy of a rolling numeric window."""
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan
    if np.nanstd(x) <= 1e-12:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    probs = hist[hist > 0] / hist.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(bins))


def _sign_entropy(x: np.ndarray) -> float:
    """Normalized Shannon entropy of {-1, 0, +1} signs."""
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan
    signs = np.sign(x)
    counts = np.array([
        np.sum(signs < 0),
        np.sum(signs == 0),
        np.sum(signs > 0),
    ], dtype=float)
    probs = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(3))


def _sample_entropy(x: np.ndarray, *, m: int = 2, r_mult: float = 0.2) -> float:
    """Small-window sample entropy with a finite-sample correction."""
    x = x[~np.isnan(x)]
    if len(x) <= m + 2:
        return np.nan
    std = np.std(x)
    if std <= 1e-12:
        return 0.0
    r = r_mult * std

    def _match_count(order: int) -> int:
        n_templates = len(x) - order + 1
        if n_templates <= 1:
            return 0
        templates = np.array([x[i:i + order] for i in range(n_templates)])
        count = 0
        for i in range(n_templates - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = _match_count(m)
    a = _match_count(m + 1)
    if b == 0:
        return np.nan
    return float(-np.log((a + 1) / (b + 1)))


def _hurst_exponent(x: np.ndarray) -> float:
    """Estimate H from log-price difference scaling."""
    x = x[~np.isnan(x)]
    if len(x) < 64:
        return np.nan
    lags = np.array([2, 4, 8, 16, 32])
    lags = lags[lags < len(x) // 2]
    tau = []
    used_lags = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        sd = np.std(diff)
        if sd > 1e-12:
            tau.append(sd)
            used_lags.append(lag)
    if len(tau) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log(used_lags), np.log(tau), 1)
    return float(slope)


def _quadratic_curvature(x: np.ndarray) -> float:
    """Quadratic coefficient over a normalized time axis."""
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return np.nan
    t = np.linspace(-1.0, 1.0, len(x))
    coef = np.polyfit(t, x, 2)
    return float(coef[0])


def _available_close_frame(panel: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    data = {}
    for sym in symbols:
        col = f"{sym}_close"
        if col in panel.columns:
            data[sym] = panel[col]
    return pd.DataFrame(data, index=panel.index)


def _rolling_rmt_stat(panel: pd.DataFrame, *, window: int, stat: str) -> pd.Series:
    symbols = ["QQQ", "SPY", "HYG", "LQD", "TLT", "SHY", "GLD", "GLDM", "GSG", "^VIX", "^VIX3M", "^TNX", "^IRX"]
    closes = _available_close_frame(panel, symbols)
    if closes.shape[1] < 3:
        return pd.Series(np.nan, index=panel.index)

    returns = np.log(closes / closes.shift(1))
    out = pd.Series(np.nan, index=panel.index, dtype=float)
    for i in range(window - 1, len(returns)):
        sub = returns.iloc[i - window + 1:i + 1].dropna(axis=1, how="any")
        if sub.shape[1] < 3 or len(sub) < window:
            continue
        corr = sub.corr().to_numpy(dtype=float)
        if not np.isfinite(corr).all():
            continue
        if stat == "market_mode":
            eigvals = np.linalg.eigvalsh(corr)
            out.iloc[i] = float(eigvals[-1] / len(eigvals))
        elif stat == "mean_abs_corr":
            upper = corr[np.triu_indices_from(corr, k=1)]
            out.iloc[i] = float(np.mean(np.abs(upper)))
    return out


def _conditional_mutual_information(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """I(X;Y|Z), in nats, for small discrete arrays."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x = x[valid].astype(int)
    y = y[valid].astype(int)
    z = z[valid].astype(int)
    n = len(x)
    if n < 60:
        return np.nan

    c_xyz: dict[tuple[int, int, int], int] = {}
    c_xz: dict[tuple[int, int], int] = {}
    c_yz: dict[tuple[int, int], int] = {}
    c_z: dict[int, int] = {}
    for xi, yi, zi in zip(x, y, z):
        c_xyz[(xi, yi, zi)] = c_xyz.get((xi, yi, zi), 0) + 1
        c_xz[(xi, zi)] = c_xz.get((xi, zi), 0) + 1
        c_yz[(yi, zi)] = c_yz.get((yi, zi), 0) + 1
        c_z[zi] = c_z.get(zi, 0) + 1

    total = 0.0
    for (xi, yi, zi), n_xyz in c_xyz.items():
        p_xyz = n_xyz / n
        p_z = c_z[zi] / n
        p_xz = c_xz[(xi, zi)] / n
        p_yz = c_yz[(yi, zi)] / n
        total += p_xyz * np.log((p_xyz * p_z) / (p_xz * p_yz))
    return float(max(total, 0.0))


# ---------------------------------------------------------------------------
# Trend / Momentum metrics
# ---------------------------------------------------------------------------

def _compute_sma_regime(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return (c.rolling(50).mean() - c.rolling(200).mean()).rename("qqq_sma50_200_regime")


register(Metric(
    name="qqq_sma50_200_regime",
    family="trend",
    compute=_compute_sma_regime,
    # Inverted from positive_is_bullish=True: empirical 5d edges -170/-139 bps (train/val)
    # across all 5 horizons. "Death cross" appears to be a reversal signal for TQQQ
    # (over-extension exhausts on multi-day horizons).
    vote=lambda s: _binary_vote(s, positive_is_bullish=False),
))


def _compute_mom_12_1(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return (c.shift(21) / c.shift(252) - 1).rename("qqq_mom_12_1")


register(Metric(
    name="qqq_mom_12_1",
    family="trend",
    compute=_compute_mom_12_1,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=1),
    # Demoted to watch: edge sign flips across sub-windows (+16, -47, +26, -183 bps);
    # the 5/5-horizons-agree in aggregate is a pooling artifact.
    status="watch",
))


def _compute_20d_slope(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _slope_20d(c).rename("qqq_20d_slope")


register(Metric(
    name="qqq_20d_slope",
    family="trend",
    compute=_compute_20d_slope,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=1),
))


def _compute_mom_term_structure(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    m1 = np.sign(c / c.shift(21) - 1)
    m3 = np.sign(c / c.shift(63) - 1)
    m12 = np.sign(c / c.shift(252) - 1)
    agree = (m1 + m3 + m12).rename("qqq_mom_term_structure")
    return agree


register(Metric(
    name="qqq_mom_term_structure",
    family="trend",
    compute=_compute_mom_term_structure,
    vote=lambda s: _threshold_vote(s, lo=-2.5, hi=2.5, lo_vote=-1, hi_vote=1),
))


# ---------------------------------------------------------------------------
# Mean Reversion metrics
# ---------------------------------------------------------------------------

def _compute_rsi2(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("qqq_rsi2")


register(Metric(
    name="qqq_rsi2",
    family="mean_rev",
    compute=_compute_rsi2,
    vote=lambda s: _threshold_vote(s, lo=20, hi=80, lo_vote=1, hi_vote=-1),
))


def _compute_bb_z20(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    return ((c - mid) / (2 * std.replace(0, np.nan))).rename("qqq_bb_z20")


register(Metric(
    name="qqq_bb_z20",
    family="mean_rev",
    compute=_compute_bb_z20,
    vote=lambda s: _threshold_vote(s, lo=-1.0, hi=1.0, lo_vote=1, hi_vote=-1),
))


# ---------------------------------------------------------------------------
# Volatility regime metrics
# ---------------------------------------------------------------------------

def _rv(close: pd.Series, window: int) -> pd.Series:
    return np.sqrt(252 * _log_ret(close).pow(2).rolling(window).mean())


def _compute_rv_20d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _rv(c, 20).rename("qqq_rv_20d")


register(Metric(
    name="qqq_rv_20d",
    family="vol",
    compute=_compute_rv_20d,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=-1),
))


def _compute_rv_60d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _rv(c, 60).rename("qqq_rv_60d")


register(Metric(
    name="qqq_rv_60d",
    family="vol",
    compute=_compute_rv_60d,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=-1),
))


def _compute_rv_ratio(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    ratio = (_rv(c, 20) / _rv(c, 60).replace(0, np.nan)).rename("qqq_rv_ratio")
    return ratio


register(Metric(
    name="qqq_rv_ratio",
    family="vol",
    compute=_compute_rv_ratio,
    vote=lambda s: _threshold_vote(s, lo=0.8, hi=1.3, lo_vote=1, hi_vote=-1),
))


def _compute_yz_vol(panel: pd.DataFrame) -> pd.Series:
    if "QQQ_open" not in panel.columns:
        return pd.Series(dtype=float)
    return _yang_zhang_vol(panel, "QQQ", 20).rename("qqq_yz_vol_20d")


register(Metric(
    name="qqq_yz_vol_20d",
    family="vol",
    compute=_compute_yz_vol,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=-1),
))


def _compute_williams_vix_fix(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    lo = panel.get("QQQ_low")
    if c is None or lo is None:
        return pd.Series(dtype=float)
    highest_close = c.rolling(22).max()
    wvf = ((highest_close - lo) / highest_close.replace(0, np.nan) * 100)
    return wvf.rename("qqq_williams_vix_fix")


register(Metric(
    name="qqq_williams_vix_fix",
    family="vol",
    compute=_compute_williams_vix_fix,
    # Inverted from sign=-1: empirical 5d edges -254/-103 bps (train/val) across all 5 horizons
    # indicate high-fear (high WVF) days precede positive TQQQ returns (buy-the-dip).
    vote=lambda s: _pctile_vote(s, lo=10, hi=90, win=252, sign=1),
))


def _compute_vix_term_structure(panel: pd.DataFrame) -> pd.Series:
    # Preferred: ^VIX / ^VIX3M
    if "^VIX_close" in panel.columns and "^VIX3M_close" in panel.columns:
        ratio = (panel["^VIX_close"] / panel["^VIX3M_close"].replace(0, np.nan))
        return ratio.rename("vix_term_structure")
    # Fallback: 20d change of VIXY/VXZ ratio
    if "VIXY_close" in panel.columns and "VXZ_close" in panel.columns:
        ratio = panel["VIXY_close"] / panel["VXZ_close"].replace(0, np.nan)
        return ratio.pct_change(20, fill_method=None).rename("vix_term_structure")
    return pd.Series(dtype=float)


register(Metric(
    name="vix_term_structure",
    family="vol",
    compute=_compute_vix_term_structure,
    vote=lambda s: _threshold_vote(s, lo=0.90, hi=1.05, lo_vote=1, hi_vote=-1),
))


# ---------------------------------------------------------------------------
# Leveraged-ETF-specific metrics
# ---------------------------------------------------------------------------

def _compute_vol_drag(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    L = 3.0
    rv = _rv(c, 20)
    drag = 0.5 * L * (L - 1) * rv ** 2
    return drag.rename("tqqq_vol_drag_est")


register(Metric(
    name="tqqq_vol_drag_est",
    family="lev_etf",
    compute=_compute_vol_drag,
    vote=lambda s: _pctile_vote(s, lo=25, hi=75, win=252, sign=-1),
))


def _compute_path_residual(panel: pd.DataFrame) -> pd.Series:
    tc = panel.get("TQQQ_close")
    qc = panel.get("QQQ_close")
    if tc is None or qc is None:
        return pd.Series(dtype=float)
    window = 60
    tqqq_ret = tc.pct_change(fill_method=None)
    qqq_ret = qc.pct_change(fill_method=None)
    tqqq_cum = tqqq_ret.rolling(window, min_periods=window).apply(lambda r: (1 + r).prod() - 1, raw=True)
    qqq_cum = qqq_ret.rolling(window, min_periods=window).apply(lambda r: (1 + r).prod() - 1, raw=True)
    return (tqqq_cum - 3 * qqq_cum).rename("tqqq_path_residual")


register(Metric(
    name="tqqq_path_residual",
    family="lev_etf",
    compute=_compute_path_residual,
    vote=lambda s: _pctile_vote(s, lo=10, hi=90, win=252, sign=1),
))


def _compute_dd_from_high(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return ((c - c.rolling(252).max()) / c.rolling(252).max()).rename("qqq_dd_from_high")


register(Metric(
    name="qqq_dd_from_high",
    family="lev_etf",
    compute=_compute_dd_from_high,
    vote=lambda s: pd.Series(0, index=s.index, dtype=int),  # watch-only
    status="watch",
))


# ---------------------------------------------------------------------------
# Cross-asset / macro metrics
# ---------------------------------------------------------------------------

def _compute_qqq_spy_slope(panel: pd.DataFrame) -> pd.Series:
    qc = panel.get("QQQ_close")
    sc = panel.get("SPY_close")
    if qc is None or sc is None:
        return pd.Series(dtype=float)
    ratio = qc / sc.replace(0, np.nan)
    return _slope_20d(ratio).rename("qqq_spy_ratio_slope")


register(Metric(
    name="qqq_spy_ratio_slope",
    family="cross",
    compute=_compute_qqq_spy_slope,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=1),
))


def _compute_yield_curve(panel: pd.DataFrame) -> pd.Series:
    # Preferred: ^TNX - ^IRX (in %)
    if "^TNX_close" in panel.columns and "^IRX_close" in panel.columns:
        return (panel["^TNX_close"] - panel["^IRX_close"]).rename("yield_curve_10y3m")
    # Fallback: 20d change of TLT/SHY ratio
    if "TLT_close" in panel.columns and "SHY_close" in panel.columns:
        ratio = panel["TLT_close"] / panel["SHY_close"].replace(0, np.nan)
        return ratio.pct_change(20, fill_method=None).rename("yield_curve_10y3m")
    return pd.Series(dtype=float)


register(Metric(
    name="yield_curve_10y3m",
    family="cross",
    compute=_compute_yield_curve,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=1),
    # Demoted to watch: val_b sub-window has zero bear votes (all 212 bull), val_a has
    # only 15 bull obs — "promising" leaderboard label is a sample-imbalance artifact.
    status="watch",
))


def _compute_tnx_chg(panel: pd.DataFrame) -> pd.Series:
    # Preferred: 20d change of ^TNX (rising rates = bearish for tech)
    if "^TNX_close" in panel.columns:
        return panel["^TNX_close"].diff(20).rename("tnx_20d_chg")
    # Fallback: -TLT 20d return (TLT falls when rates rise)
    if "TLT_close" in panel.columns:
        return (-panel["TLT_close"].pct_change(20, fill_method=None)).rename("tnx_20d_chg")
    return pd.Series(dtype=float)


register(Metric(
    name="tnx_20d_chg",
    family="cross",
    compute=_compute_tnx_chg,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=-1),
))


def _compute_hyg_lqd(panel: pd.DataFrame) -> pd.Series:
    hc = panel.get("HYG_close")
    lc = panel.get("LQD_close")
    if hc is None or lc is None:
        return pd.Series(dtype=float)
    ratio = hc / lc.replace(0, np.nan)
    return ratio.pct_change(20, fill_method=None).rename("hyg_lqd_ratio_chg")


register(Metric(
    name="hyg_lqd_ratio_chg",
    family="cross",
    compute=_compute_hyg_lqd,
    vote=lambda s: _pctile_vote(s, lo=20, hi=80, win=252, sign=1),
))


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------

def _compute_volume_z(panel: pd.DataFrame) -> pd.Series:
    v = panel.get("TQQQ_volume")
    c = panel.get("TQQQ_close")
    if v is None or c is None:
        return pd.Series(dtype=float)
    dv = v * c  # dollar volume
    mean_ = dv.rolling(20).mean()
    std_ = dv.rolling(20).std()
    return ((dv - mean_) / std_.replace(0, np.nan)).rename("tqqq_volume_z")


register(Metric(
    name="tqqq_volume_z",
    family="micro",
    compute=_compute_volume_z,
    vote=lambda s: _pctile_vote(s, lo=10, hi=90, win=252, sign=-1),
))


# ---------------------------------------------------------------------------
# Calendar — FOMC drift
# ---------------------------------------------------------------------------

# Approximate FOMC decision dates (day AFTER 2-day meeting ends) 2003–2026
# Replace with actual Fed calendar if available.
_FOMC_DATES = pd.to_datetime([
    # 2003
    "2003-01-29","2003-03-18","2003-05-06","2003-06-25","2003-08-12",
    "2003-09-16","2003-10-28","2003-12-09",
    # 2004
    "2004-01-28","2004-03-16","2004-05-04","2004-06-30","2004-08-10",
    "2004-09-21","2004-11-10","2004-12-14",
    # 2005
    "2005-02-02","2005-03-22","2005-05-03","2005-06-30","2005-08-09",
    "2005-09-20","2005-11-01","2005-12-13",
    # 2006
    "2006-01-31","2006-03-28","2006-05-10","2006-06-29","2006-08-08",
    "2006-09-20","2006-10-25","2006-12-12",
    # 2007
    "2007-01-31","2007-03-21","2007-05-09","2007-06-28","2007-08-07",
    "2007-09-18","2007-10-31","2007-12-11",
    # 2008
    "2008-01-30","2008-03-18","2008-04-30","2008-06-25","2008-08-05",
    "2008-09-16","2008-10-29","2008-12-16",
    # 2009
    "2009-01-28","2009-03-18","2009-04-29","2009-06-24","2009-08-12",
    "2009-09-23","2009-11-04","2009-12-16",
    # 2010
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10",
    "2010-09-21","2010-11-03","2010-12-14",
    # 2011
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09",
    "2011-09-21","2011-11-02","2011-12-13",
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01",
    "2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31",
    "2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30",
    "2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29",
    "2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-10",
    # 2026
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29",
    "2026-09-16","2026-10-28","2026-12-09",
])


def _compute_fomc_drift(panel: pd.DataFrame) -> pd.Series:
    idx = panel.index
    s = pd.Series(0.0, index=idx)
    for fdate in _FOMC_DATES:
        loc = idx.get_indexer([fdate], method="bfill")[0]
        if loc <= 0:
            continue
        for k in (1, 2):
            if loc - k >= 0:
                s.iloc[loc - k] = 1.0
    return s.rename("fomc_drift")


register(Metric(
    name="fomc_drift",
    family="calendar",
    compute=_compute_fomc_drift,
    vote=lambda s: s.fillna(0).astype(int),
))


# ---------------------------------------------------------------------------
# Watch-only metrics
# ---------------------------------------------------------------------------

def _compute_skew_60d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _log_ret(c).rolling(60).skew().rename("qqq_skew_60d")


register(Metric(
    name="qqq_skew_60d",
    family="vol",
    compute=_compute_skew_60d,
    vote=lambda s: pd.Series(0, index=s.index, dtype=int),
    status="watch",
))


def _compute_kurt_60d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _log_ret(c).rolling(60).kurt().rename("qqq_kurt_60d")


register(Metric(
    name="qqq_kurt_60d",
    family="vol",
    compute=_compute_kurt_60d,
    vote=_watch_vote,
    status="watch",
))


# ---------------------------------------------------------------------------
# Physics-inspired watch metrics
# ---------------------------------------------------------------------------

def _compute_sign_entropy_20d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _log_ret(c).rolling(20, min_periods=10).apply(
        _sign_entropy, raw=True
    ).rename("qqq_sign_entropy_20d")


register(Metric(
    name="qqq_sign_entropy_20d",
    family="physics_entropy",
    compute=_compute_sign_entropy_20d,
    vote=_watch_vote,
    status="watch",
))


def _compute_return_entropy_60d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _log_ret(c).rolling(60, min_periods=30).apply(
        _normalized_hist_entropy, raw=True
    ).rename("qqq_return_entropy_60d")


register(Metric(
    name="qqq_return_entropy_60d",
    family="physics_entropy",
    compute=_compute_return_entropy_60d,
    vote=_watch_vote,
    status="watch",
))


def _compute_sample_entropy_60d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return _log_ret(c).rolling(60, min_periods=45).apply(
        _sample_entropy, raw=True
    ).rename("qqq_sample_entropy_60d")


register(Metric(
    name="qqq_sample_entropy_60d",
    family="physics_entropy",
    compute=_compute_sample_entropy_60d,
    vote=_watch_vote,
    status="watch",
))


def _compute_entropy_return_ratio_60d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    entropy = _compute_return_entropy_60d(panel)
    ret_60d = c / c.shift(60) - 1
    return (ret_60d / entropy.replace(0, np.nan)).rename("qqq_entropy_return_ratio_60d")


register(Metric(
    name="qqq_entropy_return_ratio_60d",
    family="physics_entropy",
    compute=_compute_entropy_return_ratio_60d,
    vote=_watch_vote,
    status="watch",
))


def _compute_vix_to_qqq_transfer_entropy(panel: pd.DataFrame) -> pd.Series:
    qqq = panel.get("QQQ_close")
    vix = panel.get("^VIX_close")
    if qqq is None or vix is None:
        return pd.Series(dtype=float)

    qqq_sign = np.sign(_log_ret(qqq))
    vix_sign = np.sign(_log_ret(vix))
    terms = pd.DataFrame({
        "x": vix_sign.shift(1),
        "y": qqq_sign,
        "z": qqq_sign.shift(1),
    }, index=panel.index)

    window = 252
    out = pd.Series(np.nan, index=panel.index, dtype=float)
    for i in range(window - 1, len(terms)):
        sub = terms.iloc[i - window + 1:i + 1]
        out.iloc[i] = _conditional_mutual_information(
            sub["x"].to_numpy(dtype=float),
            sub["y"].to_numpy(dtype=float),
            sub["z"].to_numpy(dtype=float),
        )
    return out.rename("vix_to_qqq_transfer_entropy_252d")


register(Metric(
    name="vix_to_qqq_transfer_entropy_252d",
    family="physics_entropy",
    compute=_compute_vix_to_qqq_transfer_entropy,
    vote=_watch_vote,
    status="watch",
))


def _compute_hurst_126d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return np.log(c).rolling(126, min_periods=90).apply(
        _hurst_exponent, raw=True
    ).rename("qqq_hurst_126d")


register(Metric(
    name="qqq_hurst_126d",
    family="physics_scaling",
    compute=_compute_hurst_126d,
    vote=_watch_vote,
    status="watch",
))


def _compute_lppl_curvature_126d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    return np.log(c).rolling(126, min_periods=90).apply(
        _quadratic_curvature, raw=True
    ).rename("qqq_lppl_curvature_126d")


register(Metric(
    name="qqq_lppl_curvature_126d",
    family="physics_criticality",
    compute=_compute_lppl_curvature_126d,
    vote=_watch_vote,
    status="watch",
))


def _compute_rmt_market_mode_126d(panel: pd.DataFrame) -> pd.Series:
    return _rolling_rmt_stat(panel, window=126, stat="market_mode").rename(
        "rmt_market_mode_126d"
    )


register(Metric(
    name="rmt_market_mode_126d",
    family="physics_rmt",
    compute=_compute_rmt_market_mode_126d,
    vote=_watch_vote,
    status="watch",
))


def _compute_rmt_mean_abs_corr_126d(panel: pd.DataFrame) -> pd.Series:
    return _rolling_rmt_stat(panel, window=126, stat="mean_abs_corr").rename(
        "rmt_mean_abs_corr_126d"
    )


register(Metric(
    name="rmt_mean_abs_corr_126d",
    family="physics_rmt",
    compute=_compute_rmt_mean_abs_corr_126d,
    vote=_watch_vote,
    status="watch",
))


def _compute_vix_implied_realized_gap_20d(panel: pd.DataFrame) -> pd.Series:
    qqq = panel.get("QQQ_close")
    vix = panel.get("^VIX_close")
    if qqq is None or vix is None:
        return pd.Series(dtype=float)
    implied = vix / 100.0
    realized = _rv(qqq, 20)
    return (implied - realized).rename("vix_implied_realized_gap_20d")


register(Metric(
    name="vix_implied_realized_gap_20d",
    family="physics_options",
    compute=_compute_vix_implied_realized_gap_20d,
    vote=_watch_vote,
    status="watch",
))


def _compute_vix_vol_of_vol_20d(panel: pd.DataFrame) -> pd.Series:
    vix = panel.get("^VIX_close")
    if vix is None:
        return pd.Series(dtype=float)
    vix_ret = _log_ret(vix.replace(0, np.nan))
    return np.sqrt(252 * vix_ret.pow(2).rolling(20).mean()).rename("vix_vol_of_vol_20d")


register(Metric(
    name="vix_vol_of_vol_20d",
    family="physics_options",
    compute=_compute_vix_vol_of_vol_20d,
    vote=_watch_vote,
    status="watch",
))


def _compute_cross_asset_herding_alignment_20d(panel: pd.DataFrame) -> pd.Series:
    symbols = ["QQQ", "SPY", "HYG", "LQD", "TLT", "SHY", "GLD", "GLDM", "GSG"]
    closes = _available_close_frame(panel, symbols)
    if "QQQ" not in closes.columns or closes.shape[1] < 3:
        return pd.Series(np.nan, index=panel.index)
    signs = np.sign(np.log(closes / closes.shift(1)))
    qqq_sign = signs["QQQ"]
    aligned = signs.eq(qqq_sign, axis=0)
    nonzero = signs.ne(0) & qqq_sign.ne(0).to_numpy()[:, None]
    alignment = (aligned & nonzero).sum(axis=1) / nonzero.sum(axis=1).replace(0, np.nan)
    return alignment.rolling(20, min_periods=10).mean().rename(
        "cross_asset_herding_alignment_20d"
    )


register(Metric(
    name="cross_asset_herding_alignment_20d",
    family="physics_herding",
    compute=_compute_cross_asset_herding_alignment_20d,
    vote=_watch_vote,
    status="watch",
))


def _compute_kelly_fraction_252d(panel: pd.DataFrame) -> pd.Series:
    c = panel.get("QQQ_close")
    if c is None:
        return pd.Series(dtype=float)
    ret = c.pct_change(fill_method=None)
    mu = ret.rolling(252, min_periods=126).mean()
    var = ret.rolling(252, min_periods=126).var()
    return (mu / var.replace(0, np.nan)).clip(-3, 3).rename("qqq_kelly_fraction_252d")


register(Metric(
    name="qqq_kelly_fraction_252d",
    family="physics_sizing",
    compute=_compute_kelly_fraction_252d,
    vote=_watch_vote,
    status="watch",
))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_all(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Run every registered metric. Returns {name: series}."""
    return {name: m.compute(panel) for name, m in REGISTRY.items()}


def vote_all(panel: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Return DataFrame of votes for voting metrics.

    Parameters
    ----------
    names : list[str], optional
        Subset of metric names to compute. Defaults to all voting metrics.
    """
    result = {}
    for name, m in REGISTRY.items():
        if m.status != "voting":
            continue
        if names is not None and name not in names:
            continue
        s = m.compute(panel)
        result[name] = m.vote(s)
    return pd.DataFrame(result, index=panel.index)
