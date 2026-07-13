"""P01 — Market state: intraday volume profile, volume predictability, vol nowcast, spread state.

The client's "depending on the state of the market / market volume" input for the predictor
(P03) and scheduler (E03). Four measurements from real 15-min/daily bars, all on TQQQ/SQQQ/QQQ:

  1. Intraday volume profile: mean/median share of the day's volume per 15-min bin (the
     textbook U-shape), day-of-week effect, and per-bin dispersion.
  2. Interval-volume predictability: forecast a bin's volume as (its historical share) ×
     (a trailing EWMA of daily volume), evaluated out-of-sample; also the p10/p20 thin-tape
     floors per bin.
  3. Volatility nowcast: a trailing-window realized-vol estimate from 1-min returns, classified
     against S03's exact tercile bounds (same 20-day window, same full-sample split) so a
     regime label here means the same thing as a regime label in S03's tables.
  4. Spread state: the CS-15min half-spread curve by time-of-day bin (the historical/fallback
     estimate — findings_11 showed CS reads ~20-25% low vs SIP NBBO; live-facing use should
     prefer a fresh NBBO read, which `estimate_state(live_spread_bps=...)` supports).

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/predictor/01_market_state/build_01_market_state.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from slippage.spread import corwin_schultz_intraday, half_spread_bps  # noqa: E402
from slippage.state import session_bin_label, sigma_now_bps, classify_regime, VolRegimeBounds  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
OUT = Path(__file__).resolve().parent / "results"
EWMA_SPAN_DAYS = 10          # trailing daily-volume smoothing span
TEST_FRAC = 0.2              # last 20% of days held out for the OOS predictability check
NOWCAST_WINDOW_MIN = 120     # trailing window (minutes) for the intraday vol nowcast

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429", "QQQ": "#3fb950"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)
    for lab in (ax.yaxis.label, ax.xaxis.label, ax.title):
        lab.set_color(TEXT_COL)


# --------------------------------------------------------------------------- loaders
def load_15min(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, high, low, close, volume FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")


def load_1min(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_1min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")["close"]


def load_daily_close(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")["close"]


def s03_vol_regime_bounds(sym: str) -> VolRegimeBounds:
    """Reproduces research/03_delay_cost/build_03_delay_cost.py::vol_regime's tercile split
    exactly (20-day trailing realized vol, full-sample terciles), returned in bps units."""
    rv = load_daily_close(sym).pct_change().rolling(20).std()
    q1, q2 = rv.quantile([1 / 3, 2 / 3])
    return VolRegimeBounds(q1_bps=float(q1 * 1e4), q2_bps=float(q2 * 1e4))


# --------------------------------------------------------------------------- 1. volume profile
def session_bins(d: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the regular 09:30-15:45 session grid and tag each row with its date + bin."""
    d = d[(d.index.time >= pd.Timestamp("09:30").time()) &
          (d.index.time <= pd.Timestamp("15:45").time())].copy()
    d["date"] = d.index.normalize()
    d["bin"] = [session_bin_label(ts) for ts in d.index]
    d["dow"] = d.index.day_name()
    return d


def volume_profile(d: pd.DataFrame) -> pd.DataFrame:
    """Per-bin share of the day's volume: one row per (date, bin), plus the day's total."""
    daily_total = d.groupby("date")["volume"].transform("sum")
    d = d.copy()
    d["share"] = d["volume"] / daily_total
    return d


def profile_summary(dv: pd.DataFrame) -> pd.DataFrame:
    g = dv.groupby("bin")["share"]
    return pd.DataFrame({
        "mean": g.mean(), "median": g.median(),
        "p10": g.quantile(0.10), "p20": g.quantile(0.20),
        "std": g.std(), "n": g.size(),
    }).sort_index()


def dow_summary(dv: pd.DataFrame) -> pd.DataFrame:
    return dv.groupby(["dow", "bin"])["share"].mean().unstack("bin")


# --------------------------------------------------------------------------- 2. predictability
def predictability(dv: pd.DataFrame) -> dict:
    """OOS R^2 of predicted_volume = bin_share_mean(train) * trailing_daily_EWMA(strictly prior)."""
    daily_total = dv.groupby("date")["volume"].sum()   # each row's volume is its own bin's volume
    dates = np.sort(dv["date"].unique())
    n_test = max(1, int(len(dates) * TEST_FRAC))
    train_dates, test_dates = dates[:-n_test], dates[-n_test:]

    train = dv[dv["date"].isin(train_dates)]
    bin_share_mean = train.groupby("bin")["share"].mean()

    ewma = daily_total.ewm(span=EWMA_SPAN_DAYS, adjust=False).mean().shift(1)  # strictly prior day

    test = dv[dv["date"].isin(test_dates)].copy()
    test["pred_share"] = test["bin"].map(bin_share_mean)
    test["trailing_ewma"] = test["date"].map(ewma)
    test["pred_volume"] = test["pred_share"] * test["trailing_ewma"]
    test = test.dropna(subset=["pred_volume"])

    resid = test["volume"] - test["pred_volume"]
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((test["volume"] - test["volume"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n_test_bins": len(test), "n_test_days": len(test_dates), "r2_oos": r2}


# --------------------------------------------------------------------------- 3. vol nowcast validation
def nowcast_vs_daily_regime(sym: str, bounds: VolRegimeBounds) -> dict:
    """Agreement rate between the intraday nowcast (first NOWCAST_WINDOW_MIN of each session)
    and the day's own S03-style regime label — validates the nowcast is measuring the same thing."""
    px = load_1min(sym)
    px = px[(px.index.time >= pd.Timestamp("09:30").time()) &
            (px.index.time <= pd.Timestamp("15:45").time())]
    ret = np.log(px).diff()
    ret.index = px.index
    df = pd.DataFrame({"ret": ret})
    df["date"] = df.index.normalize()
    elapsed = (df.index - df["date"] - pd.Timedelta(hours=9, minutes=30))
    df["minute_of_session"] = elapsed.dt.total_seconds() / 60

    rv_daily = load_daily_close(sym).pct_change().rolling(20).std()
    q1, q2 = bounds.q1_bps / 1e4, bounds.q2_bps / 1e4
    daily_label = pd.cut(rv_daily, [-np.inf, q1, q2, np.inf], labels=["calm", "normal", "stress"])
    daily_label.index = daily_label.index.normalize()

    rows = []
    for date, g in df[df["minute_of_session"] < NOWCAST_WINDOW_MIN].groupby("date"):
        sig = sigma_now_bps(g["ret"].values)
        nowcast_label = classify_regime(sig, bounds)
        true_label = daily_label.get(date)
        if pd.notna(true_label):
            rows.append({"date": date, "nowcast": nowcast_label, "daily_label": true_label})
    agree = pd.DataFrame(rows)
    if agree.empty:
        return {"n_days": 0, "agreement_rate": float("nan")}
    return {"n_days": len(agree),
            "agreement_rate": float((agree["nowcast"] == agree["daily_label"]).mean())}


# --------------------------------------------------------------------------- 4. spread state
def spread_curve(d15: pd.DataFrame) -> pd.Series:
    """CS-15min half-spread by time-of-day bin (mirrors S01's intraday table)."""
    d15 = d15[(d15.index.time >= pd.Timestamp("09:30").time()) &
              (d15.index.time <= pd.Timestamp("15:45").time())].copy()
    d15["bin"] = [session_bin_label(ts) for ts in d15.index]
    s = corwin_schultz_intraday(d15[["high", "low"]], clamp_negative=False)
    hs = half_spread_bps(s)
    tmp = pd.DataFrame({"bin": d15["bin"].reindex(hs.index), "hs": hs}).dropna()
    return tmp.groupby("bin")["hs"].apply(lambda x: max(float(np.nanmean(x)), 0.0)).sort_index()


# --------------------------------------------------------------------------- plot
def make_plot(profiles: dict, spreads: dict, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    for t in TICKERS:
        p = profiles[t]
        x = range(len(p))
        ax1.plot(x, p["mean"] * 100, color=TICKER_COLORS[t], lw=1.5, label=t)
        ax1.fill_between(x, p["p10"] * 100, p["mean"] * 100, color=TICKER_COLORS[t], alpha=0.15)
    ax1.set_xticks(range(0, len(profiles["TQQQ"]), 4))
    ax1.set_xticklabels(profiles["TQQQ"].index[::4], rotation=45, fontsize=7)
    ax1.set_title("Intraday volume-share profile (mean, shaded to p10)")
    ax1.set_ylabel("% of day's volume in bin")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    for t in TICKERS:
        s = spreads[t]
        x = range(len(s))
        ax2.plot(x, s.values, color=TICKER_COLORS[t], lw=1.5, marker="o", ms=3, label=t)
    ax2.set_xticks(range(0, len(spreads["TQQQ"]), 4))
    ax2.set_xticklabels(spreads["TQQQ"].index[::4], rotation=45, fontsize=7)
    ax2.set_title("Intraday spread curve (CS-15min half-spread)")
    ax2.set_ylabel("half-spread (bps)")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    profiles, spreads, dow_tables = {}, {}, {}

    print("=== 1. Intraday volume-share profile ===\n")
    for t in TICKERS:
        d15 = load_15min(t)
        sb = session_bins(d15)
        dv = volume_profile(sb)
        prof = profile_summary(dv)
        profiles[t] = prof
        dow_tables[t] = dow_summary(dv)
        prof.to_csv(OUT / f"volume_profile_{t}.csv")
        open_bin, mid_bin = prof.loc["09:30", "mean"], prof.loc["12:30", "mean"]
        print(f"{t}: open bin (09:30) = {open_bin*100:.2f}% of day, "
              f"midday bin (12:30) = {mid_bin*100:.2f}% — ratio {open_bin/mid_bin:.1f}x")

        print("\n=== 2. Interval-volume predictability ===")
        pr = predictability(dv)
        print(f"  {t}: OOS R^2 = {pr['r2_oos']:.3f} over {pr['n_test_days']} held-out days "
              f"({pr['n_test_bins']} bin-observations)")

        print("\n=== 3. Volatility nowcast vs S03 daily regime ===")
        bounds = s03_vol_regime_bounds(t)
        nc = nowcast_vs_daily_regime(t, bounds)
        print(f"  {t}: q1={bounds.q1_bps:.1f} bps, q2={bounds.q2_bps:.1f} bps; "
              f"nowcast/daily-label agreement = {nc['agreement_rate']*100:.1f}% "
              f"({nc['n_days']} days)")

        print("\n=== 4. Spread state ===")
        sc = spread_curve(d15)
        spreads[t] = sc
        sc.to_csv(OUT / f"spread_curve_{t}.csv")
        # CS pairs consecutive bars and is indexed by the second bar, so 09:30 (the session's
        # first bar) never appears — 09:45 is the earliest bin with a spread estimate.
        print(f"  {t}: open bin (09:45) half-spread = {sc.loc['09:45']:.2f} bps, "
              f"midday (12:30) = {sc.loc['12:30']:.2f} bps\n")

    make_plot(profiles, spreads, OUT / "market_state.png")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
