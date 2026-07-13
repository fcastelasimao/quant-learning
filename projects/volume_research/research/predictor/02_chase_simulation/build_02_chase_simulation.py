"""P02 — Chase simulation: full-history cross-vs-chase entry cost, and the optimal timeout.

Replaces the n=15 live estimate of the +14 bps limit-chase drag (S08) with a full-history
simulation on 1-min bars, and answers "if we still chase, what timeout?"

At every historical entry signal (TRADES CSVs: entry_time — always a BUY, TQQQ and SQQQ are each
a long-only book) simulate two entry styles on 1-min bars.

**Data-integrity note:** the TRADES CSVs' own `decision_price` column is FMP-sourced and is
NOT on the same split-adjustment basis as this project's Alpaca-sourced DB candles — a direct
check found a stable ~2.0x ratio for TQQQ and ~0.2x for SQQQ across nearly the entire history
(only converging near 2026), consistent with differing cumulative split adjustments. Mixing the
two price sources as absolute levels is unsafe. So — exactly like S12's 0a analysis — this stage
uses ONLY the DB's own price series: "decision_price" here means the DB's 1-min bar **open** at
`entry_time`, not the CSV column of the same name.

  Style A — cross at decision: pay the spread immediately.
    cost_A = half_spread(bin) + within-minute drift proxy
    (the drift proxy is the move from decision_price to the decision-bar's own close — you can't
    literally trade at the instant of the signal, so even "crossing now" eats ~1 bar of drift)

  Style B — rest a passive buy limit AT decision_price, timeout T, then cross:
    filled (cost=0, a maker fill, no spread paid) if any 1-min bar's LOW trades <= decision_price
    within the first T minutes; otherwise cross at the price T minutes out:
    cost_B(T) = drift(decision_price -> price at T) + half_spread(bin at T)
    Run the fill test with both <= and < as a sensitivity (the boundary case is genuinely
    ambiguous on bar data — we don't have queue position).

Tier: Modeled — 1-min bars can't see intraminute queue dynamics; there is no queue-position
model here, and that assumption is carried through the findings prominently.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/predictor/02_chase_simulation/build_02_chase_simulation.py
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
from slippage.state import session_bin_label  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ"]
TIMEOUTS_MIN = [1, 2, 5, 10, 15, 30, 60]
TRADES_ROOT = Path(__file__).resolve().parents[4] / "TQQQ_SQQQ_analysis"
OUT = Path(__file__).resolve().parent / "results"

# Reference numbers this stage reconciles against.
S08_LIVE_BUY_MEAN_BPS = 14.2   # findings_08, n=15, buy fills, limit-chase
S08_LIVE_BUY_STD_BPS = 35.0
S12_AVG_15MIN_DRIFT_BPS = 1.9   # findings_12 0a decay table, TQQQ mean @15m ("~2 bps")

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429"}


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
def load_1min(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, open, close, low FROM candles_1min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")


def load_15min_spread_curve(sym: str) -> pd.Series:
    """CS-15min half-spread by time-of-day bin (mirrors P01's spread_curve)."""
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, high, low FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    d = d.set_index("dt")
    d = d[(d.index.time >= pd.Timestamp("09:30").time()) &
          (d.index.time <= pd.Timestamp("15:45").time())].copy()
    d["bin"] = [session_bin_label(ts) for ts in d.index]
    s = corwin_schultz_intraday(d[["high", "low"]], clamp_negative=False)
    hs = half_spread_bps(s)
    tmp = pd.DataFrame({"bin": d["bin"].reindex(hs.index), "hs": hs}).dropna()
    curve = tmp.groupby("bin")["hs"].apply(lambda x: max(float(np.nanmean(x)), 0.0))
    # 09:30 has no CS estimate (indexed by the pair's second bar); backfill from 09:45.
    if "09:30" not in curve.index and "09:45" in curve.index:
        curve.loc["09:30"] = curve.loc["09:45"]
    return curve.sort_index()


def load_trades(sym: str) -> pd.DataFrame:
    """Only `entry_time` is used — see the module docstring for why the CSV's own
    `decision_price` column is not safe to mix with the DB's price series."""
    path = TRADES_ROOT / "full_history_canonical" / f"TRADES_{sym}_full_history.csv"
    return pd.read_csv(path, parse_dates=["entry_time"])[["entry_time"]]


def vol_regime(sym: str) -> pd.Series:
    """Reproduces S03's exact 20-day trailing realized-vol tercile split."""
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    rv = d.set_index("dt")["close"].pct_change().rolling(20).std()
    q1, q2 = rv.quantile([1 / 3, 2 / 3])
    reg = pd.cut(rv, [-np.inf, q1, q2, np.inf], labels=["calm", "normal", "stress"])
    reg.index = reg.index.normalize()
    return reg


# --------------------------------------------------------------------------- simulation
def simulate(sym: str) -> pd.DataFrame:
    px = load_1min(sym)
    spread_curve = load_15min_spread_curve(sym)
    trades = load_trades(sym)
    idx = px.index
    pv_open, pv_close, pv_low = px["open"].values, px["close"].values, px["low"].values

    pos = idx.searchsorted(trades["entry_time"])
    aligned = (pos < len(idx)) & (idx[np.clip(pos, 0, len(idx) - 1)] == trades["entry_time"].values)
    trades = trades[aligned].reset_index(drop=True)
    pos = pos[aligned]

    bins = np.array([session_bin_label(ts) for ts in trades["entry_time"]])
    half_spread_now = spread_curve.reindex(bins).values

    rows = []
    max_h = max(TIMEOUTS_MIN)
    for i in range(len(trades)):
        p0 = pos[i]
        dp = pv_open[p0]   # DB-native "decision price" (see module docstring)
        if p0 + 1 + max_h >= len(pv_close) or not np.isfinite(dp) or dp <= 0:
            continue
        rec = {"entry_time": trades["entry_time"].iloc[i], "bin": bins[i], "decision_price": dp}

        # Style A: cross now = spread + within-minute drift proxy (decision bar's own close).
        drift_a = (pv_close[p0] - dp) / dp * 1e4
        rec["cost_A_bps"] = half_spread_now[i] + drift_a

        # Style B: passive limit at dp, timeout T, then cross. The wait window starts at bar
        # p0+1 (strictly AFTER the decision bar) — bar p0's own low is tautologically <= its
        # own open (low is defined as the minimum of the bar including the open print), so
        # including it would make every limit "fill" instantly regardless of what happens next.
        window_low = pv_low[p0 + 1:p0 + 1 + max_h]
        cum_min_low = np.minimum.accumulate(window_low)
        for T in TIMEOUTS_MIN:
            for op, suffix in ((np.less_equal, "le"), (np.less, "lt")):
                filled = bool(op(cum_min_low[T - 1], dp)) if T - 1 < len(cum_min_low) else False
                if filled:
                    cost = 0.0
                else:
                    p_T = pv_close[p0 + T]
                    bin_T = session_bin_label(idx[p0 + T])
                    hs_T = spread_curve.get(bin_T, half_spread_now[i])
                    cost = (p_T - dp) / dp * 1e4 + hs_T
                rec[f"cost_B_T{T}_{suffix}_bps"] = cost
                rec[f"filled_T{T}_{suffix}"] = filled
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, sym: str, reg: pd.Series) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["regime"] = reg.reindex(df["entry_time"].dt.normalize()).astype(object).values

    def _stats(v, label, extra=None):
        v = v.dropna()
        row = {"ticker": sym, "style": label, "n": len(v),
               "mean_bps": v.mean(), "std_bps": v.std(), "p90_bps": v.quantile(0.90)}
        if extra:
            row.update(extra)
        return row

    rows.append(_stats(df["cost_A_bps"], "A_cross"))
    for T in TIMEOUTS_MIN:
        for suffix in ("le", "lt"):
            rows.append(_stats(df[f"cost_B_T{T}_{suffix}_bps"], f"B_T{T}_{suffix}",
                              {"fill_rate": df[f"filled_T{T}_{suffix}"].mean()}))
    overall = pd.DataFrame(rows)

    by_regime_rows = []
    for T in (15,):
        for suffix in ("le",):
            for rg, g in df.groupby("regime", observed=True):
                by_regime_rows.append({"ticker": sym, "timeout": T, "regime": rg,
                                       "mean_bps": g[f"cost_B_T{T}_{suffix}_bps"].mean(), "n": len(g)})
    by_bin_rows = []
    for T in (15,):
        for suffix in ("le",):
            for b, g in df.groupby("bin"):
                by_bin_rows.append({"ticker": sym, "timeout": T, "bin": b,
                                    "mean_bps": g[f"cost_B_T{T}_{suffix}_bps"].mean(), "n": len(g)})
    return overall, pd.DataFrame(by_regime_rows), pd.DataFrame(by_bin_rows)


def make_plot(summaries: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    for sym, ov in summaries.items():
        b_le = ov[ov.style.str.startswith("B_") & ov.style.str.endswith("_le")]
        Ts = [int(s.split("_")[1][1:]) for s in b_le["style"]]
        ax.plot(Ts, b_le["mean_bps"], marker="o", color=TICKER_COLORS[sym], label=f"{sym} chase (Style B)")
        a_cost = ov[ov.style == "A_cross"]["mean_bps"].iloc[0]
        ax.axhline(a_cost, color=TICKER_COLORS[sym], ls="--", lw=1.0, alpha=0.6,
                   label=f"{sym} cross (Style A) = {a_cost:.1f} bps")
    ax.axhline(S08_LIVE_BUY_MEAN_BPS, color="#f78166", ls=":", lw=1.5,
               label=f"S08 live limit-chase = {S08_LIVE_BUY_MEAN_BPS} bps (n=15)")
    ax.set_xlabel("timeout T (min)")
    ax.set_ylabel("expected entry cost (bps)")
    ax.set_title("Cross vs chase: expected entry cost by timeout")
    _style_ax(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7, labelcolor=TEXT_COL)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for sym in TICKERS:
        print(f"=== {sym}: simulating {len(load_trades(sym))} historical entries on 1-min bars ===")
        df = simulate(sym)
        df.to_csv(OUT / f"chase_sim_events_{sym}.csv", index=False)
        reg = vol_regime(sym)
        overall, by_regime, by_bin = summarize(df, sym, reg)
        overall.to_csv(OUT / f"chase_sim_summary_{sym}.csv", index=False)
        by_regime.to_csv(OUT / f"chase_sim_by_regime_{sym}.csv", index=False)
        by_bin.to_csv(OUT / f"chase_sim_by_bin_{sym}.csv", index=False)
        summaries[sym] = overall

        a = overall[overall.style == "A_cross"].iloc[0]
        print(f"  Style A (cross): mean {a.mean_bps:.2f} bps, std {a.std_bps:.2f}, "
              f"p90 {a.p90_bps:.2f} (n={a.n:.0f})")
        for T in TIMEOUTS_MIN:
            b = overall[overall.style == f"B_T{T}_le"].iloc[0]
            print(f"  Style B T={T:2}min: mean {b.mean_bps:6.2f} bps, std {b.std_bps:5.2f}, "
                  f"p90 {b.p90_bps:6.2f}, fill_rate {b.fill_rate*100:5.1f}% (n={b.n:.0f})")
        b15 = overall[overall.style == "B_T15_le"].iloc[0]
        print(f"  Reconciliation: sim Style-B@15min mean = {b15.mean_bps:.2f} bps vs "
              f"S08 live limit-chase +{S08_LIVE_BUY_MEAN_BPS} bps (n=15) "
              f"vs S12 avg-15min-drift {S12_AVG_15MIN_DRIFT_BPS} bps. "
              f"Adverse-selection gap (S08 - sim) = {S08_LIVE_BUY_MEAN_BPS - b15.mean_bps:.2f} bps.\n")

    make_plot(summaries, OUT / "chase_simulation.png")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
