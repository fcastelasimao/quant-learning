"""E01 — Alpha-decay curve: fixes S12-0a's soft spot, produces the scheduler's key input.

S12's 0a delay-fraction table ("fraction of a trade's eventual entry->exit move already used by
an entry delay h") mixed ALL hold lengths — its 4h/1d rows were dominated by short-hold trades
(the only ones with few enough total hold days to still be "in the sample" at long delays), and
S12 itself already found that hold < 2.4h trades are net LOSERS. This stage:

  1. Recomputes the delay-fraction table conditioned on hold_days >= 1 (the P&L-bearing subset).
  2. Adds a P&L-weighted variant (each trade's fraction observation weighted by |pnl|).
  3. Fits a smooth STRETCHED exponential g(h) = 1 - exp(-(h/tau)^k) (bootstrap CI on tau, k), per
     symbol — the scheduler's alpha-forfeiture input: "how much of the edge do we give up by
     delaying entry h minutes?" The stretch exponent k > 1 captures the empirical curve's flat
     first hour + sharp 2-4h knee that a single exponential (k=1) could not: the k=1 fit
     overcharged the short-delay forfeiture 2.5-3.5x across the scheduler's 15-120 min operating
     range (g(15m)=5.1% fitted vs 1.6% empirical). Audit fix 2026-07-09.

Deliverable: `slippage/alpha_decay.py::alpha_forfeit_frac(h_min, symbol)` — the fitted g(h),
baked as per-symbol (tau, k) constants measured here (same pattern as P02's chase-drag curve).

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/scheduler/01_alpha_decay/build_01_alpha_decay.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ"]
TRADES_ROOT = Path(__file__).resolve().parents[4] / "TQQQ_SQQQ_analysis"
OUT = Path(__file__).resolve().parent / "results"
DELAY_BARS = {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16, "1d": 26, "2d": 52}
DELAY_MIN = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 390, "2d": 780}
MIN_HOLD_DAYS = 1.0
N_BOOTSTRAP = 500

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


def load_15min_close(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")["close"].sort_index()


def load_trades(sym: str) -> pd.DataFrame:
    path = TRADES_ROOT / "full_history_canonical" / f"TRADES_{sym}_full_history.csv"
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])[
        ["entry_time", "exit_time", "hold_days", "pnl"]]


def delay_fraction_events(sym: str) -> pd.DataFrame:
    """Per-trade, per-delay fraction of the eventual entry->exit move already used, restricted
    to hold_days >= MIN_HOLD_DAYS (the P&L-bearing subset, per S12)."""
    trades = load_trades(sym)
    trades = trades[trades["hold_days"] >= MIN_HOLD_DAYS].reset_index(drop=True)
    px = load_15min_close(sym)
    idx, pv = px.index, px.values

    pos_e = idx.searchsorted(trades["entry_time"])
    pos_x = idx.searchsorted(trades["exit_time"])
    aligned = (pos_e < len(idx)) & (idx[np.clip(pos_e, 0, len(idx) - 1)] == trades["entry_time"].values)
    aligned &= pos_x < len(idx)
    trades, pos_e, pos_x = trades[aligned].reset_index(drop=True), pos_e[aligned], pos_x[aligned]
    p_entry, p_exit = pv[pos_e], pv[pos_x]
    total_move = p_exit - p_entry

    rows = []
    for label, h in DELAY_BARS.items():
        fwd = pos_e + h
        within_hold = fwd <= pos_x
        fwd_c = np.clip(fwd, 0, len(pv) - 1)
        delay_move = pv[fwd_c] - p_entry
        mask = within_hold & (np.abs(total_move) > 1e-6)
        frac = np.clip(delay_move[mask] / total_move[mask], -1.0, 2.0)   # guard pathological ratios
        rows.append(pd.DataFrame({
            "delay_label": label, "delay_min": DELAY_MIN[label], "frac": frac,
            "abs_pnl": np.abs(trades["pnl"].values[mask]),
        }))
    return pd.concat(rows, ignore_index=True)


def _g(h, tau, k):
    return 1.0 - np.exp(-((np.asarray(h, float) / tau) ** k))


def fit_g(events: pd.DataFrame, *, weighted: bool = False, n_boot: int = N_BOOTSTRAP) -> dict:
    """Fit the stretched exponential g(h) = 1 - exp(-(h/tau)^k) to the median (or pnl-weighted
    median) fraction per delay, with a bootstrap CI on (tau, k) (resampling trades within each
    delay bucket)."""
    def _summary(df):
        if weighted:
            return df.groupby("delay_min").apply(
                lambda g: np.average(g["frac"], weights=g["abs_pnl"] + 1e-9), include_groups=False)
        return df.groupby("delay_min")["frac"].median()

    # tau in minutes, k the stretch exponent; bounded away from 0 for a stable fit.
    p0, bounds = [150.0, 1.5], ([1.0, 0.3], [5000.0, 6.0])
    summary = _summary(events)
    hs, fs = summary.index.values.astype(float), summary.values
    popt, _ = curve_fit(_g, hs, fs, p0=p0, bounds=bounds, maxfev=10000)
    tau_hat, k_hat = float(popt[0]), float(popt[1])

    rng = np.random.default_rng(0)
    taus, ks = [], []
    for _ in range(n_boot):
        boot = events.groupby("delay_min", group_keys=False)[events.columns].apply(
            lambda g: g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)))
        s = _summary(boot)
        try:
            p, _ = curve_fit(_g, s.index.values.astype(float), s.values, p0=[tau_hat, k_hat],
                             bounds=bounds, maxfev=10000)
            taus.append(float(p[0]))
            ks.append(float(p[1]))
        except RuntimeError:
            continue
    taus, ks = np.array(taus), np.array(ks)
    return {"tau": tau_hat, "k": k_hat,
            "tau_p05": float(np.percentile(taus, 5)), "tau_p95": float(np.percentile(taus, 95)),
            "k_p05": float(np.percentile(ks, 5)), "k_p95": float(np.percentile(ks, 95)),
            "boot_taus": taus, "boot_ks": ks, "n_boot_ok": len(taus), "summary": summary}


def make_plot(fits: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    h_grid = np.linspace(1, 800, 200)
    for sym, fit in fits.items():
        ax.scatter(fit["summary"].index, fit["summary"].values, color=TICKER_COLORS[sym],
                  s=30, zorder=3, label=f"{sym} empirical (median)")
        ax.plot(h_grid, _g(h_grid, fit["tau"], fit["k"]), color=TICKER_COLORS[sym], lw=1.5,
               label=f"{sym} fit: g(h)=1-exp(-(h/{fit['tau']:.0f})^{fit['k']:.2f})")
        # Band = 5-95th pct of the fitted curve across bootstrap (tau, k) pairs (both vary).
        boot_curves = np.array([_g(h_grid, t, k) for t, k in zip(fit["boot_taus"], fit["boot_ks"])])
        ax.fill_between(h_grid, np.percentile(boot_curves, 5, axis=0),
                        np.percentile(boot_curves, 95, axis=0),
                        color=TICKER_COLORS[sym], alpha=0.15)
    ax.set_xlabel("entry delay h (min)")
    ax.set_ylabel("fraction of eventual edge forfeited")
    ax.set_title("Alpha-decay curve g(h)=1-exp(-(h/tau)^k) — hold >= 1 day subset")
    _style_ax(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fits, fits_w = {}, {}
    for sym in TICKERS:
        events = delay_fraction_events(sym)
        events.to_csv(OUT / f"delay_fraction_events_{sym}.csv", index=False)
        n_trades = events["delay_min"].value_counts().max()

        fit = fit_g(events, weighted=False)
        fit_w = fit_g(events, weighted=True)
        fits[sym], fits_w[sym] = fit, fit_w

        print(f"=== {sym} (hold >= {MIN_HOLD_DAYS}d, n~={n_trades} trades) ===")
        print(f"  unweighted median fractions by delay:\n{fit['summary'].round(3).to_string()}")
        print(f"  fit: tau={fit['tau']:.1f} min [90% CI {fit['tau_p05']:.1f}, {fit['tau_p95']:.1f}], "
              f"k={fit['k']:.3f} [90% CI {fit['k_p05']:.3f}, {fit['k_p95']:.3f}] "
              f"({fit['n_boot_ok']}/{N_BOOTSTRAP} bootstrap fits converged)")
        print(f"  g(15m)={_g(15, fit['tau'], fit['k']):.3f}, g(60m)={_g(60, fit['tau'], fit['k']):.3f}, "
              f"g(120m)={_g(120, fit['tau'], fit['k']):.3f}, g(390m)={_g(390, fit['tau'], fit['k']):.3f}")
        print(f"  pnl-weighted fit: tau={fit_w['tau']:.1f} min "
              f"[90% CI {fit_w['tau_p05']:.1f}, {fit_w['tau_p95']:.1f}], k={fit_w['k']:.3f} "
              f"[90% CI {fit_w['k_p05']:.3f}, {fit_w['k_p95']:.3f}]\n")

        pd.DataFrame([
            {"delay_min": h, "g_unweighted": _g(h, fit["tau"], fit["k"]),
             "g_pnl_weighted": _g(h, fit_w["tau"], fit_w["k"])}
            for h in DELAY_MIN.values()
        ]).to_csv(OUT / f"g_curve_{sym}.csv", index=False)

    make_plot(fits, OUT / "alpha_decay.png")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
