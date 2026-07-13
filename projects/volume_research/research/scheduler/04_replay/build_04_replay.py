"""E04 — Historical replay harness: prove the scheduler on history.

Replays `schedule_order` at every historical TQQQ/SQQQ entry (1-min bars), for a size sweep
($100k -> $20M), against two baselines: a naive single fill at the 15-min decision cadence, and a
day-VWAP fill (the patient-algo benchmark). **Interruptions are replayed from the actual trade
exits** (the real event, not E02's modeled hazard) — the event-stream interface E02's module
docstring promised: any slice scheduled after the real `exit_time` is treated as cancelled.

This harness is deliberately the "code ready for someone to do backtest/live experiments"
artifact. To point it at new data: replace `load_trades`/`load_1min`/`load_15min_volume` with
whatever loads real (or live) entries and bars, and replace the `exit_time` column with the
actual interruption event stream (stop-fires / signal-flips) from that source — everything
downstream of those three loaders is unchanged.

State construction (documented simplifications, consistent with earlier stages):
  - "decision price" = the DB's own 1-min bar open at entry_time (P02's fix for the CSV
    decision_price/DB price-basis mismatch — see findings_02_chase_simulation.md).
  - sigma_now = a REAL trailing-120-min nowcast via `state.sigma_now_bps` on real 1-min returns
    (not a placeholder).
  - expected_interval_volume = the REAL 15-min bar's own volume at entry_time (used as if it were
    the "predicted" volume for that bin — a documented proxy, not a live nowcast).
  - spread = the representative one-way half-spread from findings_01 (0.74 TQQQ / 1.00 SQQQ), not
    a live NBBO lookup for every historical timestamp.
  - edge_bps = 50 (illustrative, as in E03).

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/scheduler/04_replay/build_04_replay.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from slippage.schedule import schedule_order  # noqa: E402
from slippage.state import MarketState, session_bin_label, sigma_now_bps  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ"]
TRADES_ROOT = Path(__file__).resolve().parents[4] / "TQQQ_SQQQ_analysis"
OUT = Path(__file__).resolve().parent / "results"

SPREAD_BPS = {"TQQQ": 0.74, "SQQQ": 1.00}
EDGE_BPS = 50.0
NOTIONAL_SWEEP = [1e5, 1e6, 5e6, 20e6]
NOWCAST_WINDOW_MIN = 120


def load_1min(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, open, close, volume FROM candles_1min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")


def load_15min_volume(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, volume FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")["volume"].sort_index()


def load_trades(sym: str) -> pd.DataFrame:
    path = TRADES_ROOT / "full_history_canonical" / f"TRADES_{sym}_full_history.csv"
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])[["entry_time", "exit_time"]]


def vol_regime(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    rv = d.set_index("dt")["close"].pct_change().rolling(20).std()
    q1, q2 = rv.quantile([1 / 3, 2 / 3])
    reg = pd.cut(rv, [-np.inf, q1, q2, np.inf], labels=["calm", "normal", "stress"])
    reg.index = reg.index.normalize()
    return reg


def replay_symbol(sym: str) -> pd.DataFrame:
    px = load_1min(sym)
    idx, pv_open, pv_close, pv_vol = px.index, px["open"].values, px["close"].values, px["volume"].values
    vol15 = load_15min_volume(sym)
    trades = load_trades(sym)
    reg = vol_regime(sym)
    spread = SPREAD_BPS[sym]

    pos_e = idx.searchsorted(trades["entry_time"])
    aligned = (pos_e < len(idx)) & (idx[np.clip(pos_e, 0, len(idx) - 1)] == trades["entry_time"].values)
    trades, pos_e = trades[aligned].reset_index(drop=True), pos_e[aligned]

    rows = []
    for i in range(len(trades)):
        p0 = pos_e[i]
        entry_time = trades["entry_time"].iloc[i]
        exit_time = trades["exit_time"].iloc[i]
        if p0 < NOWCAST_WINDOW_MIN or p0 + 1 >= len(pv_close):
            continue
        decision_price = pv_open[p0]
        if not np.isfinite(decision_price) or decision_price <= 0:
            continue

        window = np.log(px["close"].iloc[p0 - NOWCAST_WINDOW_MIN:p0]).diff().dropna()
        sigma = sigma_now_bps(window)
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        bin_label = session_bin_label(entry_time)
        vol_bin = float(vol15.get(pd.Timestamp(entry_time).floor("15min"), np.nan))
        if not np.isfinite(vol_bin) or vol_bin <= 0:
            continue
        regime = reg.get(entry_time.normalize(), "normal")
        if pd.isna(regime):
            regime = "normal"

        # session close (last bar of this calendar day) for the day-VWAP baseline.
        day_mask = idx.normalize() == entry_time.normalize()
        day_idx = np.where(day_mask)[0]
        session_end = day_idx[-1] if len(day_idx) else min(p0 + 26, len(pv_close) - 1)
        vwap_window = slice(p0, session_end + 1)
        vwap_vol = pv_vol[vwap_window]
        vwap = (float(np.sum(pv_close[vwap_window] * vwap_vol) / vwap_vol.sum())
               if vwap_vol.sum() > 0 else pv_close[session_end])

        interrupt_min = (exit_time - entry_time).total_seconds() / 60.0

        state = MarketState(ts=entry_time, symbol=sym, bin_label=bin_label,
                            expected_interval_volume=vol_bin, thin_volume_p10=vol_bin * 0.4,
                            thin_volume_p20=vol_bin * 0.6, sigma_now_bps=sigma, regime=str(regime),
                            spread_bps=spread)

        for notional in NOTIONAL_SWEEP:
            sched = schedule_order(notional, "buy", state, price=decision_price, edge_bps=EDGE_BPS)
            cutoff = min(sched.horizon_min, interrupt_min) if sched.feasible else 0.0

            filled_notional, weighted_cost = 0.0, 0.0
            for sl in sched.slices:
                if sl.time_offset_min > cutoff:
                    continue
                bar = np.clip(p0 + int(round(sl.time_offset_min)), 0, len(pv_close) - 1)
                drift = (pv_close[bar] - decision_price) / decision_price * 1e4
                cost = drift + spread
                weighted_cost += cost * sl.child_notional_usd
                filled_notional += sl.child_notional_usd
            fill_rate = filled_notional / notional if notional > 0 else 0.0
            realized_scheduled_bps = weighted_cost / filled_notional if filled_notional > 0 else np.nan

            naive_bar = np.clip(p0 + 15, 0, len(pv_close) - 1)
            naive_bps = (pv_close[naive_bar] - decision_price) / decision_price * 1e4 + spread
            vwap_bps = (vwap - decision_price) / decision_price * 1e4 + spread

            rows.append({"symbol": sym, "entry_time": entry_time, "regime": regime,
                        "notional": notional, "h_star": sched.horizon_min,
                        "n_slices": len(sched.slices), "feasible": sched.feasible,
                        "fill_rate": fill_rate, "realized_scheduled_bps": realized_scheduled_bps,
                        "naive_15min_bps": naive_bps, "day_vwap_bps": vwap_bps})
    return pd.DataFrame(rows)


def _usd(x):
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x/1e3:.0f}k"


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
LINE_COLORS = {"scheduled_bps": "#58a6ff", "naive_15min_bps": "#f0b429", "day_vwap_bps": "#3fb950"}
LINE_LABELS = {"scheduled_bps": "scheduled (filled portion)", "naive_15min_bps": "naive 15-min fill",
               "day_vwap_bps": "day-VWAP"}


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


def make_plot(res_dir: Path):
    """Regenerate replay_summary.png from summary_by_size.csv (no DB access, no re-replay)."""
    df = pd.read_csv(res_dir / "summary_by_size.csv")
    symbols = list(df["symbol"].unique())
    fig, axes = plt.subplots(1, len(symbols), figsize=(7 * len(symbols), 5))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.atleast_1d(axes)
    for ax, sym in zip(axes, symbols):
        sub = df[df.symbol == sym].reset_index(drop=True)
        x = np.arange(len(sub))
        # fill rate as faded background bars (right axis)
        ax2 = ax.twinx()
        ax2.bar(x, sub["fill_rate"] * 100, width=0.55, color="#8b949e", alpha=0.22, zorder=1)
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("fill rate (%)", color="#8b949e", fontsize=8)
        ax2.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        for xi, fr in zip(x, sub["fill_rate"]):
            ax2.text(xi, fr * 100 + 2, f"{fr*100:.0f}%", ha="center", va="bottom",
                     fontsize=7, color="#8b949e")
        # the three cost series
        for col in LINE_COLORS:
            ax.plot(x, sub[col], color=LINE_COLORS[col], lw=1.6, marker="o", ms=4, zorder=3,
                    label=LINE_LABELS[col] if sym == symbols[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["notional"])
        ax.set_ylabel("realized cost (bps; drift + spread — impact NOT charged)")
        ax.set_title(f"{sym} — scheduler wins small, real interruptions bite at size")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        _style_ax(ax)
    fig.suptitle("E04 — historical replay vs baselines (bars = scheduled fill rate; baselines "
                 "always fill 100%)", color=TEXT_COL, fontsize=11)
    fig.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL,
               loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(res_dir / "replay_summary.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows = []
    for sym in TICKERS:
        print(f"Replaying {sym}...")
        df = replay_symbol(sym)
        all_rows.append(df)
        print(f"  {len(df)} (trade x size) events in {time.time()-t0:.1f}s")
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT / "replay_events.csv", index=False)

    print(f"\n{'='*90}\nREALIZED cost by notional (mean, bps) — scheduled (on filled portion) vs "
          f"naive-15min vs day-VWAP\n{'='*90}")
    summary = full.groupby(["symbol", "notional"]).agg(
        n=("realized_scheduled_bps", "size"),
        fill_rate=("fill_rate", "mean"),
        scheduled_bps=("realized_scheduled_bps", "mean"),
        naive_15min_bps=("naive_15min_bps", "mean"),
        day_vwap_bps=("day_vwap_bps", "mean"),
    ).reset_index()
    summary["notional"] = summary["notional"].map(_usd)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "summary_by_size.csv", index=False)

    print(f"\n{'='*90}\nBy regime (all sizes pooled)\n{'='*90}")
    by_regime = full.groupby(["symbol", "regime"], observed=True).agg(
        n=("realized_scheduled_bps", "size"), fill_rate=("fill_rate", "mean"),
        scheduled_bps=("realized_scheduled_bps", "mean"),
        naive_15min_bps=("naive_15min_bps", "mean"), day_vwap_bps=("day_vwap_bps", "mean"),
    ).reset_index()
    print(by_regime.to_string(index=False))
    by_regime.to_csv(OUT / "summary_by_regime.csv", index=False)
    make_plot(OUT)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s. Outputs written to {OUT}")


if __name__ == "__main__":
    main()
