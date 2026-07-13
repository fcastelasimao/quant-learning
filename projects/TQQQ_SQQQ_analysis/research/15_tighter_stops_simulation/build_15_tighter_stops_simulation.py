"""15_tighter_stops_simulation: walk 15-min bars inside each trade.

Depends on:
  - full_history_canonical/TRADES_<SYM>_full_history.csv (regime-labeled subset)
  - Shared QuantFinance data store (DB_<SYM>_historical_data.db)
    candles_15min table for TQQQ and SQQQ. Refresh with
    `quantcore-ingest --intervals 15min --symbols TQQQ SQQQ`.


For each trade (entry_time T0 → exit_time T1, entry_price P0, recorded pnl_pct):
  - Pull TQQQ/SQQQ 15-min bars with dt in (T0, T1].
  - Compute intraday_max_loss = min(bar.low / P0 - 1) across those bars.
  - For each candidate stop_pct in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0} %:
      - If intraday_max_loss <= -stop_pct, the simulated stop would have triggered.
      - simulated_pnl_pct = -stop_pct  (clean fill at the stop price; idealized)
      - Otherwise: simulated_pnl_pct = original pnl_pct.
  - Aggregate by symbol: total simulated pnl, n_stopped_out, n_winners_stopped, n_losers_caught.

What this answers:
  - Is there a tighter stop level that would have aggregate-improved net pnl?
  - How many actual winners would the tighter stop have killed?
  - Per-year breakdown of net change.

CAVEATS:
  - Assumes immediate fill at exactly the stop price. Real stops slip; this is an
    upper bound on stop benefit.
  - 15-min bars are coarse. A drop INTRA-bar that touches the stop and reverses
    in the same bar would have triggered in reality but isn't visible to us.
    Our metric uses bar.low so it captures bar-low; this may over-count stop
    triggers (over-estimating losses saved).
  - Trade entry_price approximated by avg_order_price; fee/slippage on entry not
    re-applied to the simulated alt exit.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quant")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from quantcore import config as _qc_config

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent.parent
CANON = PROJ / "full_history_canonical"
DATA_DIR = _qc_config.data_dir()
OUT = ROOT

STOP_LEVELS_PCT = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def load_15min(symbol: str) -> pd.DataFrame:
    db = DATA_DIR / f"DB_{symbol}_historical_data.db"
    conn = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT et_datetime, open, high, low, close FROM candles_15min ORDER BY ts",
        conn,
    )
    conn.close()
    df["dt"] = pd.to_datetime(df["et_datetime"])
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def load_trades(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(CANON / f"TRADES_{symbol}_full_history.csv",
                     parse_dates=["entry_time", "exit_time"])
    df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    return df


def simulate_one(trade: pd.Series, bars: pd.DataFrame) -> dict:
    """Returns per-trade stats: intraday_max_loss + sim pnl for each stop level."""
    T0, T1 = trade["entry_time"], trade["exit_time"]
    P0 = float(trade["avg_order_price"])
    if not (P0 > 0) or pd.isna(T0) or pd.isna(T1):
        return None
    intra = bars[(bars["dt"] > T0) & (bars["dt"] <= T1)]
    if intra.empty:
        return None
    lows = intra["low"].values
    # For long position: max loss along the trade.
    intraday_max_loss = float(np.min(lows / P0) - 1.0)
    out = {
        "entry_time": T0,
        "year": int(T0.year),
        "n_bars": len(intra),
        "P0": P0,
        "orig_pnl_pct": float(trade["pnl_pct"]),
        "intraday_max_loss_pct": intraday_max_loss * 100,
        "is_loser": bool(trade["is_loser"]),
    }
    for s in STOP_LEVELS_PCT:
        stop_hit = (intraday_max_loss * 100) <= -s
        sim_pnl = -s if stop_hit else float(trade["pnl_pct"])
        out[f"sim_pnl_stop_{s}"] = sim_pnl
        out[f"stop_hit_{s}"] = stop_hit
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    yearly_rows = []
    per_trade_dump = []

    for sym in ("TQQQ", "SQQQ"):
        print(f"\n{sym}: loading bars + trades...")
        bars = load_15min(sym)
        trades = load_trades(sym)
        print(f"  bars={len(bars)}, trades={len(trades)}")

        records = []
        for _, tr in trades.iterrows():
            r = simulate_one(tr, bars)
            if r is None:
                continue
            r["symbol"] = sym
            records.append(r)
        rec = pd.DataFrame(records)
        print(f"  simulatable trades: {len(rec)} ({len(rec)/len(trades):.1%})")
        per_trade_dump.append(rec)

        # Aggregate per stop level
        orig_total = float(rec["orig_pnl_pct"].sum())
        for s in STOP_LEVELS_PCT:
            sim_total = float(rec[f"sim_pnl_stop_{s}"].sum())
            n_hit = int(rec[f"stop_hit_{s}"].sum())
            stopped = rec[rec[f"stop_hit_{s}"]]
            n_winners_killed = int((stopped["orig_pnl_pct"] > 0).sum())
            n_losers_caught = int((stopped["orig_pnl_pct"] < 0).sum())
            # Of those losers caught, how many were severe (<=-1%)?
            n_severe_caught = int((stopped["orig_pnl_pct"] <= -1).sum())
            # Sum of orig pnl for trades that got stopped
            sum_orig_pnl_of_stopped = float(stopped["orig_pnl_pct"].sum())
            saved_pnl = sum_orig_pnl_of_stopped - (-s * n_hit)  # what the stops avoided vs what they took
            summary_rows.append({
                "symbol": sym,
                "stop_pct": s,
                "n_trades": len(rec),
                "n_stop_hit": n_hit,
                "n_winners_killed": n_winners_killed,
                "n_losers_caught": n_losers_caught,
                "n_severe_caught": n_severe_caught,
                "trigger_rate": n_hit / len(rec) if len(rec) else np.nan,
                "orig_total_pnl_pct": orig_total,
                "sim_total_pnl_pct": sim_total,
                "delta_total_pnl_pct": sim_total - orig_total,
                "delta_per_trade_pp": (sim_total - orig_total) / len(rec) if len(rec) else np.nan,
            })
            # Per-year
            for y, ysub in rec.groupby("year"):
                sim_y = float(ysub[f"sim_pnl_stop_{s}"].sum())
                orig_y = float(ysub["orig_pnl_pct"].sum())
                yearly_rows.append({
                    "symbol": sym,
                    "stop_pct": s,
                    "year": int(y),
                    "n_trades": len(ysub),
                    "n_stop_hit": int(ysub[f"stop_hit_{s}"].sum()),
                    "orig_total_pnl_pct": orig_y,
                    "sim_total_pnl_pct": sim_y,
                    "delta_pnl_pct": sim_y - orig_y,
                })

    pd.DataFrame(summary_rows).to_csv(OUT / "stop_sweep_summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUT / "stop_sweep_yearly.csv", index=False)
    pd.concat(per_trade_dump, ignore_index=True).to_csv(OUT / "per_trade_intraday_path.csv", index=False)

    summ = pd.DataFrame(summary_rows)
    print("\nSummary:")
    print(summ[["symbol", "stop_pct", "n_stop_hit", "n_winners_killed", "n_losers_caught",
                "n_severe_caught", "delta_total_pnl_pct", "delta_per_trade_pp"]].to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sym, color in zip(axes, ("TQQQ", "SQQQ"), ("#2563eb", "#dc2626")):
        d = summ[summ["symbol"] == sym].sort_values("stop_pct")
        ax.bar(d["stop_pct"].astype(str), d["delta_total_pnl_pct"], color=color, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("tighter stop (% from entry)")
        ax.set_ylabel("Δ total pnl_pct (sim − original)")
        ax.set_title(f"{sym}: net effect of imposing a tighter stop")
        for _, r in d.iterrows():
            ax.annotate(f"{int(r['n_stop_hit'])}/{int(r['n_winners_killed'])}",
                        (str(r["stop_pct"]), r["delta_total_pnl_pct"]),
                        xytext=(0, 4 if r["delta_total_pnl_pct"] >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=7)
    fig.suptitle("Tighter-stop sweep — annotation shows n_stop_hit / n_winners_killed")
    fig.tight_layout()
    fig.savefig(OUT / "stop_sweep_delta.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
