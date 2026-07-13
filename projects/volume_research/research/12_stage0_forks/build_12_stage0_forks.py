"""Stage 0 — resolve the two forks that gate Phase 2 of the Almgren-adoption plan.

  0a ALPHA HORIZON: is the 15-min strategy's edge captured intraday (minutes), or does it
     accumulate over days? Determines whether the creation/redemption door (hours-to-overnight
     fill time) is even temporally reachable before the edge decays.
     Method: real 15-min market data (not the backtest's synthetic flat-haircut fills) —
       (a) forward-return decay shape from each entry_time,
       (b) fraction of a trade's eventual (entry->exit) move already realized after a delay,
       (c) where the trade log's P&L actually lives, by hold-time bucket.

  0b ETF PERMANENT IMPACT: for an arbitrage-pinned ETF, does a large TQQQ print leave a
     persistent price deviation (like a single stock's permanent impact), or does AP arbitrage
     revert it once the print isn't backed by a genuine move in the underlying (QQQ)?
     Method: event study on real Alpaca tick data — excess return (TQQQ return - 3*QQQ return)
     around large prints, short horizon (1 min) vs long horizon (30 min).

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/12_stage0_forks/build_12_stage0_forks.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantcore import config  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
TQQQ_SQQQ_ROOT = Path(__file__).resolve().parents[3] / "TQQQ_SQQQ_analysis"
LEVERAGE = {"TQQQ": 3.0, "SQQQ": -3.0}   # signed exposure to QQQ

# ---------------------------------------------------------------- 0a: alpha horizon
DELAY_BARS = {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16, "1d": 26}
DECAY_BARS = {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16, "1d": 26, "2d": 52,
              "3d": 78, "5d": 130, "10d": 260}
HOLD_BUCKETS = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 25]
HOLD_LABELS = ["<2.4h", "2.4-6h", "6-12h", "12-24h", "1-2d", "2-3d", "3-5d", "5d+"]


def load_15min_close(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")["close"].sort_index()


def load_trade_log(sym: str) -> pd.DataFrame:
    path = TQQQ_SQQQ_ROOT / "full_history_canonical" / f"TRADES_{sym}_full_history.csv"
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])


def decay_curve(sym: str, trades: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    """Mean/median forward log-return (bps) from each entry, by horizon — the decay shape."""
    idx, pv = px.index, px.values
    pos = idx.searchsorted(trades["entry_time"])
    aligned = (pos < len(idx)) & (idx[np.clip(pos, 0, len(idx) - 1)] == trades["entry_time"].values)
    pos = pos[aligned]
    rows = {}
    for label, h in DECAY_BARS.items():
        fwd = pos + h
        ok = fwd < len(pv)
        ret = np.full(len(pos), np.nan)
        ret[ok] = np.log(pv[fwd[ok]] / pv[pos[ok]])
        rows[label] = ret
    df = pd.DataFrame(rows)
    return pd.DataFrame({"mean_bps": df.mean() * 1e4, "median_bps": df.median() * 1e4,
                         "n": df.notna().sum()})


def delay_cost(sym: str, trades: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    """Fraction of a trade's eventual entry->exit move already realized after delaying entry."""
    idx, pv = px.index, px.values
    pos_e = idx.searchsorted(trades["entry_time"])
    pos_x = idx.searchsorted(trades["exit_time"])
    aligned = (pos_e < len(idx)) & (idx[np.clip(pos_e, 0, len(idx) - 1)] == trades["entry_time"].values)
    aligned &= pos_x < len(idx)
    pos_e, pos_x = pos_e[aligned], pos_x[aligned]
    p_entry, p_exit = pv[pos_e], pv[pos_x]
    total_move = p_exit - p_entry
    rows = []
    for label, h in DELAY_BARS.items():
        fwd = pos_e + h
        within_hold = fwd <= pos_x
        fwd_c = np.clip(fwd, 0, len(pv) - 1)
        delay_move = pv[fwd_c] - p_entry
        mask = within_hold & (np.abs(total_move) > 1e-6)
        frac = delay_move[mask] / total_move[mask]
        rows.append({"delay": label, "n_trades_spanning_delay": int(mask.sum()),
                    "median_frac_of_move_used": float(np.nanmedian(frac))})
    return pd.DataFrame(rows)


def pnl_by_hold_bucket(sym: str, trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["hold_bucket"] = pd.cut(t["hold_days"], HOLD_BUCKETS, labels=HOLD_LABELS)
    g = t.groupby("hold_bucket", observed=True).agg(
        n=("pnl_pct", "size"), total_pnl=("pnl", "sum"), mean_pnl_pct=("pnl_pct", "mean"))
    g["pct_of_total_pnl"] = 100 * g["total_pnl"] / t["pnl"].sum()
    g["pct_of_n"] = 100 * g["n"] / len(t)
    return g.reset_index()


def run_stage_0a():
    print("=== 0a — ALPHA HORIZON ===\n")
    for sym in ["TQQQ", "SQQQ"]:
        trades = load_trade_log(sym)
        px = load_15min_close(sym)

        decay = decay_curve(sym, trades, px)
        print(f"--- {sym}: forward-return decay from entry (bps, signed = trade direction) ---")
        print(decay.round(2).to_string())

        dc = delay_cost(sym, trades, px)
        print(f"\n--- {sym}: median fraction of eventual entry->exit move already used, "
              f"by entry-execution delay ---")
        print(dc.round(3).to_string(index=False))

        pnl = pnl_by_hold_bucket(sym, trades)
        print(f"\n--- {sym}: where the P&L lives, by hold-time bucket ---")
        print(pnl.round(2).to_string(index=False))
        print()

        decay.to_csv(OUT / f"0a_decay_{sym}.csv")
        dc.to_csv(OUT / f"0a_delay_cost_{sym}.csv", index=False)
        pnl.to_csv(OUT / f"0a_pnl_by_hold_{sym}.csv", index=False)


# ---------------------------------------------------------------- 0b: ETF permanent impact
SAMPLE_DAYS_0B = 4   # a couple extra to absorb market holidays (e.g. July 3 2026 observed for July 4)
LARGE_PRINT_PCTILE = 99.0
SHORT_HORIZON_MIN = 1
LONG_HORIZON_MIN = 30
SESSION_START_UTC = (13, 35)   # 09:35 ET (EDT, UTC-4)
SESSION_END_UTC = (19, 55)     # 15:55 ET


def _alpaca_headers() -> dict:
    txt = config.api_keys_path().read_text()
    def g(k):
        m = re.search(rf"^{k}=(.*)$", txt, re.M)
        return m.group(1).strip() if m else ""
    kid = g("ALPACA_API_KEY") or g("APCA_API_KEY_ID")
    sec = g("ALPACA_SECRET_KEY") or g("APCA_API_SECRET_KEY")
    if not (kid and sec):
        raise RuntimeError("no Alpaca keys in api_keys.env")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def _recent_weekdays(n: int) -> list[datetime]:
    days, d = [], datetime.now(timezone.utc).date() - timedelta(days=2)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return [datetime(x.year, x.month, x.day) for x in days]


def fetch_day_trades(sym: str, day: datetime, headers: dict, cap_pages: int = 100) -> pd.DataFrame:
    """Full trading-day tape (paginated), feed=sip (free once >15 min old)."""
    start = day.replace(hour=13, minute=30, tzinfo=timezone.utc)
    end = day.replace(hour=20, minute=0, tzinfo=timezone.utc)
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/trades"
    params = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "limit": 10000, "feed": "sip"}
    out, pages = [], 0
    while True:
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=60)
                r.raise_for_status()
                j = r.json()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        out += j.get("trades") or []
        pages += 1
        tok = j.get("next_page_token")
        if not tok or pages >= cap_pages:
            break
        params["page_token"] = tok
    if not out:
        return pd.DataFrame(columns=["t", "p", "s"])
    t = pd.to_datetime([x["t"] for x in out]).tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame({"t": t, "p": [x["p"] for x in out], "s": [x["s"] for x in out]})
    return df.sort_values("t").reset_index(drop=True)


def _price_at_or_after(df: pd.DataFrame, ts) -> float:
    pos = df["t"].values.searchsorted(np.datetime64(ts))
    if pos >= len(df):
        return float("nan")
    return float(df["p"].iloc[pos])


def _price_at_or_before(df: pd.DataFrame, ts) -> float:
    pos = df["t"].values.searchsorted(np.datetime64(ts), side="right") - 1
    if pos < 0:
        return float("nan")
    return float(df["p"].iloc[pos])


def permanent_impact_events(sym: str, day_trades: pd.DataFrame, qqq_trades: pd.DataFrame) -> pd.DataFrame:
    if day_trades.empty or qqq_trades.empty:
        return pd.DataFrame()
    day = day_trades.copy()
    session_mask = (
        (day["t"].dt.hour > SESSION_START_UTC[0]) |
        ((day["t"].dt.hour == SESSION_START_UTC[0]) & (day["t"].dt.minute >= SESSION_START_UTC[1]))
    ) & (
        (day["t"].dt.hour < SESSION_END_UTC[0]) |
        ((day["t"].dt.hour == SESSION_END_UTC[0]) & (day["t"].dt.minute <= SESSION_END_UTC[1]))
    )
    day = day[session_mask].reset_index(drop=True)
    if day.empty:
        return pd.DataFrame()

    thresh = np.percentile(day["s"], LARGE_PRINT_PCTILE)
    large = day[day["s"] >= thresh]

    L = LEVERAGE[sym]
    rows = []
    for _, ev in large.iterrows():
        t0 = ev["t"]
        p_before_tqqq = _price_at_or_before(day, t0 - pd.Timedelta(seconds=1))
        p_short_tqqq = _price_at_or_after(day, t0 + pd.Timedelta(minutes=SHORT_HORIZON_MIN))
        p_long_tqqq = _price_at_or_after(day, t0 + pd.Timedelta(minutes=LONG_HORIZON_MIN))
        p_before_qqq = _price_at_or_before(qqq_trades, t0 - pd.Timedelta(seconds=1))
        p_short_qqq = _price_at_or_after(qqq_trades, t0 + pd.Timedelta(minutes=SHORT_HORIZON_MIN))
        p_long_qqq = _price_at_or_after(qqq_trades, t0 + pd.Timedelta(minutes=LONG_HORIZON_MIN))
        if any(np.isnan(x) for x in (p_before_tqqq, p_short_tqqq, p_long_tqqq,
                                      p_before_qqq, p_short_qqq, p_long_qqq)):
            continue
        ret_short_tqqq = p_short_tqqq / p_before_tqqq - 1
        ret_long_tqqq = p_long_tqqq / p_before_tqqq - 1
        ret_short_qqq = p_short_qqq / p_before_qqq - 1
        ret_long_qqq = p_long_qqq / p_before_qqq - 1
        excess_short = ret_short_tqqq - L * ret_short_qqq
        excess_long = ret_long_tqqq - L * ret_long_qqq
        print_dir = np.sign(ev["p"] - p_before_tqqq) or 1.0
        rows.append({"t": t0, "size": ev["s"], "excess_short_bps": excess_short * 1e4 * print_dir,
                    "excess_long_bps": excess_long * 1e4 * print_dir})
    return pd.DataFrame(rows)


def run_stage_0b():
    print("\n=== 0b — ETF PERMANENT IMPACT (event study vs 3x-QQQ proxy) ===\n")
    headers = _alpaca_headers()
    days = _recent_weekdays(SAMPLE_DAYS_0B)
    all_events = {"TQQQ": [], "SQQQ": []}
    for day in days:
        print(f"  fetching {day.date()} ...")
        qqq = fetch_day_trades("QQQ", day, headers)
        if qqq.empty:
            print(f"    no QQQ trades ({day.date()} likely a market holiday) — skipping")
            continue
        for sym in ["TQQQ", "SQQQ"]:
            tape = fetch_day_trades(sym, day, headers)
            ev = permanent_impact_events(sym, tape, qqq)
            if not ev.empty:
                all_events[sym].append(ev)
        time.sleep(0.2)

    for sym in ["TQQQ", "SQQQ"]:
        if not all_events[sym]:
            print(f"{sym}: no qualifying events.")
            continue
        ev = pd.concat(all_events[sym], ignore_index=True)
        ev.to_csv(OUT / f"0b_events_{sym}.csv", index=False)
        n = len(ev)
        short_mean, short_se = ev["excess_short_bps"].mean(), ev["excess_short_bps"].sem()
        long_mean, long_se = ev["excess_long_bps"].mean(), ev["excess_long_bps"].sem()
        reversion_frac = 1 - (long_mean / short_mean) if short_mean else float("nan")
        print(f"--- {sym}: n={n} large prints (>= p{LARGE_PRINT_PCTILE} size) across "
              f"{len(days)} sample days ---")
        print(f"  excess return (TQQQ_ret - {LEVERAGE[sym]}*QQQ_ret), signed by print direction:")
        print(f"    short ({SHORT_HORIZON_MIN}min): {short_mean:+.2f} ± {short_se:.2f} bps")
        print(f"    long  ({LONG_HORIZON_MIN}min): {long_mean:+.2f} ± {long_se:.2f} bps")
        print(f"    reversion fraction (1 - long/short): {reversion_frac:.2f}")
        print()


# ---------------------------------------------------------------- plot
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


def make_plot(res_dir: Path):
    """Regenerate stage0_forks.png from the saved 0a/0b CSVs (no DB or Alpaca access)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor(DARK_BG)

    # (1) 0a: where the P&L lives, by hold bucket — the multi-day-alpha verdict.
    # Reindex both tickers onto the canonical bucket order (a bucket can be absent for one
    # ticker — e.g. SQQQ has no 5d+ holds), so the paired bars stay aligned.
    pnls = {sym: pd.read_csv(res_dir / f"0a_pnl_by_hold_{sym}.csv").set_index("hold_bucket")
            for sym in ("TQQQ", "SQQQ")}
    buckets = [b for b in HOLD_LABELS if any(b in p.index for p in pnls.values())]
    x = np.arange(len(buckets))
    for k, sym in enumerate(("TQQQ", "SQQQ")):
        vals = pnls[sym]["pct_of_total_pnl"].reindex(buckets).fillna(0.0)
        ax1.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=TICKER_COLORS[sym], label=sym)
    ax1.axhline(0, color=GRID_COL, lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=30, fontsize=7)
    ax1.set_ylabel("% of total P&L")
    ax1.set_title("0a — P&L by hold time (<2.4h holds are net losers)")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    # (2) 0a: fraction of the eventual move used up, by entry delay.
    for sym in ("TQQQ", "SQQQ"):
        dc = pd.read_csv(res_dir / f"0a_delay_cost_{sym}.csv")
        ax2.plot(range(len(dc)), dc["median_frac_of_move_used"] * 100,
                 color=TICKER_COLORS[sym], lw=1.5, marker="o", ms=4, label=sym)
        ax2.set_xticks(range(len(dc)))
        ax2.set_xticklabels(dc["delay"], fontsize=8)
    ax2.set_ylabel("median % of entry→exit move already used")
    ax2.set_xlabel("entry-execution delay")
    ax2.set_title("0a — delay cost (all surviving trades; see E01 for hold≥1d)")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    # (3) 0b: permanent-impact event study — excess move at 1 min vs 30 min after large prints.
    labels3, means, ses, cols = [], [], [], []
    for sym in ("TQQQ", "SQQQ"):
        f = res_dir / f"0b_events_{sym}.csv"
        if not f.exists():
            continue
        ev = pd.read_csv(f)
        for col, lab in (("excess_short_bps", "1min"), ("excess_long_bps", "30min")):
            labels3.append(f"{sym}\n{lab}")
            means.append(ev[col].mean())
            ses.append(ev[col].sem())
            cols.append(TICKER_COLORS[sym])
    if means:
        x3 = np.arange(len(means))
        ax3.bar(x3, means, yerr=ses, width=0.6, color=cols, capsize=4,
                error_kw=dict(ecolor=TEXT_COL, lw=1))
        ax3.axhline(0, color=GRID_COL, lw=0.8)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(labels3, fontsize=8)
        ax3.set_ylabel("excess return vs 3×QQQ (bps, signed by print)")
        ax3.set_title("0b — permanent impact ≈ 0 (mean ± SE after large prints)")
        _style_ax(ax3)

    fig.tight_layout()
    fig.savefig(res_dir / "stage0_forks.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run_stage_0a()
    run_stage_0b()
    make_plot(OUT)
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
