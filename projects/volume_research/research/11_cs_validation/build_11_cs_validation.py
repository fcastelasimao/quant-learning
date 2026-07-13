"""Stage 13 — validate the Corwin–Schultz 15-min half-spread against ground truth.

CS-15min gives ~0.74/1.00/0.72 bp half for TQQQ/SQQQ/QQQ. This checks that number two ways:

  TIER 1 (no external data):
    - tick-floor bound: the quoted spread cannot be below one tick ($0.01 / price). Where CS sits
      relative to the penny floor tells us if it is at the floor (accurate) or biased.
    - CS by resolution: 1-min vs 15-min within-session, to show where CS stabilises.
    - (context) live market-order sells averaged ~0 bps (findings_08) → realized cost ≤ a few bps;
      synthetic recovery ±20% is in tests/test_spread.py. Cited, not recomputed.

  TIER 2 (gold-standard, Alpaca SIP quotes = real NBBO):
    - pull historical bid/ask for a sample of short intraday windows, compute the time-weighted
      quoted half-spread, and compare directly to CS. Skips gracefully if the API is unreachable.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/11_cs_validation/build_11_cs_validation.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slippage.spread import corwin_schultz_intraday, half_spread_bps  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
TICK = 0.01                       # US equity min tick above $1
OUT = Path(__file__).resolve().parent / "results"

# CS-15min reference (findings_01/04) — what we are validating.
CS_HALF_BPS = {"TQQQ": 0.74, "SQQQ": 1.00, "QQQ": 0.72}

# Tier-2 sampling: a few recent trading days × times of day, 1-min quote windows each.
ALPACA_QUOTES = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"
SAMPLE_DAYS = 8                   # most recent N weekdays before "today"
SAMPLE_TIMES_ET = ["10:00", "12:30", "15:30"]   # open-ish / midday / close-ish
ET_OFFSET = -4                    # EDT (summer). ET = UTC-4.


# --------------------------------------------------------------------------- Tier 1
def load_candles(sym: str, interval: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql(f"SELECT et_datetime, high, low, close FROM candles_{interval} ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")


def cs_half(df: pd.DataFrame) -> float:
    s = corwin_schultz_intraday(df[["high", "low"]], clamp_negative=False)
    return float(max(np.nanmean(half_spread_bps(s)), 0.0))


def tier1(sym: str) -> dict:
    d1 = load_candles(sym, "15min")
    price = float(d1["close"].iloc[-1])
    tick_full = TICK / price * 1e4          # one-tick full spread, bps
    tick_half = tick_full / 2.0
    cs15 = cs_half(d1)
    try:
        cs1 = cs_half(load_candles(sym, "1min"))
    except Exception:
        cs1 = float("nan")
    return {"price": price, "tick_half_bps": tick_half, "cs_15min": cs15,
            "cs_1min": cs1, "cs_over_tickfloor": cs15 / tick_half if tick_half else float("nan")}


# --------------------------------------------------------------------------- Tier 2
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


def fetch_quotes(sym: str, start: datetime, end: datetime, headers: dict) -> pd.DataFrame:
    """One page (≤10k) of SIP quotes for a short window → DataFrame[t, bid, ask]."""
    p = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000, "feed": "sip"}
    r = requests.get(ALPACA_QUOTES.format(sym=sym), headers=headers, params=p, timeout=30)
    r.raise_for_status()
    qs = r.json().get("quotes") or []
    if not qs:
        return pd.DataFrame(columns=["t", "bid", "ask"])
    df = pd.DataFrame({"t": pd.to_datetime([q["t"] for q in qs]),
                       "bid": [q["bp"] for q in qs], "ask": [q["ap"] for q in qs]})
    return df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]


def window_twa_half_bps(q: pd.DataFrame, end: datetime) -> float:
    """Time-weighted quoted half-spread (bps) over a window: weight each quote by how long it
    stood until the next update."""
    if len(q) < 2:
        return float("nan")
    mid = (q["ask"] + q["bid"]) / 2.0
    half = ((q["ask"] - q["bid"]) / 2.0 / mid * 1e4).to_numpy()
    t = q["t"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()   # naive UTC ns
    end_ts = np.datetime64(end.replace(tzinfo=None))
    dt = np.append(np.diff(t), end_ts - t[-1]) / np.timedelta64(1, "s")
    dt = np.clip(dt, 0, None)
    w = dt.sum()
    return float(np.average(half, weights=dt)) if w > 0 else float(np.mean(half))


def _recent_weekdays(n: int) -> list[datetime]:
    days, d = [], datetime.now(timezone.utc).date() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return [datetime(x.year, x.month, x.day) for x in days]


def tier2(sym: str, headers: dict) -> dict:
    halves = []
    for day in _recent_weekdays(SAMPLE_DAYS):
        for hhmm in SAMPLE_TIMES_ET:
            h, m = map(int, hhmm.split(":"))
            start = day.replace(hour=h - ET_OFFSET, minute=m, tzinfo=timezone.utc)
            end = start + timedelta(minutes=1)
            try:
                q = fetch_quotes(sym, start, end, headers)
                v = window_twa_half_bps(q, end)
                if np.isfinite(v):
                    halves.append(v)
            except Exception:
                continue
    if not halves:
        return {}
    a = np.array(halves)
    return {"n_windows": len(a), "quoted_half_mean": float(a.mean()),
            "quoted_half_median": float(np.median(a)), "quoted_half_p90": float(np.percentile(a, 90))}


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
BAR_COLORS = {"cs_15min_half_bps": "#58a6ff", "quoted_half_mean": "#3fb950",
              "quoted_half_median": "#2ea043", "quoted_half_p90": "#f0b429"}
BAR_LABELS = {"cs_15min_half_bps": "CS-15min (estimate)", "quoted_half_mean": "SIP NBBO mean",
              "quoted_half_median": "SIP NBBO median", "quoted_half_p90": "SIP NBBO p90"}


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
    """Regenerate cs_validation.png from cs_validation.csv (no Alpaca access needed)."""
    df = pd.read_csv(res_dir / "cs_validation.csv")
    if "quoted_half_mean" not in df.columns:
        return   # Tier 2 was skipped (Alpaca unreachable) — nothing worth plotting
    cols = list(BAR_COLORS)
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    for j, c in enumerate(cols):
        xs = x + (j - (len(cols) - 1) / 2) * w
        ax.bar(xs, df[c], width=w * 0.92, color=BAR_COLORS[c], label=BAR_LABELS[c])
        for xi, v in zip(xs, df[c]):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT_COL)
    for k, (xi, r) in enumerate(zip(x, df.itertuples())):
        ax.plot([xi - 0.45, xi + 0.45], [r.tick_half_bps] * 2, color="#f85149", lw=1.0, ls="--",
                label="penny tick floor" if k == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.ticker}\nCS/SIP = {r.cs_over_sip:.2f}" for r in df.itertuples()],
                       fontsize=8.5)
    ax.set_ylabel("half-spread (bps, one-way)")
    ax.set_title("S11 — Corwin–Schultz estimate vs real SIP NBBO half-spread (21 windows)")
    _style_ax(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)
    fig.tight_layout()
    fig.savefig(res_dir / "cs_validation.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== TIER 1 — tick-floor + resolution (no external data) ===\n")
    print(f"{'tk':5} {'price':>8} {'tickfloor½':>11} {'CS-15m':>8} {'CS/floor':>9} {'CS-1m':>8}")
    rows = []
    t1 = {}
    for sym in TICKERS:
        r = tier1(sym)
        t1[sym] = r
        print(f"{sym:5} {r['price']:>8.2f} {r['tick_half_bps']:>11.3f} {r['cs_15min']:>8.3f} "
              f"{r['cs_over_tickfloor']:>9.2f} {r['cs_1min']:>8.3f}")
    print("\n  (CS/floor ≈1 → CS sits at the penny floor = accurate; ≫1 → CS likely overstates; "
          "<1 → below one tick, i.e. CS understates the minimum.)")
    print("  Context: findings_08 market-order sells ~0±2 bps (realized cost ≤ a few bps); "
          "synthetic recovery ±20% in tests/test_spread.py.")

    print("\n=== TIER 2 — Alpaca SIP quotes (real NBBO) vs CS ===\n")
    try:
        headers = _alpaca_headers()
        print(f"{'tk':5} {'CS-15m':>8} {'SIP½ mean':>10} {'median':>8} {'p90':>7} {'n':>4}  verdict")
        for sym in TICKERS:
            t2 = tier2(sym, headers)
            if not t2:
                print(f"{sym:5} {CS_HALF_BPS[sym]:>8.2f}   (no quotes returned)")
                continue
            cs = CS_HALF_BPS[sym]
            ratio = cs / t2["quoted_half_mean"] if t2["quoted_half_mean"] else float("nan")
            verdict = ("CS≈SIP ✓" if 0.6 <= ratio <= 1.6 else
                       ("CS high" if ratio > 1.6 else "CS low"))
            print(f"{sym:5} {cs:>8.2f} {t2['quoted_half_mean']:>10.3f} "
                  f"{t2['quoted_half_median']:>8.3f} {t2['quoted_half_p90']:>7.3f} "
                  f"{t2['n_windows']:>4}  {verdict} (CS/SIP={ratio:.2f})")
            rows.append({"ticker": sym, "cs_15min_half_bps": cs, **t2, "cs_over_sip": ratio})
    except Exception as e:
        print(f"  Tier 2 skipped — Alpaca unreachable: {type(e).__name__}: {str(e)[:120]}")

    # persist
    out = pd.DataFrame([{"ticker": s, **t1[s]} for s in TICKERS])
    if rows:
        out = out.merge(pd.DataFrame(rows), on="ticker", how="left")
    out.to_csv(OUT / "cs_validation.csv", index=False)
    make_plot(OUT)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
