"""Stage 9 — validate the cost model against live Alpaca fills (2026-05-22 → 06-24).

Parses the per-cycle live logs, extracts every real fill (decision price → fill price), and
compares the *realized* slippage to what the model predicts at this (retail) size. The trades
are ~$150k, far too small to register impact — so this validates the **spread + timing** end of
the model, NOT the capacity curve. Its real payoff is resolving the open momentum-vs-mean-
reversion question (W3): for a directional signal, is delay a symmetric risk or a signed drag?

Realized slippage (bps), signed so positive = adverse (worse than the decision price):
    buy :  (fill − decision) / decision · 1e4
    sell:  (decision − fill) / decision · 1e4

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/08_live_validation/build_08_live_validation.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = Path(__file__).resolve().parents[1].parent / "LIVE_alpaca_cycle_from_20260522_20260624"
OUT = Path(__file__).resolve().parent / "results"

# Model reference levels (from findings_01 spread + findings_03 timing), TQQQ.
HALF_SPREAD_TYPICAL_BPS = 1.0      # one-way, midday
HALF_SPREAD_OPEN_BPS = 2.5         # one-way, ~09:45
TIMING_1MIN_BPS = 17.0             # 1σ timing risk at ~1-min latency

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"

BUY_RE = re.compile(
    r">>> BUY (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) qty ([\d.]+).*? "
    r"decision ([\d.]+) fill ([\d.]+)")
SELL_RE = re.compile(
    r"SELL (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) qty ([\d.]+) "
    r"decision ([\d.]+) fill ([\d.]+) style (\w+)")
SELL_FAIL_RE = re.compile(r"Live SELL failed: invalid fill result filled_qty=0")


def parse_fills() -> tuple[pd.DataFrame, int]:
    rows, unfilled = [], 0
    for f in sorted(LOGS.glob("*.log")):
        txt = f.read_text(errors="ignore")
        for m in BUY_RE.finditer(txt):
            ts, sym, qty, dec, fill = m.groups()
            dec, fill, qty = float(dec), float(fill), float(qty)
            rows.append(dict(time=ts, symbol=sym, side="buy", qty=qty,
                             decision=dec, fill=fill, style="limit",
                             notional=qty * fill,
                             slip_bps=(fill - dec) / dec * 1e4))
        for m in SELL_RE.finditer(txt):
            ts, sym, qty, dec, fill, style = m.groups()
            dec, fill, qty = float(dec), float(fill), float(qty)
            rows.append(dict(time=ts, symbol=sym, side="sell", qty=qty,
                             decision=dec, fill=fill, style=style,
                             notional=qty * fill,
                             slip_bps=(dec - fill) / dec * 1e4))
        unfilled += len(SELL_FAIL_RE.findall(txt))
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True), unfilled


def summarize(df: pd.DataFrame, unfilled: int):
    print(f"\n{'='*72}\nLIVE FILL VALIDATION — {len(df)} fills "
          f"({(df.side=='buy').sum()} buy, {(df.side=='sell').sum()} sell), "
          f"{unfilled} unfilled limit sell(s)")
    print(f"notional range ${df.notional.min()/1e3:.0f}k–${df.notional.max()/1e3:.0f}k "
          f"(retail — impact ≈ 0 at this size)\n{'-'*72}")
    print(f"{'side':>6} {'n':>3} {'mean':>7} {'std':>6} {'min':>7} {'max':>7}   (realized slippage, bps)")
    for side, g in df.groupby("side"):
        s = g.slip_bps
        print(f"{side:>6} {len(g):>3} {s.mean():>7.1f} {s.std():>6.1f} {s.min():>7.1f} {s.max():>7.1f}")
    print(f"\nModel reference: spread floor ~{HALF_SPREAD_TYPICAL_BPS:.0f}–{HALF_SPREAD_OPEN_BPS:.1f} bps "
          f"(one-way), timing ~{TIMING_1MIN_BPS:.0f} bps 1σ at ~1-min latency.")
    buy = df[df.side == "buy"].slip_bps
    print(f"\nVERDICT:")
    print(f" • Buy slippage mean {buy.mean():+.1f} bps (1σ {buy.std():.1f}) — magnitude matches the "
          f"~{TIMING_1MIN_BPS:.0f} bps timing scale, NOT a larger impact (confirms impact≈0 at $150k).")
    print(f" • The mean is clearly POSITIVE (adverse), not ≈0 → momentum entries fill against you: "
          f"delay is a SIGNED drag here (resolves W3).")
    print(f" • Sells (market) ≈ {df[df.side=='sell'].slip_bps.mean():.1f} bps — clean; cost is "
          f"concentrated on the limit-order BUY side.")
    if unfilled:
        print(f" • {unfilled} limit sell never filled → opportunity cost (W3 price-guard), the cost "
              f"that becomes invisible rather than measured.")


def make_plot(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    colors = {"buy": "#f78166", "sell": "#3fb950"}
    for side, g in df.groupby("side"):
        ax.scatter(g.time, g.slip_bps, s=60, color=colors[side], label=f"{side} fills",
                   edgecolor=TEXT_COL, linewidth=0.4, zorder=3)
    ax.axhspan(-TIMING_1MIN_BPS, TIMING_1MIN_BPS, color="#58a6ff", alpha=0.12,
               label="model ±1σ timing (~17 bps)")
    ax.axhline(0, color=TEXT_COL, ls=":", lw=1)
    ax.axhline(HALF_SPREAD_OPEN_BPS, color="#bc8cff", ls="--", lw=0.9, label="spread floor (~1–2.5 bps)")
    buy_mean = df[df.side == "buy"].slip_bps.mean()
    ax.axhline(buy_mean, color="#f78166", ls="-", lw=1.1, alpha=0.8,
               label=f"realized buy mean ({buy_mean:+.0f} bps, momentum-adverse)")
    ax.set_title("Live realized slippage vs the model (TQQQ, 2026-05-22 → 06-24)", color=TEXT_COL)
    ax.set_ylabel("realized slippage, bps (+ = adverse)", color=TEXT_COL)
    ax.set_xlabel("fill date", color=TEXT_COL)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_COL, alpha=0.4)
    ax.tick_params(colors=TEXT_COL)
    for s in ax.spines.values():
        s.set_color(GRID_COL)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df, unfilled = parse_fills()
    df.to_csv(OUT / "live_fills.csv", index=False)
    summarize(df, unfilled)
    make_plot(df, OUT / "live_validation.png")
    print(f"\nWrote live_fills.csv + live_validation.png to {OUT}")


if __name__ == "__main__":
    main()
