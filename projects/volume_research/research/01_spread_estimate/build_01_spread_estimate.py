"""Block 1 / Stage 2 — Corwin–Schultz spread estimate for TQQQ / SQQQ / QQQ.

Computes a time-varying bid-ask spread from OHLC candles. We use **15-min bars**:
raw daily-bar Corwin–Schultz is unusable for these ultra-liquid ETFs because the
~1 bp spread is swamped by the 300–800 bp daily range (the signed mean estimate
goes negative — see findings). At 15-min resolution the estimate is stable and
matches known microstructure (wide at the open, ~sub-bp midday).

Outputs: a daily spread series (per-day mean of the intraday estimate), a by-year
+ stress-percentile summary, an intraday time-of-day curve, a dark-theme PNG, and
a console summary comparing against the backtest's flat 5 bps entry.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/01_spread_estimate/build_01_spread_estimate.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slippage.spread import corwin_schultz_intraday, half_spread_bps  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
FLAT_ENTRY_BPS = 5.0  # the size-blind entry cost baked into current backtests
OUT = Path(__file__).resolve().parent / "results"

# Dark theme (mirrors all-weather/engine/plotting.py).
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429", "QQQ": "#3fb950"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)


def load_candles(symbol: str, interval: str) -> pd.DataFrame:
    """OHLCV for a ticker/interval, indexed by ET datetime."""
    db = config.data_dir() / f"DB_{symbol}_historical_data.db"
    with sqlite3.connect(db) as conn:
        df = pd.read_sql(
            f"SELECT et_datetime, open, high, low, close, volume "
            f"FROM candles_{interval} ORDER BY ts",
            conn,
        )
    df["dt"] = pd.to_datetime(df["et_datetime"])
    return df.set_index("dt")[["open", "high", "low", "close", "volume"]]


def intraday_halfspread(symbol: str) -> pd.Series:
    """Per-pair half-spread (bps) from 15-min bars, within-session, *unclamped*.

    CS is unbiased only when signed per-pair estimates are averaged and the
    aggregate is clamped — so we keep negatives here and clamp at aggregation.
    """
    df = load_candles(symbol, "15min")
    s = corwin_schultz_intraday(df[["high", "low"]], clamp_negative=False)
    hs = half_spread_bps(s).dropna()
    hs.name = "half_spread_bps"
    return hs


def _agg(hs: pd.Series, ticker: str, period: str) -> dict:
    # Mean is the CS point estimate (clamp at zero). Percentiles describe the
    # noisy right tail / stress behaviour.
    return {"ticker": ticker, "period": period, "n": int(len(hs)),
            "mean_bps": max(hs.mean(), 0.0), "median_bps": max(hs.median(), 0.0),
            "p90_bps": hs.quantile(0.90), "p95_bps": hs.quantile(0.95)}


def summarise(intraday: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for ticker, hs in intraday.items():
        rows.append(_agg(hs, ticker, "overall"))
        for year, gy in hs.groupby(hs.index.year):
            rows.append(_agg(gy, ticker, str(year)))
    return pd.DataFrame(rows)


def tod_table(intraday: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for ticker, hs in intraday.items():
        tod = hs.groupby(hs.index.time).mean().clip(lower=0)
        for tm, v in tod.items():
            rows.append({"time": tm.strftime("%H:%M"), "half_spread_bps": v, "ticker": ticker})
    return pd.DataFrame(rows)


def daily_series(intraday: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Per-day mean of the intraday half-spread, floored at 0 — the CSV daily series.
    (A single day is noisy; aggregate further before relying on it.)"""
    return {t: hs.resample("D").mean().dropna().clip(lower=0) for t, hs in intraday.items()}


def make_plot(pairs: dict[str, pd.Series], tod: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.patch.set_facecolor(DARK_BG)

    # Clamp at the monthly aggregate (not per pair/day) to avoid clamp bias.
    for ticker, hs in pairs.items():
        monthly = hs.resample("ME").mean().clip(lower=0)
        ax1.plot(monthly.index, monthly.values, color=COLORS[ticker], lw=1.2, label=ticker)
    ax1.axhline(FLAT_ENTRY_BPS, color="#f78166", ls="--", lw=1.0,
                label=f"flat backtest entry ({FLAT_ENTRY_BPS:.0f} bps)")
    ax1.set_title("Corwin–Schultz half-spread from 15-min bars — monthly mean (bps)")
    ax1.set_ylabel("half-spread (bps)")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    for ticker, g in tod.groupby("ticker"):
        ax2.plot(g["time"], g["half_spread_bps"], color=COLORS[ticker], lw=1.4, marker="o",
                 ms=3, label=ticker)
    ax2.set_title("Intraday half-spread by time of day — 15-min bars (bps)")
    ax2.set_ylabel("half-spread (bps)")
    ax2.set_xlabel("ET time")
    ax2.tick_params(axis="x", rotation=45)
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    intraday = {t: intraday_halfspread(t) for t in TICKERS}
    daily = daily_series(intraday)
    summary = summarise(intraday)
    tod = tod_table(intraday)

    daily_df = pd.concat(
        [d.to_frame().assign(ticker=t) for t, d in daily.items()]
    ).reset_index(names="date")
    daily_df.to_csv(OUT / "cs_spread_daily.csv", index=False)
    summary.to_csv(OUT / "cs_spread_summary.csv", index=False)
    tod.to_csv(OUT / "cs_spread_intraday_tod.csv", index=False)
    make_plot(intraday, tod, OUT / "cs_spread.png")

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    overall = summary[summary["period"] == "overall"].set_index("ticker")
    print("\n=== 15-min Corwin–Schultz half-spread, overall (bps) ===")
    print(overall[["n", "mean_bps", "median_bps", "p90_bps", "p95_bps"]])
    print(f"\nBacktest assumes a flat {FLAT_ENTRY_BPS:.0f} bps entry (size-blind).")
    for t in TICKERS:
        m, p95 = overall.loc[t, "mean_bps"], overall.loc[t, "p95_bps"]
        verdict = "under" if m < FLAT_ENTRY_BPS else "over"
        print(f"  {t}: mean {m:.2f} bps ({verdict} flat), 95th-pct {p95:.2f} bps")
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
