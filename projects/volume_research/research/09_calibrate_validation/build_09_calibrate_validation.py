"""Stage 11 — validate `slippage.calibrate()` against the real DB pulls.

`calibrate()` is the one library path exercised only by synthetic test frames. This probe runs
it on the *actual* daily + 15-min DBs for TQQQ/SQQQ/QQQ and checks it reproduces the numbers the
research chain already published:

  - $ADV / thin-$ADV        vs findings_04 (mean / p10 of daily $volume)
  - σ stress (p90)          vs findings_04
  - 15-min half-spread      vs findings_01 (Corwin–Schultz, aggregate-clamped)
  - σ normal                reported as BOTH summaries of the right-skewed rolling-vol series:
        * mean   = expected-cost / time-average moment (the library default, the headline)
        * median = typical-day moment (what the published chain used, 339/341/113)

The mean/median σ gap (~9%) is a summary-statistic choice, not a bug — see findings_09. Everything
else should reproduce tightly; a FAIL there is a real regression.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/09_calibrate_validation/build_09_calibrate_validation.py
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
from slippage import calibrate  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
OUT = Path(__file__).resolve().parent / "results"

# Published reference values. σ median / stress + $ADV from findings_04; half-spread from
# findings_01 (15-min CS aggregate mean, one-way). Tolerances: measured statistics reproduce
# tightly; half-spread is noisier; the σ-mean is informational (new headline, no prior point).
EXPECTED = {
    "TQQQ": dict(sigma_median=339, sigma_stress=510, adv=4.9e9, adv_thin=2.9e9, half_spread=0.74),
    "SQQQ": dict(sigma_median=341, sigma_stress=510, adv=2.8e9, adv_thin=1.0e9, half_spread=1.00),
    "QQQ":  dict(sigma_median=113, sigma_stress=172, adv=25.8e9, adv_thin=12.6e9, half_spread=0.72),
}
TOL = {"adv": 0.05, "adv_thin": 0.05, "sigma_stress": 0.05, "sigma_median": 0.03,
       "half_spread": 0.10}


def load_daily(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close, volume FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")[["close", "volume"]]


def load_15min(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, high, low FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")[["high", "low"]]


def _pct(measured: float, expected: float) -> float:
    return (measured - expected) / expected * 100.0


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429", "QQQ": "#3fb950"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_COL)
    for lab in (ax.yaxis.label, ax.xaxis.label, ax.title):
        lab.set_color(TEXT_COL)


def make_plot(res_dir: Path):
    """Regenerate calibration_check.png from calibration_check.csv (no DB access)."""
    df = pd.read_csv(res_dir / "calibration_check.csv")
    df = df[df["expected"].notna()].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    y = list(range(len(df)))
    ax.barh(y, df["gap_pct"], color=[TICKER_COLORS[t] for t in df["ticker"]], height=0.6)
    for i, r in df.iterrows():
        for side in (-1, 1):
            ax.plot([side * r["tol_pct"]] * 2, [i - 0.35, i + 0.35], color=TEXT_COL, lw=0.9)
    ax.axvline(0, color=GRID_COL, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.ticker}  {r.metric}" for r in df.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("measured vs published gap (%)")
    ax.set_title("S09 — calibrate() vs the published S01/S04 constants "
                 "(bar inside the |ticks| = within tolerance = PASS)")
    _style_ax(ax)
    ax.grid(axis="x", color=GRID_COL, alpha=0.5, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(res_dir / "calibration_check.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_pass = True
    for sym in TICKERS:
        cal = calibrate(load_daily(sym), load_15min(sym))
        exp = EXPECTED[sym]
        measured = {
            "sigma_mean": cal.sigma_daily_bps,          # new headline (expected-cost)
            "sigma_median": cal.sigma_daily_median_bps,  # typical-day, vs published
            "sigma_stress": cal.sigma_stress_bps,
            "adv": cal.adv_usd,
            "adv_thin": cal.adv_thin_usd,
            "half_spread": cal.half_spread_bps,
        }
        for metric, tol in TOL.items():
            gap = _pct(measured[metric], exp[metric])
            ok = abs(gap) <= tol * 100.0
            all_pass &= ok
            rows.append({"ticker": sym, "metric": metric, "measured": measured[metric],
                         "expected": exp[metric], "gap_pct": round(gap, 2),
                         "tol_pct": tol * 100.0, "pass": ok})
        # σ mean is informational (no prior point estimate — it IS the new headline).
        rows.append({"ticker": sym, "metric": "sigma_mean", "measured": measured["sigma_mean"],
                     "expected": None, "gap_pct": None, "tol_pct": None, "pass": None})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_check.csv", index=False)
    make_plot(OUT)

    print("=== calibrate() vs published chain ===\n")
    hdr = f"{'ticker':6} {'metric':13} {'measured':>12} {'expected':>12} {'gap%':>7}  result"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        meas = f"{r['measured']:.3g}" if r["measured"] is not None else "-"
        exp = f"{r['expected']:.3g}" if r["expected"] is not None else "-"
        gap = f"{r['gap_pct']:+.2f}" if r["gap_pct"] is not None else "-"
        res = "info" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
        print(f"{r['ticker']:6} {r['metric']:13} {meas:>12} {exp:>12} {gap:>7}  {res}")

    print("\nσ normal is reported as a pair (right-skewed rolling vol):")
    for sym in TICKERS:
        cal = calibrate(load_daily(sym), load_15min(sym))
        print(f"  {sym:5}: mean {cal.sigma_daily_bps:5.0f} bps (expected-cost headline)  |  "
              f"median {cal.sigma_daily_median_bps:5.0f} bps (typical-day, published)")

    print(f"\n{'ALL REPRODUCE ✓' if all_pass else 'REGRESSION — see FAIL rows ✗'}")
    print(f"Outputs written to {OUT}")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
