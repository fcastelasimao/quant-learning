#!/usr/bin/env python
"""P04 — Recurring TCA monitor: realized vs predicted slippage, with drift alerting.

Generalizes S08's one-off live-fill parser (research/08_live_validation/build_08_live_validation.py)
into a rerunnable, idempotent monitor.

    python scripts/tca_monitor.py <log_dir> [--state-dir DIR] [--window 20]

Idempotent: keeps a processed-files ledger (`processed_logs.txt`) in `--state-dir` so a rerun only
parses *.log files it hasn't seen before, and APPENDS to a running `fills.csv` there — cheap to
run often (e.g. daily) rather than reparsing the whole log directory every time.

Per fill: realized slippage (S08's regex parser, unchanged) vs `predict_slippage()`'s prediction
under the logged conditions (symbol, side, notional, decision price as `price`, order style
mapped to `order_type`). Rolling `--window` (default 20) fills per (side, order_type) group:
compares the rolling mean of (realized - predicted mean) against the model's own predicted
timing sigma. **Alerts** (nonzero exit code + printed summary) when that rolling residual mean
strays beyond 2x the model's predicted sigma — the model no longer explains what's happening.

How/when to schedule this recurrently (cron, launchd, CI) is an owner decision — this script
only implements the one-shot check; it does not schedule itself.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slippage.predict import predict_slippage  # noqa: E402
from slippage.state import MarketState, session_bin_label  # noqa: E402

DEFAULT_STATE_DIR = (Path(__file__).resolve().parents[1] /
                     "research" / "predictor" / "04_tca_monitor" / "results")
DEFAULT_WINDOW = 20

# Representative per-symbol reference constants (S08's fills are retail-size, "too small to
# register impact" per findings_08 — so a large placeholder volume keeps impact ~0, matching
# that finding, rather than trying to reconstruct a live volume nowcast for arbitrary past
# log timestamps). spread from findings_01; sigma from findings_09's "normal" mean.
_SPREAD_BPS = {"TQQQ": 0.74, "SQQQ": 1.00}
_SIGMA_NORMAL_BPS = {"TQQQ": 370.0, "SQQQ": 372.0}
_LARGE_VOLUME_SHARES = 1e9

# S08's exact parser (research/08_live_validation/build_08_live_validation.py) — duplicated
# rather than imported since research/NN_*/ folders aren't packages.
BUY_RE = re.compile(
    r">>> BUY (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) qty ([\d.]+).*? "
    r"decision ([\d.]+) fill ([\d.]+)")
SELL_RE = re.compile(
    r"SELL (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) qty ([\d.]+) "
    r"decision ([\d.]+) fill ([\d.]+) style (\w+)")


def _state_for(symbol: str, ts: pd.Timestamp) -> MarketState:
    spread = _SPREAD_BPS.get(symbol, _SPREAD_BPS["TQQQ"])
    sigma = _SIGMA_NORMAL_BPS.get(symbol, _SIGMA_NORMAL_BPS["TQQQ"])
    return MarketState(
        ts=ts, symbol=symbol, bin_label=session_bin_label(ts),
        expected_interval_volume=_LARGE_VOLUME_SHARES,
        thin_volume_p10=_LARGE_VOLUME_SHARES * 0.4, thin_volume_p20=_LARGE_VOLUME_SHARES * 0.6,
        sigma_now_bps=sigma, regime="normal", spread_bps=spread,
    )


def parse_new_fills(log_dir: Path, already_processed: set[str]) -> tuple[pd.DataFrame, list[str]]:
    rows, newly_processed = [], []
    for f in sorted(log_dir.glob("*.log")):
        if f.name in already_processed:
            continue
        newly_processed.append(f.name)
        txt = f.read_text(errors="ignore")
        for m in BUY_RE.finditer(txt):
            ts, sym, qty, dec, fill = m.groups()
            dec, fill, qty = float(dec), float(fill), float(qty)
            rows.append(dict(time=ts, symbol=sym, side="buy", style="limit",
                             notional=qty * fill, decision=dec,
                             slip_bps=(fill - dec) / dec * 1e4))
        for m in SELL_RE.finditer(txt):
            ts, sym, qty, dec, fill, style = m.groups()
            dec, fill, qty = float(dec), float(fill), float(qty)
            rows.append(dict(time=ts, symbol=sym, side="sell", style=style,
                             notional=qty * fill, decision=dec,
                             slip_bps=(dec - fill) / dec * 1e4))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df, newly_processed


def predict_for_fill(row: pd.Series) -> dict:
    order_type = "limit_chase" if row["style"] == "limit" else "cross"
    state = _state_for(row["symbol"], row["time"])
    return predict_slippage(row["notional"], row["side"], order_type, state,
                            price=row["decision"], latency_min=15.0)


def load_state(state_dir: Path) -> tuple[set[str], pd.DataFrame]:
    ledger_path = state_dir / "processed_logs.txt"
    fills_path = state_dir / "fills.csv"
    processed = set(ledger_path.read_text().splitlines()) if ledger_path.exists() else set()
    fills = pd.read_csv(fills_path, parse_dates=["time"]) if fills_path.exists() else pd.DataFrame()
    return processed, fills


def save_state(state_dir: Path, processed: set[str], fills: pd.DataFrame) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "processed_logs.txt").write_text("\n".join(sorted(processed)) + "\n")
    fills.to_csv(state_dir / "fills.csv", index=False)


def check_drift(fills: pd.DataFrame, window: int) -> tuple[pd.DataFrame, bool]:
    """Per (side, order_type), rolling-window mean residual vs the model's 2sigma band."""
    rows, alert = [], False
    for (side, style), g in fills.groupby(["side", "style"]):
        g = g.sort_values("time").tail(window)
        if len(g) < window:
            continue   # not enough fills yet for a stable rolling read
        resid_mean = g["residual_bps"].mean()
        sigma_ref = g["timing_sigma_bps"].mean()
        breach = bool(abs(resid_mean) > 2 * sigma_ref)
        alert = alert or breach
        rows.append({"side": side, "style": style, "n": len(g),
                    "rolling_resid_mean_bps": resid_mean, "model_sigma_bps": sigma_ref,
                    "breach_2sigma": breach})
    return pd.DataFrame(rows), alert


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
GROUP_COLORS = {("buy", "limit"): "#f0b429", ("sell", "market"): "#58a6ff",
                ("sell", "aggressive_limit"): "#3fb950"}


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


def make_plot(fills: pd.DataFrame, state_dir: Path):
    """Regenerate tca_monitor.png from the cumulative fills ledger (refreshed every run)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(DARK_BG)

    # (1) per-fill: realized slippage over time, colored by group; model mean as reference.
    for (side, style), g in fills.groupby(["side", "style"]):
        col = GROUP_COLORS.get((side, style), TEXT_COL)
        g = g.sort_values("time")
        ax1.scatter(g["time"], g["slip_bps"], s=22, color=col, label=f"{side}/{style} (n={len(g)})")
        ax1.plot(g["time"], g["predicted_mean_bps"], color=col, lw=1.0, ls="--", alpha=0.7)
    ax1.axhline(0, color=GRID_COL, lw=0.8)
    ax1.set_ylabel("slippage (bps; dots = realized, dashed = predicted mean)")
    ax1.set_title("per-fill realized vs predicted")
    ax1.tick_params(axis="x", rotation=30)
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7.5, labelcolor=TEXT_COL)

    # (2) per-group means: realized vs predicted, residual annotated.
    stats = (fills.groupby(["side", "style"])
             .agg(realized=("slip_bps", "mean"), predicted=("predicted_mean_bps", "mean"),
                  n=("slip_bps", "size")).reset_index())
    x = np.arange(len(stats))
    ax2.bar(x - 0.18, stats["realized"], width=0.36, color="#f85149", label="realized mean")
    ax2.bar(x + 0.18, stats["predicted"], width=0.36, color="#58a6ff", label="predicted mean")
    for xi, r in zip(x, stats.itertuples()):
        ax2.text(xi, max(r.realized, r.predicted),
                 f"resid {r.realized - r.predicted:+.1f}", ha="center", va="bottom",
                 fontsize=7.5, color=TEXT_COL)
    ax2.axhline(0, color=GRID_COL, lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r.side}\n{r.style}\n(n={r.n})" for r in stats.itertuples()], fontsize=7.5)
    ax2.set_ylabel("mean slippage (bps)")
    ax2.set_title("group means — the buy/limit residual is P02's adverse-selection gap")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.suptitle("P04 — TCA monitor: realized vs predict_slippage() on live fills",
                 color=TEXT_COL, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(state_dir / "tca_monitor.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = ap.parse_args()

    processed, fills = load_state(args.state_dir)
    new_df, newly_processed = parse_new_fills(args.log_dir, processed)

    if not new_df.empty:
        preds = new_df.apply(predict_for_fill, axis=1)
        new_df["predicted_mean_bps"] = preds.apply(lambda p: p["mean_bps"])
        new_df["timing_sigma_bps"] = preds.apply(lambda p: p["components"]["timing_sigma_bps"])
        new_df["residual_bps"] = new_df["slip_bps"] - new_df["predicted_mean_bps"]
        fills = pd.concat([fills, new_df], ignore_index=True)

    processed |= set(newly_processed)
    save_state(args.state_dir, processed, fills)

    print(f"TCA monitor: {len(newly_processed)} new log(s) parsed, {len(new_df)} new fill(s), "
          f"{len(fills)} total fills tracked ({args.state_dir}).")
    if fills.empty:
        print("No fills yet.")
        return 0

    make_plot(fills, args.state_dir)
    drift, alert = check_drift(fills, args.window)
    if not drift.empty:
        print(f"\nRolling {args.window}-fill drift check (realized - predicted vs model's 2sigma band):")
        print(drift.to_string(index=False))
    else:
        print(f"\nFewer than {args.window} fills in every (side, style) group — drift check skipped.")

    if alert:
        print("\nALERT: rolling residual mean has left the model's 2sigma band for at least one "
              "(side, style) group — the model no longer explains recent fills.")
        return 1
    print("\nNo drift alert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
