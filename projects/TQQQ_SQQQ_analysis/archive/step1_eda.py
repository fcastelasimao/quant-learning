"""
Step 1 EDA — validate canonical trade logs and produce edge-prior plot.
Outputs go to step1_outputs/. Reads but never writes to canonical CSVs.
"""
import os
import sys
import sqlite3
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

from quantcore import config as _qc_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_qc_config.data_dir())
OUT_DIR = os.path.join(BASE_DIR, "step1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CANONICAL_FILES = {
    "TQQQ": os.path.join(BASE_DIR, "TRADES_TQQQ_canonical.csv"),
    "SQQQ": os.path.join(BASE_DIR, "TRADES_SQQQ_canonical.csv"),
}
RAW_FILES = {
    "TQQQ": os.path.join(BASE_DIR, "TRADES_TQQQ_backtest_alpaca.csv"),
    "SQQQ": os.path.join(BASE_DIR, "TRADES_SQQQ_backtest_alpaca.csv"),
}
DB_FILES = {
    "TQQQ": os.path.join(DATA_DIR, "DB_TQQQ_historical_data.db"),
    "SQQQ": os.path.join(DATA_DIR, "DB_SQQQ_historical_data.db"),
    "^IRX": os.path.join(DATA_DIR, "DB_^IRX_historical_data.db"),
}

CRITICAL_COLS = [
    "RSI_entry", "decision_price", "avg_order_price",
    "exit_decision_price", "exit_avg_order_price", "pnl_pct",
    "entry_time", "exit_time",
]

report_lines: list[str] = []


def r(text: str = "") -> None:
    report_lines.append(text)


def progress(msg: str) -> None:
    print(f"[step1] {msg}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 7.0  SETUP  (done above)
# ---------------------------------------------------------------------------

progress("Starting Step 1 EDA")
r("# Step 1 — EDA Report")
r()

# ---------------------------------------------------------------------------
# 7.1  FMP DB inventory
# ---------------------------------------------------------------------------
progress("7.1  FMP DB inventory")
r("## Data availability")
r()

DB_WINDOW = ("2020-01-03 00:00:00", "2026-05-08 23:59:59")

for db_key, db_path in DB_FILES.items():
    r(f"### {db_key}")
    if not os.path.exists(db_path):
        r(f"  **WARNING:** {db_path} not found — skipping.")
        r()
        continue
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        # count rows in window
        cur.execute(
            "SELECT COUNT(*) FROM candles_1d "
            "WHERE et_datetime BETWEEN ? AND ?",
            DB_WINDOW,
        )
        total_rows = cur.fetchone()[0]
        r(f"  - Rows in window {DB_WINDOW[0][:10]} → {DB_WINDOW[1][:10]}: **{total_rows:,}**")
        # null adj_close (only meaningful for equity symbols)
        if db_key in ("TQQQ", "SQQQ"):
            cur.execute(
                "SELECT COUNT(*) FROM candles_1d "
                "WHERE et_datetime BETWEEN ? AND ? AND adj_close IS NULL",
                DB_WINDOW,
            )
            null_adj = cur.fetchone()[0]
            r(f"  - Rows with `adj_close IS NULL`: **{null_adj}** (expected 0)")
        con.close()
    except Exception as exc:
        r(f"  **ERROR querying {db_key}:** {exc}")
    r()

# ---------------------------------------------------------------------------
# 7.2  Load canonical files
# ---------------------------------------------------------------------------
progress("7.2  Loading canonical files")
r("## Canonical trade logs")
r()

canons: dict[str, pd.DataFrame] = {}
canonical_runs: dict[str, str] = {}

for sym, path in CANONICAL_FILES.items():
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Assert single run
    unique_runs = df["run_started_at"].unique()
    if len(unique_runs) != 1:
        sys.exit(f"FATAL: {sym} canonical has {len(unique_runs)} distinct run_started_at values: {unique_runs}")

    # Assert single symbol
    unique_syms = df["symbol"].unique()
    if len(unique_syms) != 1 or unique_syms[0] != sym:
        sys.exit(f"FATAL: {sym} canonical has unexpected symbols: {unique_syms}")

    # Assert mode == backtest
    bad_mode = df[df["mode"] != "backtest"]
    if len(bad_mode) > 0:
        sys.exit(f"FATAL: {sym} canonical has {len(bad_mode)} rows with mode != 'backtest'")

    canons[sym] = df
    canonical_runs[sym] = str(df["run_started_at"].iloc[0])

    span_start = df["entry_time"].min()
    span_end = df["exit_time"].max()
    r(f"### {sym}")
    r(f"  - Rows: **{len(df):,}**")
    r(f"  - Source run: `{canonical_runs[sym]}`")
    r(f"  - Span: `{span_start}` → `{span_end}`")
    r(f"  - Columns: {df.shape[1]}")
    r()

# ---------------------------------------------------------------------------
# 7.3  Null check on critical columns
# ---------------------------------------------------------------------------
progress("7.3  Null checks")
r("## Validation results")
r()
r("### Nulls in critical columns")
r()

for sym, df in canons.items():
    r(f"**{sym}**")
    for col in CRITICAL_COLS:
        if col not in df.columns:
            r(f"  - `{col}`: column MISSING")
            continue
        n_null = df[col].isna().sum()
        note = ""
        if n_null > 0:
            first5 = df.index[df[col].isna()][:5].tolist()
            note = f" — first 5 row indices: {first5}"
        r(f"  - `{col}`: {n_null} null(s){note}")
    r()

# ---------------------------------------------------------------------------
# 7.4  Timestamp validation
# ---------------------------------------------------------------------------
progress("7.4  Timestamp validation")
r("### Timestamp validity")
r()

NYSE_OPEN = pd.Timedelta(hours=9, minutes=30)
NYSE_CLOSE = pd.Timedelta(hours=16)
VALID_MINUTES = {0, 15, 30, 45}


def ts_checks(df: pd.DataFrame, sym: str) -> None:
    r(f"**{sym}**")

    # exit < entry
    bad_order = df[df["exit_time"] < df["entry_time"]]
    r(f"  - `exit_time < entry_time`: **{len(bad_order)}**"
      + (f" — first 5: {bad_order.index[:5].tolist()}" if len(bad_order) else ""))

    for col in ("entry_time", "exit_time"):
        ts = df[col]

        # Outside NYSE regular hours (09:30–16:00) or not weekday
        tod = ts.dt.hour * 60 + ts.dt.minute  # minutes since midnight
        outside_hours = df[
            (ts.dt.dayofweek >= 5) |
            (tod < 9 * 60 + 30) |
            (tod > 16 * 60)
        ]
        r(f"  - `{col}` outside NYSE hours or weekend: **{len(outside_hours)}**"
          + (f" — first 5 indices: {outside_hours.index[:5].tolist()}" if len(outside_hours) else ""))

        # Not on 15-min boundary
        off_boundary = df[~ts.dt.minute.isin(VALID_MINUTES)]
        r(f"  - `{col}` minute not in {{0,15,30,45}}: **{len(off_boundary)}**"
          + (f" — first 5 indices: {off_boundary.index[:5].tolist()}" if len(off_boundary) else ""))
    r()


for sym, df in canons.items():
    ts_checks(df, sym)

# ---------------------------------------------------------------------------
# 7.5  pnl_pct sanity check
# ---------------------------------------------------------------------------
progress("7.5  pnl_pct sanity check")
r("### pnl_pct math check")
r()

for sym, df in canons.items():
    recomputed = df["exit_avg_order_price"] / df["avg_order_price"] - 1
    diff = (recomputed - df["pnl_pct"]).abs()

    n_large = (diff > 0.001).sum()
    r(f"**{sym}** (formula: exit_avg/avg − 1 vs stored pnl_pct)")
    r(f"  - Rows with |diff| > 0.001: **{n_large}**")
    r(f"  - diff max={diff.max():.6f}, mean={diff.mean():.6f}, p99={diff.quantile(0.99):.6f}")

    # Detect if pnl_pct might be in percent form (×100 scaling)
    mean_abs_pct = df["pnl_pct"].abs().mean()
    mean_abs_frac = recomputed.abs().mean()
    scaling_note = ""
    if mean_abs_pct > mean_abs_frac * 50:
        # pnl_pct looks percentage-scaled; redo diff
        diff_pct = (recomputed * 100 - df["pnl_pct"]).abs()
        n_large_pct = (diff_pct > 0.001).sum()
        scaling_note = (
            f"  - **NOTE:** pnl_pct mean≈{mean_abs_pct:.4f} vs recomputed mean≈{mean_abs_frac:.4f} "
            f"— pnl_pct appears to be in percentage form (×100). "
            f"Re-checking with diff = |recomputed×100 − pnl_pct|: "
            f"{n_large_pct} rows > 0.001; max={diff_pct.max():.6f}, mean={diff_pct.mean():.6f}"
        )

    if n_large > 0:
        cols_show = ["avg_order_price", "exit_avg_order_price", "pnl_pct"]
        bad5 = df[diff > 0.001][cols_show].head()
        bad5 = bad5.copy()
        bad5["recomputed"] = recomputed[bad5.index]
        r(f"  - First 5 rows with |diff| > 0.001:")
        r(f"```\n{bad5.to_string()}\n```")
    if scaling_note:
        r(scaling_note)
    r()

# ---------------------------------------------------------------------------
# 7.6  Outlier flags
# ---------------------------------------------------------------------------
progress("7.6  Outlier flags")
r("### Outliers (reported, not removed)")
r()
r("**NOTE:** pnl_pct is in percentage form (×100). The threshold |pnl_pct| > 0.50 catches "
  "returns > 0.5%, not > 50%. Virtually all TQQQ/SQQQ trades exceed 0.5% so this count "
  "is expected to be large. For >50% outliers, the equivalent threshold would be >50.0.")
r()

for sym, df in canons.items():
    r(f"**{sym}**")

    big_pnl = df[df["pnl_pct"].abs() > 0.50]
    r(f"  - |pnl_pct| > 0.50: **{len(big_pnl)}**"
      + (f"\n```\n{big_pnl[['entry_time','exit_time','pnl_pct']].head().to_string()}\n```" if len(big_pnl) else ""))

    df = df.copy()
    df["_dur_days"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 86400
    long_trades = df[df["_dur_days"] > 5]
    r(f"  - Duration > 5 calendar days: **{len(long_trades)}**"
      + (f"\n```\n{long_trades[['entry_time','exit_time','_dur_days']].head().to_string()}\n```" if len(long_trades) else ""))
    # put the column back in canons
    canons[sym]["_dur_days"] = df["_dur_days"]
    r()

# ---------------------------------------------------------------------------
# 7.7  Self-overlap within symbol
# ---------------------------------------------------------------------------
progress("7.7  Self-overlap within symbol")
r("## Trade structure")
r()
r("### Self-overlap (within symbol)")
r()

for sym, df in canons.items():
    df = df.sort_values("entry_time").reset_index(drop=True)

    # consecutive check
    n_overlap = int(((df["entry_time"].iloc[1:].values) < df["exit_time"].iloc[:-1].values).sum())

    r(f"**{sym}** ({len(df)} trades sorted by entry_time)")
    r(f"  - Consecutive-pair overlaps (entry[i+1] < exit[i]): **{n_overlap}**")

    if n_overlap > 0:
        # Event sweep for max depth and longest contiguous overlap span
        events = []
        for _, row in df.iterrows():
            events.append((row["entry_time"], 1))
            events.append((row["exit_time"], -1))
        events.sort(key=lambda x: (x[0], x[1]))  # closes before opens at same time

        depth = 0
        max_depth = 0
        overlap_start = None
        max_overlap_span = pd.Timedelta(0)

        for t, delta in events:
            if depth <= 1 and depth + delta > 1:
                overlap_start = t
            elif depth > 1 and depth + delta <= 1 and overlap_start is not None:
                span = t - overlap_start
                if span > max_overlap_span:
                    max_overlap_span = span
                overlap_start = None
            depth += delta
            if depth > max_depth:
                max_depth = depth

        r(f"  - Max simultaneous open positions: **{max_depth}**")
        r(f"  - Longest contiguous overlap span: **{max_overlap_span}**")
        r(f"  - **Impact on Step 2:** position sizing must handle stacked positions.")
    r()

# ---------------------------------------------------------------------------
# 7.8  Cross-symbol overlap report
# ---------------------------------------------------------------------------
progress("7.8  Cross-symbol overlap")
r("### Cross-symbol overlap (TQQQ ∩ SQQQ)")
r()

tqqq_df = canons["TQQQ"][["trade_id", "entry_time", "exit_time"]].copy()
sqqq_df = canons["SQQQ"][["trade_id", "entry_time", "exit_time"]].copy()

# Vectorised interval overlap via numpy broadcast
t_en = tqqq_df["entry_time"].values.astype("datetime64[ns]")
t_ex = tqqq_df["exit_time"].values.astype("datetime64[ns]")
s_en = sqqq_df["entry_time"].values.astype("datetime64[ns]")
s_ex = sqqq_df["exit_time"].values.astype("datetime64[ns]")

# overlap[i,j] = True iff TQQQ[i] and SQQQ[j] overlap
overlap_mat = (t_en[:, None] < s_ex[None, :]) & (s_en[None, :] < t_ex[:, None])

ti_idx, si_idx = np.where(overlap_mat)

if len(ti_idx) > 0:
    o_start = np.maximum(t_en[ti_idx], s_en[si_idx])
    o_end = np.minimum(t_ex[ti_idx], s_ex[si_idx])
    overlap_secs = (o_end - o_start).astype("int64") / 1e9

    overlap_df = pd.DataFrame({
        "tqqq_trade_id": tqqq_df["trade_id"].iloc[ti_idx].values,
        "tqqq_entry": tqqq_df["entry_time"].iloc[ti_idx].values,
        "tqqq_exit": tqqq_df["exit_time"].iloc[ti_idx].values,
        "sqqq_trade_id": sqqq_df["trade_id"].iloc[si_idx].values,
        "sqqq_entry": sqqq_df["entry_time"].iloc[si_idx].values,
        "sqqq_exit": sqqq_df["exit_time"].iloc[si_idx].values,
        "overlap_seconds": overlap_secs,
    })
    overlap_df.to_csv(os.path.join(OUT_DIR, "overlap_report.csv"), index=False)

    n_pairs = len(overlap_df)
    frac_tqqq = overlap_mat.any(axis=1).mean()
    total_overlap_h = overlap_secs.sum() / 3600

    # max simultaneous across both symbols via event sweep
    all_events = []
    for _, row in tqqq_df.iterrows():
        all_events.append((row["entry_time"], 1))
        all_events.append((row["exit_time"], -1))
    for _, row in sqqq_df.iterrows():
        all_events.append((row["entry_time"], 1))
        all_events.append((row["exit_time"], -1))
    all_events.sort(key=lambda x: (x[0], x[1]))
    depth = 0
    max_combined_depth = 0
    for _, delta in all_events:
        depth += delta
        max_combined_depth = max(max_combined_depth, depth)

    r(f"  - Overlapping pairs: **{n_pairs:,}**")
    r(f"  - Fraction of TQQQ trades overlapping ≥1 SQQQ trade: **{frac_tqqq:.1%}**")
    r(f"  - Total overlapping wall-time: **{total_overlap_h:,.1f} hours**")
    r(f"  - Max simultaneous open positions (both symbols): **{max_combined_depth}**")
else:
    overlap_df = pd.DataFrame(columns=[
        "tqqq_trade_id", "tqqq_entry", "tqqq_exit",
        "sqqq_trade_id", "sqqq_entry", "sqqq_exit", "overlap_seconds",
    ])
    overlap_df.to_csv(os.path.join(OUT_DIR, "overlap_report.csv"), index=False)
    r("  - Overlapping pairs: **0**")
r()

# ---------------------------------------------------------------------------
# 7.9  Disagreement diagnostic
# ---------------------------------------------------------------------------
progress("7.9  Disagreement diagnostic")
r("### Disagreements vs other runs")
r()

for sym in ("TQQQ", "SQQQ"):
    canon = canons[sym]
    canon_run = canonical_runs[sym]
    raw_path = RAW_FILES[sym]

    raw = pd.read_csv(raw_path, parse_dates=["entry_time", "exit_time"])
    raw["run_started_at"] = raw["run_started_at"].astype(str)

    # Per-run trade counts
    run_counts = raw.groupby("run_started_at").size().rename("other_run_trade_count").reset_index()
    run_counts.columns = ["run_started_at", "other_run_trade_count"]

    # Canonical fingerprint
    canon_fp = canon[["entry_time", "exit_time", "avg_order_price",
                       "exit_avg_order_price", "RSI_entry"]].copy()
    canon_fp = canon_fp.rename(columns={
        "avg_order_price": "canonical_avg",
        "exit_avg_order_price": "canonical_exit_avg",
        "RSI_entry": "canonical_RSI",
    })

    # Non-canonical rows
    other = raw[raw["run_started_at"] != canon_run][
        ["run_started_at", "entry_time", "exit_time",
         "avg_order_price", "exit_avg_order_price", "RSI_entry"]
    ].copy()

    # Merge on (entry_time, exit_time)
    merged = other.merge(canon_fp, on=["entry_time", "exit_time"], how="inner")
    if len(merged) == 0:
        disagree = pd.DataFrame()
    else:
        merged["diff_avg"] = (merged["avg_order_price"] - merged["canonical_avg"]).abs()
        merged["diff_exit_avg"] = (merged["exit_avg_order_price"] - merged["canonical_exit_avg"]).abs()
        merged["diff_RSI"] = (merged["RSI_entry"] - merged["canonical_RSI"]).abs()

        disagree = merged[
            (merged["diff_avg"] > 1e-6) |
            (merged["diff_exit_avg"] > 1e-6) |
            (merged["diff_RSI"] > 1e-6)
        ].copy()

        if len(disagree) > 0:
            disagree = disagree.merge(run_counts, on="run_started_at", how="left")
            disagree = disagree.rename(columns={
                "run_started_at": "other_run",
                "avg_order_price": "other_avg",
                "exit_avg_order_price": "other_exit_avg",
                "RSI_entry": "other_RSI",
            })
            disagree = disagree[[
                "entry_time", "exit_time",
                "canonical_avg", "canonical_exit_avg", "canonical_RSI",
                "other_run", "other_run_trade_count",
                "other_avg", "other_exit_avg", "other_RSI",
                "diff_avg", "diff_exit_avg", "diff_RSI",
            ]]

    out_path = os.path.join(OUT_DIR, f"disagreements_{sym}.csv")
    disagree.to_csv(out_path, index=False)

    r(f"**{sym}**")
    r(f"  - Non-canonical rows with matching (entry_time, exit_time): checked")
    r(f"  - Disagreements (any field differs > 1e-6): **{len(disagree)}**")
    if len(disagree) > 0:
        r(f"  - Max |diff_avg|: {disagree['diff_avg'].max():.8f}")
        r(f"  - Max |diff_exit_avg|: {disagree['diff_exit_avg'].max():.8f}")
        r(f"  - Max |diff_RSI|: {disagree['diff_RSI'].max():.8f}")
        other_runs_list = disagree["other_run"].unique().tolist()
        r(f"  - Runs that disagreed ({len(other_runs_list)}): {other_runs_list[:10]}"
          + (" …" if len(other_runs_list) > 10 else ""))
    r()

# ---------------------------------------------------------------------------
# 7.10  Distribution profiling
# ---------------------------------------------------------------------------
progress("7.10  Distribution profiling")
r("## Distributions")
r()

for sym, df in canons.items():
    r(f"### {sym}")
    r()

    # RSI_entry
    rsi = df["RSI_entry"].dropna()
    r("**RSI_entry**")
    r(f"  mean={rsi.mean():.2f}, median={rsi.median():.2f}, std={rsi.std():.2f}, "
      f"min={rsi.min():.2f}, max={rsi.max():.2f}, IQR={rsi.quantile(0.75)-rsi.quantile(0.25):.2f}")
    r()

    # pnl_pct
    pnl = df["pnl_pct"].dropna()
    win_rate = (pnl > 0).mean()
    r("**pnl_pct**")
    r(f"  mean={pnl.mean():.5f}, median={pnl.median():.5f}, std={pnl.std():.5f}, "
      f"win_rate={win_rate:.1%}, skew={pnl.skew():.3f}, excess_kurt={pnl.kurtosis():.3f}")
    r()

    # Trade duration
    dur = df["_dur_days"]
    r("**Trade duration (calendar days)**")
    r(f"  median={dur.median():.2f}, p90={dur.quantile(0.90):.2f}, max={dur.max():.2f}")
    r()

    # exit_reason
    r("**exit_reason value_counts**")
    vc = df["exit_reason"].value_counts()
    for k, v in vc.items():
        r(f"  - {k}: {v}")
    r()

    # regime_entry
    r("**regime_entry value_counts**")
    if df["regime_entry"].notna().any():
        vc2 = df["regime_entry"].value_counts()
        for k, v in vc2.items():
            r(f"  - {k}: {v}")
    else:
        r("  - all NaN")
    r()

    # hour_of_entry
    r("**hour_of_entry value_counts**")
    vc3 = df["hour_of_entry"].value_counts().sort_index()
    for k, v in vc3.items():
        r(f"  - hour {k}: {v}")
    r()

# ---------------------------------------------------------------------------
# 7.11  Edge prior
# ---------------------------------------------------------------------------
progress("7.11  Edge prior — pnl_pct vs RSI_entry")
r("## Edge prior — pnl_pct vs RSI_entry")
r()

# Combine symbols
combined = pd.concat(
    [canons["TQQQ"].assign(symbol="TQQQ"), canons["SQQQ"].assign(symbol="SQQQ")],
    ignore_index=True,
)

# 5-point bins: [0,5), [5,10), ..., [95,100]
bin_edges = list(range(0, 101, 5))
bin_labels = [f"[{b},{b+5})" for b in range(0, 100, 5)]
combined["rsi_bin"] = pd.cut(
    combined["RSI_entry"], bins=bin_edges, right=False, labels=bin_labels
)

binned = (
    combined.groupby(["symbol", "rsi_bin"], observed=True)["pnl_pct"]
    .agg(
        n_trades="count",
        mean_pnl_pct="mean",
        median_pnl_pct="median",
        std_pnl_pct="std",
        win_rate=lambda x: (x > 0).mean(),
    )
    .reset_index()
)
binned["sem"] = binned["std_pnl_pct"] / np.sqrt(binned["n_trades"])

binned_path = os.path.join(OUT_DIR, "prior_pnl_vs_rsi_binned.csv")
binned.to_csv(binned_path, index=False)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle("Edge Prior: pnl_pct vs RSI_entry", fontsize=14, fontweight="bold")

# pnl_pct is in percentage form (×100), so ±10 = ±10%. Plan specified [-0.10,0.10]
# for fraction form; we use [-10, 10] to produce a meaningful plot.
CLIP_LO, CLIP_HI = -10.0, 10.0

for ax, sym in zip(axes, ["TQQQ", "SQQQ"]):
    sub = combined[combined["symbol"] == sym].dropna(subset=["RSI_entry", "pnl_pct"])
    x = sub["RSI_entry"].values
    y = sub["pnl_pct"].values

    # Count clipped points
    n_clipped = int(((y < CLIP_LO) | (y > CLIP_HI)).sum())

    # Scatter (clipped view)
    ax.scatter(x, np.clip(y, CLIP_LO, CLIP_HI), alpha=0.25, s=10,
               color="steelblue", label="_nolegend_")

    # Binned mean ±1 SEM
    bsym = binned[binned["symbol"] == sym].copy()
    bin_mids = [b + 2.5 for b in range(0, 100, 5)]
    bsym["bin_mid"] = bin_mids[:len(bsym)]
    valid_b = bsym[bsym["n_trades"] > 0]
    ax.errorbar(
        valid_b["bin_mid"], valid_b["mean_pnl_pct"],
        yerr=valid_b["sem"],
        fmt="o-", color="darkorange", ms=5, lw=1.5,
        label="Bin mean ±1 SEM", zorder=4,
    )

    # Linear regression
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    rsq = r_value ** 2
    xrange = np.array([0, 100])
    ax.plot(xrange, intercept + slope * xrange, "r--", lw=1.5,
            label=f"OLS fit", zorder=5)

    # Reference line
    ax.axhline(0, color="black", lw=0.8, ls="--")

    # Formatting
    ax.set_xlim(0, 100)
    ax.set_ylim(CLIP_LO, CLIP_HI)
    ax.set_xlabel("RSI_entry")
    if sym == "TQQQ":
        ax.set_ylabel("pnl_pct")
    ax.set_title(f"{sym}: slope={slope:.5f}, p={p_value:.3f}, R²={rsq:.3f}")
    ax.legend(fontsize=8)
    ax.text(
        0.02, 0.02,
        f"{n_clipped} pts clipped",
        transform=ax.transAxes, fontsize=8, color="grey", va="bottom",
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # Report regression
    r(f"**{sym}**")
    r(f"  - slope={slope:.6f}, intercept={intercept:.6f}, p={p_value:.4f}, R²={rsq:.4f}")
    n_bins_with_data = int((valid_b["n_trades"] > 0).sum())
    r(f"  - Bins with data: {n_bins_with_data}/{len(bsym)}")
    r()

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "prior_pnl_vs_rsi.png")
plt.savefig(plot_path, dpi=150)
plt.close()

r(f"See `prior_pnl_vs_rsi.png` for the scatter/regression/binned-mean plot.")
r()

# Compact binned table in report
r("**Binned mean pnl_pct by RSI bin (5-pt bins)**")
r()
r("| symbol | rsi_bin | n | mean_pnl_pct | median_pnl_pct | win_rate | sem |")
r("|--------|---------|---|-------------|---------------|----------|-----|")
for _, row in binned.iterrows():
    r(f"| {row['symbol']} | {row['rsi_bin']} | {int(row['n_trades'])} "
      f"| {row['mean_pnl_pct']:.5f} | {row['median_pnl_pct']:.5f} "
      f"| {row['win_rate']:.1%} | {row['sem']:.5f} |")
r()

# ---------------------------------------------------------------------------
# 7.12  Headline and conclusions
# ---------------------------------------------------------------------------
progress("7.12  Writing headline and conclusions")

# Collect regression results for headline
reg_results = {}
for sym in ("TQQQ", "SQQQ"):
    sub = combined[combined["symbol"] == sym].dropna(subset=["RSI_entry", "pnl_pct"])
    slope, intercept, r_value, p_value, _ = stats.linregress(
        sub["RSI_entry"].values, sub["pnl_pct"].values
    )
    reg_results[sym] = {"slope": slope, "p": p_value, "rsq": r_value ** 2}

# Insert headline at beginning of report
headline_lines = []
headline_lines.append("")
headline_lines.append("## Headline")
headline_lines.append("")

for sym, res in reg_results.items():
    sig_note = "statistically significant (p<0.05)" if res["p"] < 0.05 else "NOT statistically significant (p≥0.05)"
    headline_lines.append(
        f"- **{sym}**: slope={res['slope']:.5f}, p={res['p']:.3f}, R²={res['rsq']:.4f} — {sig_note}."
    )
headline_lines.append("")

# Insert after title
insert_idx = 1
for i, line in enumerate(report_lines):
    if line.startswith("## Data availability"):
        insert_idx = i
        break
report_lines[insert_idx:insert_idx] = headline_lines

# Conclusions section
r("## Conclusions")
r()

for sym, res in reg_results.items():
    direction = "positive" if res["slope"] > 0 else "negative"
    if res["p"] < 0.05:
        r(f"- **{sym} shows a {direction} and statistically significant RSI slope** "
          f"(slope={res['slope']:.5f}, p={res['p']:.3f}). "
          f"The relationship explains R²={res['rsq']:.4f} of variance — "
          f"small in absolute terms, but confirms a detectable signal worth sweeping.")
    else:
        r(f"- **{sym}: no statistically significant RSI edge detected** "
          f"(slope={res['slope']:.5f}, p={res['p']:.3f}, R²={res['rsq']:.4f}). "
          f"Proceed to the threshold sweep cautiously — any effect found may be noise.")
r()
r("- Data integrity: see Validation results section for any violations found.")
r("- Cross-symbol overlap: see Trade structure section. "
  "Step 2 combined-portfolio run must handle simultaneous TQQQ+SQQQ positions.")
r("- Disagreement diagnostic: see disagreements_TQQQ.csv and disagreements_SQQQ.csv "
  "for any price/RSI inconsistencies across runs.")
r()

# ---------------------------------------------------------------------------
# Write eda_report.md
# ---------------------------------------------------------------------------
progress("Writing eda_report.md")
report_path = os.path.join(OUT_DIR, "eda_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
expected_files = [
    "eda_report.md",
    "prior_pnl_vs_rsi.png",
    "prior_pnl_vs_rsi_binned.csv",
    "disagreements_TQQQ.csv",
    "disagreements_SQQQ.csv",
    "overlap_report.csv",
]

progress("Verifying outputs...")
all_ok = True
for fname in expected_files:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f"  OK  {fname} ({size:,} bytes)")
    else:
        print(f"  MISSING  {fname}")
        all_ok = False

if all_ok:
    progress("Step 1 complete — all 6 output files produced.")
else:
    sys.exit("Step 1 FAILED — some output files are missing.")
