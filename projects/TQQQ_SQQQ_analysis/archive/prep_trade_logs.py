"""
Prepare newly received TQQQ/SQQQ Excel trade logs for the RSI overlay workflow.

This script intentionally does not require RSI columns. It normalizes the
current Excel schema, runs data-quality checks, writes canonical interim CSVs,
and creates a readiness report that makes the remaining RSI dependency explicit.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJ = BASE_DIR.parent
DEFAULT_INPUT_DIR = PROJ / "full_history_canonical" / "trades_backtest"
DEFAULT_OUTPUT_DIR = BASE_DIR / "prep_outputs"

FILENAME_RE = re.compile(
    r"^(?P<symbol>TQQQ|SQQQ)_backtest_intraday_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"(?P<run_id>\d+(?:_\d+)?)\.xlsx$"
)

RENAME_MAP = {
    "entry_price": "avg_order_price",
    "avg_price": "source_avg_price",
    "capital_at_entry": "capital_before",
    "capital_after_entry": "capital_after",
    "exit_price": "exit_avg_order_price",
    "profit": "pnl",
    "return_pct": "pnl_pct",
    "reason": "exit_reason",
}

NUMERIC_COLUMNS = [
    "decision_price",
    "avg_order_price",
    "source_avg_price",
    "shares",
    "capital_before",
    "capital_after",
    "exit_avg_order_price",
    "capital_end",
    "pnl",
    "pnl_pct",
    "slippage_buy",
    "total_slippage_buy",
    "slippage_buy_pct",
    "slippage_sell",
    "total_slippage_sell",
    "slippage_sell_pct",
    "cumulative_profit",
    "exit_trigger_price",
    "exit_bid_at_trigger",
    "exit_ask_at_trigger",
    "exit_mid_at_trigger",
    "exit_spread_at_trigger",
    "exit_slippage_vs_bid",
    "exit_latency_ms_trigger_to_submit",
    "exit_latency_ms_submit_to_fill",
]

CANONICAL_ORDER = [
    "symbol",
    "trade_id",
    "source_file",
    "source_run_id",
    "source_period_start",
    "source_period_end",
    "entry_time",
    "decision_price",
    "avg_order_price",
    "shares",
    "capital_before",
    "capital_after",
    "exit_time",
    "exit_avg_order_price",
    "capital_end",
    "pnl",
    "pnl_pct",
    "RSI_entry",
    "rsi_available",
    "exit_reason",
]


@dataclass(frozen=True)
class SourceMeta:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and validate new Excel trade logs before RSI data arrives."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def source_meta(path: Path) -> SourceMeta:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected trade-log filename: {path.name}")
    return SourceMeta(
        symbol=match.group("symbol"),
        start=pd.Timestamp(match.group("start")),
        end=pd.Timestamp(match.group("end")),
        run_id=match.group("run_id"),
    )


def parse_dayfirst(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), dayfirst=True, errors="coerce")


def load_trade_log(path: Path) -> pd.DataFrame:
    meta = source_meta(path)
    raw = pd.read_excel(path, sheet_name=0, dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns=RENAME_MAP).copy()

    df["entry_time"] = parse_dayfirst(df["entry_time"])
    df["exit_time"] = parse_dayfirst(df["exit_time"])
    if "decision_price" not in df.columns:
        df["decision_price"] = df["avg_order_price"]
    df["symbol"] = meta.symbol
    df["source_file"] = path.name
    df["source_run_id"] = meta.run_id
    df["source_period_start"] = meta.start
    df["source_period_end"] = meta.end

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "RSI_entry" not in df.columns:
        df["RSI_entry"] = np.nan
    df["rsi_available"] = df["RSI_entry"].notna()

    df = df.sort_values(["symbol", "entry_time", "exit_time"]).reset_index(drop=True)
    return df


def load_all(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx trade logs found in {input_dir}")
    frames = [load_trade_log(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "entry_time", "exit_time"]).reset_index(drop=True)
    combined["trade_id"] = combined.groupby("symbol").cumcount() + 1

    ordered = [col for col in CANONICAL_ORDER if col in combined.columns]
    remaining = [col for col in combined.columns if col not in ordered]
    return combined[ordered + remaining]


def check_rows(name: str, severity: str, failing_count: int, total: int, detail: str) -> dict:
    return {
        "check": name,
        "severity": severity,
        "status": "PASS" if failing_count == 0 else "FAIL",
        "failing_count": int(failing_count),
        "total_rows": int(total),
        "detail": detail,
    }


def self_overlap_count(df: pd.DataFrame) -> int:
    count = 0
    for _, group in df.sort_values("entry_time").groupby("symbol"):
        prev_exit = None
        for row in group.itertuples(index=False):
            if prev_exit is not None and row.entry_time < prev_exit:
                count += 1
            prev_exit = row.exit_time
    return count


def cross_symbol_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    tqqq = df[df["symbol"] == "TQQQ"][["trade_id", "entry_time", "exit_time"]].copy()
    sqqq = df[df["symbol"] == "SQQQ"][["trade_id", "entry_time", "exit_time"]].copy()
    rows = []
    for t in tqqq.itertuples(index=False):
        candidates = sqqq[(sqqq["entry_time"] <= t.exit_time) & (sqqq["exit_time"] >= t.entry_time)]
        for s in candidates.itertuples(index=False):
            overlap_start = max(t.entry_time, s.entry_time)
            overlap_end = min(t.exit_time, s.exit_time)
            seconds = (overlap_end - overlap_start).total_seconds()
            if seconds > 0:
                rows.append(
                    {
                        "tqqq_trade_id": t.trade_id,
                        "tqqq_entry": t.entry_time,
                        "tqqq_exit": t.exit_time,
                        "sqqq_trade_id": s.trade_id,
                        "sqqq_entry": s.entry_time,
                        "sqqq_exit": s.exit_time,
                        "overlap_seconds": seconds,
                    }
                )
    return pd.DataFrame(rows)


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    checks = []
    checks.append(
        check_rows(
            "RSI_entry availability",
            "BLOCKER_FOR_RSI_OVERLAY",
            int(df["RSI_entry"].isna().sum()),
            total,
            "Expected to fail for current files; sleeve gates cannot be evaluated until RSI_entry is supplied.",
        )
    )
    for col in [
        "entry_time",
        "exit_time",
        "avg_order_price",
        "exit_avg_order_price",
        "capital_before",
        "capital_end",
        "pnl",
        "pnl_pct",
    ]:
        checks.append(
            check_rows(
                f"{col} non-null",
                "ERROR",
                int(df[col].isna().sum()),
                total,
                "Required for canonical trade replay and summary analysis.",
            )
        )

    checks.append(
        check_rows(
            "exit_time >= entry_time",
            "ERROR",
            int((df["exit_time"] < df["entry_time"]).sum()),
            total,
            "Negative holding periods indicate date parsing or source-data issues.",
        )
    )

    weekend_entry = df["entry_time"].dt.dayofweek >= 5
    weekend_exit = df["exit_time"].dt.dayofweek >= 5
    checks.append(
        check_rows(
            "weekday timestamps",
            "WARN",
            int((weekend_entry | weekend_exit).sum()),
            total,
            "Weekend timestamps can reflect parsing errors or non-market artifacts.",
        )
    )

    bar_minutes = {0, 15, 30, 45}
    off_bar = ~df["entry_time"].dt.minute.isin(bar_minutes) | ~df["exit_time"].dt.minute.isin(bar_minutes)
    checks.append(
        check_rows(
            "15-minute bar alignment",
            "WARN",
            int(off_bar.sum()),
            total,
            "Existing strategy works on 15-minute bars; off-grid timestamps need review.",
        )
    )

    recomputed = (df["exit_avg_order_price"] / df["avg_order_price"] - 1.0) * 100.0
    return_diff = (recomputed - df["pnl_pct"]).abs()
    return_tolerance = 0.10  # percentage points, i.e. 10 bps
    checks.append(
        check_rows(
            "pnl_pct matches exit/entry fills",
            "WARN",
            int((return_diff > return_tolerance).sum()),
            total,
            f"Tolerance is 10 bps in percentage-return units. Max diff={return_diff.max():.6f}.",
        )
    )

    duplicate_keys = df.duplicated(["symbol", "entry_time", "exit_time", "avg_order_price", "exit_avg_order_price"])
    checks.append(
        check_rows(
            "duplicate trade keys",
            "WARN",
            int(duplicate_keys.sum()),
            total,
            "Duplicate keys may be real repeated trades, but should be inspected before canonical use.",
        )
    )

    checks.append(
        check_rows(
            "self-overlap within symbol",
            "WARN",
            self_overlap_count(df),
            total,
            "A non-zero value requires capital allocation logic that supports concurrent same-symbol trades.",
        )
    )
    return pd.DataFrame(checks)


def cagr(start_value: float, end_value: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if start_value <= 0 or end_value <= 0 or pd.isna(start) or pd.isna(end):
        return math.nan
    years = (end - start).total_seconds() / (365.25 * 86400)
    if years <= 0:
        return math.nan
    return (end_value / start_value) ** (1.0 / years) - 1.0


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, source_file), group in df.sort_values("entry_time").groupby(["symbol", "source_file"]):
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "symbol": symbol,
                "source_file": source_file,
                "n_trades": len(group),
                "entry_start": group["entry_time"].min(),
                "exit_end": group["exit_time"].max(),
                "starting_capital": first["capital_before"],
                "ending_capital": last["capital_end"],
                "cagr_from_source_capital": cagr(
                    first["capital_before"], last["capital_end"], group["entry_time"].min(), group["exit_time"].max()
                ),
                "mean_pnl_pct": group["pnl_pct"].mean(),
                "median_pnl_pct": group["pnl_pct"].median(),
                "win_rate": float((group["pnl"] > 0).mean()),
                "min_pnl_pct": group["pnl_pct"].min(),
                "max_pnl_pct": group["pnl_pct"].max(),
                "median_hold_days": (
                    group["exit_time"].sub(group["entry_time"]).dt.total_seconds().median() / 86400.0
                ),
                "p90_hold_days": (
                    group["exit_time"].sub(group["entry_time"]).dt.total_seconds().quantile(0.90) / 86400.0
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    view = df[list(columns)].copy()
    return view.to_markdown(index=False)


def write_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    checks: pd.DataFrame,
    overlaps: pd.DataFrame,
    output_dir: Path,
) -> None:
    rsi_missing = int(df["RSI_entry"].isna().sum())
    report = []
    report.append("# New Trade Data Readiness Report\n")
    report.append("## Status\n")
    report.append(
        f"- Loaded **{len(df):,}** trades from **{df['source_file'].nunique()}** Excel files.\n"
    )
    report.append(
        f"- `RSI_entry` is missing for **{rsi_missing:,} / {len(df):,}** trades, so RSI-gated sleeve tests remain blocked.\n"
    )
    report.append(
        "- Non-RSI preparation is complete: dates, schema normalization, trade summaries, and quality checks are available.\n"
    )
    report.append("\n## Source Summary\n\n")
    report.append(
        markdown_table(
            summary,
            [
                "symbol",
                "source_file",
                "n_trades",
                "entry_start",
                "exit_end",
                "starting_capital",
                "ending_capital",
                "mean_pnl_pct",
                "win_rate",
            ],
        )
    )
    report.append("\n\n## Quality Checks\n\n")
    report.append(markdown_table(checks, ["check", "severity", "status", "failing_count", "detail"]))
    report.append("\n\n## Cross-symbol Overlap\n\n")
    if overlaps.empty:
        report.append("- No TQQQ/SQQQ overlaps detected.\n")
    else:
        t_overlap = overlaps["tqqq_trade_id"].nunique()
        s_overlap = overlaps["sqqq_trade_id"].nunique()
        report.append(f"- Overlapping TQQQ/SQQQ trade pairs: **{len(overlaps):,}**\n")
        report.append(f"- TQQQ trades with at least one overlap: **{t_overlap:,}**\n")
        report.append(f"- SQQQ trades with at least one overlap: **{s_overlap:,}**\n")
        report.append(
            f"- Total overlap time: **{overlaps['overlap_seconds'].sum() / 3600.0:,.1f} hours**\n"
        )
    report.append("\n## Waiting On RSI\n\n")
    report.append("Once RSI columns arrive, the remaining work is mechanical:\n")
    report.append("1. Join or append `RSI_entry` to `canonical_trades_no_rsi.csv` by symbol and timestamp.\n")
    report.append("2. Re-run this prep script and confirm the RSI availability check passes.\n")
    report.append("3. Re-run the existing sleeve sweeps on the longer 2013-2026 history.\n")
    report.append("4. Add SQQQ standalone and combined-portfolio allocation tests.\n")
    (output_dir / "readiness_report.md").write_text("".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trades = load_all(args.input_dir)
    summary = summarize(trades)
    checks = quality_checks(trades)
    overlaps = cross_symbol_overlaps(trades)

    trades.to_csv(args.output_dir / "canonical_trades_no_rsi.csv", index=False)
    summary.to_csv(args.output_dir / "source_summary.csv", index=False)
    checks.to_csv(args.output_dir / "quality_checks.csv", index=False)
    overlaps.to_csv(args.output_dir / "cross_symbol_overlaps.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "n_files": int(trades["source_file"].nunique()),
                "n_trades": int(len(trades)),
                "rsi_available_rows": int(trades["rsi_available"].sum()),
                "rsi_missing_rows": int(trades["RSI_entry"].isna().sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(trades, summary, checks, overlaps, args.output_dir)

    print(f"Prepared {len(trades):,} trades from {trades['source_file'].nunique()} files.")
    print(f"Outputs written to {args.output_dir}")
    print("RSI overlay status: BLOCKED until RSI_entry is supplied.")


if __name__ == "__main__":
    main()
