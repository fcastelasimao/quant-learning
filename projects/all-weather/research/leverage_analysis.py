"""
leverage_analysis.py
====================
Pure-data helpers for the leverage-comparison notebook.

No matplotlib, no marimo imports — every function receives plain DataFrames
and returns DataFrames or scalar values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = "S&P 500 (SPY)"
BASE = "My Strategy (Base)"
PLOT_EXCLUDED: set[str] = {BENCHMARK}
PALETTE = [
    "#58a6ff", "#f78166", "#3fb950", "#f0b429",
    "#d2a8ff", "#ff7b72", "#56d4dd", "#a5d6ff",
]
SELECTOR_LABELS: dict[str, str] = {
    "default_30_50_20": "Default 30/50, +20%",
    "robust_calmar_region": "Robust Calmar Region",
    "best_maxdd_preservation": "Best Drawdown Preservation",
    "best_calmar": "Best Calmar",
    "best_cagr_with_maxdd_guard": "Best CAGR With MaxDD Guard",
    "base": "Base",
}


def latest_bundle(root: Path) -> str:
    """Return the path of the most recently modified valid result bundle under root."""
    if not root.exists():
        return ""
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    return str(max(dirs, key=lambda p: p.stat().st_mtime)) if dirs else ""


def visible_strategies(strategies) -> list[str]:
    """Return strategies in display order, excluding PLOT_EXCLUDED."""
    ordered: list[str] = [BASE] if BASE in strategies else []
    ordered += sorted(s for s in strategies if s not in set(ordered) | PLOT_EXCLUDED)
    return ordered


def default_focus_strategies(strategies) -> list[str]:
    """Return the default pre-selected strategies for the portfolio view."""
    wanted = [BASE, "My Strategy + GLD RSI Overlay", "My Strategy + SPY RSI Overlay",
              "My Strategy + QQQ RSI Overlay"]
    found = [s for s in wanted if s in set(strategies)]
    return found or visible_strategies(strategies)


def portfolio_view_strategies(selected, include_benchmark: bool, available) -> list[str]:
    """Filter selected strategies to those present in available, optionally adding the benchmark."""
    avail = set(available)
    strategies = [s for s in selected if s in avail]
    if include_benchmark and BENCHMARK in avail and BENCHMARK not in strategies:
        strategies.append(BENCHMARK)
    return strategies


def maybe_filter_benchmark(df: pd.DataFrame, include_benchmark: bool) -> pd.DataFrame:
    """Remove BENCHMARK rows from df unless include_benchmark is True."""
    if include_benchmark or df.empty or "Strategy" not in df:
        return df
    return df[~df["Strategy"].isin(PLOT_EXCLUDED)].copy()


def slice_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Filter df to rows where Date is within [start, end]."""
    if df.empty or "Date" not in df:
        return df
    return df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))].copy()


def ticker_from_strategy(strategy: str) -> str:
    """Extract the ETF ticker from an overlay strategy label."""
    if strategy == BASE:
        return "BASE"
    if strategy.startswith("My Strategy + ") and " RSI Overlay" in strategy:
        return strategy.replace("My Strategy + ", "").replace(" RSI Overlay", "")
    return strategy


def label_selector(selector: str) -> str:
    """Return a human-readable label for a selector key."""
    return SELECTOR_LABELS.get(selector, str(selector).replace("_", " ").title())


def fmt_pct(value, digits: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.{digits}f}%"


def fmt_num(value, digits: int = 3) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.{digits}f}"


def fmt_pp(value, digits: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):+.{digits}f} pp"


def strategy_label(strategy: str) -> str:
    """Return a short human-readable label for a strategy string."""
    if strategy == BENCHMARK:
        return "SPY benchmark"
    ticker = ticker_from_strategy(strategy)
    return "Base" if ticker == "BASE" else f"{ticker} overlay"


def presentation_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df restricted to the columns that exist in it."""
    return df[[col for col in cols if col in df.columns]].copy()


def compute_heatmap_pivot(
    data: pd.DataFrame,
    mode: str,
    entry: float,
    exit_: float,
    leverage: float,
    metric: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Build a pivot table for the threshold-grid heatmap and return display labels.

    Returns (pivot, xlabel, ylabel, title_suffix).
    """
    if mode == "Leverage x Exit":
        subset = data[data["Entry Threshold"] == entry]
        pivot = subset.pivot_table(
            index="Overlay Weight (%)", columns="Exit Threshold",
            values=metric, aggfunc="max",
        ).sort_index(ascending=True)
        return pivot, "Exit threshold", "Overlay weight (%)", f"entry {entry:g}"
    elif mode == "Leverage x Entry":
        subset = data[data["Exit Threshold"] == exit_]
        pivot = subset.pivot_table(
            index="Overlay Weight (%)", columns="Entry Threshold",
            values=metric, aggfunc="max",
        ).sort_index(ascending=True)
        return pivot, "Entry threshold", "Overlay weight (%)", f"exit {exit_:g}"
    else:  # Exit x Entry
        subset = data[data["Overlay Weight (%)"] == leverage]
        pivot = subset.pivot_table(
            index="Exit Threshold", columns="Entry Threshold",
            values=metric, aggfunc="max",
        ).sort_index(ascending=True)
        return pivot, "Entry threshold", "Exit threshold", f"leverage {leverage:g}%"


def scale_overlay_leverage(
    daily: pd.DataFrame,
    manifest: dict,
    new_leverage_pct: float,
) -> pd.DataFrame:
    """Return daily with overlay-strategy returns scaled to new_leverage_pct.

    Scales each overlay strategy's excess return (overlay minus base) by
    new_leverage_pct / old_leverage_pct. Entry/exit signal days are unchanged;
    only the size of the exposure is adjusted. When new_leverage_pct equals the
    bundle default the original series is returned unchanged.
    """
    specs = manifest.get("overlay_specs", [])
    if not specs or daily.empty:
        return daily

    wide = daily.pivot(index="Date", columns="Strategy", values="Value")
    base_col = next((c for c in wide.columns if "(Base)" in c), None)
    if base_col is None:
        return daily

    base_ret = wide[base_col].pct_change().fillna(0.0)
    result = wide.copy()

    for spec in specs:
        old_lev_pct = float(spec.get("overlay_weight", 0.20)) * 100.0
        if abs(old_lev_pct) < 1e-9:
            continue
        scale = new_leverage_pct / old_lev_pct
        if abs(scale - 1.0) < 1e-9:
            continue

        ticker = spec.get("ticker", "")
        overlay_col = next((c for c in wide.columns if f"+ {ticker} RSI" in c), None)
        if overlay_col is None:
            continue

        overlay_ret = wide[overlay_col].pct_change().fillna(0.0)
        new_ret = base_ret + scale * (overlay_ret - base_ret)
        initial = wide[overlay_col].iloc[0]
        result[overlay_col] = initial * (1.0 + new_ret).cumprod()

    stacked = result.stack().reset_index()
    stacked.columns = ["Date", "Strategy", "Value"]
    return stacked


def derive_leverage_tables(
    manifest: dict,
    overlay_summary: pd.DataFrame,
    pass_fail: pd.DataFrame,
) -> dict:
    """Compute all derived display tables for the executive-summary section.

    Returns a dict with keys:
        allocation, overlay_specs, default_rank, pass_table, verdict_table, cards
    """
    allocation = pd.DataFrame([
        {"Asset": asset, "Weight": weight, "Weight (%)": f"{weight:.1%}"}
        for asset, weight in manifest["allocation"].items()
    ])
    overlay_specs = pd.DataFrame(manifest["overlay_specs"])

    default_rank = overlay_summary[overlay_summary["Ticker"] != "BASE"].copy()
    default_rank["RF Cost Drag (pp)"] = (
        default_rank["CAGR (%)"] - default_rank["RF Opportunity Cost CAGR (%)"]
    ).round(4)
    default_rank = default_rank.sort_values(["Calmar", "CAGR (%)"], ascending=[False, False])

    best_oos = None
    worst_reject = None

    if pass_fail.empty:
        pass_table = pd.DataFrame()
        verdict_table = pd.DataFrame()
    else:
        pass_table = pass_fail.copy()
        pass_table["Verdict"] = np.where(pass_table["Overall Pass"], "Passed OOS", "Failed OOS")
        pass_table["Caveat"] = ""
        pass_table.loc[pass_table["Low Trade Count Splits"] > 0, "Caveat"] = "Low trade count"
        pass_table.loc[pass_table["MaxDD Breach >3pp"], "Caveat"] = "MaxDD breach"
        pass_table = pass_table.sort_values(
            ["Overall Pass", "Splits Passed", "Worst OOS Calmar Delta"],
            ascending=[False, False, False],
        )
        best_oos = pass_table[pass_table["Overall Pass"]].head(1)
        worst_reject = pass_table[~pass_table["Overall Pass"]].sort_values("Worst OOS Calmar Delta").head(1)

        verdict_rows = []
        for ticker in sorted(pass_table["Ticker"].dropna().unique()):
            ticker_pass = pass_table[(pass_table["Ticker"] == ticker) & (pass_table["Overall Pass"])]
            default_pass = ticker_pass[ticker_pass["Selector"] == "default_30_50_20"]
            if ticker == "GLD" and not default_pass.empty:
                verdict = "Keep Researching"
                reason = "Cleanest OOS result; default rule passes all splits."
            elif ticker in {"SPY", "QQQ"} and not default_pass.empty:
                verdict = "Use as Benchmark"
                reason = "Default passes OOS but adds equity risk."
            elif ticker == "TIP" and not ticker_pass.empty:
                verdict = "Appendix Only"
                reason = "Some robust rows pass, but impact is modest."
            elif not ticker_pass.empty:
                verdict = "Appendix Only"
                reason = "A selector passes, but caveats are material."
            else:
                verdict = "Reject"
                reason = "No reliable OOS pass under current gates."
            verdict_rows.append({
                "Ticker": ticker, "Verdict": verdict, "Reason": reason,
                "Best Splits Passed": int(pass_table[pass_table["Ticker"] == ticker]["Splits Passed"].max()),
            })
        verdict_table = pd.DataFrame(verdict_rows)

    best_default = default_rank.head(1)
    worst_default = default_rank.tail(1)
    rf_drag = default_rank.sort_values("RF Cost Drag (pp)", ascending=False).head(1)
    low_trade_count = int(pass_table["Low Trade Count Splits"].gt(0).sum()) if not pass_table.empty else 0

    cards = pd.DataFrame([
        {
            "Card": "Best default overlay",
            "Conclusion": (
                f"{best_default.iloc[0]['Ticker']} ({fmt_num(best_default.iloc[0]['Calmar'])} Calmar)"
                if not best_default.empty else "n/a"
            ),
        },
        {
            "Card": "Best OOS pass",
            "Conclusion": (
                f"{best_oos.iloc[0]['Ticker']} - {best_oos.iloc[0]['Selector Label']}"
                if best_oos is not None and not best_oos.empty
                else "No OOS bundle loaded"
            ),
        },
        {
            "Card": "Worst rejected overlay",
            "Conclusion": (
                f"{worst_reject.iloc[0]['Ticker']} - {worst_reject.iloc[0]['Selector Label']}"
                if worst_reject is not None and not worst_reject.empty
                else f"{worst_default.iloc[0]['Ticker']} default"
            ),
        },
        {
            "Card": "Largest RF opportunity-cost drag",
            "Conclusion": (
                f"{rf_drag.iloc[0]['Ticker']} ({fmt_pp(-rf_drag.iloc[0]['RF Cost Drag (pp)'])})"
                if not rf_drag.empty else "n/a"
            ),
        },
        {
            "Card": "Low-trade-count warning",
            "Conclusion": f"{low_trade_count} passing/review rows carry a low-trade caveat",
        },
    ])

    return {
        "allocation": allocation,
        "overlay_specs": overlay_specs,
        "default_rank": default_rank,
        "pass_table": pass_table,
        "verdict_table": verdict_table,
        "cards": cards,
    }
