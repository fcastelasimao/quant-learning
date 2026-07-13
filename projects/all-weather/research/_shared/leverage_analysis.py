"""
leverage_analysis.py
====================
Pure-data helpers for the leverage-comparison notebook.

No matplotlib, no marimo imports — every function receives plain DataFrames
and returns DataFrames or scalar values.
"""

from __future__ import annotations

from pathlib import Path
import json

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
    "default_30_50_20": "Default 30/50, +20% (control only)",
    "robust_calmar_region": "Robust Calmar Region",
    "best_maxdd_preservation": "Best Drawdown Preservation",
    "best_calmar": "Best Calmar",
    "best_cagr_with_maxdd_guard": "Best CAGR With MaxDD Guard",
    "simple_stable_region": "Simple Stable Region",
    "base": "Base",
}
CONTROL_CANDIDATES: set[str] = {
    "default_30_50_20",
    "SPY+GLD default 20% total cap",
    "Default 30/50, +20% (control only)",
}

BROKER_LIMIT_PROFILES: dict[str, dict[str, float | bool | str]] = {
    "IBKR Safe": {
        "max_cap_pct": 30.0,
        "max_sleeve_pct": 30.0,
        "unrestricted": False,
        "description": "IBKR conservative rules-based proxy: cap and sleeves <= 30%.",
    },
    "Strict Pilot": {
        "max_cap_pct": 20.0,
        "max_sleeve_pct": 20.0,
        "unrestricted": False,
        "description": "First live/paper pilot proxy: cap and sleeves <= 20%.",
    },
    "Research Unrestricted": {
        "max_cap_pct": float("inf"),
        "max_sleeve_pct": float("inf"),
        "unrestricted": True,
        "description": "Show every research row; tag >30% as aggressive.",
    },
}


def latest_bundle(root: Path) -> str:
    """Return the path of the most recently modified valid result bundle under root."""
    if not root.exists():
        return ""
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    return str(max(dirs, key=lambda p: p.stat().st_mtime)) if dirs else ""


def latest_bundle_for_profile(root: Path, profile: str) -> str:
    """Return the newest result bundle matching the selected broker profile when possible."""
    if not root.exists():
        return ""
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if not dirs:
        return ""
    limits = broker_profile_limits(profile)
    if bool(limits["unrestricted"]):
        return str(max(dirs, key=lambda p: p.stat().st_mtime))
    target_cap = float(limits["max_cap_pct"]) / 100.0
    target_sleeve = float(limits["max_sleeve_pct"]) / 100.0
    matches = []
    for path in dirs:
        try:
            manifest = json.loads((path / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        constraints = manifest.get("broker_constraints") or {}
        cap = constraints.get("max_global_cap")
        sleeve = constraints.get("max_sleeve_weight")
        if cap is None or sleeve is None:
            continue
        if abs(float(cap) - target_cap) <= 1e-9 and abs(float(sleeve) - target_sleeve) <= 1e-9:
            matches.append(path)
    return str(max(matches or dirs, key=lambda p: p.stat().st_mtime))


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


def is_control_candidate(name: object) -> bool:
    text = str(name)
    return text in CONTROL_CANDIDATES or "default 30/50" in text.lower()


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


def broker_profile_options() -> list[str]:
    """Return broker-limit profile names in UI display order."""
    return list(BROKER_LIMIT_PROFILES)


def broker_profile_limits(profile: str) -> dict[str, float | bool | str]:
    """Return the configured broker-limit profile, defaulting to IBKR Safe."""
    return BROKER_LIMIT_PROFILES.get(profile, BROKER_LIMIT_PROFILES["IBKR Safe"])


def max_allowed_leverage_pct(values: list[float], profile: str) -> float:
    """Return the largest available leverage not above the broker profile cap."""
    if not values:
        return 20.0
    clean = sorted(float(v) for v in values)
    limit = float(broker_profile_limits(profile)["max_cap_pct"])
    if np.isinf(limit):
        return 20.0 if 20.0 in clean else clean[0]
    allowed = [v for v in clean if v <= limit + 1e-9]
    if not allowed:
        return clean[0]
    preferred = 20.0 if limit <= 20.0 + 1e-9 else 30.0
    allowed_preferred = [v for v in allowed if v <= preferred + 1e-9]
    return max(allowed_preferred) if allowed_preferred else max(allowed)


def _pct_from_decimal_or_pct(series: pd.Series, pct_name: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values if pct_name else values * 100.0


def broker_limit_columns(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Return leverage/cap columns as (name, already_pct, limit_kind)."""
    specs = []
    for col in ["Overlay Weight (%)", "Max Overlay Weight (%)"]:
        if col in df:
            specs.append((col, True, "sleeve"))
    for col in ["Overlay Weight", "SPY Weight", "GLD Weight"]:
        if col in df:
            specs.append((col, False, "sleeve"))
    if "Global Cap" in df:
        specs.append(("Global Cap", False, "cap"))
    return specs


def tag_broker_limits(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Annotate rows with broker-limit status without dropping anything."""
    out = df.copy()
    limits = broker_profile_limits(profile)
    max_cap = float(limits["max_cap_pct"])
    max_sleeve = float(limits["max_sleeve_pct"])
    cols = broker_limit_columns(out)

    if out.empty:
        out["Broker Profile"] = profile
        out["Broker Allowed"] = pd.Series(dtype=bool)
        out["Aggressive Research"] = pd.Series(dtype=bool)
        out["Promotion Tier"] = pd.Series(dtype=object)
        return out

    allowed = pd.Series(True, index=out.index)
    aggressive = pd.Series(False, index=out.index)
    max_row_leverage = pd.Series(0.0, index=out.index)
    for col, already_pct, kind in cols:
        pct = _pct_from_decimal_or_pct(out[col], already_pct)
        max_row_leverage = pd.Series(
            np.maximum(max_row_leverage.to_numpy(dtype=float), pct.fillna(0.0).to_numpy(dtype=float)),
            index=out.index,
        )
        threshold = max_cap if kind == "cap" else max_sleeve
        if not np.isinf(threshold):
            allowed &= pct.isna() | (pct <= threshold + 1e-9)
        aggressive |= pct.fillna(0.0) > 30.0 + 1e-9

    if "Aggressive Cap >30%" in out:
        flagged_aggressive = out["Aggressive Cap >30%"].astype(str).str.lower().isin(["true", "1", "yes"])
        aggressive |= flagged_aggressive
        if not bool(limits["unrestricted"]):
            allowed &= ~flagged_aggressive

    if bool(limits["unrestricted"]):
        allowed = pd.Series(True, index=out.index)

    out["Broker Profile"] = profile
    out["Max Broker-Relevant Leverage (%)"] = max_row_leverage.round(4)
    out["Broker Allowed"] = allowed.astype(bool)
    out["Aggressive Research"] = aggressive.astype(bool)
    out["Promotion Tier"] = np.where(out["Aggressive Research"], "aggressive research", "broker-safe candidate")
    return out


def filter_broker_limits(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Annotate and filter rows according to the selected broker profile."""
    tagged = tag_broker_limits(df, profile)
    if bool(broker_profile_limits(profile)["unrestricted"]):
        return tagged
    return tagged[tagged["Broker Allowed"]].copy()


def broker_filter_impact(df: pd.DataFrame, profile: str, label: str = "Rows") -> pd.DataFrame:
    """Build a compact before/after summary for the active broker filter."""
    tagged = tag_broker_limits(df, profile)
    shown = int(tagged["Broker Allowed"].sum()) if "Broker Allowed" in tagged else len(tagged)
    hidden = int(len(tagged) - shown)
    limits = broker_profile_limits(profile)
    return pd.DataFrame([{
        "Dataset": label,
        "Broker Profile": profile,
        "Max Cap Allowed (%)": "unrestricted" if np.isinf(float(limits["max_cap_pct"])) else float(limits["max_cap_pct"]),
        "Max Sleeve Allowed (%)": "unrestricted" if np.isinf(float(limits["max_sleeve_pct"])) else float(limits["max_sleeve_pct"]),
        "Rows Available": int(len(tagged)),
        "Rows Shown": shown if not bool(limits["unrestricted"]) else int(len(tagged)),
        "Rows Hidden": 0 if bool(limits["unrestricted"]) else hidden,
        "Aggressive Rows": int(tagged.get("Aggressive Research", pd.Series(dtype=bool)).sum()),
    }])


def build_candidate_funnel(
    mixed_pass_fail: pd.DataFrame,
    mixed_sweep_pass_fail: pd.DataFrame,
    mixed_sweep_stability: pd.DataFrame,
    profile: str,
) -> pd.DataFrame:
    """Combine mixed OOS and disciplined sweep summaries into one review table."""
    rows: list[dict[str, object]] = []
    if not mixed_pass_fail.empty:
        for _, row in mixed_pass_fail.iterrows():
            candidate = row.get("Name", "")
            rows.append({
                "Source": f"Mixed OOS {row.get('Source', '')}".strip(),
                "Candidate": candidate,
                "Selector": candidate,
                "Benchmark": row.get("Benchmark", "base"),
                "Control Only": bool(row.get("Control Only", is_control_candidate(candidate))),
                "Global Cap": row.get("Global Cap", np.nan),
                "SPY Rule": row.get("SPY Rule", ""),
                "GLD Rule": row.get("GLD Rule", ""),
                "Splits Passed": row.get("Splits Passed", np.nan),
                "Splits Tested": row.get("Splits Tested", np.nan),
                "Average OOS Calmar": row.get("Average OOS Calmar", np.nan),
                "Average OOS Calmar Delta": row.get("Average OOS Calmar Delta", np.nan),
                "Worst OOS Calmar Delta": row.get("Worst OOS Calmar Delta", np.nan),
                "Worst OOS MaxDD Delta (%)": row.get("Worst OOS MaxDD Delta (%)", np.nan),
                "Average OOS CAGR Delta (%)": row.get("Average OOS CAGR Delta (%)", np.nan),
                "RF Cost Pass Splits": row.get("RF Cost Pass Splits", np.nan),
                "Min OOS Trade Episodes": row.get("Min OOS Trade Episodes", np.nan),
                "Average OOS Exposure (%)": row.get("Average OOS Exposure (%)", np.nan),
                "Low Trade Count Splits": row.get("Low Trade Count Splits", np.nan),
                "MaxDD Breach >3pp": row.get("MaxDD Breach >3pp", False),
                "Overall Pass": row.get("Overall Pass", False),
                "Promotion Tier": "broker-safe candidate",
            })
    if not mixed_sweep_pass_fail.empty:
        stability = (
            mixed_sweep_stability.set_index("Selector")
            if not mixed_sweep_stability.empty and "Selector" in mixed_sweep_stability
            else pd.DataFrame()
        )
        for _, row in mixed_sweep_pass_fail.iterrows():
            selector = row.get("Selector", "")
            stable_row = stability.loc[selector] if not stability.empty and selector in stability.index else {}
            candidate = label_selector(selector)
            rows.append({
                "Source": "Disciplined Sweep",
                "Candidate": candidate,
                "Selector": selector,
                "Benchmark": row.get("Benchmark", "base"),
                "Control Only": bool(row.get("Control Only", is_control_candidate(selector))),
                "Global Cap": row.get("Global Cap", np.nan),
                "SPY Rule": row.get("SPY Rule", ""),
                "GLD Rule": row.get("GLD Rule", ""),
                "Splits Passed": row.get("Structural Splits Passed", np.nan),
                "Splits Tested": row.get("Structural Splits Tested", np.nan),
                "Average OOS Calmar": row.get("Average OOS Calmar", np.nan),
                "Average OOS Calmar Delta": row.get("Average OOS Calmar Delta", np.nan),
                "Worst OOS Calmar Delta": row.get("Worst OOS Calmar Delta", np.nan),
                "Worst OOS MaxDD Delta (%)": row.get("Worst OOS MaxDD Delta (%)", np.nan),
                "Average OOS CAGR Delta (%)": row.get("Average OOS CAGR Delta (%)", np.nan),
                "RF Cost Pass Splits": row.get("RF Cost Pass Splits", np.nan),
                "Min OOS Trade Episodes": row.get("Min OOS Trade Episodes", np.nan),
                "Average OOS Exposure (%)": row.get("Average OOS Exposure (%)", np.nan),
                "Low Trade Count Splits": np.nan,
                "MaxDD Breach >3pp": row.get("Worst OOS MaxDD Delta (%)", 0) < -3.0,
                "Overall Pass": row.get("Overall Pass", False),
                "Annual Calmar Improvement Years": row.get("Annual Calmar Improvement Years", np.nan),
                "Stable Neighborhood Pass": row.get("Stable Neighborhood Pass", np.nan),
                "Most Common Config": stable_row.get("Most Common Config", "") if isinstance(stable_row, pd.Series) else "",
                "Promotion Tier": row.get("Promotion Tier", ""),
                "Aggressive Cap >30%": row.get("Aggressive Cap >30%", False),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[~out["Control Only"].fillna(False).astype(bool)].copy()
    if out.empty:
        return out
    out["Caveat"] = ""
    out.loc[pd.to_numeric(out.get("Low Trade Count Splits"), errors="coerce").fillna(0) > 0, "Caveat"] = "Low trade count"
    breach = (
        out["MaxDD Breach >3pp"].fillna(False).astype(bool)
        if "MaxDD Breach >3pp" in out
        else pd.Series(False, index=out.index)
    )
    out.loc[breach, "Caveat"] = "MaxDD breach"
    out["Broker Profile"] = profile
    aggressive_cap = (
        out["Aggressive Cap >30%"].fillna(False).astype(bool)
        if "Aggressive Cap >30%" in out
        else pd.Series(False, index=out.index)
    )
    out["Broker Allowed"] = ~aggressive_cap
    if bool(broker_profile_limits(profile)["unrestricted"]):
        out["Broker Allowed"] = True
    out = out[out["Broker Allowed"]].copy()
    if out.empty:
        return out
    return out.sort_values(
        ["Overall Pass", "Average OOS Calmar", "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)"],
        ascending=[False, False, False, False],
        na_position="last",
    )


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


def build_is_vs_oos_comparison(
    oos_df: pd.DataFrame,
    name_col: str,
) -> pd.DataFrame:
    """Build a tidy IS-vs-OOS Calmar table for mixed overlay candidates."""
    if oos_df.empty or name_col not in oos_df.columns:
        return pd.DataFrame()
    rows = []
    for _, row in oos_df.iterrows():
        name = str(row[name_col])
        is_calmar = float(row["IS Calmar"])
        oos_calmar = float(row["OOS Overlay Calmar"])
        delta = oos_calmar - is_calmar
        rows.append({
            name_col: name,
            "Label": SELECTOR_LABELS.get(name, name),
            "Split": row["Split"],
            "IS Calmar": round(is_calmar, 4),
            "OOS Calmar": round(oos_calmar, 4),
            "Delta": round(delta, 4),
            "Overfitting Signal": delta < -0.05,
        })
    return pd.DataFrame(rows)
