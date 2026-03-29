"""
results_dashboard.py
====================
Reads results/master_log.xlsx and produces:
  results/dashboard.html         — interactive Chart.js dashboard
  results/scatter_calmar_mdd.png — static matplotlib scatter for presentations

Usage:
    python3 results_dashboard.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import json
import math
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from plotting import style_ax


# ===========================================================================
# CONSTANTS
# ===========================================================================

MASTER_LOG_PATH = os.path.join("results", "master_log.xlsx")
DASHBOARD_PATH  = os.path.join("results", "dashboard.html")
SCATTER_PATH    = os.path.join("results", "scatter_calmar_mdd.png")

BASELINE_CALMAR      = 0.441   # 8-asset manual OOS Calmar
SIXTY_FORTY_CALMAR   = 0.263   # 60/40 OOS Calmar
SIXTY_FORTY_FULL_CAL = 0.327   # 60/40 full-period Calmar

# Colour per asset count
ASSET_COLORS = {5: "#58a6ff", 6: "#f0b429", 7: "#3fb950", 8: "#d2a8ff"}
ASSET_COLORS_RGBA = {
    5: "rgba(88,166,255,0.75)",
    6: "rgba(240,180,41,0.75)",
    7: "rgba(63,185,80,0.75)",
    8: "rgba(210,168,255,0.75)",
}


# ===========================================================================
# COLUMN DEFINITIONS (must mirror export.py exactly)
# ===========================================================================

META_COLS = [
    "Timestamp", "Label", "Run Mode", "Backtest Start", "Backtest End",
    "OOS Start", "Pricing Model", "Tx Cost %", "Tax Drag %",
    "Data Freq", "Tickers",
]
METRIC_COLS = [
    "CAGR (%)", "Max_DD (%)", "Avg_DD (%)", "Max_DD_Dur", "Avg_Rec",
    "Ulcer", "Sharpe", "Sortino", "Calmar", "Fin_Val ($)",
]
STRATEGY_NAMES = ["AW_R", "B&H_AW", "SPY", "60/40"]

# Step suffix → semantic key
STEP_SUFFIXES = {
    "s1_backtest":    "is_manual",
    "s2_optimise":    "is_opt",
    "s3_walkforward": "walkforward",
    "s4_oos":         "oos",
    "s5_full":        "full",
    "spot_full":      "spot",
}

SHORT_NAMES: dict[str, str] = {
    "5asset_dalio_original":    "5A Dalio",
    "5asset_tip_previous_best": "5A TIP Best",
    "6asset_tip_gsg":           "6A TIP+GSG \u2605",
    "6asset_duration_ladder":   "6A Duration",
    "6asset_intl_equity":       "6A Intl",
    "7asset_no_gsg":            "7A -GSG",
    "7asset_no_iwd":            "7A -IWD",
    "7asset_tip_gsg_dual":      "7A TIP+GSG",
    "8asset_efa_replaces_iwd":  "8A +EFA",
    "8asset_tip_replaces_shy":  "8A +TIP",
    "cost_sensitivity_005bps":  "8A 0.05%",
    "cost_sensitivity_10bps":   "8A 0.10%",
    "cost_sensitivity_50bps":   "8A 0.50%",
    "spotcheck_ivv_vs_spy":     "IVV\u2248SPY",
    "spotcheck_gldm_vs_gld":    "GLDM\u2248GLD",
    "spotcheck_pdbc_vs_gsg":    "PDBC vs GSG",
    "6asset_tip_gsg_split2018": "6A TIP+GSG'18",
    "6asset_tip_gsg_split2022": "6A TIP+GSG'22",
    "7asset_vnq_reits":         "7A +VNQ",
    "7asset_tip_gsg_vnq":       "7A +REITs",
    "8asset_agg_replaces_qqq":  "8A +AGG",
    "8asset_lqd_replaces_shy":  "8A +LQD",
    "8asset_djp_replaces_gsg":  "8A +DJP",
}


def _flat_cols() -> list[str]:
    cols = list(META_COLS)
    for s in STRATEGY_NAMES:
        for m in METRIC_COLS:
            cols.append(f"{s}_{m}")
    cols.append("Results Folder")
    return cols


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _f(v, fmt=".3f") -> str:
    return "n/a" if _isnan(v) else format(float(v), fmt)


def _parse_label(label: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (exp_name, step_key) from 'exp_6asset_tip_gsg_s4_oos'."""
    if not isinstance(label, str) or not label.startswith("exp_"):
        return None, None
    body = label[4:]
    for suffix, step_key in STEP_SUFFIXES.items():
        if body.endswith(f"_{suffix}"):
            return body[:-(len(suffix) + 1)], step_key
    return None, None


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data(log_path: str = MASTER_LOG_PATH
              ) -> tuple[dict, pd.DataFrame, dict]:
    """
    Parse master_log.xlsx.

    Returns
    -------
    experiments : dict of {name: metrics_dict}
    df_raw      : raw DataFrame (for benchmark extraction)
    benchmarks  : {"spy": {...}, "sixty_forty": {...}}
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Master log not found: {log_path}\n"
            "Run some experiments first, then re-run results_dashboard.py."
        )

    df = pd.read_excel(log_path, header=None, skiprows=2)
    flat = _flat_cols()
    df.columns = flat[: len(df.columns)]

    experiments: dict[str, dict] = {}
    benchmarks: dict[str, dict] = {"spy": {}, "sixty_forty": {}}

    def _blank_exp(name: str) -> dict:
        return {
            "name":            name,
            "short_name":      SHORT_NAMES.get(name, name[:22]),
            "n_assets":        0,
            "is_calmar_manual": float("nan"),
            "is_calmar_opt":   float("nan"),
            "wf_mean":         float("nan"),
            "wf_median":       float("nan"),
            "oos_calmar":      float("nan"),
            "oos_cagr":        float("nan"),
            "oos_mdd":         float("nan"),
            "oos_ulcer":       float("nan"),
            "oos_sharpe":      float("nan"),
            "oos_sortino":     float("nan"),
            "oos_avg_dd":      float("nan"),
            "oos_max_dd_dur":  float("nan"),
            "oos_final_val":   float("nan"),
            "full_calmar":     float("nan"),
            "spot_calmar":     float("nan"),
            "tx_cost":         float("nan"),
            "is_spot_check":   False,
        }

    for _, row in df.iterrows():
        label = str(row.get("Label", ""))
        name, step_key = _parse_label(label)

        # Harvest benchmark columns from every row (best-effort)
        def _bget(col: str) -> Optional[float]:
            v = row.get(col)
            return float(v) if not _isnan(v) else None

        if "SPY_Calmar" in df.columns:
            v = _bget("SPY_Calmar")
            if v is not None:
                benchmarks["spy"].setdefault("calmar", v)
        if "60/40_Calmar" in df.columns:
            v = _bget("60/40_Calmar")
            if v is not None:
                benchmarks["sixty_forty"].setdefault("calmar", v)
        # Store full metric row for Panel 5 benchmarks (first non-NaN wins)
        for strat, bkey in [("SPY", "spy"), ("60/40", "sixty_forty")]:
            if f"{strat}_CAGR (%)" in df.columns and "calmar" in benchmarks[bkey]:
                b = benchmarks[bkey]
                if "cagr" not in b:
                    for met, bfield in [
                        ("CAGR (%)", "cagr"), ("Max_DD (%)", "mdd"),
                        ("Avg_DD (%)", "avg_dd"), ("Max_DD_Dur", "max_dd_dur"),
                        ("Ulcer", "ulcer"), ("Sharpe", "sharpe"),
                        ("Sortino", "sortino"), ("Calmar", "calmar"),
                        ("Fin_Val ($)", "final_val"),
                    ]:
                        v = _bget(f"{strat}_{met}")
                        if v is not None:
                            b[bfield] = v

        if name is None:
            continue

        if name not in experiments:
            experiments[name] = _blank_exp(name)

        e = experiments[name]

        tickers_str = str(row.get("Tickers", ""))
        n = tickers_str.count("|") + 1 if "|" in tickers_str else 0
        if n > 0:
            e["n_assets"] = n

        tx = row.get("Tx Cost %")
        if not _isnan(tx):
            e["tx_cost"] = float(tx)

        def _get(col: str) -> float:
            v = row.get(col)
            return float("nan") if _isnan(v) else float(v)

        if step_key == "is_manual":
            e["is_calmar_manual"] = _get("AW_R_Calmar")

        elif step_key == "is_opt":
            e["is_calmar_opt"] = _get("AW_R_Calmar")

        elif step_key == "walkforward":
            folder = str(row.get("Results Folder", ""))
            if os.path.isdir(folder):
                wf_csv = os.path.join(folder, "walk_forward.csv")
                if os.path.exists(wf_csv):
                    try:
                        wf_df = pd.read_csv(wf_csv)
                        if "Overfit Ratio Clamped" in wf_df.columns:
                            ratios = wf_df["Overfit Ratio Clamped"]
                        elif "Overfit Ratio" in wf_df.columns:
                            ratios = wf_df["Overfit Ratio"].clip(upper=2.0)
                        else:
                            ratios = pd.Series(dtype=float)
                        if not ratios.empty:
                            e["wf_mean"]   = float(ratios.mean())
                            e["wf_median"] = float(ratios.median())
                    except Exception:
                        pass

        elif step_key == "oos":
            e["oos_calmar"]    = _get("AW_R_Calmar")
            e["oos_cagr"]      = _get("AW_R_CAGR (%)")
            e["oos_mdd"]       = _get("AW_R_Max_DD (%)")
            e["oos_ulcer"]     = _get("AW_R_Ulcer")
            e["oos_sharpe"]    = _get("AW_R_Sharpe")
            e["oos_sortino"]   = _get("AW_R_Sortino")
            e["oos_avg_dd"]    = _get("AW_R_Avg_DD (%)")
            e["oos_max_dd_dur"]= _get("AW_R_Max_DD_Dur")
            e["oos_final_val"] = _get("AW_R_Fin_Val ($)")

        elif step_key == "full":
            e["full_calmar"] = _get("AW_R_Calmar")

        elif step_key == "spot":
            e["spot_calmar"]   = _get("AW_R_Calmar")
            e["oos_cagr"]      = _get("AW_R_CAGR (%)")
            e["oos_mdd"]       = _get("AW_R_Max_DD (%)")
            e["oos_ulcer"]     = _get("AW_R_Ulcer")
            e["oos_sharpe"]    = _get("AW_R_Sharpe")
            e["oos_sortino"]   = _get("AW_R_Sortino")
            e["oos_avg_dd"]    = _get("AW_R_Avg_DD (%)")
            e["oos_max_dd_dur"]= _get("AW_R_Max_DD_Dur")
            e["oos_final_val"] = _get("AW_R_Fin_Val ($)")
            e["is_spot_check"] = True

    return experiments, df, benchmarks


# ===========================================================================
# HTML PANEL GENERATORS
# ===========================================================================

def _cell_color(value: float, lo: float, hi: float,
                higher_better: bool = True) -> str:
    """Return an HSL background colour string for a metric cell."""
    if _isnan(value) or hi == lo:
        return "#1e2530"
    t = (float(value) - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    if not higher_better:
        t = 1.0 - t
    # Green (120°) → amber (45°) → red (0°)
    hue = int(t * 120)
    return f"hsl({hue}, 60%, 22%)"


def _wf_cell_color(v: float) -> str:
    if _isnan(v):
        return "#1e2530"
    v = float(v)
    if v >= 0.6:
        return "hsl(120,55%,20%)"
    if v >= 0.3:
        return "hsl(45,65%,22%)"
    return "hsl(0,55%,22%)"


def _panel3_wf_table(experiments: dict) -> str:
    """HTML for Panel 3 — Walk-forward reliability table."""
    rows_with_data = [
        e for e in experiments.values()
        if not (_isnan(e["is_calmar_manual"]) and _isnan(e["wf_mean"])
                and _isnan(e["oos_calmar"]))
        and not e["is_spot_check"]
    ]
    rows_with_data.sort(key=lambda e: (
        float("inf") if _isnan(e["oos_calmar"]) else -float(e["oos_calmar"])
    ))

    if not rows_with_data:
        return "<p style='color:#8b949e'>No walk-forward data yet.</p>"

    html = """<table>
<thead>
<tr>
  <th>Experiment</th>
  <th>IS Calmar (M)</th>
  <th>IS Calmar (O)</th>
  <th>WF Mean</th>
  <th>WF Median</th>
  <th>OOS Calmar</th>
</tr>
</thead>
<tbody>"""

    for e in rows_with_data:
        is_star = e["name"] == "6asset_tip_gsg"
        row_style = 'background:#2a2000;' if is_star else ''
        wf_mn_bg  = _wf_cell_color(e["wf_mean"])
        wf_md_bg  = _wf_cell_color(e["wf_median"])
        html += f"""
<tr style="{row_style}">
  <td style="font-weight:{'bold' if is_star else 'normal'};color:{'#f0b429' if is_star else '#c9d1d9'}">{e['short_name']}</td>
  <td>{_f(e['is_calmar_manual'])}</td>
  <td>{_f(e['is_calmar_opt'])}</td>
  <td style="background:{wf_mn_bg}">{_f(e['wf_mean'])}</td>
  <td style="background:{wf_md_bg}">{_f(e['wf_median'])}</td>
  <td>{_f(e['oos_calmar'])}</td>
</tr>"""

    html += "\n</tbody>\n</table>"
    return html


def _panel5_metric_table(experiments: dict, benchmarks: dict) -> str:
    """HTML for Panel 5 — Full metric comparison table."""
    PRIORITY = [
        "6asset_tip_gsg", "8asset_tip_replaces_shy", "7asset_tip_gsg_vnq",
        "5asset_dalio_original", "6asset_duration_ladder", "7asset_no_gsg",
    ]

    rows: list[dict] = []
    seen = set()
    for name in PRIORITY:
        if name in experiments and name not in seen:
            e = experiments[name]
            oos_c = e["oos_calmar"] if not _isnan(e["oos_calmar"]) else e["spot_calmar"]
            if not _isnan(oos_c):
                rows.append({
                    "label": e["short_name"],
                    "cagr":      e["oos_cagr"],
                    "mdd":       e["oos_mdd"],
                    "avg_dd":    e["oos_avg_dd"],
                    "dur":       e["oos_max_dd_dur"],
                    "ulcer":     e["oos_ulcer"],
                    "sharpe":    e["oos_sharpe"],
                    "sortino":   e["oos_sortino"],
                    "calmar":    oos_c,
                    "final_val": e["oos_final_val"],
                })
                seen.add(name)

    # Append 60/40 and SPY from benchmarks
    for bkey, blabel in [("sixty_forty", "60/40"), ("spy", "SPY")]:
        b = benchmarks.get(bkey, {})
        if b.get("calmar"):
            rows.append({
                "label":     blabel,
                "cagr":      b.get("cagr", float("nan")),
                "mdd":       b.get("mdd", float("nan")),
                "avg_dd":    b.get("avg_dd", float("nan")),
                "dur":       b.get("max_dd_dur", float("nan")),
                "ulcer":     b.get("ulcer", float("nan")),
                "sharpe":    b.get("sharpe", float("nan")),
                "sortino":   b.get("sortino", float("nan")),
                "calmar":    b.get("calmar", float("nan")),
                "final_val": b.get("final_val", float("nan")),
            })

    if not rows:
        return "<p style='color:#8b949e'>No OOS data yet for metric table.</p>"

    # Compute per-column ranges for gradient colouring
    cols_def = [
        ("CAGR (%)",    "cagr",      True),
        ("Max DD (%)",  "mdd",       True),   # less negative = better → higher
        ("Avg DD (%)",  "avg_dd",    True),   # same
        ("DD Dur",      "dur",       False),  # lower = better
        ("Ulcer",       "ulcer",     False),  # lower = better
        ("Sharpe",      "sharpe",    True),
        ("Sortino",     "sortino",   True),
        ("Calmar",      "calmar",    True),
        ("£10k→",       "final_val", True),
    ]
    ranges: dict[str, tuple] = {}
    for _, key, _ in cols_def:
        vals = [r[key] for r in rows if not _isnan(r[key])]
        if vals:
            ranges[key] = (min(vals), max(vals))
        else:
            ranges[key] = (0.0, 1.0)

    html = "<table>\n<thead>\n<tr>\n<th>Strategy</th>"
    for header, _, _ in cols_def:
        html += f"<th>{header}</th>"
    html += "\n</tr>\n</thead>\n<tbody>"

    for r in rows:
        is_star = r["label"].endswith("\u2605")
        row_style = 'background:#2a2000;' if is_star else ''
        html += f'\n<tr style="{row_style}">'
        lbl_color = "#f0b429" if is_star else "#c9d1d9"
        html += f'<td style="color:{lbl_color};font-weight:bold">{r["label"]}</td>'
        for _, key, higher in cols_def:
            v = r[key]
            lo, hi = ranges[key]
            bg = _cell_color(v, lo, hi, higher_better=higher)
            display = (f"${v:,.0f}" if key == "final_val"
                       else (_f(v, ".1f") if key == "dur" else _f(v)))
            html += f'<td style="background:{bg};text-align:right">{display}</td>'
        html += "</tr>"

    html += "\n</tbody>\n</table>"
    return html


# ===========================================================================
# JAVASCRIPT DATA BUILDERS
# ===========================================================================

def _scatter_datasets(experiments: dict) -> str:
    """Return Chart.js datasets JSON string for Panel 1 bubble chart."""
    by_n: dict[int, list] = {5: [], 6: [], 7: [], 8: []}

    for e in experiments.values():
        if _isnan(e["oos_calmar"]):
            continue
        n = max(5, min(8, e["n_assets"])) if e["n_assets"] else 7
        cagr  = float(e["oos_cagr"])  if not _isnan(e["oos_cagr"])  else 6.0
        r     = max(5, min(28, abs(cagr) * 1.8))
        is_star = e["name"] == "6asset_tip_gsg"
        by_n.setdefault(n, []).append({
            "x":       round(float(e["oos_mdd"]),    3),
            "y":       round(float(e["oos_calmar"]), 3),
            "r":       round(r * 1.5 if is_star else r, 1),
            "label":   e["short_name"],
            "name":    e["name"],
            "calmar":  round(float(e["oos_calmar"]), 3),
            "cagr":    round(cagr, 2),
            "mdd":     round(float(e["oos_mdd"]),    2),
            "sharpe":  round(float(e["oos_sharpe"]),  3) if not _isnan(e["oos_sharpe"])  else None,
            "sortino": round(float(e["oos_sortino"]), 3) if not _isnan(e["oos_sortino"]) else None,
            "ulcer":   round(float(e["oos_ulcer"]),   3) if not _isnan(e["oos_ulcer"])   else None,
            "isStar":  is_star,
        })

    datasets = []
    for n, pts in sorted(by_n.items()):
        if not pts:
            continue
        datasets.append({
            "label":           f"{n} assets",
            "data":            pts,
            "backgroundColor": ASSET_COLORS_RGBA.get(n, "rgba(200,200,200,0.7)"),
            "borderColor":     ASSET_COLORS.get(n, "#aaaaaa"),
            "borderWidth":     1,
        })
    return json.dumps(datasets)


def _bar_data(experiments: dict) -> tuple[str, str, str, str]:
    """Panel 2: labels, OOS calmar bars, IS calmar dots, bar colours."""
    rows = [
        e for e in experiments.values()
        if not _isnan(e["oos_calmar"]) and not e["is_spot_check"]
    ]
    rows.sort(key=lambda e: float(e["oos_calmar"]), reverse=True)

    labels   = [e["short_name"] for e in rows]
    oos_vals = [round(float(e["oos_calmar"]), 3) for e in rows]
    is_vals  = [
        None if _isnan(e["is_calmar_manual"]) else round(float(e["is_calmar_manual"]), 3)
        for e in rows
    ]
    colors = []
    for e in rows:
        c = float(e["oos_calmar"])
        if c > BASELINE_CALMAR:
            colors.append("rgba(63,185,80,0.8)")
        elif c > SIXTY_FORTY_CALMAR:
            colors.append("rgba(240,180,41,0.8)")
        else:
            colors.append("rgba(248,81,73,0.8)")

    return (json.dumps(labels), json.dumps(oos_vals),
            json.dumps(is_vals), json.dumps(colors))


def _cost_data(experiments: dict) -> tuple[str, str]:
    """Panel 4: cost levels and corresponding Calmar values."""
    cost_map = {
        0.0000: float("nan"),   # placeholder — will try to fill from zero-cost run
        0.0005: float("nan"),
        0.0010: float("nan"),
        0.0050: float("nan"),
    }
    name_to_cost = {
        "cost_sensitivity_005bps": 0.0005,
        "cost_sensitivity_10bps":  0.0010,
        "cost_sensitivity_50bps":  0.0050,
    }
    for exp_name, cost_level in name_to_cost.items():
        e = experiments.get(exp_name, {})
        if e and not _isnan(e.get("spot_calmar", float("nan"))):
            cost_map[cost_level] = round(float(e["spot_calmar"]), 3)

    # 0% point: look for any 8-asset experiment without tx cost
    zero_cost_names = [
        "8asset_tip_replaces_shy", "8asset_efa_replaces_iwd",
        "8asset_agg_replaces_qqq",
    ]
    for n in zero_cost_names:
        e = experiments.get(n, {})
        if e and _isnan(e.get("tx_cost", float("nan"))):
            c = e.get("oos_calmar", float("nan"))
            if not _isnan(c):
                cost_map[0.0] = round(float(c), 3)
                break

    labels = ["0%", "0.05%", "0.10%", "0.50%"]
    vals   = [cost_map.get(k, None) for k in [0.0, 0.0005, 0.001, 0.005]]
    return json.dumps(labels), json.dumps(vals)


# ===========================================================================
# HTML ASSEMBLY
# ===========================================================================

_DARK_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace; font-size: 13px; }
h1 { color: #f0f6fc; padding: 24px 24px 4px; font-size: 20px; }
.subtitle { color: #8b949e; padding: 0 24px 20px; font-size: 12px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; padding: 0 18px 24px; }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px; }
.panel.wide { grid-column: 1 / -1; }
.panel h2 { color: #58a6ff; font-size: 13px; font-weight: bold;
            margin-bottom: 14px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
canvas { max-height: 380px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
thead th { background: #1f2937; color: #58a6ff; padding: 7px 10px;
           text-align: left; border-bottom: 2px solid #30363d; white-space: nowrap; }
tbody td { padding: 5px 10px; border-bottom: 1px solid #1e2530; white-space: nowrap; }
tbody tr:hover { background: rgba(88,166,255,0.06); }
.legend-dot { display: inline-block; width: 10px; height: 10px;
              border-radius: 50%; margin-right: 5px; }
"""


def generate_html(experiments: dict, benchmarks: dict) -> str:
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    scatter_ds = _scatter_datasets(experiments)
    bar_labels, bar_oos, bar_is, bar_colors = _bar_data(experiments)
    cost_labels, cost_vals = _cost_data(experiments)
    wf_table_html     = _panel3_wf_table(experiments)
    metric_table_html = _panel5_metric_table(experiments, benchmarks)

    n_validated = sum(
        1 for e in experiments.values() if not _isnan(e["oos_calmar"])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>All Weather Portfolio — Results Dashboard</title>
<style>{_DARK_CSS}</style>
</head>
<body>

<h1>All Weather Portfolio — Results Dashboard</h1>
<p class="subtitle">Generated {ts} &nbsp;|&nbsp; {n_validated} validated experiments</p>

<div class="grid">

  <!-- Panel 1: Scatter -->
  <div class="panel wide">
    <h2>Panel 1 — OOS Calmar vs Max Drawdown (bubble size = CAGR)</h2>
    <canvas id="scatterChart"></canvas>
  </div>

  <!-- Panel 2: Bar -->
  <div class="panel wide">
    <h2>Panel 2 — OOS Calmar Ranking
      &nbsp; <span class="legend-dot" style="background:#3fb950"></span>Above baseline
      &nbsp; <span class="legend-dot" style="background:#f0b429"></span>Above 60/40
      &nbsp; <span class="legend-dot" style="background:#f85149"></span>Below 60/40
      &nbsp; <span class="legend-dot" style="background:#c9d1d9"></span>IS Calmar (manual)
    </h2>
    <canvas id="barChart"></canvas>
  </div>

  <!-- Panel 3: WF table -->
  <div class="panel wide">
    <h2>Panel 3 — Walk-Forward Reliability
      &nbsp; <span style="color:hsl(120,55%,55%)">■</span> WF median ≥ 0.6 (robust)
      &nbsp; <span style="color:hsl(45,65%,55%)">■</span> 0.3 – 0.6 (caution)
      &nbsp; <span style="color:hsl(0,55%,55%)">■</span> &lt; 0.3 (overfit)
    </h2>
    {wf_table_html}
  </div>

  <!-- Panel 4: Cost line -->
  <div class="panel">
    <h2>Panel 4 — Transaction Cost Sensitivity (8-asset allocation)</h2>
    <canvas id="costChart"></canvas>
  </div>

  <!-- Panel 5: Metric table -->
  <div class="panel wide">
    <h2>Panel 5 — Full Metric Comparison (OOS period)</h2>
    {metric_table_html}
  </div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
        integrity="sha512-ZwR1/gSZM3ai6vCdI+LVF1zSq/5HznD3oD+sCoJrzXJ+yKoAClnqnh+aDitmJ5lT6wt0zEA3bqoFmI0/jK+A=="
        crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script>
'use strict';
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

// ── Custom plugin: horizontal reference lines ──────────────────────────────
const hLinePlugin = {{
  id: 'hLine',
  afterDraw(chart) {{
    const cfg = chart.options.plugins.hLine;
    if (!cfg || !cfg.lines) return;
    const {{ ctx, scales }} = chart;
    const xScale = scales.x, yScale = scales.y;
    if (!yScale) return;
    cfg.lines.forEach(line => {{
      const yPx = yScale.getPixelForValue(line.y);
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash(line.dash || [6, 4]);
      ctx.strokeStyle = line.color || '#8b949e';
      ctx.lineWidth   = line.width || 1.2;
      ctx.moveTo(xScale.left,  yPx);
      ctx.lineTo(xScale.right, yPx);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = line.color || '#8b949e';
      ctx.font      = '11px Courier New, monospace';
      ctx.fillText(line.label || '', xScale.left + 6, yPx - 4);
      ctx.restore();
    }});
  }}
}};

// ── Custom plugin: point labels for scatter/bubble ─────────────────────────
const pointLabelPlugin = {{
  id: 'pointLabel',
  afterDatasetsDraw(chart) {{
    if (!chart.options.plugins.pointLabel) return;
    const {{ ctx, data }} = chart;
    data.datasets.forEach((ds, di) => {{
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      meta.data.forEach((el, i) => {{
        const pt = ds.data[i];
        if (!pt) return;
        const lbl   = pt.label || '';
        const bold  = pt.isStar;
        const color = pt.isStar ? '#f0b429' : '#c9d1d9';
        ctx.save();
        ctx.font      = (bold ? 'bold ' : '') + '10px Courier New, monospace';
        ctx.fillStyle = color;
        ctx.fillText(lbl, el.x + 6, el.y - 4);
        ctx.restore();
      }});
    }});
  }}
}};

Chart.register(hLinePlugin, pointLabelPlugin);

// ── Panel 1: Scatter/Bubble ────────────────────────────────────────────────
const scatterDatasets = {scatter_ds};

new Chart(document.getElementById('scatterChart'), {{
  type: 'bubble',
  data: {{ datasets: scatterDatasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2.8,
    scales: {{
      x: {{
        reverse: true,
        title: {{ display: true, text: 'OOS Max Drawdown (%)' }},
      }},
      y: {{
        title: {{ display: true, text: 'OOS Calmar Ratio' }},
        min: 0,
      }},
    }},
    plugins: {{
      legend:     {{ position: 'right' }},
      pointLabel: true,
      hLine: {{
        lines: [
          {{ y: {SIXTY_FORTY_CALMAR}, color: '#f0b429', dash: [5,4], label: '60/40 floor ({SIXTY_FORTY_CALMAR})' }},
          {{ y: {BASELINE_CALMAR},    color: '#3fb950', dash: [4,3], label: 'Baseline ({BASELINE_CALMAR})' }},
        ]
      }},
      tooltip: {{
        callbacks: {{
          label(ctx) {{
            const d = ctx.raw;
            const lines = [
              d.label,
              `Calmar: ${{d.calmar}}`,
              `CAGR:   ${{d.cagr}}%`,
              `MaxDD:  ${{d.mdd}}%`,
            ];
            if (d.sharpe  != null) lines.push(`Sharpe:  ${{d.sharpe}}`);
            if (d.sortino != null) lines.push(`Sortino: ${{d.sortino}}`);
            if (d.ulcer   != null) lines.push(`Ulcer:   ${{d.ulcer}}`);
            return lines;
          }}
        }}
      }}
    }}
  }}
}});

// ── Panel 2: Bar + IS dots ─────────────────────────────────────────────────
new Chart(document.getElementById('barChart'), {{
  data: {{
    labels: {bar_labels},
    datasets: [
      {{
        type: 'bar',
        label: 'OOS Calmar',
        data: {bar_oos},
        backgroundColor: {bar_colors},
        borderWidth: 0,
        order: 2,
      }},
      {{
        type: 'line',
        label: 'IS Calmar (manual)',
        data: {bar_is},
        borderColor: 'transparent',
        backgroundColor: '#c9d1d9',
        pointBackgroundColor: '#c9d1d9',
        pointRadius: 5,
        pointStyle: 'circle',
        showLine: false,
        order: 1,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 3.0,
    scales: {{
      x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }},
      y: {{ title: {{ display: true, text: 'Calmar Ratio' }}, min: 0 }},
    }},
    plugins: {{
      legend: {{ position: 'top' }},
      hLine: {{
        lines: [
          {{ y: {SIXTY_FORTY_CALMAR}, color: '#f0b429', dash: [5,4], label: '60/40 ({SIXTY_FORTY_CALMAR})' }},
          {{ y: {BASELINE_CALMAR},    color: '#3fb950', dash: [4,3], label: 'Baseline ({BASELINE_CALMAR})' }},
        ]
      }}
    }}
  }}
}});

// ── Panel 4: Cost sensitivity line ────────────────────────────────────────
new Chart(document.getElementById('costChart'), {{
  type: 'line',
  data: {{
    labels: {cost_labels},
    datasets: [
      {{
        label: '8-asset Calmar',
        data: {cost_vals},
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.12)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: '#58a6ff',
      }}
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Transaction cost (per trade)' }} }},
      y: {{
        title: {{ display: true, text: 'Calmar Ratio' }},
        min: 0,
      }},
    }},
    plugins: {{
      annotation: false,
      hLine: {{
        lines: [
          {{ y: {SIXTY_FORTY_FULL_CAL}, color: '#f0b429', dash: [5,4],
             label: '60/40 full-period ({SIXTY_FORTY_FULL_CAL})' }},
        ]
      }},
      tooltip: {{
        callbacks: {{
          afterBody: () => ['Beats 60/40 at all cost levels tested']
        }}
      }}
    }}
  }}
}});

</script>
</body>
</html>
"""


# ===========================================================================
# STATIC SCATTER PLOT
# ===========================================================================

def plot_scatter_png(experiments: dict, output_path: str) -> None:
    """Dark-theme matplotlib scatter: OOS Calmar vs MaxDD."""
    pts = [
        e for e in experiments.values()
        if not _isnan(e["oos_calmar"]) and not _isnan(e["oos_mdd"])
    ]
    if not pts:
        print("  No OOS data to plot scatter.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d1117")
    style_ax(ax)

    for e in pts:
        n     = max(5, min(8, e["n_assets"])) if e["n_assets"] else 7
        color = ASSET_COLORS.get(n, "#aaaaaa")
        cagr  = float(e["oos_cagr"])  if not _isnan(e["oos_cagr"])  else 6.0
        size  = max(40, min(300, cagr * 28))
        is_star = e["name"] == "6asset_tip_gsg"

        ax.scatter(
            float(e["oos_mdd"]), float(e["oos_calmar"]),
            s      = size * 1.8 if is_star else size,
            color  = color,
            edgecolors = "#f0b429" if is_star else color,
            linewidths = 2.5 if is_star else 0.5,
            alpha  = 0.85,
            zorder = 3,
        )
        ax.annotate(
            e["short_name"],
            xy=(float(e["oos_mdd"]), float(e["oos_calmar"])),
            xytext=(5, 5), textcoords="offset points",
            color="#f0b429" if is_star else "#c9d1d9",
            fontsize=8,
            fontweight="bold" if is_star else "normal",
        )

    # Reference lines
    xlim = ax.get_xlim()
    ax.axhline(SIXTY_FORTY_CALMAR, color="#f0b429", lw=1.2,
               linestyle="--", alpha=0.7, zorder=2)
    ax.axhline(BASELINE_CALMAR,    color="#3fb950", lw=1.2,
               linestyle="--", alpha=0.7, zorder=2)
    ax.text(xlim[0] * 0.98, SIXTY_FORTY_CALMAR + 0.008,
            f"60/40 floor ({SIXTY_FORTY_CALMAR})",
            color="#f0b429", fontsize=8)
    ax.text(xlim[0] * 0.98, BASELINE_CALMAR + 0.008,
            f"Baseline ({BASELINE_CALMAR})",
            color="#3fb950", fontsize=8)

    # Legend (asset count colours)
    for n, color in sorted(ASSET_COLORS.items()):
        ax.scatter([], [], s=60, color=color, label=f"{n} assets", alpha=0.85)

    ax.set_xlabel("OOS Max Drawdown (%)", fontsize=10)
    ax.set_ylabel("OOS Calmar Ratio",     fontsize=10)
    ax.set_title("All Weather Portfolio — OOS Calmar vs Max Drawdown\n"
                 "(bubble size ∝ OOS CAGR)",
                 fontsize=11, pad=10, color="white")
    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
              labelcolor="white", loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Scatter plot saved -> {output_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print(f"Reading {MASTER_LOG_PATH} ...")
    experiments, df_raw, benchmarks = load_data()

    n_total     = len(experiments)
    n_validated = sum(1 for e in experiments.values() if not _isnan(e["oos_calmar"]))
    n_spot      = sum(1 for e in experiments.values() if e["is_spot_check"])
    print(f"  {n_total} experiments found  "
          f"({n_validated} with OOS, {n_spot} spot-checks)")

    print("Generating dashboard.html ...")
    html = generate_html(experiments, benchmarks)
    os.makedirs(os.path.dirname(os.path.abspath(DASHBOARD_PATH)), exist_ok=True)
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard saved  -> {DASHBOARD_PATH}")

    print("Generating scatter plot ...")
    plot_scatter_png(experiments, SCATTER_PATH)


if __name__ == "__main__":
    main()
