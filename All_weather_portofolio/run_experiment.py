"""
run_experiment.py
=================
Batch experiment runner for the All Weather portfolio project.

Runs the full IS → optimise → walk-forward → OOS → full-backtest pipeline for
a configurable list of asset universe experiments. Never modifies config.py on
disk; config overrides are applied in-memory and restored in a finally block.

Steps per experiment
--------------------
  1. IS backtest      (backtest mode,     BACKTEST_START → OOS_START)
  2. IS optimise      (optimise mode,     BACKTEST_START → OOS_START)
  3. Walk-forward     (walk_forward mode, BACKTEST_START → OOS_START)
  4. Confirmation     (human gate: yes / skip / quit)
  5. OOS evaluate     (oos_evaluate mode, OOS_START → BACKTEST_END)
  6. Full backtest    (full_backtest mode, BACKTEST_START → BACKTEST_END)

Usage
-----
    python3 run_experiment.py                             # all experiments
    python3 run_experiment.py --dry-run                  # preview only
    python3 run_experiment.py --experiments NAME [NAME ...]
"""

from __future__ import annotations
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import config
from data      import fetch_prices
from backtest  import run_backtest, compute_stats
from optimiser import optimise_allocation
from validation import run_walk_forward
from plotting  import plot_backtest
from export    import (make_results_dir, export_results, append_to_master_log,
                       print_header, print_stats, start_run_log, stop_run_log)

# ===========================================================================
# BASELINE REFERENCE
# ===========================================================================

BASELINE_NAME    = "8asset_manual_v1"
BASELINE_IS_CAL  = 0.336
BASELINE_OOS_CAL = 0.441


# ===========================================================================
# EXPERIMENTS
# ===========================================================================

EXPERIMENTS = [
    # ---- 5-asset: Dalio's original All Weather --------------------------------
    {
        "name":        "5asset_dalio_original",
        "description": "Dalio's original 5-asset All Weather (SPY/TLT/IEF/GLD/GSG)",
        "allocation": {
            "SPY": 0.300,
            "TLT": 0.400,
            "IEF": 0.150,
            "GLD": 0.075,
            "GSG": 0.075,
        },
        "asset_class_groups": {
            "stocks":             ["SPY"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.50,
            "long_bonds":         0.50,
            "intermediate_bonds": 0.30,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # ---- 5-asset: previous best from TIP analysis ----------------------------
    {
        "name":        "5asset_tip_previous_best",
        "description": "5-asset: SPY/QQQ/TLT/TIP/GLD — previous best variant",
        "allocation": {
            "SPY": 0.142,
            "QQQ": 0.203,
            "TLT": 0.300,
            "TIP": 0.142,
            "GLD": 0.213,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["TIP"],
            "gold":               ["GLD"],
        },
        "asset_class_max_weight": {
            "stocks":             0.45,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.30,
            "gold":               0.30,
        },
    },

    # ---- 6-asset: add TIP (inflation-protected) and GSG (commodities) --------
    {
        "name":        "6asset_tip_gsg",
        "description": "6-asset: SPY/QQQ + TLT + TIP + GLD + GSG",
        "allocation": {
            "SPY": 0.150,
            "QQQ": 0.150,
            "TLT": 0.300,
            "TIP": 0.150,
            "GLD": 0.150,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
            "commodities":        0.20,
        },
    },

    # ---- 6-asset: bond duration ladder (no commodities) ---------------------
    {
        "name":        "6asset_duration_ladder",
        "description": "6-asset: SPY/QQQ + TLT/IEF/SHY duration ladder + GLD",
        "allocation": {
            "SPY": 0.200,
            "QQQ": 0.150,
            "TLT": 0.250,
            "IEF": 0.150,
            "SHY": 0.100,
            "GLD": 0.150,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
        },
        "asset_class_max_weight": {
            "stocks":             0.45,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.35,
            "gold":               0.25,
        },
    },

    # ---- 6-asset: international equity added --------------------------------
    {
        "name":        "6asset_intl_equity",
        "description": "6-asset: SPY/EFA (intl) + TLT/IEF + GLD + GSG",
        "allocation": {
            "SPY": 0.150,
            "EFA": 0.100,
            "TLT": 0.250,
            "IEF": 0.150,
            "GLD": 0.200,
            "GSG": 0.150,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "EFA"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.30,
            "commodities":        0.25,
        },
    },

    # ---- 7-asset: 8-asset baseline minus GSG (commodities removed) ----------
    {
        "name":        "7asset_no_gsg",
        "description": "7-asset: 8-asset baseline without GSG (commodities removed)",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "IWD": 0.100,
            "TLT": 0.250,
            "IEF": 0.150,
            "SHY": 0.050,
            "GLD": 0.200,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
        },
    },

    # ---- 7-asset: 8-asset baseline minus IWD (value equity removed) ---------
    {
        "name":        "7asset_no_iwd",
        "description": "7-asset: 8-asset baseline without IWD (US value equity removed)",
        "allocation": {
            "SPY": 0.150,
            "QQQ": 0.150,
            "TLT": 0.250,
            "IEF": 0.100,
            "SHY": 0.050,
            "GLD": 0.150,
            "GSG": 0.150,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # ---- 7-asset: TIP/GSG dual addition (SPY/QQQ + bond trio + GLD + GSG) --
    {
        "name":        "7asset_tip_gsg_dual",
        "description": "7-asset: SPY/QQQ + TLT/IEF/TIP + GLD + GSG",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "TLT": 0.250,
            "IEF": 0.100,
            "TIP": 0.100,
            "GLD": 0.200,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
            "commodities":        0.20,
        },
    },

    # ---- 8-asset: EFA replaces IWD ------------------------------------------
    {
        "name":        "8asset_efa_replaces_iwd",
        "description": "8-asset: EFA (intl developed equity) instead of IWD (US value)",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "EFA": 0.100,
            "TLT": 0.250,
            "IEF": 0.100,
            "SHY": 0.050,
            "GLD": 0.150,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "EFA"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # ---- 8-asset: TIP replaces SHY ------------------------------------------
    {
        "name":        "8asset_tip_replaces_shy",
        "description": "8-asset: TIP (inflation-protected bonds) instead of SHY (short-term)",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "IWD": 0.100,
            "TLT": 0.250,
            "IEF": 0.100,
            "TIP": 0.050,
            "GLD": 0.150,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # =========================================================================
    # GROUP D — Transaction cost sensitivity (spot-check, 8-asset manual alloc)
    # =========================================================================

    {
        "name":        "cost_sensitivity_005bps",
        "description": "8-asset manual: 0.05% tx cost (zero-commission broker, spread only)",
        "group":       "D_costs",
        "mode":        "spot_check",
        "transaction_cost_pct": 0.0005,
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name":        "cost_sensitivity_10bps",
        "description": "8-asset manual: 0.1% tx cost (realistic UK retail with FX conversion)",
        "group":       "D_costs",
        "mode":        "spot_check",
        "transaction_cost_pct": 0.001,
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name":        "cost_sensitivity_50bps",
        "description": "8-asset manual: 0.5% tx cost (traditional broker worst case)",
        "group":       "D_costs",
        "mode":        "spot_check",
        "transaction_cost_pct": 0.005,
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # =========================================================================
    # GROUP C — ETF substitution spot-checks
    # =========================================================================

    {
        "name":        "spotcheck_ivv_vs_spy",
        "description": "IVV replacing SPY — confirm equivalence, full 2006-2026",
        "group":       "C_etf_check",
        "mode":        "spot_check",
        "allocation": {
            "IVV": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["IVV", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name":        "spotcheck_gldm_vs_gld",
        "description": "GLDM replacing GLD — confirm equivalence (2018-2026 only, GLDM inception)",
        "group":       "C_etf_check",
        "mode":        "spot_check",
        "backtest_start": "2018-01-01",
        "oos_start":      "2023-01-01",
        "backtest_end":   "2026-01-01",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLDM": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLDM"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name":        "spotcheck_pdbc_vs_gsg",
        "description": "PDBC replacing GSG — contango mitigation test (2014-2026)",
        "group":       "C_etf_check",
        "mode":        "spot_check",
        "backtest_start": "2014-01-01",
        "oos_start":      "2020-01-01",
        "backtest_end":   "2026-01-01",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "PDBC": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["PDBC"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # =========================================================================
    # GROUP A — 6asset_tip_gsg robustness: alternative IS/OOS splits
    # =========================================================================

    {
        "name":        "6asset_tip_gsg_split2018",
        "description": "6asset_tip_gsg: alternate OOS split 2018-2026 — robustness check",
        "group":       "A_robustness",
        "backtest_start": "2006-01-01",
        "oos_start":      "2018-01-01",
        "backtest_end":   "2026-01-01",
        "allocation": {
            "SPY": 0.150, "QQQ": 0.150, "TLT": 0.300,
            "TIP": 0.150, "GLD": 0.150, "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
            "commodities":        0.20,
        },
    },
    {
        "name":        "6asset_tip_gsg_split2022",
        "description": "6asset_tip_gsg: stress OOS split 2022-2026 — rate shock survival test",
        "group":       "A_robustness",
        "backtest_start": "2006-01-01",
        "oos_start":      "2022-01-01",
        "backtest_end":   "2026-01-01",
        "allocation": {
            "SPY": 0.150, "QQQ": 0.150, "TLT": 0.300,
            "TIP": 0.150, "GLD": 0.150, "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
            "commodities":        0.20,
        },
    },

    # =========================================================================
    # GROUP B — New asset class candidates (full 2006-2026, full pipeline)
    # =========================================================================

    {
        "name":        "7asset_vnq_reits",
        "description": "Add VNQ (REITs) — real asset diversification, 7 assets",
        "group":       "B_new_assets",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "TLT": 0.25,
            "IEF": 0.10, "GLD": 0.15, "GSG": 0.10, "VNQ": 0.15,
        },
        "asset_class_groups": {
            "stocks":       ["SPY", "QQQ"],
            "long_bonds":   ["TLT"],
            "intermediate_bonds": ["IEF"],
            "gold":         ["GLD"],
            "commodities":  ["GSG"],
            "real_estate":  ["VNQ"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
            "real_estate":        0.20,
        },
    },
    {
        "name":        "7asset_tip_gsg_vnq",
        "description": "Best 6-asset (tip_gsg) + VNQ REITs — can REITs improve further?",
        "group":       "B_new_assets",
        "allocation": {
            "SPY": 0.12, "QQQ": 0.12, "TLT": 0.25,
            "TIP": 0.12, "GLD": 0.13, "GSG": 0.10, "VNQ": 0.16,
        },
        "asset_class_groups": {
            "stocks":         ["SPY", "QQQ"],
            "long_bonds":     ["TLT"],
            "inflation_bonds": ["TIP"],
            "gold":           ["GLD"],
            "commodities":    ["GSG"],
            "real_estate":    ["VNQ"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "inflation_bonds":    0.25,
            "gold":               0.25,
            "commodities":        0.20,
            "real_estate":        0.20,
        },
    },
    {
        "name":        "8asset_agg_replaces_qqq",
        "description": "Replace QQQ with AGG (aggregate bonds) — broad bond vs tech concentration",
        "group":       "B_new_assets",
        "allocation": {
            "SPY": 0.15, "AGG": 0.15, "IWD": 0.10, "TLT": 0.20,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY", "AGG"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.35,
            "long_bonds":         0.30,
            "intermediate_bonds": 0.45,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name":        "8asset_lqd_replaces_shy",
        "description": "Replace SHY with LQD (investment grade corporate bonds) — corp vs govt short",
        "group":       "B_new_assets",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.08, "TLT": 0.20,
            "IEF": 0.12, "LQD": 0.10, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "LQD"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.30,
            "intermediate_bonds": 0.30,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name":        "8asset_djp_replaces_gsg",
        "description": "DJP replacing GSG — Bloomberg commodity index vs GSCI (balanced vs energy-heavy)",
        "group":       "B_new_assets",
        "backtest_start": "2006-07-01",
        "oos_start":      "2020-01-01",
        "backtest_end":   "2026-01-01",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "DJP": 0.10,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["DJP"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
    {
        "name": "spotcheck_gld_2018_baseline",
        "description": "GLD on 2018-2026 only — baseline for GLDM comparison",
        "mode": "spot_check",
        "backtest_start": "2018-01-01",
        "oos_start": "2023-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ","IWD"], "long_bonds": ["TLT"],
            "intermediate_bonds": ["IEF","SHY"], "gold": ["GLD"],
            "commodities": ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40,
            "intermediate_bonds": 0.25, "gold": 0.20, "commodities": 0.20,
        },
        "group": "C_etf_check",
    },

    # --- GROUP E: 7-asset robustness and new combinations ---

    {
        "name": "7asset_tip_gsg_vnq_split2018",
        "description": "7A tip_gsg_vnq robustness: OOS 2018-2026 — alternative split",
        "backtest_start": "2006-01-01",
        "oos_start": "2018-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.12, "QQQ": 0.12, "TLT": 0.25,
            "TIP": 0.12, "GLD": 0.13, "GSG": 0.10, "VNQ": 0.16,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ"], "long_bonds": ["TLT"],
            "inflation_bonds": ["TIP"], "gold": ["GLD"],
            "commodities": ["GSG"], "real_estate": ["VNQ"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40, "inflation_bonds": 0.25,
            "gold": 0.25, "commodities": 0.20, "real_estate": 0.20,
        },
        "group": "E_7asset",
    },
    {
        "name": "7asset_tip_gsg_vnq_split2022",
        "description": "7A tip_gsg_vnq robustness: OOS 2022-2026 — rate shock stress test",
        "mode": "full_pipeline",
        "backtest_start": "2006-01-01",
        "oos_start": "2022-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.12, "QQQ": 0.12, "TLT": 0.25,
            "TIP": 0.12, "GLD": 0.13, "GSG": 0.10, "VNQ": 0.16,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ"], "long_bonds": ["TLT"],
            "inflation_bonds": ["TIP"], "gold": ["GLD"],
            "commodities": ["GSG"], "real_estate": ["VNQ"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40, "inflation_bonds": 0.25,
            "gold": 0.25, "commodities": 0.20, "real_estate": 0.20,
        },
        "group": "E_7asset",
    },
    {
        "name": "7asset_tip_djp",
        "description": "TIP + DJP (Bloomberg commodities) — inflation bonds + balanced commodity",
        "backtest_start": "2006-07-01",
        "oos_start": "2020-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.12, "QQQ": 0.13, "IWD": 0.08, "TLT": 0.27,
            "TIP": 0.13, "GLD": 0.15, "DJP": 0.12,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ","IWD"], "long_bonds": ["TLT"],
            "inflation_bonds": ["TIP"], "gold": ["GLD"], "commodities": ["DJP"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40, "inflation_bonds": 0.25,
            "gold": 0.25, "commodities": 0.20,
        },
        "group": "E_7asset",
    },
    {
        "name": "7asset_6tip_plus_ief",
        "description": "Best 6-asset + IEF duration buffer — does adding intermediate bonds help?",
        "allocation": {
            "SPY": 0.12, "QQQ": 0.12, "TLT": 0.22,
            "IEF": 0.10, "TIP": 0.12, "GLD": 0.14, "GSG": 0.18,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ"], "long_bonds": ["TLT"],
            "intermediate_bonds": ["IEF","TIP"], "gold": ["GLD"], "commodities": ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40,
            "intermediate_bonds": 0.30, "gold": 0.25, "commodities": 0.25,
        },
        "group": "E_7asset",
    },
    {
        "name": "7asset_6tip_plus_shy",
        "description": "Best 6-asset + SHY stability anchor — rate shock buffer",
        "allocation": {
            "SPY": 0.13, "QQQ": 0.13, "TLT": 0.25,
            "TIP": 0.12, "SHY": 0.07, "GLD": 0.15, "GSG": 0.15,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ"], "long_bonds": ["TLT"],
            "short_bonds": ["SHY"], "inflation_bonds": ["TIP"],
            "gold": ["GLD"], "commodities": ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40, "short_bonds": 0.15,
            "inflation_bonds": 0.25, "gold": 0.25, "commodities": 0.20,
        },
        "group": "E_7asset",
    },
    {
        "name": "8asset_manual_split2022",
        "description": "8-asset manual: OOS 2022-2026 stress test — fair comparison vs 6asset_tip_gsg_split2022",
        "mode": "full_pipeline",
        "backtest_start": "2006-01-01",
        "oos_start": "2022-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.10, "QQQ": 0.15, "IWD": 0.10, "TLT": 0.25,
            "IEF": 0.10, "SHY": 0.05, "GLD": 0.15, "GSG": 0.10,
        },
        "asset_class_groups": {
            "stocks": ["SPY", "QQQ", "IWD"], "long_bonds": ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold": ["GLD"], "commodities": ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40,
            "intermediate_bonds": 0.25, "gold": 0.20, "commodities": 0.20,
        },
        "group": "A_robustness",
    },
    # --- GROUP F: 2022 stress tests for all paper trading candidates ---

    {
        "name": "7asset_tip_djp_split2022",
        "description": "7asset_tip_djp: OOS 2022-2026 stress test",
        "backtest_start": "2006-07-01",
        "oos_start": "2022-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.12, "QQQ": 0.13, "IWD": 0.08, "TLT": 0.27,
            "TIP": 0.13, "GLD": 0.15, "DJP": 0.12,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ","IWD"], "long_bonds": ["TLT"],
            "inflation_bonds": ["TIP"], "gold": ["GLD"], "commodities": ["DJP"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40, "inflation_bonds": 0.25,
            "gold": 0.25, "commodities": 0.20,
        },
        "group": "F_stress2022",
    },
    {
        "name": "7asset_6tip_plus_shy_split2022",
        "description": "7asset_6tip_plus_shy: OOS 2022-2026 stress test",
        "backtest_start": "2006-01-01",
        "oos_start": "2022-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.13, "QQQ": 0.13, "TLT": 0.25,
            "TIP": 0.12, "SHY": 0.07, "GLD": 0.15, "GSG": 0.15,
        },
        "asset_class_groups": {
            "stocks": ["SPY","QQQ"], "long_bonds": ["TLT"],
            "short_bonds": ["SHY"], "inflation_bonds": ["TIP"],
            "gold": ["GLD"], "commodities": ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks": 0.40, "long_bonds": 0.40, "short_bonds": 0.15,
            "inflation_bonds": 0.25, "gold": 0.25, "commodities": 0.20,
        },
        "group": "F_stress2022",
    },
    {
        "name": "5asset_dalio_split2022",
        "description": "Dalio original: OOS 2022-2026 stress test",
        "backtest_start": "2006-01-01",
        "oos_start": "2022-01-01",
        "backtest_end": "2026-01-01",
        "allocation": {
            "SPY": 0.300, "TLT": 0.400, "IEF": 0.150,
            "GLD": 0.075, "GSG": 0.075,
        },
        "asset_class_groups": {
            "stocks": ["SPY"], "long_bonds": ["TLT"],
            "intermediate_bonds": ["IEF"], "gold": ["GLD"], "commodities": ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks": 0.50, "long_bonds": 0.50,
            "intermediate_bonds": 0.30, "gold": 0.20, "commodities": 0.20,
        },
        "group": "F_stress2022",
    },
]


# ===========================================================================
# HELPERS
# ===========================================================================

def _read_wf_summary(results_dir: str) -> dict:
    """
    Read walk_forward.csv from results_dir and return a summary dict.

    Returns
    -------
    dict with keys: mean_ratio, median_ratio, opt_beat, n_windows, verdict
    """
    csv_path = os.path.join(results_dir, "walk_forward.csv")
    if not os.path.exists(csv_path):
        return {
            "mean_ratio": float("nan"), "median_ratio": float("nan"),
            "opt_beat": 0, "n_windows": 0, "verdict": "no_data",
        }

    df = pd.read_csv(csv_path)

    OVERFIT_CAP = 2.0
    if "Overfit Ratio Clamped" in df.columns:
        ratios = df["Overfit Ratio Clamped"]
    elif "Overfit Ratio" in df.columns:
        ratios = df["Overfit Ratio"].clip(upper=OVERFIT_CAP)
    else:
        return {
            "mean_ratio": float("nan"), "median_ratio": float("nan"),
            "opt_beat": 0, "n_windows": 0, "verdict": "no_data",
        }

    mean_ratio   = float(ratios.mean())
    median_ratio = float(ratios.median())

    if "Opt Beat Original" in df.columns:
        opt_beat = int((df["Opt Beat Original"] == True).sum())  # noqa: E712
    elif ("Optimised Test Calmar" in df.columns
          and "Original Test Calmar" in df.columns):
        opt_beat = int(
            (df["Optimised Test Calmar"] > df["Original Test Calmar"]).sum()
        )
    else:
        opt_beat = 0

    n_windows = len(df)

    if mean_ratio >= 0.7:
        verdict = "robust"
    elif mean_ratio >= 0.5:
        verdict = "moderate"
    else:
        verdict = "overfit"

    return {
        "mean_ratio":   mean_ratio,
        "median_ratio": median_ratio,
        "opt_beat":     opt_beat,
        "n_windows":    n_windows,
        "verdict":      verdict,
    }


def _confirmation_gate(exp_name: str,
                       is_stats: list,
                       opt_alloc: dict,
                       wf_summary: dict,
                       auto_yes: bool = False) -> str:
    """
    Print a pre-OOS summary and ask the user whether to proceed.

    Returns one of: "yes", "skip", "quit"

    If auto_yes is True the prompt is skipped and "yes" is returned
    automatically, enabling unattended overnight runs.
    """
    print("\n" + "=" * 68)
    print(f"  PRE-OOS REVIEW — {exp_name}")
    print("=" * 68)

    if is_stats:
        aw = is_stats[0]
        print(f"  IS Backtest:  CAGR={aw.cagr:.2f}%  "
              f"MaxDD={aw.max_drawdown:.2f}%  Calmar={aw.calmar:.3f}")

    if opt_alloc:
        print(f"\n  Optimised weights:")
        for t, w in opt_alloc.items():
            print(f"    {t:5s}  {w:.1%}")

    if wf_summary.get("n_windows", 0) > 0:
        print(f"\n  Walk-forward:  "
              f"mean_ratio={wf_summary['mean_ratio']:.3f}  "
              f"median_ratio={wf_summary['median_ratio']:.3f}  "
              f"opt_beat={wf_summary['opt_beat']}/{wf_summary['n_windows']}  "
              f"verdict={wf_summary['verdict']}")

    if auto_yes:
        print("\n  --auto-yes flag set: proceeding to OOS automatically.")
        return "yes"

    print("\n  Proceed to OOS evaluation?")
    print("  [yes / skip / quit]", flush=True)
    try:
        answer = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  (stdin closed / interrupted — proceeding to OOS automatically)")
        return "yes"

    if answer in ("yes", "y"):
        return "yes"
    if answer in ("skip", "s"):
        return "skip"
    if answer in ("quit", "q", "exit"):
        return "quit"

    print(f"  Unrecognised input '{answer}', treating as skip.")
    return "skip"


# ===========================================================================
# PER-EXPERIMENT RUNNER
# ===========================================================================

def run_experiment(exp: dict, auto_yes: bool = False) -> dict:
    """
    Run the full 6-step validated workflow for one experiment.

    Returns a result dict for inclusion in the summary table.

    Config overrides are applied in-memory and always restored in the finally
    block — config.py is never touched on disk.

    If auto_yes is True the confirmation gate is bypassed and OOS always runs.
    """
    name  = exp["name"]
    alloc = exp["allocation"]

    assert abs(sum(alloc.values()) - 1.0) < 1e-6, (
        f"Allocation for '{name}' sums to {sum(alloc.values()):.6f}, not 1.0"
    )

    row: dict = {
        "name":          name,
        "n_assets":      len(alloc),
        "is_calmar":     float("nan"),
        "opt_calmar":    float("nan"),
        "wf_mean_ratio": float("nan"),
        "wf_med_ratio":  float("nan"),
        "oos_calmar":    float("nan"),
        "spot_calmar":   float("nan"),
        "status":        "pending",
        "elapsed_s":     0.0,
    }

    t_exp_start = time.time()

    # ---- Save original config values ----------------------------------------
    _orig_allocation      = config.TARGET_ALLOCATION
    _orig_groups          = config.ASSET_CLASS_GROUPS
    _orig_max_weight      = config.ASSET_CLASS_MAX_WEIGHT
    _orig_run_mode        = config.RUN_MODE
    _orig_backtest_start  = config.BACKTEST_START
    _orig_oos_start       = config.OOS_START
    _orig_backtest_end    = config.BACKTEST_END
    _orig_tx_cost         = config.TRANSACTION_COST_PCT

    try:
        # ---- Apply in-memory overrides --------------------------------------
        config.TARGET_ALLOCATION      = alloc
        config.ASSET_CLASS_GROUPS     = exp["asset_class_groups"]
        config.ASSET_CLASS_MAX_WEIGHT = exp["asset_class_max_weight"]
        if exp.get("backtest_start"):
            config.BACKTEST_START = exp["backtest_start"]
        if exp.get("oos_start"):
            config.OOS_START      = exp["oos_start"]
        if exp.get("backtest_end"):
            config.BACKTEST_END   = exp["backtest_end"]
        if exp.get("transaction_cost_pct") is not None:
            config.TRANSACTION_COST_PCT = exp["transaction_cost_pct"]

        # ------------------------------------------------------------------
        # Fetch prices once for the full date range, then slice per step
        # ------------------------------------------------------------------
        all_tickers = list(dict.fromkeys(
            list(alloc.keys()) + [config.BENCHMARK_TICKER, "TLT"]
        ))
        print(f"\nFetching prices: {' '.join(all_tickers)}")
        prices_full = fetch_prices(
            all_tickers, config.BACKTEST_START, config.BACKTEST_END
        )

        port_cols = list(alloc.keys())
        bench_col = config.BENCHMARK_TICKER

        is_mask   = ((prices_full.index >= config.BACKTEST_START) &
                     (prices_full.index <  config.OOS_START))
        oos_mask  = ((prices_full.index >= config.OOS_START) &
                     (prices_full.index <  config.BACKTEST_END))
        full_mask = ((prices_full.index >= config.BACKTEST_START) &
                     (prices_full.index <  config.BACKTEST_END))

        is_port   = prices_full.loc[is_mask,   port_cols]
        is_bench  = prices_full.loc[is_mask,   bench_col]
        is_tlt    = prices_full.loc[is_mask,   "TLT"]
        oos_port  = prices_full.loc[oos_mask,  port_cols]
        oos_bench = prices_full.loc[oos_mask,  bench_col]
        oos_tlt   = prices_full.loc[oos_mask,  "TLT"]
        full_port  = prices_full.loc[full_mask, port_cols]
        full_bench = prices_full.loc[full_mask, bench_col]
        full_tlt   = prices_full.loc[full_mask, "TLT"]

        # ------------------------------------------------------------------
        # SPOT-CHECK MODE: single full_backtest, no IS/OOS split
        # ------------------------------------------------------------------
        if exp.get("mode") == "spot_check":
            t_step = time.time()
            config.RUN_MODE = "full_backtest"
            step_label  = f"exp_{name}_spot_full"
            results_dir = make_results_dir(step_label)

            tee = None
            try:
                tee = start_run_log(results_dir)
                print_header(f"SPOT CHECK — {name}")
                backtest_spot = run_backtest(
                    full_port, full_bench, alloc,
                    tlt_prices           = full_tlt,
                    transaction_cost_pct = config.TRANSACTION_COST_PCT,
                    tax_drag_pct         = config.TAX_DRAG_PCT,
                )
                spot_stats = compute_stats(backtest_spot)
                print_stats(spot_stats)
                export_results(backtest_spot, pd.DataFrame(), spot_stats,
                               alloc, results_dir, step_label)
                append_to_master_log(results_dir, spot_stats, alloc, step_label)
                plot_backtest(backtest_spot, spot_stats, results_dir, step_label, alloc)
                row["spot_calmar"] = spot_stats[0].calmar
            finally:
                if tee is not None:
                    stop_run_log(tee)

            print(f"  Spot check complete in {time.time() - t_step:.1f}s")
            row["status"] = "spot_check_done"
            return row   # triggers outer finally to restore config

        # ------------------------------------------------------------------
        # Step 1: IS backtest
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "backtest"
        step_label  = f"exp_{name}_s1_backtest"
        results_dir = make_results_dir(step_label)

        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 1/6  IS BACKTEST — {name}")
            backtest_is = run_backtest(
                is_port, is_bench, alloc,
                tlt_prices           = is_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            is_stats = compute_stats(backtest_is)
            print_stats(is_stats)
            export_results(backtest_is, pd.DataFrame(), is_stats,
                           alloc, results_dir, step_label)
            append_to_master_log(results_dir, is_stats, alloc, step_label)
            plot_backtest(backtest_is, is_stats, results_dir, step_label, alloc)
            row["is_calmar"] = is_stats[0].calmar
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 1 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 2: IS optimise
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "optimise"
        step_label  = f"exp_{name}_s2_optimise"
        results_dir = make_results_dir(step_label)

        opt_alloc = dict(alloc)   # will be updated to optimised weights below
        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 2/6  IS OPTIMISE — {name}")
            optimised = optimise_allocation(
                prices           = is_port,
                benchmark_prices = is_bench,
                allocation       = dict(alloc),
                method           = config.OPT_METHOD,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                min_cagr         = config.OPT_MIN_CAGR,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
            )
            opt_alloc.update(optimised)

            # Backtest the optimised weights to record IS Calmar
            bt_opt = run_backtest(
                is_port, is_bench, opt_alloc,
                tlt_prices           = is_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            opt_stats = compute_stats(bt_opt)
            print_stats(opt_stats)
            export_results(bt_opt, pd.DataFrame(), opt_stats,
                           opt_alloc, results_dir, step_label)
            append_to_master_log(results_dir, opt_stats, opt_alloc, step_label)
            plot_backtest(bt_opt, opt_stats, results_dir, step_label, opt_alloc)
            row["opt_calmar"] = opt_stats[0].calmar
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 2 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 3: Walk-forward validation
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "walk_forward"
        step_label = f"exp_{name}_s3_walkforward"
        wf_dir     = make_results_dir(step_label)

        wf_summary: dict = {
            "mean_ratio": float("nan"), "median_ratio": float("nan"),
            "opt_beat": 0, "n_windows": 0, "verdict": "no_data",
        }
        tee = None
        try:
            tee = start_run_log(wf_dir)
            print_header(f"STEP 3/6  WALK-FORWARD — {name}")
            run_walk_forward(
                prices           = is_port,
                benchmark_prices = is_bench,
                allocation       = dict(alloc),
                train_years      = config.WF_TRAIN_YEARS,
                test_years       = config.WF_TEST_YEARS,
                step_years       = config.WF_STEP_YEARS,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
                results_dir      = wf_dir,
                tlt_prices       = is_tlt,
            )
            wf_summary = _read_wf_summary(wf_dir)
            row["wf_mean_ratio"] = wf_summary["mean_ratio"]
            row["wf_med_ratio"]  = wf_summary["median_ratio"]
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 3 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 4: Human confirmation gate
        # ------------------------------------------------------------------
        answer = _confirmation_gate(name, is_stats, opt_alloc, wf_summary,
                                    auto_yes=auto_yes)
        if answer == "quit":
            row["status"] = "quit"
            return row
        if answer == "skip":
            row["status"] = "skipped"
            return row

        # ------------------------------------------------------------------
        # Step 5: OOS evaluate (with optimised weights)
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "oos_evaluate"
        step_label  = f"exp_{name}_s4_oos"
        results_dir = make_results_dir(step_label)

        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 5/6  OOS EVALUATE — {name}")
            backtest_oos = run_backtest(
                oos_port, oos_bench, opt_alloc,
                tlt_prices           = oos_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            oos_stats = compute_stats(backtest_oos)
            print_stats(oos_stats)
            export_results(backtest_oos, pd.DataFrame(), oos_stats,
                           opt_alloc, results_dir, step_label)
            append_to_master_log(results_dir, oos_stats, opt_alloc, step_label)
            plot_backtest(backtest_oos, oos_stats, results_dir, step_label, opt_alloc)
            row["oos_calmar"] = oos_stats[0].calmar
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 5 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 6: Full backtest (with optimised weights)
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "full_backtest"
        step_label  = f"exp_{name}_s5_full"
        results_dir = make_results_dir(step_label)

        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 6/6  FULL BACKTEST — {name}")
            backtest_full = run_backtest(
                full_port, full_bench, opt_alloc,
                tlt_prices           = full_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            full_stats = compute_stats(backtest_full)
            print_stats(full_stats)
            export_results(backtest_full, pd.DataFrame(), full_stats,
                           opt_alloc, results_dir, step_label)
            append_to_master_log(results_dir, full_stats, opt_alloc, step_label)
            plot_backtest(backtest_full, full_stats, results_dir, step_label, opt_alloc)
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 6 complete in {time.time() - t_step:.1f}s")

        row["status"] = "done"

    except Exception as exc:
        row["status"] = f"error: {exc}"
        traceback.print_exc()

    finally:
        # Always restore config — even on early return or exception
        config.TARGET_ALLOCATION      = _orig_allocation
        config.ASSET_CLASS_GROUPS     = _orig_groups
        config.ASSET_CLASS_MAX_WEIGHT = _orig_max_weight
        config.RUN_MODE               = _orig_run_mode
        config.BACKTEST_START         = _orig_backtest_start
        config.OOS_START              = _orig_oos_start
        config.BACKTEST_END           = _orig_backtest_end
        config.TRANSACTION_COST_PCT   = _orig_tx_cost
        row["elapsed_s"] = time.time() - t_exp_start

    return row


# ===========================================================================
# SUMMARY
# ===========================================================================

def _print_summary(rows: list[dict], output_path: str) -> None:
    """Print and save the final experiment summary table."""

    def _nan(v: float) -> bool:
        return isinstance(v, float) and np.isnan(v)

    def _f(v: float) -> str:
        return f"{v:5.3f}" if not _nan(v) else "  n/a"

    def _oos_cell(r: dict) -> str:
        """OOS column: validated OOS calmar, or spot calmar with '(S)' tag."""
        oos = r.get("oos_calmar", float("nan"))
        if not _nan(oos):
            return f"{oos:5.3f}"
        sc = r.get("spot_calmar", float("nan"))
        if not _nan(sc):
            return f"{sc:.3f}(S)"
        return "  n/a"

    sep    = "=" * 80
    header = (
        f"{sep}\n"
        f"EXPERIMENT RESULTS SUMMARY\n"
        f"{sep}\n"
        f"Baseline ({BASELINE_NAME}):  "
        f"IS Cal={BASELINE_IS_CAL:.3f}   OOS Cal={BASELINE_OOS_CAL:.3f}\n"
        f"  (S) = spot-check full_backtest; not a validated OOS result\n"
        f"{sep}\n"
        f"{'Name':<32}  {'N':>2}  "
        f"{'IS(M)':>5}  {'IS(O)':>5}  "
        f"{'WF(mn)':>6}  {'WF(md)':>6}  "
        f"{'OOS':>8}  {'Status'}\n"
        f"{'-' * 80}"
    )

    lines = [header]
    for r in rows:
        line = (
            f"{r['name']:<32}  {r['n_assets']:>2}  "
            f"{_f(r['is_calmar'])}  {_f(r['opt_calmar'])}  "
            f"{_f(r['wf_mean_ratio'])}  {_f(r['wf_med_ratio'])}  "
            f"{_oos_cell(r):>8}  {r['status']}"
        )
        lines.append(line)

    lines.append(sep)
    table = "\n".join(lines)

    print("\n" + table)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSummary saved to {output_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for All Weather portfolio variants."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview experiments without running anything, then exit.",
    )
    parser.add_argument(
        "--experiments", nargs="+", metavar="NAME",
        help="Run only these named experiments (default: run all).",
    )
    parser.add_argument(
        "--auto-yes", action="store_true",
        help="Automatically proceed to OOS for every experiment "
             "without prompting. Use for overnight unattended runs.",
    )
    args = parser.parse_args()

    # ---- Filter experiments -------------------------------------------------
    if args.experiments:
        requested = set(args.experiments)
        selected  = [e for e in EXPERIMENTS if e["name"] in requested]
        unknown   = requested - {e["name"] for e in selected}
        if unknown:
            print(f"WARNING: Unknown experiment name(s): {', '.join(sorted(unknown))}")
            print("Available names:")
            for e in EXPERIMENTS:
                print(f"  {e['name']}")
        if not selected:
            print("No matching experiments found. Exiting.")
            sys.exit(1)
    else:
        selected = EXPERIMENTS

    # ---- Dry run ------------------------------------------------------------
    if args.dry_run:
        print("DRY RUN — no experiments will be executed.\n")
        print(f"{'Name':<35}  {'N':>2}  {'Mode':<14}  {'Tickers'}")
        print("-" * 85)
        for exp in selected:
            tickers = " ".join(exp["allocation"].keys())
            n       = len(exp["allocation"])
            total   = sum(exp["allocation"].values())
            mode    = exp.get("mode", "full_pipeline")
            tag     = "" if abs(total - 1.0) < 1e-6 else "  *** BAD SUM ***"
            print(f"{exp['name']:<35}  {n:>2}  {mode:<14}  {tickers}{tag}")
        print(f"\n{len(selected)} experiment(s) would be run.")
        sys.exit(0)

    # ---- Run all selected experiments ---------------------------------------
    t_total = time.time()
    rows: list[dict] = []

    for i, exp in enumerate(selected, 1):
        group_tag = f"  [{exp['group']}]" if exp.get("group") else ""
        print(f"\n{'#' * 72}")
        print(f"  EXPERIMENT {i}/{len(selected)}: {exp['name']}{group_tag}")
        print(f"  {exp.get('description', '')}")
        print(f"{'#' * 72}")

        row = run_experiment(exp, auto_yes=args.auto_yes)
        rows.append(row)

        elapsed_str = f"{row['elapsed_s']:.1f}s"
        if row["elapsed_s"] >= 60:
            elapsed_str += f"  ({row['elapsed_s'] / 60:.1f}min)"
        print(f"\n  Elapsed: {elapsed_str}  |  Status: {row['status']}")

        if row["status"] == "quit":
            print("\nUser requested quit. Stopping.")
            break

    # ---- Summary ------------------------------------------------------------
    total_elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {total_elapsed:.1f}s  ({total_elapsed / 60:.1f}min)")

    summary_path = os.path.join(
        "results",
        f"experiment_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    )
    _print_summary(rows, summary_path)


if __name__ == "__main__":
    main()
