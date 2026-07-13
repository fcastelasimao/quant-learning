"""Stage 7 — cost-aware sizing with λ.

Stage 6 deployed all-in (f≈0.95·AUM) regardless of size, so net Sharpe collapses past the
capacity point. Stage 7 asks the sizing question instead: at a given AUM, what fraction `f` of
capital should each trade deploy?

Per-trade mean–variance objective (return on AUM units):

    U(f) = f·μ_edge  −  f·c(f·AUM)  −  λ · f² · (σ_trade² + τ²(f·AUM))
           └ edge ──┘   └ cost drag ┘   └──── risk penalty (λ) ─────┘

  - μ_edge   = gross expected return per trade
  - c(Q)     = round-trip mean cost rate (spread+impact, 15-min fill) — convex (√-law), so it
               caps f on its own even at λ=0
  - σ_trade  = per-trade gross return std ; τ = timing 1σ (independent)
  - λ        = sizing risk-aversion (the same mean-variance knob as execution, one layer up)

Two facts drive the result:
  1. A uniform f leaves **Sharpe unchanged** (it scales mean and std together) — so cost-aware
     sizing does not rescue Sharpe by shrinking; it keeps the *per-deployed-$* edge intact by
     not over-trading. What it trades away is **return on total AUM** (idle capital).
  2. The cost-optimal traded notional Q*=f*·AUM is ~**constant** beyond a threshold AUM →
     "trade a fixed dollar size, let the deployed fraction fall as 1/AUM."

Composition with the signal: final size = min(1 − p_severe, f*) — the confidence signal and the
cost cap are independent shrink factors; the binding one wins.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/07_cost_aware_sizing/build_07_cost_aware_sizing.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slippage import CostModel  # noqa: E402

# Reuse Stage-6 data loaders / cost wrappers (not a package — load by path).
_spec = importlib.util.spec_from_file_location(
    "build_06", ROOT / "research/06_capacity_curve/build_06_capacity_curve.py")
b06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b06)

AUM_GRID = np.array([1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
LAMBDAS = [0.0, 5.0, 20.0]            # sizing risk-aversion (per-trade return mean-variance)
Y_CENTRAL = 0.5
F_GRID = np.linspace(0.005, 1.0, 400)
OUT = Path(__file__).resolve().parent / "results"
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"


def timing_var_rt(notional, p, Y=Y_CENTRAL):
    """Round-trip timing variance (frac²) at the 15-min cadence."""
    return CostModel(p).roundtrip(notional, horizon_min=b06.DECISION_CADENCE_MIN, Y=Y).timing_var_frac2


def optimal_fraction(aum, mu_edge, sigma_trade, p, lam, Y=Y_CENTRAL):
    """f* maximising the per-trade mean–variance utility, on a grid in (0,1]."""
    q = F_GRID * aum
    cost = np.array([b06.roundtrip_no_lambda(qi, p, Y) for qi in q])     # mean drag rate
    tvar = np.array([timing_var_rt(qi, p, Y) for qi in q])
    util = F_GRID * mu_edge - F_GRID * cost - lam * F_GRID**2 * (sigma_trade**2 + tvar)
    i = int(np.argmax(util))
    return float(F_GRID[i]), float(cost[i]), float(tvar[i])


def build_symbol(symbol):
    trades = b06.load_gross(symbol)
    mu_edge = float(trades["r_gross"].mean())
    sigma_trade = float(trades["r_gross"].std(ddof=1))
    mean_d, var_d, n_days, n_trades = b06.daily_stats(trades)
    turnover = n_trades / (n_days / 252.0)
    p = b06.params_for(symbol)

    rows = []
    for aum in AUM_GRID:
        rec = {"aum": aum}
        for lam in LAMBDAS:
            f, cost, tvar = optimal_fraction(aum, mu_edge, sigma_trade, p, lam)
            q_star = f * aum
            # Sharpe uses the cost rate at Q* (uniform f cancels in mean/std).
            sharpe = b06.net_sharpe(mean_d, var_d, n_days, n_trades, cost, tvar)
            # net annualised return on *total* AUM: deployed fraction × per-trade net × turnover
            ret_aum = f * (mu_edge - cost) * turnover
            rec[f"f_lam{lam:g}"] = f
            rec[f"qstar_lam{lam:g}"] = q_star
            rec[f"sharpe_lam{lam:g}"] = sharpe
            rec[f"ret_lam{lam:g}"] = ret_aum
        rows.append(rec)
    return {
        "mu_edge": mu_edge, "sigma_trade": sigma_trade, "turnover": turnover,
        "gross_sharpe": mean_d / np.sqrt(var_d) * np.sqrt(252),
        "df": pd.DataFrame(rows),
    }


def fmt_aum(a):
    return f"${a/1e9:.1f}B" if a >= 1e9 else (f"${a/1e6:g}M" if a >= 1e6 else f"${a/1e3:g}k")


def fmt_q(q):
    return f"${q/1e6:.1f}M" if q >= 1e6 else f"${q/1e3:.0f}k"


def main():
    OUT.mkdir(exist_ok=True)
    results = {s: build_symbol(s) for s in b06.MARKET}

    for sym, R in results.items():
        df = R["df"]
        print(f"\n{'='*82}\n{sym}  —  μ_edge {R['mu_edge']*1e4:.0f} bps/trade | "
              f"σ_trade {R['sigma_trade']*1e2:.2f}%/trade | {R['turnover']:.0f} rt/yr | "
              f"gross Sharpe {R['gross_sharpe']:.2f}")
        print(f"{'-'*82}\nCOST-AWARE SIZING — f* (deploy frac), Q* (trade size), net Sharpe, "
              f"net ann. return on AUM")
        for lam in LAMBDAS:
            print(f"\n  λ={lam:g}")
            print(f"    {'AUM':>7} | {'f*':>6} | {'Q*':>8} | {'net Sh':>7} | {'ret/AUM':>8}")
            for _, r in df.iterrows():
                print(f"    {fmt_aum(r['aum']):>7} | {r[f'f_lam{lam:g}']:>6.3f} | "
                      f"{fmt_q(r[f'qstar_lam{lam:g}']):>8} | {r[f'sharpe_lam{lam:g}']:>7.2f} | "
                      f"{r[f'ret_lam{lam:g}']*100:>7.1f}%")
        df.to_csv(OUT / f"cost_aware_sizing_{sym}.csv", index=False)

    plot(results)
    print(f"\nWrote CSVs + cost_aware_sizing.png to {OUT}")


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=DARK_BG)
    colors = ["#3fb950", "#58a6ff", "#d29922"]
    for col, sym in enumerate(b06.MARKET):
        df = results[sym]["df"]
        ax_f, ax_r = axes[0, col], axes[1, col]
        for lam, c in zip(LAMBDAS, colors):
            ax_f.plot(df["aum"], df[f"f_lam{lam:g}"], lw=1.8, color=c, label=f"λ={lam:g}")
            ax_r.plot(df["aum"], df[f"ret_lam{lam:g}"] * 100, lw=1.8, color=c, label=f"λ={lam:g}")
        ax_f.set_title(f"{sym} — cost-optimal deploy fraction f*", color=TEXT_COL)
        ax_f.set_ylabel("f* (fraction of AUM)", color=TEXT_COL)
        ax_r.set_title(f"{sym} — net ann. return on AUM", color=TEXT_COL)
        ax_r.set_ylabel("net return on AUM", color=TEXT_COL)
        for a in (ax_f, ax_r):
            a.axvline(b06.ETF_CEILING, color="#8b949e", lw=1, ls=":", alpha=0.7)
            a.set_xscale("log")
            a.set_xlabel("AUM ($)", color=TEXT_COL)
            a.set_facecolor(PANEL_BG)
            a.grid(True, color=GRID_COL, alpha=0.4)
            a.tick_params(colors=TEXT_COL)
            for s in a.spines.values():
                s.set_color(GRID_COL)
            a.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cost_aware_sizing.png", dpi=130, facecolor=DARK_BG)


if __name__ == "__main__":
    main()
