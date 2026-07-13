"""Stage 6 — the net-Sharpe-vs-AUM capacity curve (the headline).

Takes the strategy's *gross* (pre-cost) returns and re-charges execution cost with the
size-aware model (Stage 5), sweeping AUM from $100k to $1B. Per-trade notional grows with
AUM, so the modelled cost grows, and the strategy's Sharpe decays — the capacity curve.

Two views, to make the role of risk-aversion λ explicit:

  WITHOUT λ  — expected-cost basis. Execution pinned to the strategy's 15-min decision
               cadence. Only the mean drag (spread + impact) hits the Sharpe numerator;
               timing is a separate risk, not in the Sharpe. λ-free, robust. Reported as a
               band over Y∈{0.3,0.5,1.0} plus a stress line (high-σ + thin volume).

  WITH λ     — execution speed is *chosen* by the Almgren–Chriss optimum optimal_participation(Q,λ).
               Impact at that speed → numerator drag; timing 1σ at that speed → added return
               *variance* in the denominator. Sharpe falls as λ rises. Run a small λ grid.

Inputs (reconstructed / measured, not re-fitted):
  - gross returns: TQQQ/SQQQ canonical trade logs, decision_price → exit_decision_price.
  - turnover & per-trade size: from the same logs (≈175 / ≈145 round-trips/yr; notional≈0.95·AUM).
  - σ, ADV, half-spread: measured in Block 1/4 (findings_04), Y adopted from literature. σ
    normal = window mean (expected-cost headline); median σ carried as a typical-day sensitivity.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/06_capacity_curve/build_06_capacity_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slippage import CostModel, MarketParams  # noqa: E402

# ---------------------------------------------------------------------------
# Measured market params (findings_04_impact_capacity.md). σ in bps/day, ADV in $.
# half-spread (one-way, bps) from findings_01. These are inputs, re-verified in Block 4.
# σ normal is reported as a pair (the rolling-vol series is right-skewed; findings_09):
#   sigma        = window MEAN   — expected-cost / time-average moment → the HEADLINE curve.
#   sigma_median = window MEDIAN — typical-day moment → a labelled SENSITIVITY line.
# Both sit inside the dominant Y-band, so the capacity verdict is unchanged either way.
# ---------------------------------------------------------------------------
MARKET = {
    "TQQQ": dict(sigma=370.0, sigma_median=339.0, sigma_stress=510.0,
                 adv=4.9e9, adv_thin=2.9e9, half_spread=0.74),
    "SQQQ": dict(sigma=372.0, sigma_median=341.0, sigma_stress=510.0,
                 adv=2.8e9, adv_thin=1.0e9, half_spread=1.00),
}
TRADES_DIR = Path("/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/"
                  "TQQQ_SQQQ_analysis/full_history_canonical")

DEPLOY_FRAC = 0.95          # per-trade notional ≈ 0.95·AUM (measured, ~5% cash buffer)
DECISION_CADENCE_MIN = 15   # the strategy decides/must fill within 15 min (no-λ view)
AUM_GRID = np.array([1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
Y_BAND = {"low": 0.3, "central": 0.5, "high": 1.0}
LAMBDAS = [0.0, 1.0, 3.0]
ETF_CEILING = 50e6          # single-name √-law valid to ~$50M for 3× ETFs (W6)

OUT = Path(__file__).resolve().parent / "results"
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"


# ---------------------------------------------------------------------------
# Strategy gross returns
# ---------------------------------------------------------------------------
def load_gross(symbol: str) -> pd.DataFrame:
    """Per-trade gross (pre-slippage) return from decision prices, + exit date."""
    df = pd.read_csv(TRADES_DIR / f"TRADES_{symbol}_full_history.csv")
    out = pd.DataFrame({
        "exit_date": pd.to_datetime(df["exit_time"]).dt.normalize(),
        "entry_time": pd.to_datetime(df["entry_time"]),
        "r_gross": (df["exit_decision_price"] - df["decision_price"]) / df["decision_price"],
    }).dropna(subset=["exit_date", "r_gross"])
    return out


def daily_stats(trades: pd.DataFrame):
    """Constant-notional daily PnL stats from gross per-trade returns.

    Returns (mean_daily, var_daily, n_business_days, n_trades). Subtracting a per-trade
    drag shifts the mean; timing adds independent variance — both applied in net_sharpe.
    """
    start = trades["entry_time"].min().normalize()
    end = trades["exit_date"].max()
    bdays = pd.bdate_range(start, end)
    daily = trades.groupby("exit_date")["r_gross"].sum().reindex(bdays, fill_value=0.0)
    return float(daily.mean()), float(daily.var(ddof=1)), len(bdays), len(trades)


def net_sharpe(mean_d, var_d, n_days, n_trades, drag_rt_frac, timing_var_rt_frac2):
    """Annualised net Sharpe from gross daily stats + a per-trade round-trip drag and
    timing variance. Drag (mean cost) lowers the numerator; timing (independent, mean-0)
    raises the denominator. trades/day scales per-trade quantities to the daily series."""
    trades_per_day = n_trades / n_days
    mean_net = mean_d - drag_rt_frac * trades_per_day
    var_net = var_d + timing_var_rt_frac2 * trades_per_day
    return mean_net / np.sqrt(var_net) * np.sqrt(252) if var_net > 0 else float("nan")


# ---------------------------------------------------------------------------
# Cost per round-trip at a given AUM
# ---------------------------------------------------------------------------
def roundtrip_no_lambda(notional, p: MarketParams, Y):
    """Expected round-trip cost (frac), execution pinned to the 15-min cadence.
    Thin wrapper over CostModel.roundtrip (the formula lives in the library)."""
    rt = CostModel(p).roundtrip(notional, horizon_min=DECISION_CADENCE_MIN, Y=Y)
    return rt.expected_slippage_bps / 1e4


def roundtrip_with_lambda(notional, p: MarketParams, Y, lam):
    """A–C optimal-speed round-trip: returns (drag_frac, timing_var_rt_frac2).
    Impact at the λ-optimal POV is a mean drag; timing 1σ there is independent risk."""
    rt = CostModel(p).roundtrip_optimal(notional, lam=lam, Y=Y)
    return rt.expected_slippage_bps / 1e4, rt.timing_var_frac2


def params_for(symbol, *, stress=False, typical=False) -> MarketParams:
    """MarketParams for a symbol. Default σ = mean (headline); `typical` uses the median σ
    (typical-day sensitivity); `stress` uses p90 σ + thin ADV."""
    m = MARKET[symbol]
    if stress:
        sigma = m["sigma_stress"]
    elif typical:
        sigma = m["sigma_median"]
    else:
        sigma = m["sigma"]
    return MarketParams(
        sigma_daily_bps=sigma,
        adv_usd=m["adv_thin"] if stress else m["adv"],
        half_spread_bps=m["half_spread"],
    )


# ---------------------------------------------------------------------------
# Build the curves
# ---------------------------------------------------------------------------
def build_symbol(symbol):
    trades = load_gross(symbol)
    mean_d, var_d, n_days, n_trades = daily_stats(trades)
    gross_sharpe = mean_d / np.sqrt(var_d) * np.sqrt(252)

    rows_nolam, rows_lam = [], []
    for aum in AUM_GRID:
        q = DEPLOY_FRAC * aum
        # ---- no-λ: Y band (normal) + stress (central Y, stress params) ----
        rec = {"aum": aum}
        p_norm = params_for(symbol)
        for key, Y in Y_BAND.items():
            drag = roundtrip_no_lambda(q, p_norm, Y)
            rec[f"sharpe_{key}"] = net_sharpe(mean_d, var_d, n_days, n_trades, drag, 0.0)
            rec[f"cost_bps_{key}"] = drag * 1e4
        drag_s = roundtrip_no_lambda(q, params_for(symbol, stress=True), Y_BAND["central"])
        rec["sharpe_stress"] = net_sharpe(mean_d, var_d, n_days, n_trades, drag_s, 0.0)
        rec["cost_bps_stress"] = drag_s * 1e4
        # σ sensitivity: typical-day (median σ) at central Y — headline is the mean above.
        drag_typ = roundtrip_no_lambda(q, params_for(symbol, typical=True), Y_BAND["central"])
        rec["sharpe_central_median"] = net_sharpe(mean_d, var_d, n_days, n_trades, drag_typ, 0.0)
        rec["cost_bps_central_median"] = drag_typ * 1e4
        rows_nolam.append(rec)
        # ---- with-λ: central Y, λ grid (impact→drag, timing→variance) ----
        recl = {"aum": aum}
        for lam in LAMBDAS:
            drag, tvar = roundtrip_with_lambda(q, p_norm, Y_BAND["central"], lam)
            recl[f"sharpe_lam{lam:g}"] = net_sharpe(mean_d, var_d, n_days, n_trades, drag, tvar)
            recl[f"drag_bps_lam{lam:g}"] = drag * 1e4
            recl[f"timing_bps_lam{lam:g}"] = np.sqrt(tvar / 2) * 1e4  # per-side 1σ
        rows_lam.append(recl)

    return {
        "gross_sharpe": gross_sharpe, "n_trades": n_trades, "n_days": n_days,
        "turnover": n_trades / (n_days / 252.0),
        "nolam": pd.DataFrame(rows_nolam), "lam": pd.DataFrame(rows_lam),
    }


def fmt_aum(a):
    return f"${a/1e9:.1f}B" if a >= 1e9 else (f"${a/1e6:g}M" if a >= 1e6 else f"${a/1e3:g}k")


def main():
    OUT.mkdir(exist_ok=True)
    results = {s: build_symbol(s) for s in MARKET}

    for sym, R in results.items():
        print(f"\n{'='*78}\n{sym}  —  gross Sharpe {R['gross_sharpe']:.2f} "
              f"| {R['turnover']:.0f} round-trips/yr | {R['n_trades']} trades / {R['n_days']} bdays")
        print(f"{'-'*78}\nNET SHARPE vs AUM  —  WITHOUT λ (expected-cost, 15-min cadence)")
        print("  headline σ = mean (expected-cost); 'typ Sh' = median-σ typical-day sensitivity")
        nl = R["nolam"]
        print(f"{'AUM':>7} | {'Sh(Y=.5)':>9} {'[Y=.3':>7} {'Y=1]':>7} | "
              f"{'cost bps':>8} | {'typ Sh':>7} | {'stress Sh':>9}")
        for _, r in nl.iterrows():
            ceil = "  ⚠>$50M" if r["aum"] > ETF_CEILING else ""
            print(f"{fmt_aum(r['aum']):>7} | {r['sharpe_central']:>9.2f} "
                  f"{r['sharpe_high']:>7.2f} {r['sharpe_low']:>7.2f} | "
                  f"{r['cost_bps_central']:>8.1f} | {r['sharpe_central_median']:>7.2f} | "
                  f"{r['sharpe_stress']:>9.2f}{ceil}")
        print(f"\nNET SHARPE vs AUM  —  WITH λ (central Y; impact→drag, timing→variance)")
        lm = R["lam"]
        hdr = " ".join(f"λ={l:g}".rjust(8) for l in LAMBDAS)
        print(f"{'AUM':>7} | {hdr} | {'drag/POVbps λ=1':>16}")
        for _, r in lm.iterrows():
            sh = " ".join(f"{r[f'sharpe_lam{l:g}']:>8.2f}" for l in LAMBDAS)
            print(f"{fmt_aum(r['aum']):>7} | {sh} | "
                  f"{r['drag_bps_lam1']:>7.1f}/{r['timing_bps_lam1']:>6.1f}")
        nl.to_csv(OUT / f"capacity_no_lambda_{sym}.csv", index=False)
        lm.to_csv(OUT / f"capacity_with_lambda_{sym}.csv", index=False)

    plot(results)
    print(f"\nWrote CSVs + capacity_curve.png to {OUT}")


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=DARK_BG)
    for col, sym in enumerate(MARKET):
        R = results[sym]
        nl, lm = R["nolam"], R["lam"]
        # top: no-λ band + stress
        ax = axes[0, col]
        ax.fill_between(nl["aum"], nl["sharpe_low"], nl["sharpe_high"],
                        color="#58a6ff", alpha=0.18, label="Y∈[0.3,1.0] band")
        ax.plot(nl["aum"], nl["sharpe_central"], color="#58a6ff", lw=2, label="central Y=0.5")
        ax.plot(nl["aum"], nl["sharpe_stress"], color="#f85149", lw=1.6, ls="--", label="stress")
        ax.axhline(R["gross_sharpe"], color="#3fb950", lw=1, ls=":", label=f"gross {R['gross_sharpe']:.2f}")
        ax.set_title(f"{sym} — net Sharpe vs AUM (no λ)", color=TEXT_COL)
        # bottom: λ grid
        ax2 = axes[1, col]
        for lam, c in zip(LAMBDAS, ["#3fb950", "#58a6ff", "#d29922"]):
            ax2.plot(lm["aum"], lm[f"sharpe_lam{lam:g}"], lw=1.8, color=c, label=f"λ={lam:g}")
        ax2.set_title(f"{sym} — net Sharpe vs AUM (with λ)", color=TEXT_COL)
        for a in (ax, ax2):
            a.axvline(ETF_CEILING, color="#8b949e", lw=1, ls=":", alpha=0.7)
            a.set_xscale("log")
            a.set_xlabel("AUM ($)", color=TEXT_COL)
            a.set_ylabel("net Sharpe", color=TEXT_COL)
            a.set_facecolor(PANEL_BG)
            a.grid(True, color=GRID_COL, alpha=0.4)
            a.tick_params(colors=TEXT_COL)
            for s in a.spines.values():
                s.set_color(GRID_COL)
            a.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "capacity_curve.png", dpi=130, facecolor=DARK_BG)


if __name__ == "__main__":
    main()
