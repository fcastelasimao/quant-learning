"""C02 — Capacity curve refresh: re-run S06 with everything learned. RUNS LAST.

Re-runs the S06 net-Sharpe-vs-AUM capacity chain (research/06_capacity_curve) with four
additions:

  1. **Envelope band** (C01) instead of the sqrt-law Y-band alone: band = min/max across
     {sqrt@Y=0.3, sqrt@Y=1.0, Almgren temporary}, central estimate unchanged (sqrt@Y=0.5).
  2. **Added metrics**: return-on-AUM, max drawdown, Sortino, alongside Sharpe — all from the
     full daily net-P&L series (S06 only kept the mean/var, discarding the path).
  3. **Edge-sensitivity row**: capacity re-shown at 50% of the measured gross edge.
  4. **Execution basis**: cost each trade at the SCHEDULER's chosen horizon h* (E03's
     `_choose_horizon`, using the strategy's own measured average edge as `edge_bps` — not the
     illustrative 50 bps used in E03/E04's demonstrations) instead of the hard 15-min pin.

The OLD 15-min-pinned, Y-band-only table (research/06_capacity_curve) is left untouched and is
still the "previous" reference — this stage does not overwrite it, per the plan's instruction to
keep it for continuity.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/capacity/02_capacity_refresh/build_02_capacity_refresh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from slippage import CostModel, MarketParams  # noqa: E402
from slippage.schedule import _choose_horizon, alpha_interruption_bps  # noqa: E402
from slippage.state import MarketState  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
TRADES_DIR = Path("/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/"
                  "TQQQ_SQQQ_analysis/full_history_canonical")

# Identical to research/06_capacity_curve/build_06_capacity_curve.py's MARKET dict.
MARKET = {
    "TQQQ": dict(sigma=370.0, sigma_median=339.0, sigma_stress=510.0,
                 adv=4.9e9, adv_thin=2.9e9, half_spread=0.74),
    "SQQQ": dict(sigma=372.0, sigma_median=341.0, sigma_stress=510.0,
                 adv=2.8e9, adv_thin=1.0e9, half_spread=1.00),
}
PRICE = {"TQQQ": 77.0, "SQQQ": 39.5}   # representative, for the MarketState -> $ conversion only

DEPLOY_FRAC = 0.95
AUM_GRID = np.array([1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
ETF_CEILING = 50e6


# --------------------------------------------------------------------------- gross returns
def load_gross(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(TRADES_DIR / f"TRADES_{symbol}_full_history.csv")
    out = pd.DataFrame({
        "exit_date": pd.to_datetime(df["exit_time"]).dt.normalize(),
        "entry_time": pd.to_datetime(df["entry_time"]),
        "r_gross": (df["exit_decision_price"] - df["decision_price"]) / df["decision_price"],
    }).dropna(subset=["exit_date", "r_gross"])
    return out


def params_for(symbol, *, stress=False) -> MarketParams:
    m = MARKET[symbol]
    return MarketParams(sigma_daily_bps=m["sigma_stress"] if stress else m["sigma"],
                        adv_usd=m["adv_thin"] if stress else m["adv"],
                        half_spread_bps=m["half_spread"])


def state_for(symbol) -> MarketState:
    """A representative MarketState for the scheduler's horizon search (mirrors E03's worked
    examples: interval volume = ADV_shares/26 bins/day, a flat 'typical bin' proxy)."""
    p = params_for(symbol)
    interval_volume_shares = (p.adv_usd / PRICE[symbol]) / 26.0
    return MarketState(ts=pd.Timestamp("2026-01-05 10:00:00"), symbol=symbol, bin_label="10:00",
                       expected_interval_volume=interval_volume_shares,
                       thin_volume_p10=interval_volume_shares * 0.4,
                       thin_volume_p20=interval_volume_shares * 0.6,
                       sigma_now_bps=p.sigma_daily_bps, regime="normal", spread_bps=p.half_spread_bps)


# --------------------------------------------------------------------------- full daily series + metrics
def build_daily_net_series(trades: pd.DataFrame, drag_rt_frac: float, edge_scale: float = 1.0) -> pd.Series:
    """Full daily NET return series: gross - (mean drag x trades that day). Expected-cost basis
    (no simulated timing-noise realization) -- consistent with S06's own no-lambda convention.
    `edge_scale` multiplies the gross leg only (the Item-3 50%-edge sensitivity)."""
    start = trades["entry_time"].min().normalize()
    end = trades["exit_date"].max()
    bdays = pd.bdate_range(start, end)
    daily_gross = trades.groupby("exit_date")["r_gross"].sum().reindex(bdays, fill_value=0.0)
    daily_n = trades.groupby("exit_date").size().reindex(bdays, fill_value=0)
    return edge_scale * daily_gross - drag_rt_frac * daily_n


def metrics(daily_net: pd.Series) -> dict:
    mean_d, var_d = daily_net.mean(), daily_net.var(ddof=1)
    sharpe = mean_d / np.sqrt(var_d) * np.sqrt(252) if var_d > 0 else float("nan")
    downside = daily_net[daily_net < 0]
    d_std = downside.std(ddof=1)
    sortino = mean_d / d_std * np.sqrt(252) if len(downside) > 1 and d_std > 0 else float("nan")
    cum = (1.0 + daily_net).cumprod()
    maxdd = float((cum / cum.cummax() - 1.0).min())
    return {"sharpe": sharpe, "sortino": sortino, "maxdd_pct": maxdd * 100,
           "return_on_aum_pct": mean_d * 252 * 100}


# --------------------------------------------------------------------------- cost at h* (Item 4) + envelope (Item 1)
def cost_at_scheduler_horizon(notional, p: MarketParams, state: MarketState, edge_bps, mode="cancel"):
    """h* from schedule.py's own horizon search; TOTAL round-trip cost at that horizon
    (execution + alpha-forfeiture + interruption, matching schedule.py's own `_objective` —
    NOT execution cost alone, which would silently bank the scheduler's benefit without paying
    for it: E04 showed the alpha/interruption cost of a longer horizon is real and, at large
    size, can exceed the execution-cost saving). Envelope spans execution-only; alpha/interruption
    don't depend on the impact model, so they're added identically to every band edge."""
    h_star = _choose_horizon(notional, "buy", state, PRICE[state.symbol], edge_bps, mode)
    alpha_bps, interrupt_bps = alpha_interruption_bps(h_star, state, edge_bps, mode)
    addon_bps = alpha_bps + interrupt_bps

    rt_lo = CostModel(p).roundtrip(notional, horizon_min=h_star, Y=0.3)
    rt_mid = CostModel(p).roundtrip(notional, horizon_min=h_star, Y=0.5)
    rt_hi = CostModel(p).roundtrip(notional, horizon_min=h_star, Y=1.0)
    rt_alm = CostModel(p).roundtrip(notional, horizon_min=h_star, impact_model="almgren")
    exec_lo = min(rt_lo.expected_slippage_bps, rt_hi.expected_slippage_bps, rt_alm.expected_slippage_bps)
    exec_hi = max(rt_lo.expected_slippage_bps, rt_hi.expected_slippage_bps, rt_alm.expected_slippage_bps)
    total_mid = rt_mid.expected_slippage_bps + addon_bps
    return h_star, total_mid / 1e4, (exec_lo + addon_bps) / 1e4, (exec_hi + addon_bps) / 1e4, addon_bps


# --------------------------------------------------------------------------- build
def build_symbol(symbol):
    trades = load_gross(symbol)
    edge_bps = float(trades["r_gross"].mean() * 1e4)   # the strategy's OWN measured average edge
    p = params_for(symbol)
    state = state_for(symbol)

    rows = []
    for aum in AUM_GRID:
        q = DEPLOY_FRAC * aum
        h_star, drag_mid, env_lo, env_hi, addon_bps = cost_at_scheduler_horizon(q, p, state, edge_bps)

        net_full = build_daily_net_series(trades, drag_mid)
        m_full = metrics(net_full)
        net_half_edge = build_daily_net_series(trades, drag_mid, edge_scale=0.5)
        m_half = metrics(net_half_edge)
        net_env_lo = build_daily_net_series(trades, env_lo)
        m_env_lo = metrics(net_env_lo)
        net_env_hi = build_daily_net_series(trades, env_hi)
        m_env_hi = metrics(net_env_hi)

        rows.append({
            "aum": aum, "h_star_min": h_star,
            "sharpe_central": m_full["sharpe"], "sharpe_env_lo": m_env_lo["sharpe"],
            "sharpe_env_hi": m_env_hi["sharpe"], "sortino_central": m_full["sortino"],
            "maxdd_pct_central": m_full["maxdd_pct"], "return_on_aum_pct_central": m_full["return_on_aum_pct"],
            "cost_bps_central": drag_mid * 1e4, "cost_bps_env_lo": env_lo * 1e4, "cost_bps_env_hi": env_hi * 1e4,
            "alpha_plus_interrupt_bps": addon_bps, "sharpe_half_edge": m_half["sharpe"],
        })
    return {"edge_bps": edge_bps, "table": pd.DataFrame(rows)}


def fmt_aum(a):
    return f"${a/1e9:.1f}B" if a >= 1e9 else (f"${a/1e6:g}M" if a >= 1e6 else f"${a/1e3:g}k")


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)
    for lab in (ax.yaxis.label, ax.xaxis.label, ax.title):
        lab.set_color(TEXT_COL)


def make_plot(res_dir: Path):
    """Regenerate capacity_refresh.png from the capacity_refresh_{SYM}.csv files."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax, sym in zip(axes, MARKET):
        t = pd.read_csv(res_dir / f"capacity_refresh_{sym}.csv")
        col = TICKER_COLORS[sym]
        # env_lo = optimistic cost edge -> HIGHER Sharpe; env_hi = pessimistic -> LOWER.
        ax.fill_between(t["aum"], t["sharpe_env_hi"], t["sharpe_env_lo"], color=col, alpha=0.18,
                        label="envelope band (sqrt Y∈[0.3,1.0] ∪ Almgren)")
        ax.plot(t["aum"], t["sharpe_central"], color=col, lw=1.8, marker="o", ms=4,
                label="central (sqrt Y=0.5, at scheduler h*)")
        ax.plot(t["aum"], t["sharpe_half_edge"], color=col, lw=1.2, ls="--", alpha=0.8,
                label="50% edge sensitivity")
        ax.axhline(0, color="#f85149", lw=0.9, ls=":")
        ax.axvline(ETF_CEILING, color=GRID_COL, lw=1.0, ls="--")
        ax.text(ETF_CEILING, ax.get_ylim()[1], " $50M ETF ceiling →  invalid",
                fontsize=7, color=TEXT_COL, va="top", ha="left")
        for _, r in t.iterrows():
            ax.annotate(f"{r['h_star_min']:.0f}m", (r["aum"], r["sharpe_central"]),
                        textcoords="offset points", xytext=(0, 7), fontsize=6.5,
                        color=TEXT_COL, ha="center")
        ax.set_xscale("log")
        ax.set_xlabel("AUM ($)")
        ax.set_ylabel("net Sharpe")
        ax.set_title(f"{sym} — labels = the scheduler's chosen h*")
        _style_ax(ax)
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7.5, labelcolor=TEXT_COL,
                  loc="lower left")
    fig.suptitle("C02 — capacity curve refresh: scheduler horizon + full (exec+alpha+interruption) "
                 "cost, envelope band", color=TEXT_COL, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(res_dir / "capacity_refresh.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in MARKET:
        R = build_symbol(sym)
        t = R["table"]
        t.to_csv(OUT / f"capacity_refresh_{sym}.csv", index=False)

        print(f"\n{'='*100}\n{sym} — measured avg edge = {R['edge_bps']:.1f} bps/trade\n{'-'*100}")
        print("cost = execution (at h*) + alpha-forfeit + interruption, ALL included (E04: the "
              "execution-only saving is not the real benefit)")
        print(f"{'AUM':>7} | {'h*(min)':>8} | {'a+i bps':>8} | {'Sh(env mid)':>11} {'[env lo':>8} "
              f"{'env hi]':>8} | {'Sortino':>8} {'maxDD%':>8} {'ret/AUM%':>9} | {'Sh(50% edge)':>12}")
        for _, r in t.iterrows():
            ceil = "  ⚠>$50M" if r["aum"] > ETF_CEILING else ""
            print(f"{fmt_aum(r['aum']):>7} | {r['h_star_min']:>8.0f} | {r['alpha_plus_interrupt_bps']:>8.1f} | "
                  f"{r['sharpe_central']:>11.2f} {r['sharpe_env_lo']:>8.2f} {r['sharpe_env_hi']:>8.2f} | "
                  f"{r['sortino_central']:>8.2f} {r['maxdd_pct_central']:>8.1f} "
                  f"{r['return_on_aum_pct_central']:>9.1f} | {r['sharpe_half_edge']:>12.2f}{ceil}")
    make_plot(OUT)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
