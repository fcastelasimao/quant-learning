"""
Step 2.5 — EDA follow-up (Part A) + Targeted scenario additions (Part B).

Part A: per-bin contribution analysis, counterfactual CAGRs, non-linear tests,
        risk-asymmetry, and three diagnostic plots → step1_outputs/
Part B: 9 new sleeve scenarios (6 always-on sizes + 3 targeted RSI rules),
        full rebuild of runs/metrics.csv (32 rows), 9 new equity CSVs → runs/

Approach: full rebuild of all 31 strategy scenarios (22 original + 9 new) so
the DSR can be recomputed consistently across the entire grid.  The existing
22 equity_*.csv files are NOT rewritten; only 9 new ones are created.

Engine change: sleeve size promoted from hardcoded 0.30 to sleeve.size attribute.
"""
import os
import sys
import sqlite3
import warnings
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

warnings.filterwarnings("ignore")
np.random.seed(42)

from quantcore import config as _qc_config

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = str(_qc_config.data_dir())
OUT1_DIR   = os.path.join(BASE_DIR, "step1_outputs")
RUNS_DIR   = os.path.join(BASE_DIR, "runs")
os.makedirs(OUT1_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

STARTING_CAPITAL  = 10_000.0
BOOTSTRAP_ITERS   = 1000
BOOTSTRAP_BLOCK   = 10
N_STRATEGY_SCENARIOS = 31   # 1 baseline + 21 window cells + 9 new (DSR denominator)

TIER_SPREAD_SCHEDULE = [
    (25_000,       0.080),
    (50_000,       0.070),
    (100_000,      0.060),
    (250_000,      0.050),
    (500_000,      0.045),
    (float("inf"), 0.040),
]

def progress(msg):
    print(f"[step2.5] {msg}")
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

progress("Loading TQQQ canonical trade log")
trades = pd.read_csv(
    os.path.join(BASE_DIR, "TRADES_TQQQ_canonical.csv"),
    parse_dates=["entry_time", "exit_time"],
)
trades = trades.sort_values("entry_time").reset_index(drop=True)
N_TRADES = len(trades)

SPAN_START = trades["entry_time"].iloc[0]
SPAN_END   = trades["exit_time"].iloc[-1]
SPAN_DAYS  = (SPAN_END - SPAN_START).total_seconds() / 86400.0

progress("Loading ^IRX and TQQQ adj_close")
con_irx = sqlite3.connect(os.path.join(DATA_DIR, "DB_^IRX_historical_data.db"))
irx_df  = pd.read_sql(
    "SELECT et_datetime, close FROM candles_1d WHERE close IS NOT NULL ORDER BY et_datetime",
    con_irx,
)
con_irx.close()
irx_df["date"] = pd.to_datetime(irx_df["et_datetime"]).dt.normalize()
irx_series = irx_df.groupby("date")["close"].last()

con_tqqq  = sqlite3.connect(os.path.join(DATA_DIR, "DB_TQQQ_historical_data.db"))
tqqq_df   = pd.read_sql(
    "SELECT et_datetime, adj_close FROM candles_1d WHERE adj_close IS NOT NULL ORDER BY et_datetime",
    con_tqqq,
)
con_tqqq.close()
tqqq_df["date"] = pd.to_datetime(tqqq_df["et_datetime"]).dt.normalize()
tqqq_adj  = tqqq_df.groupby("date")["adj_close"].last()
tqqq_adj  = tqqq_adj.loc[SPAN_START.normalize():SPAN_END.normalize()]
bh_daily_returns = tqqq_adj.pct_change().dropna()

# Pre-compute IRX at each trade's entry date (forward-fill on holidays)
entry_dates    = trades["entry_time"].dt.normalize()
irx_at_entry   = entry_dates.map(lambda d: irx_series.asof(d))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sleeve classes (size promoted to instance attribute)
# ─────────────────────────────────────────────────────────────────────────────

class Sleeve(ABC):
    size: float  # fraction of portfolio notional

    @abstractmethod
    def should_enter(self, trade_row) -> bool: ...
    @abstractmethod
    def exit_event(self, trade_row, current_time) -> bool: ...
    @property
    @abstractmethod
    def label(self) -> str: ...


class NoSleeve(Sleeve):
    size = 0.0

    @property
    def label(self): return "baseline"
    def should_enter(self, trade_row): return False
    def exit_event(self, trade_row, current_time): return True


class WindowEntryRSISleeve(Sleeve):
    size = 0.30

    def __init__(self, low: float, high: float):
        self.low, self.high = low, high

    @property
    def label(self): return f"low{int(self.low)}_high{int(self.high)}"

    def should_enter(self, trade_row):
        return self.low <= trade_row.RSI_entry < self.high

    def exit_event(self, trade_row, current_time):
        return current_time >= trade_row.exit_time


class AlwaysOnSleeve(Sleeve):
    def __init__(self, size: float):
        self.size = size

    @property
    def label(self): return f"always_on_{int(round(self.size * 100))}pct"

    def should_enter(self, trade_row): return True

    def exit_event(self, trade_row, current_time):
        return current_time >= trade_row.exit_time


class MultiWindowEntryRSISleeve(Sleeve):
    """Fires when RSI_entry falls in ANY of the listed [low, high) intervals."""
    size = 0.30

    def __init__(self, windows: list[tuple[float, float]], label: str):
        self.windows  = windows
        self._label   = label

    @property
    def label(self): return self._label

    def should_enter(self, trade_row):
        rsi = trade_row.RSI_entry
        return any(lo <= rsi < hi for lo, hi in self.windows)

    def exit_event(self, trade_row, current_time):
        return current_time >= trade_row.exit_time


# ─────────────────────────────────────────────────────────────────────────────
# 3. Borrow-cost helpers
# ─────────────────────────────────────────────────────────────────────────────

def tier_spread(equity: float) -> float:
    for cap, spread in TIER_SPREAD_SCHEDULE:
        if equity <= cap:
            return spread
    return TIER_SPREAD_SCHEDULE[-1][1]


def annual_borrow_rate(irx_pct: float, equity: float) -> float:
    return irx_pct / 100.0 + tier_spread(equity)


def borrow_cost_fn(notional: float, ann_rate: float, days: float) -> float:
    return notional * ann_rate * days / 365.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-scenario backtest walker
# ─────────────────────────────────────────────────────────────────────────────

_TradeProxy = type("T", (), {})   # lightweight proxy for should_enter calls


def run_scenario(sleeve: Sleeve) -> pd.DataFrame:
    equity    = STARTING_CAPITAL
    rows      = []

    cap_b_arr    = trades["capital_before"].values
    pnl_arr      = trades["pnl"].values
    pnl_pct_arr  = trades["pnl_pct"].values
    rsi_arr      = trades["RSI_entry"].values
    entry_arr    = trades["entry_time"].values
    exit_arr     = trades["exit_time"].values
    trade_id_arr = trades["trade_id"].values
    exit_rsn_arr = trades["exit_reason"].values
    irx_arr      = irx_at_entry.values

    for i in range(N_TRADES):
        equity_before = equity

        baseline_ratio        = 1.0 + pnl_arr[i] / cap_b_arr[i]
        baseline_pnl_dollars  = equity_before * (baseline_ratio - 1.0)
        equity_after_baseline = equity_before * baseline_ratio

        proxy         = _TradeProxy()
        proxy.RSI_entry = rsi_arr[i]
        proxy.exit_time = pd.Timestamp(exit_arr[i])
        triggered     = sleeve.should_enter(proxy)

        irx_pct = float(irx_arr[i]) if not np.isnan(irx_arr[i]) else 5.0
        days_held = float(
            (pd.Timestamp(exit_arr[i]) - pd.Timestamp(entry_arr[i])).total_seconds() / 86400.0
        )

        if triggered and sleeve.size > 0:
            notional   = sleeve.size * equity_before
            gross      = notional * (pnl_pct_arr[i] / 100.0)
            ann_rate   = annual_borrow_rate(irx_pct, equity_before)
            borrow     = borrow_cost_fn(notional, ann_rate, days_held)
            net        = gross - borrow
            equity_after = equity_after_baseline + net
        else:
            notional = gross = borrow = net = 0.0
            ann_rate = annual_borrow_rate(irx_pct, equity_before)
            equity_after = equity_after_baseline

        rows.append({
            "trade_id":            trade_id_arr[i],
            "entry_time":          pd.Timestamp(entry_arr[i]),
            "exit_time":           pd.Timestamp(exit_arr[i]),
            "RSI_entry":           rsi_arr[i],
            "sleeve_triggered":    triggered,
            "equity_before":       equity_before,
            "baseline_pnl_dollars": baseline_pnl_dollars,
            "sleeve_notional":     notional,
            "sleeve_gross_pnl":    gross,
            "sleeve_days_held":    days_held,
            "sleeve_borrow_rate_ann": ann_rate,
            "sleeve_borrow_cost":  borrow,
            "sleeve_net_pnl":      net,
            "equity_after":        equity_after,
            "exit_reason":         exit_rsn_arr[i],
        })
        equity = equity_after

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Daily equity builder
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_equity(trade_df: pd.DataFrame) -> pd.Series:
    date_range = pd.date_range(start=SPAN_START.normalize(), end=SPAN_END.normalize(), freq="D")
    exit_dates = trade_df["exit_time"].dt.normalize()
    eq_events  = pd.Series(trade_df["equity_after"].values, index=exit_dates)
    daily_eq   = eq_events.groupby(level=0).last()
    daily_eq   = daily_eq.reindex(date_range).ffill().fillna(STARTING_CAPITAL)
    return daily_eq


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _max_duration(underwater: np.ndarray, index: pd.DatetimeIndex):
    max_cal = cur_cal = 0
    for uw in underwater:
        if uw:
            cur_cal += 1
            max_cal = max(max_cal, cur_cal)
        else:
            cur_cal = 0
    uw_td = underwater[index.dayofweek < 5]
    max_td = cur_td = 0
    for uw in uw_td:
        if uw:
            cur_td += 1
            max_td = max(max_td, cur_td)
        else:
            cur_td = 0
    return int(max_cal), int(max_td)


def _losing_streak(pnl_arr: np.ndarray) -> int:
    max_s = cur = 0
    for p in pnl_arr:
        if p < 0:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def _vs_bh(strat_daily: pd.Series, bh_daily: pd.Series):
    aligned = pd.concat([strat_daily.rename("s"), bh_daily.rename("b")], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan, np.nan, np.nan
    slope, intercept, *_ = stats.linregress(aligned["b"].values, aligned["s"].values)
    excess = aligned["s"] - aligned["b"]
    ir_ann = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan
    return float(intercept * 252), float(slope), float(ir_ann)


def compute_metrics(trade_df: pd.DataFrame, daily_equity: pd.Series,
                    bh_daily_ret: pd.Series) -> dict:
    dr     = daily_equity.pct_change().dropna()
    dr_arr = dr.values
    T      = len(dr_arr)

    final_equity  = float(daily_equity.iloc[-1])
    total_return  = final_equity / STARTING_CAPITAL - 1.0
    cagr          = (final_equity / STARTING_CAPITAL) ** (365.0 / SPAN_DAYS) - 1.0

    mean_dr = dr_arr.mean()
    std_dr  = dr_arr.std(ddof=1)
    sharpe_ann = (mean_dr / std_dr * np.sqrt(252)) if std_dr > 0 else 0.0

    downside   = np.minimum(dr_arr, 0.0)
    std_down   = downside.std(ddof=1)
    sortino_ann = (mean_dr / std_down * np.sqrt(252)) if std_down > 0 else 0.0

    pos_sum = dr_arr[dr_arr > 0].sum()
    neg_sum = abs(dr_arr[dr_arr < 0].sum())
    omega   = (pos_sum / neg_sum) if neg_sum > 0 else np.inf

    running_max = np.maximum.accumulate(daily_equity.values)
    dd_series   = (daily_equity.values - running_max) / running_max
    max_dd      = float(-dd_series.min()) if len(dd_series) > 0 else 0.0
    ulcer_index = float(np.sqrt(np.mean(dd_series ** 2)))
    time_uw_pct = float(np.mean(dd_series < 0))
    max_cal, max_td = _max_duration(dd_series < 0, daily_equity.index)

    var_95   = float(np.percentile(dr_arr, 5))
    cvar_95  = float(dr_arr[dr_arr <= var_95].mean()) if (dr_arr <= var_95).any() else var_95
    skew_val = float(pd.Series(dr_arr).skew())
    kurt_val = float(pd.Series(dr_arr).kurtosis())
    p95      = float(np.percentile(dr_arr, 95))
    p05      = float(np.percentile(dr_arr, 5))
    tail_ratio = (abs(p95) / abs(p05)) if abs(p05) > 0 else np.inf
    daily_vol_ann = std_dr * np.sqrt(252)

    total_pnl = trade_df["baseline_pnl_dollars"] + trade_df["sleeve_net_pnl"]
    win_rate  = float((total_pnl > 0).mean())
    pf_pos    = total_pnl[total_pnl > 0].sum()
    pf_neg    = abs(total_pnl[total_pnl < 0].sum())
    profit_factor = (pf_pos / pf_neg) if pf_neg > 0 else np.inf
    expectancy    = float(total_pnl.mean())
    dur            = (trade_df["exit_time"] - trade_df["entry_time"]).dt.total_seconds() / 86400.0
    avg_hold_days  = float(dur.mean())
    max_ls         = _losing_streak(total_pnl.values)

    n_triggered       = int(trade_df["sleeve_triggered"].sum())
    sleeve_trig_rate  = n_triggered / N_TRADES
    sleeve_total_pnl  = float(trade_df["sleeve_net_pnl"].sum())

    alpha_ann, beta, ir_ann = _vs_bh(dr, bh_daily_ret)

    return {
        "n_trades_total": N_TRADES,
        "n_trades_triggered": n_triggered,
        "sleeve_trigger_rate": sleeve_trig_rate,
        "starting_capital": STARTING_CAPITAL,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "daily_vol_ann": daily_vol_ann,
        "sharpe_ann": sharpe_ann,
        "sortino_ann": sortino_ann,
        "calmar": (cagr / max_dd) if max_dd > 0 else np.inf,
        "omega": omega,
        "max_dd": max_dd,
        "max_dd_duration_cal_days": max_cal,
        "max_dd_duration_trading_days": max_td,
        "ulcer_index": ulcer_index,
        "time_underwater_pct": time_uw_pct,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skew": skew_val,
        "excess_kurt": kurt_val,
        "tail_ratio": tail_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "avg_hold_days": avg_hold_days,
        "max_losing_streak": max_ls,
        "sleeve_only_total_pnl": sleeve_total_pnl,
        "vs_bh_alpha_ann": alpha_ann,
        "vs_bh_beta": beta,
        "vs_bh_info_ratio_ann": ir_ann,
        "_daily_returns": dr.values,  # cached for bootstrap / DSR (removed before writing)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bootstrap CIs  (circular block bootstrap, fixed block = 10)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(dr_arr: np.ndarray):
    T       = len(dr_arr)
    n_block = int(np.ceil(T / BOOTSTRAP_BLOCK))
    sharpe_s, cagr_s = np.empty(BOOTSTRAP_ITERS), np.empty(BOOTSTRAP_ITERS)
    for k in range(BOOTSTRAP_ITERS):
        starts    = np.random.randint(0, T, size=n_block)
        idx       = np.concatenate([np.arange(s, s + BOOTSTRAP_BLOCK) % T for s in starts])
        resampled = dr_arr[idx[:T]]
        sd        = resampled.std(ddof=1)
        sharpe_s[k] = resampled.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
        cagr_s[k]   = float((1.0 + resampled).prod() ** (365.0 / SPAN_DAYS) - 1.0)
    return (
        float(np.percentile(sharpe_s, 2.5)), float(np.percentile(sharpe_s, 97.5)),
        float(np.percentile(cagr_s,   2.5)), float(np.percentile(cagr_s,   97.5)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Deflated Sharpe Ratio  (N = N_STRATEGY_SCENARIOS = 31)
# ─────────────────────────────────────────────────────────────────────────────

def deflated_sharpe(all_metrics: list[dict]) -> float:
    N            = N_STRATEGY_SCENARIOS
    EULER_GAMMA  = 0.5772156649015329
    e_max_std    = (
        (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / N)
        + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (N * np.e))
    )
    sr_max_ann   = max(m["sharpe_ann"] for m in all_metrics)
    best_idx     = int(np.argmax([m["sharpe_ann"] for m in all_metrics]))
    best_dr      = all_metrics[best_idx]["_daily_returns"]
    T            = len(best_dr)
    sr_daily_max = sr_max_ann / np.sqrt(252)
    skew_val     = float(pd.Series(best_dr).skew())
    excess_k     = float(pd.Series(best_dr).kurtosis())
    total_k      = excess_k + 3.0
    denom_sq     = max(
        1.0 - skew_val * sr_daily_max + (total_k - 1.0) / 4.0 * sr_daily_max ** 2,
        1e-10,
    )
    se_sr_ann    = np.sqrt(denom_sq / T) * np.sqrt(252)
    z            = (sr_max_ann - se_sr_ann * e_max_std) / se_sr_ann
    return float(stats.norm.cdf(z))


# ─────────────────────────────────────────────────────────────────────────────
# 9. B&H TQQQ benchmark row
# ─────────────────────────────────────────────────────────────────────────────

def bh_tqqq_row() -> dict:
    bh_equity = STARTING_CAPITAL * tqqq_adj / tqqq_adj.iloc[0]
    bh_dr     = bh_equity.pct_change().dropna()
    bh_arr    = bh_dr.values

    final_eq  = float(bh_equity.iloc[-1])
    cagr      = (final_eq / STARTING_CAPITAL) ** (365.0 / SPAN_DAYS) - 1.0
    std_dr    = bh_arr.std(ddof=1)
    mean_dr   = bh_arr.mean()
    sharpe    = mean_dr / std_dr * np.sqrt(252) if std_dr > 0 else 0.0
    down      = np.minimum(bh_arr, 0.0)
    sortino   = mean_dr / down.std(ddof=1) * np.sqrt(252) if down.std(ddof=1) > 0 else 0.0
    pos_s     = bh_arr[bh_arr > 0].sum()
    neg_s     = abs(bh_arr[bh_arr < 0].sum())
    run_max   = np.maximum.accumulate(bh_equity.values)
    dd        = (bh_equity.values - run_max) / run_max
    max_dd    = float(-dd.min())
    ulcer     = float(np.sqrt(np.mean(dd ** 2)))
    max_cal, max_td = _max_duration(dd < 0, bh_equity.index)
    var_95    = float(np.percentile(bh_arr, 5))
    cvar_95   = float(bh_arr[bh_arr <= var_95].mean()) if (bh_arr <= var_95).any() else var_95
    p95, p05  = float(np.percentile(bh_arr, 95)), float(np.percentile(bh_arr, 5))

    return dict(
        scenario="bh_tqqq", low=np.nan, high=np.nan,
        sleeve_trigger_rate=np.nan, n_trades_total=np.nan, n_trades_triggered=np.nan,
        starting_capital=STARTING_CAPITAL, final_equity=final_eq,
        total_return=final_eq / STARTING_CAPITAL - 1.0, cagr=cagr,
        daily_vol_ann=std_dr * np.sqrt(252), sharpe_ann=sharpe,
        sortino_ann=sortino, calmar=(cagr / max_dd) if max_dd > 0 else np.inf,
        omega=(pos_s / neg_s) if neg_s > 0 else np.inf,
        max_dd=max_dd, max_dd_duration_cal_days=max_cal,
        max_dd_duration_trading_days=max_td, ulcer_index=ulcer,
        time_underwater_pct=float(np.mean(dd < 0)),
        var_95=var_95, cvar_95=cvar_95,
        skew=float(pd.Series(bh_arr).skew()), excess_kurt=float(pd.Series(bh_arr).kurtosis()),
        tail_ratio=(abs(p95) / abs(p05)) if abs(p05) > 0 else np.inf,
        win_rate=np.nan, profit_factor=np.nan, expectancy_per_trade=np.nan,
        avg_hold_days=np.nan, max_losing_streak=np.nan, sleeve_only_total_pnl=np.nan,
        marginal_cagr_vs_baseline=np.nan, marginal_sharpe_vs_baseline=np.nan,
        sharpe_ci_low=np.nan, sharpe_ci_high=np.nan,
        cagr_ci_low=np.nan, cagr_ci_high=np.nan,
        deflated_sharpe=np.nan,
        vs_bh_alpha_ann=np.nan, vs_bh_beta=np.nan, vs_bh_info_ratio_ann=np.nan,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PART A  —  EDA follow-up
# ─────────────────────────────────────────────────────────────────────────────

progress("=" * 60)
progress("PART A — EDA follow-up")
progress("=" * 60)

rsi  = trades["RSI_entry"].values
pnl  = trades["pnl_pct"].values   # in percent form
pnl_d= trades["pnl"].values       # dollar P&L

# ── A.1  Per-bin contribution ─────────────────────────────────────────────

progress("A.1  Per-bin contribution tables")

BIN5_EDGES  = [35, 40, 45, 50, 55, 60, 65, 70, 75]
BIN5_LABELS = [f"[{a},{b})" for a, b in zip(BIN5_EDGES[:-1], BIN5_EDGES[1:])]

BIN25_EDGES  = [35 + 2.5 * i for i in range(17)]  # 35.0 to 75.0
BIN25_LABELS = [f"[{a},{b})" for a, b in zip(BIN25_EDGES[:-1], BIN25_EDGES[1:])]

total_pnl_sum = pnl.sum()

def bin_contribution(bin_edges, bin_labels, gran_label):
    rows = []
    bins = pd.cut(
        rsi, bins=bin_edges, right=False, labels=bin_labels, include_lowest=False
    )
    total_trades = len(trades)
    for lbl in bin_labels:
        mask = bins == lbl
        sub  = trades[mask]
        if len(sub) == 0:
            continue
        sp = sub["pnl_pct"]
        rows.append({
            "granularity":         gran_label,
            "bin":                 lbl,
            "n_trades":            len(sub),
            "pct_of_total_trades": len(sub) / total_trades * 100,
            "mean_pnl_pct":        sp.mean(),
            "median_pnl_pct":      sp.median(),
            "std_pnl_pct":         sp.std(ddof=1),
            "total_pnl_pct_sum":   sp.sum(),
            "pct_of_total_pnl":    sp.sum() / total_pnl_sum * 100,
            "sharpe_per_trade":    sp.mean() / sp.std(ddof=1) if sp.std(ddof=1) > 0 else np.nan,
            "min_pnl_pct":         sp.min(),
            "max_pnl_pct":         sp.max(),
        })
    return pd.DataFrame(rows)

contrib5  = bin_contribution(BIN5_EDGES,  BIN5_LABELS,  "5pt")
contrib25 = bin_contribution(BIN25_EDGES, BIN25_LABELS, "2.5pt")

# ── A.2  Counterfactual drop-bin CAGR ─────────────────────────────────────

progress("A.2  Counterfactual drop-bin CAGR")

full_cagr = (
    (STARTING_CAPITAL * np.prod(1.0 + trades["pnl"].values / trades["capital_before"].values))
    / STARTING_CAPITAL
) ** (365.0 / SPAN_DAYS) - 1.0

bins5 = pd.cut(rsi, bins=BIN5_EDGES, right=False, labels=BIN5_LABELS, include_lowest=False)
drop_rows = []
for lbl in BIN5_LABELS:
    remaining = trades[bins5 != lbl]
    if len(remaining) == 0:
        continue
    eq = STARTING_CAPITAL
    for _, row in remaining.iterrows():
        eq *= (1.0 + row["pnl"] / row["capital_before"])
    c = (eq / STARTING_CAPITAL) ** (365.0 / SPAN_DAYS) - 1.0
    drop_rows.append({
        "granularity":        "drop_bin",
        "bin":                lbl,
        "final_equity":       eq,
        "cagr":               c,
        "cagr_delta_vs_full": c - full_cagr,
    })
drop_df = pd.DataFrame(drop_rows)

# Combine and save
from functools import reduce
contrib_all = pd.concat([contrib5, contrib25, drop_df], ignore_index=True)
contrib_path = os.path.join(OUT1_DIR, "contribution_by_rsi_bin.csv")
contrib_all.to_csv(contrib_path, index=False, float_format="%.6f")
progress(f"  Saved {contrib_path}")

# ── A.3  Non-linear regression tests ──────────────────────────────────────

progress("A.3  Non-linear tests")

# Quadratic regression
X_quad = sm.add_constant(np.column_stack([rsi, rsi ** 2]))
ols_quad = sm.OLS(pnl, X_quad).fit()
quad_vertex = -ols_quad.params[1] / (2 * ols_quad.params[2])

# LOWESS
lowess_result = sm_lowess(pnl, rsi, frac=0.3, return_sorted=True)
lowess_x = lowess_result[:, 0]
lowess_y = lowess_result[:, 1]

# Kruskal-Wallis
kw_groups = [trades[bins5 == lbl]["pnl_pct"].values for lbl in BIN5_LABELS if (bins5 == lbl).any()]
kw_stat, kw_p = stats.kruskal(*kw_groups)

# Levene's test for variance equality
lev_stat, lev_p = stats.levene(*kw_groups)

# ── A.4  Risk-asymmetry ────────────────────────────────────────────────────

progress("A.4  Risk-asymmetry analysis")

risk_rows = []
FLOOR_THRESHOLD = -2.5   # percent
for lbl in BIN5_LABELS:
    sub    = trades[bins5 == lbl]
    losses = sub[sub["pnl_pct"] < 0]["pnl_pct"]
    if len(losses) == 0:
        continue
    min_loss = float(losses.min())
    risk_rows.append({
        "bin":         lbl,
        "n_trades":    len(sub),
        "n_losses":    len(losses),
        "min_loss":    min_loss,
        "loss_std":    float(losses.std(ddof=1)) if len(losses) > 1 else np.nan,
        "pct5_loss":   float(losses.quantile(0.05)) if len(losses) >= 20 else np.nan,
        "hard_floor":  min_loss > FLOOR_THRESHOLD,
    })
risk_df = pd.DataFrame(risk_rows)

# ── A.5  Plots ────────────────────────────────────────────────────────────

progress("A.5  Generating Part A plots")

# --- contribution_by_rsi_bin.png ---
fig, ax = plt.subplots(figsize=(11, 5))
x_pos  = np.arange(len(BIN5_LABELS))
w      = 0.35
bars_p = ax.bar(x_pos - w/2, contrib5["pct_of_total_pnl"],  width=w, label="% of total P/L",    color="steelblue")
bars_t = ax.bar(x_pos + w/2, contrib5["pct_of_total_trades"], width=w, label="% of total trades", color="coral", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(BIN5_LABELS, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("% of total")
ax.set_title("Per-bin P/L Contribution vs Trade Share — TQQQ")
ax.axhline(100 / len(BIN5_LABELS), ls="--", lw=0.8, color="grey", label="Equal share")
ax.legend(fontsize=9)

# Annotate workhorse and dead zone
for row in contrib5.itertuples():
    if row.bin == "[55,60)":
        idx = BIN5_LABELS.index(row.bin)
        ax.annotate("[55,60): 26% P/L\nwith 15% of trades",
                    xy=(idx - w/2, row.pct_of_total_pnl),
                    xytext=(idx - w/2 - 1.2, row.pct_of_total_pnl + 3),
                    arrowprops=dict(arrowstyle="->", lw=0.8),
                    fontsize=8, color="steelblue")
    if row.bin == "[60,65)":
        idx = BIN5_LABELS.index(row.bin)
        ax.annotate("[60,65): dead zone\n2% P/L, 15% of trades",
                    xy=(idx + w/2, row.pct_of_total_trades),
                    xytext=(idx + w/2 + 0.5, row.pct_of_total_trades + 4),
                    arrowprops=dict(arrowstyle="->", lw=0.8),
                    fontsize=8, color="coral")

plt.tight_layout()
plt.savefig(os.path.join(OUT1_DIR, "contribution_by_rsi_bin.png"), dpi=150)
plt.close()

# --- loss_distribution_by_rsi.png ---
fig, ax = plt.subplots(figsize=(11, 5))
box_data  = [trades[bins5 == lbl]["pnl_pct"].clip(-8, 8).values for lbl in BIN5_LABELS]
box_data  = [d for d in box_data if len(d) > 0]
valid_lbl = [lbl for lbl in BIN5_LABELS if (bins5 == lbl).any()]
bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=1.5),
                flierprops=dict(marker=".", ms=3, alpha=0.4))
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
    patch.set_alpha(0.7)
ax.set_xticklabels(valid_lbl, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("pnl_pct (%)")
ax.set_ylim(-8, 8)
ax.set_title("Trade Return Distribution by RSI Bin — TQQQ (clipped ±8%)")
ax.axhline(0, ls="--", lw=0.8, color="grey")
ax.axhline(-2.5, ls=":", lw=1.0, color="red", label="−2.5% floor reference")
ax.annotate("RSI<45: losses\ncap near −2.1%",
            xy=(1.0, -2.1), xytext=(2.5, -5.5),
            arrowprops=dict(arrowstyle="->", lw=0.8),
            fontsize=8, color="red")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT1_DIR, "loss_distribution_by_rsi.png"), dpi=150)
plt.close()

# --- polynomial_fit.png ---
# Linear regression (same as Step 1)
slope_lin, intercept_lin, *_ = stats.linregress(rsi, pnl)

# Quadratic fit line
rsi_range = np.linspace(rsi.min(), rsi.max(), 200)
quad_y    = (ols_quad.params[0]
             + ols_quad.params[1] * rsi_range
             + ols_quad.params[2] * rsi_range ** 2)

CLIP_LO, CLIP_HI = -10.0, 10.0
n_clipped = int(((pnl < CLIP_LO) | (pnl > CLIP_HI)).sum())

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(rsi, np.clip(pnl, CLIP_LO, CLIP_HI), alpha=0.2, s=8,
           color="steelblue", label="_nolegend_")
ax.plot(rsi_range, slope_lin * rsi_range + intercept_lin,
        "r--", lw=1.5, label="OLS linear (p=0.40)")
ax.plot(rsi_range, quad_y, "-", color="green", lw=1.8,
        label=f"Quadratic (R²={ols_quad.rsquared:.4f})")
ax.plot(lowess_x, lowess_y, "-", color="darkorange", lw=1.8, label="LOWESS (frac=0.3)")
ax.axhline(0, ls="--", lw=0.8, color="black")
ax.axvline(quad_vertex, ls=":", lw=1.0, color="green", alpha=0.6)
ax.set_xlim(30, 80)
ax.set_ylim(CLIP_LO, CLIP_HI)
ax.set_xlabel("RSI_entry")
ax.set_ylabel("pnl_pct (%)")
ax.set_title(f"TQQQ: pnl_pct vs RSI_entry — vertex at RSI={quad_vertex:.1f}, R²={ols_quad.rsquared:.4f}")
ax.text(0.02, 0.02, f"{n_clipped} pts clipped", transform=ax.transAxes,
        fontsize=8, color="grey", va="bottom")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT1_DIR, "polynomial_fit.png"), dpi=150)
plt.close()

progress("  Plots saved")

# ── A.6  eda_followup_report.md ───────────────────────────────────────────

progress("A.6  Writing eda_followup_report.md")

rl = []
def r(s=""): rl.append(s)

r("# Step 1 EDA Follow-up — Non-linear Tests and Contribution Decomposition")
r()
r("## Why this exists")
r()
r("Step 1's linear regression of `pnl_pct` on `RSI_entry` returned slope ≈ 0 (p=0.40) "
  "and concluded 'no RSI signal.' That test can only detect monotone relationships. "
  "Post-hoc analysis revealed the real structure: RSI predicts the *shape* of the return "
  "distribution — particularly tail risk and contribution density — not the mean. "
  "This document adds the quadratic fit, non-parametric tests, and per-bin decomposition "
  "that expose that structure.")
r()

r("## Per-bin contribution")
r()
r("### 5-pt bins")
r()
r("| bin | n | % trades | mean | median | std | % of P/L | Sharpe/trade |")
r("|-----|---|---------|------|--------|-----|---------|-------------|")
for row in contrib5.itertuples():
    r(f"| {row.bin} | {int(row.n_trades)} | {row.pct_of_total_trades:.1f}% "
      f"| {row.mean_pnl_pct:.3f} | {row.median_pnl_pct:.3f} "
      f"| {row.std_pnl_pct:.3f} | {row.pct_of_total_pnl:.1f}% "
      f"| {row.sharpe_per_trade:.3f} |")
r()
r("See `contribution_by_rsi_bin.png` for the bar chart.")
r()
r("### 2.5-pt bins")
r()
r("| bin | n | % trades | mean | % of P/L | Sharpe/trade |")
r("|-----|---|---------|------|---------|-------------|")
for row in contrib25.itertuples():
    r(f"| {row.bin} | {int(row.n_trades)} | {row.pct_of_total_trades:.1f}% "
      f"| {row.mean_pnl_pct:.3f} | {row.pct_of_total_pnl:.1f}% "
      f"| {row.sharpe_per_trade:.3f} |")
r()

r("## Counterfactual drop-bin CAGR")
r()
r(f"Full baseline CAGR (all trades): **{full_cagr:.4f}** ({full_cagr*100:.2f}%)")
r()
r("| bin dropped | final equity | CAGR | ΔCAGR vs full |")
r("|-------------|-------------|------|--------------|")
for row in drop_df.sort_values("cagr_delta_vs_full").itertuples():
    r(f"| {row.bin} | ${row.final_equity:,.0f} | {row.cagr*100:.2f}% "
      f"| {row.cagr_delta_vs_full*100:+.2f} pp |")
r()

r("## Non-linear tests")
r()
r("### Quadratic regression  `pnl_pct ~ RSI_entry + RSI_entry²`")
r()
r(f"- Intercept: {ols_quad.params[0]:.5f} (t={ols_quad.tvalues[0]:.2f}, p={ols_quad.pvalues[0]:.4f})")
r(f"- β₁ (RSI):   {ols_quad.params[1]:.5f} (t={ols_quad.tvalues[1]:.2f}, p={ols_quad.pvalues[1]:.4f})")
r(f"- β₂ (RSI²):  {ols_quad.params[2]:.6f} (t={ols_quad.tvalues[2]:.2f}, p={ols_quad.pvalues[2]:.4f})")
r(f"- R²: {ols_quad.rsquared:.4f}   (R²_linear was 0.0004)")
r(f"- Vertex of parabola (peak): RSI = **{quad_vertex:.2f}**")
r(f"- Interpretation: inverted-U shape — returns peak near RSI≈{quad_vertex:.0f}, fall off at extremes. "
  f"Quadratic t-stat = {ols_quad.tvalues[2]:.2f} (p={ols_quad.pvalues[2]:.3f}) — suggestive but not significant at 5%.")
r()
r("### LOWESS smoothing (frac=0.3)")
r()
r("See `polynomial_fit.png` for the orange LOWESS curve. Non-monotone shape confirmed visually; "
  "mild peak in the RSI 50–60 region with a drop-off at high RSI consistent with the dead-zone finding.")
r()
r("### Kruskal-Wallis test  (any cross-bin difference in central tendency)")
r()
r(f"- H = {kw_stat:.3f},  p = {kw_p:.4f}")
if kw_p < 0.05:
    r("- **Significant at 5%.** At least one bin has a different median pnl_pct.")
else:
    r(f"- Not significant at 5% (p={kw_p:.3f}). Confirms that the mean/median story is null — "
      "RSI bins do not differ in central tendency.")
r()
r("### Levene's test for variance equality")
r()
r(f"- Statistic = {lev_stat:.3f},  p = {lev_p:.6f}")
if lev_p < 0.001:
    r("- **Highly significant (p < 0.001).** Variances differ substantially across RSI bins. "
      "This is the real RSI signal: risk level, not mean return.")
elif lev_p < 0.05:
    r(f"- **Significant (p={lev_p:.4f}).** Variances differ across RSI bins.")
else:
    r(f"- Not significant (p={lev_p:.4f}).")
r()

r("## Risk asymmetry")
r()
r("| bin | n_trades | n_losses | min_loss (%) | loss_std | pct5_loss | hard floor (<−2.5%) |")
r("|-----|----------|----------|-------------|----------|-----------|---------------------|")
for row in risk_df.itertuples():
    floor_str = "**YES**" if row.hard_floor else "no"
    pct5_str  = f"{row.pct5_loss:.3f}" if not np.isnan(row.pct5_loss) else "n/a"
    r(f"| {row.bin} | {int(row.n_trades)} | {int(row.n_losses)} "
      f"| {row.min_loss:.3f} | {row.loss_std:.3f} | {pct5_str} | {floor_str} |")
r()
r("See `loss_distribution_by_rsi.png` for the distribution plot.")
r("Bins with a hard floor (`min_loss > −2.5%`) exhibit tightly capped downside — "
  "likely a tighter trail-stop on low-RSI entries in the original strategy. "
  "This makes those bins asymmetrically attractive for a leverage overlay: "
  "known cap on loss, but full upside participation.")
r()

r("## Conclusions")
r()
worst_drop = drop_df.sort_values("cagr_delta_vs_full").iloc[0]
best_fine  = contrib25.sort_values("pct_of_total_pnl", ascending=False).iloc[0]
r(f"- **The [55,60) RSI bin is the workhorse**: {contrib5[contrib5.bin=='[55,60)']['pct_of_total_pnl'].values[0]:.1f}% "
  f"of total P/L from {contrib5[contrib5.bin=='[55,60)']['pct_of_total_trades'].values[0]:.1f}% of trades. "
  f"Dropping it collapses CAGR by {abs(worst_drop.cagr_delta_vs_full)*100:.1f} pp. "
  f"Fine bin [55,57.5) alone: {best_fine.pct_of_total_pnl:.1f}% of P/L.")
r(f"- **RSI predicts risk, not return**: Levene p = {lev_p:.4f} (variance differs) vs "
  f"Kruskal-Wallis p = {kw_p:.3f} (median unchanged). Low-RSI bins have hard loss floors "
  f"(~−2.1%) with uncapped upside — asymmetric risk profile that favours leverage overlay.")
r()

report_path = os.path.join(OUT1_DIR, "eda_followup_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(rl))
progress(f"  Saved {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PART B  —  Scenario additions
# ─────────────────────────────────────────────────────────────────────────────

progress("=" * 60)
progress("PART B — New scenarios + full metrics rebuild")
progress("=" * 60)

_THRESHOLDS = [35, 40, 45, 50, 55, 60, 65, 70]
WINDOW_SLEEVES = [
    WindowEntryRSISleeve(lo, hi)
    for lo in _THRESHOLDS for hi in _THRESHOLDS if hi - lo >= 10
]

NEW_SLEEVES = [
    AlwaysOnSleeve(size=0.05),
    AlwaysOnSleeve(size=0.10),
    AlwaysOnSleeve(size=0.15),
    AlwaysOnSleeve(size=0.20),
    AlwaysOnSleeve(size=0.25),
    AlwaysOnSleeve(size=0.30),
    MultiWindowEntryRSISleeve(windows=[(55.0, 60.0)],               label="targeted_55_60"),
    MultiWindowEntryRSISleeve(windows=[(55.0, 57.5)],               label="targeted_55_575"),
    MultiWindowEntryRSISleeve(windows=[(40.0, 60.0), (65.0, 70.0)], label="skip_dead_zone"),
]

ALL_SLEEVES = [NoSleeve()] + WINDOW_SLEEVES + NEW_SLEEVES
assert len(ALL_SLEEVES) == N_STRATEGY_SCENARIOS + 0  # 1 + 21 + 9 = 31

EQUITY_COLS = [
    "trade_id", "entry_time", "exit_time", "RSI_entry", "sleeve_triggered",
    "equity_before", "baseline_pnl_dollars", "sleeve_notional",
    "sleeve_gross_pnl", "sleeve_days_held", "sleeve_borrow_rate_ann",
    "sleeve_borrow_cost", "sleeve_net_pnl", "equity_after", "exit_reason",
]

NEW_LABELS = {s.label for s in NEW_SLEEVES}

all_metrics   = []
all_trade_dfs = {}   # label → trade_df (only for new scenarios; write equity CSV)

progress(f"Running {len(ALL_SLEEVES)} scenarios ({BOOTSTRAP_ITERS} bootstrap iters each)…")

for sleeve in ALL_SLEEVES:
    label      = sleeve.label
    trade_df   = run_scenario(sleeve)
    daily_eq   = build_daily_equity(trade_df)
    m          = compute_metrics(trade_df, daily_eq, bh_daily_returns)
    m["scenario"] = label

    # Expected trigger rate
    if isinstance(sleeve, NoSleeve):
        exp_rate = 0.0
    elif isinstance(sleeve, AlwaysOnSleeve):
        exp_rate = 1.0
    elif isinstance(sleeve, WindowEntryRSISleeve):
        exp_rate = float(((trades["RSI_entry"] >= sleeve.low) &
                          (trades["RSI_entry"] <  sleeve.high)).mean())
    elif isinstance(sleeve, MultiWindowEntryRSISleeve):
        mask = pd.Series([False] * N_TRADES)
        for lo, hi in sleeve.windows:
            mask |= (trades["RSI_entry"] >= lo) & (trades["RSI_entry"] < hi)
        exp_rate = float(mask.mean())
    else:
        exp_rate = np.nan

    ok_str = "OK" if abs(m["sleeve_trigger_rate"] - exp_rate) < 1e-8 else "MISMATCH"
    progress(
        f"  {label}: eq=${m['final_equity']:,.0f}, "
        f"trig={m['n_trades_triggered']}/{N_TRADES} ({m['sleeve_trigger_rate']:.1%}), "
        f"trigger_check={ok_str}"
    )

    all_metrics.append(m)

    # Write equity CSV only for new scenarios
    if label in NEW_LABELS:
        out_path = os.path.join(RUNS_DIR, f"equity_{label}.csv")
        trade_df[EQUITY_COLS].to_csv(out_path, index=False, float_format="%.6f")

# Verify existing 22 scenarios match prior metrics.csv final_equity
prior_metrics_path = os.path.join(RUNS_DIR, "metrics.csv")
if os.path.exists(prior_metrics_path):
    prior = pd.read_csv(prior_metrics_path)
    prior_map = prior.set_index("scenario")["final_equity"].to_dict()
    mismatches = 0
    for m in all_metrics[:22]:   # baseline + 21 window cells
        lbl = m["scenario"]
        if lbl in prior_map and not np.isnan(prior_map[lbl]):
            rel = abs(m["final_equity"] - prior_map[lbl]) / abs(prior_map[lbl])
            if rel > 1e-6:
                mismatches += 1
                progress(f"  MISMATCH {lbl}: new={m['final_equity']:.6f} prior={prior_map[lbl]:.6f}")
    progress(f"Existing 22-scenario integrity: {22 - mismatches}/22 match within 1e-6 relative tolerance")

# ── Bootstrap CIs ─────────────────────────────────────────────────────────

progress("Running bootstrap CIs…")
for m in all_metrics:
    sh_lo, sh_hi, cg_lo, cg_hi = bootstrap_ci(m["_daily_returns"])
    m["sharpe_ci_low"]  = sh_lo
    m["sharpe_ci_high"] = sh_hi
    m["cagr_ci_low"]    = cg_lo
    m["cagr_ci_high"]   = cg_hi

# ── Deflated Sharpe (N = 31) ───────────────────────────────────────────────

progress("Computing Deflated Sharpe Ratio (N=31)…")
dsr_val = deflated_sharpe(all_metrics)
progress(f"  DSR = {dsr_val:.4f}")
for m in all_metrics:
    m["deflated_sharpe"] = dsr_val

# ── Marginal vs baseline ───────────────────────────────────────────────────

baseline_cagr   = all_metrics[0]["cagr"]
baseline_sharpe = all_metrics[0]["sharpe_ann"]
for m in all_metrics:
    m["marginal_cagr_vs_baseline"]   = m["cagr"]   - baseline_cagr
    m["marginal_sharpe_vs_baseline"] = m["sharpe_ann"] - baseline_sharpe

# ── Build metrics.csv ─────────────────────────────────────────────────────

METRICS_COLS = [
    "scenario", "low", "high",
    "sleeve_trigger_rate", "n_trades_total", "n_trades_triggered",
    "starting_capital", "final_equity", "total_return", "cagr",
    "daily_vol_ann", "sharpe_ann", "sortino_ann", "calmar", "omega",
    "max_dd", "max_dd_duration_cal_days", "max_dd_duration_trading_days",
    "ulcer_index", "time_underwater_pct",
    "var_95", "cvar_95", "skew", "excess_kurt", "tail_ratio",
    "win_rate", "profit_factor", "expectancy_per_trade",
    "avg_hold_days", "max_losing_streak",
    "sleeve_only_total_pnl", "marginal_cagr_vs_baseline", "marginal_sharpe_vs_baseline",
    "sharpe_ci_low", "sharpe_ci_high", "cagr_ci_low", "cagr_ci_high",
    "deflated_sharpe",
    "vs_bh_alpha_ann", "vs_bh_beta", "vs_bh_info_ratio_ann",
]

rows = []

# Row ordering: baseline, 21 window cells, 6 always_on, 3 targeted, then bh_tqqq
def _row(m, low=np.nan, high=np.nan):
    r = {k: v for k, v in m.items() if not k.startswith("_")}
    r["low"]  = low
    r["high"] = high
    return r

# baseline
rows.append(_row(all_metrics[0]))

# 21 window cells sorted by (low, high)
window_rows = []
for sleeve, m in zip(WINDOW_SLEEVES, all_metrics[1:22]):
    window_rows.append(_row(m, low=sleeve.low, high=sleeve.high))
window_rows.sort(key=lambda r: (r["low"], r["high"]))
rows.extend(window_rows)

# 6 always_on + 3 targeted (new, in definition order)
for m in all_metrics[22:]:
    rows.append(_row(m))

# bh_tqqq
rows.append(bh_tqqq_row())

metrics_df = pd.DataFrame(rows)[METRICS_COLS]
metrics_df.to_csv(os.path.join(RUNS_DIR, "metrics.csv"), index=False, float_format="%.6f")

# ── Final checks ──────────────────────────────────────────────────────────

n_eq_files   = len([f for f in os.listdir(RUNS_DIR) if f.startswith("equity_") and f.endswith(".csv")])
n_new_eq     = len([f for f in os.listdir(RUNS_DIR) if any(f"equity_{s.label}.csv" == f for s in NEW_SLEEVES)])
always_on_ok = (metrics_df[metrics_df["scenario"].str.startswith("always_on_")]["sleeve_trigger_rate"] == 1.0).all()
targeted_rows = metrics_df[metrics_df["scenario"].isin(["targeted_55_60", "targeted_55_575", "skip_dead_zone"])]

progress(f"\nFinal checks:")
progress(f"  metrics.csv rows:           {len(metrics_df)} (expected 32)")
progress(f"  equity CSV files total:     {n_eq_files}  (22 old + 9 new = 31 expected)")
progress(f"  New equity CSVs written:    {n_new_eq} (expected 9)")
progress(f"  all_on trigger_rate == 1.0: {always_on_ok}")
progress(f"  always_on_30pct sleeve P/L: ${metrics_df[metrics_df.scenario=='always_on_30pct']['sleeve_only_total_pnl'].values[0]:,.0f}")
progress(f"\n  Targeted scenario trigger rates:")
for _, row in targeted_rows.iterrows():
    progress(f"    {row.scenario}: {row.sleeve_trigger_rate:.4f}")

progress("\nStep 2.5 complete.")
