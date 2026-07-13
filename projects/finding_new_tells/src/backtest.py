"""Walk-forward backtest engine for TQQQ research framework.

Fill rule: decide at t's close, fill at (t+1)'s open.
Costs: 1 bp commission + 5 bp slippage on TQQQ (6 bp round-trip per leg).
Benchmarks: TQQQ buy-and-hold, QQQ buy-and-hold.

Auto-appends one row to MASTER_LOG.csv at end of every run.
"""
from __future__ import annotations

import argparse
import csv
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel, SYMBOLS_ALL
from quantcore import stats as qcs
from metrics import REGISTRY, vote_all
from regime import build_regime_features, HSMM5, walk_forward_states
from strategy import decide_series, _DEFAULT_ALPHA, _DEFAULT_TAU

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMISSION_BPS = 1.0    # commission per leg (buy or sell)
SLIPPAGE_BPS   = 5.0    # slippage per leg
COST_BPS       = COMMISSION_BPS + SLIPPAGE_BPS   # 6 bps per leg, 12 bps round-trip

MASTER_LOG_PATH = Path(__file__).resolve().parents[1] / "MASTER_LOG.csv"

# ---------------------------------------------------------------------------
# Drawdown utilities
# ---------------------------------------------------------------------------

def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown from peak as a fraction (negative values)."""
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def _max_dd_duration(equity: pd.Series) -> int:
    """Longest peak-to-trough-to-recovery in trading days."""
    peak = equity.iloc[0]
    peak_t = 0
    in_dd = False
    max_dur = 0

    vals = equity.values
    n = len(vals)
    dd_start = 0
    for t in range(1, n):
        if vals[t] >= peak:
            if in_dd:
                dur = t - dd_start
                max_dur = max(max_dur, dur)
                in_dd = False
            peak = vals[t]
            peak_t = t
        else:
            if not in_dd:
                in_dd = True
                dd_start = peak_t
    # If still in drawdown at end
    if in_dd:
        dur = n - 1 - dd_start
        max_dur = max(max_dur, dur)
    return max_dur


def _current_dd_duration(equity: pd.Series) -> int:
    """Duration of the current (ongoing) drawdown in trading days."""
    vals = equity.values
    peak = vals[0]
    dd_start = 0
    in_dd = False
    for t in range(1, len(vals)):
        if vals[t] >= peak:
            peak = vals[t]
            in_dd = False
        else:
            if not in_dd:
                in_dd = True
                dd_start = t
    return (len(vals) - 1 - dd_start) if in_dd else 0


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _perf_stats(equity: pd.Series, *, trading_days: int = 252) -> dict:
    """Compute standard performance statistics from a NAV series."""
    rets = equity.pct_change(fill_method=None).dropna()
    n = len(rets)
    if n < 2:
        return {k: np.nan for k in [
            "cagr", "sharpe", "sortino", "calmar",
            "maxdd_pct", "maxdd_duration_days", "current_dd_duration_days",
        ]}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = n / trading_days
    cagr = (1 + total_return) ** (1 / max(years, 1e-6)) - 1

    excess = rets - 0.0   # risk-free ≈ 0 (simplification)
    sharpe = qcs.sharpe(rets, periods_per_year=trading_days, rf_annual=0.0) if excess.std() > 0 else np.nan

    downside = rets[rets < 0]
    sortino_denom = np.sqrt((downside ** 2).mean() * trading_days) if len(downside) > 0 else np.nan
    sortino = qcs.sortino(rets, periods_per_year=trading_days, rf_annual=0.0, downside="rms", threshold=0.0) if sortino_denom and sortino_denom > 0 else np.nan

    maxdd = qcs.max_drawdown(equity)
    calmar = cagr / abs(maxdd) if maxdd < -1e-6 else np.nan

    return {
        "cagr":                    round(cagr, 6),
        "sharpe":                  round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        "sortino":                 round(sortino, 4) if not np.isnan(sortino) else np.nan,
        "calmar":                  round(calmar, 4) if not np.isnan(calmar) else np.nan,
        "maxdd_pct":               round(maxdd * 100, 4),
        "maxdd_duration_days":     _max_dd_duration(equity),
        "current_dd_duration_days": _current_dd_duration(equity),
    }


# ---------------------------------------------------------------------------
# Return decomposition
# ---------------------------------------------------------------------------

def _gap_intraday_decomp(
    open_prices: pd.Series,
    close_prices: pd.Series,
    positions: pd.Series,
) -> tuple[float, float]:
    """Decompose strategy return into gap (overnight) and intraday components.

    gap_ret     = sum of (open[t] / close[t-1] - 1) * position[t-1]
    intraday_ret = sum of (close[t] / open[t]  - 1) * position[t]
    position ∈ {0, 1}.
    """
    close_lag = close_prices.shift(1)
    gap = ((open_prices / close_lag - 1) * positions.shift(1)).fillna(0).sum()
    intraday = ((close_prices / open_prices - 1) * positions).fillna(0).sum()
    return float(gap), float(intraday)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity:           pd.Series
    benchmark_tqqq:   pd.Series
    benchmark_qqq:    pd.Series
    positions:        pd.Series          # 0 or 1
    signals:          pd.DataFrame       # p_buy, p_hold, p_sell, score, action
    trade_ledger:     pd.DataFrame
    perf:             dict
    gap_return_pct:   float
    intraday_return_pct: float
    regime_states:    pd.Series | None = field(default=None)
    regime_probs:     pd.DataFrame | None = field(default=None)
    split:            str = "unknown"


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    panel: pd.DataFrame,
    *,
    tau: float = _DEFAULT_TAU,
    alpha: np.ndarray | None = None,
    metric_names: list[str] | None = None,
    use_regime: bool = True,
    regime_min_train_years: int = 3,
    split: str = "unknown",
) -> BacktestResult:
    """Run walk-forward backtest.

    Signals are computed using all data up to t (causal).
    Positions are taken at t+1's open.
    Costs are deducted on position changes (6 bps per leg).

    Parameters
    ----------
    panel : wide DataFrame with TQQQ/QQQ OHLCV columns.
    tau : softmax temperature.
    alpha : per-regime hold-bias vector (5,).
    metric_names : subset of REGISTRY voting metrics.
    use_regime : if True, fit HSMM5 walk-forward. If False, use sideways (state=2).
    split : label for MASTER_LOG ('train', 'val', 'test').
    """
    if alpha is None:
        alpha = _DEFAULT_ALPHA.copy()

    if metric_names is None:
        metric_names = [n for n, m in REGISTRY.items() if m.status == "voting"]

    # Trim to TQQQ's first valid row to avoid NaN warmup contamination
    if "TQQQ_open" in panel.columns:
        first_valid = panel["TQQQ_open"].first_valid_index()
        if first_valid is not None:
            panel = panel.loc[first_valid:].copy()

    # --- Regime ---
    if use_regime:
        state_series, proba_df = walk_forward_states(
            panel, min_train_years=regime_min_train_years
        )
    else:
        state_series = pd.Series(2, index=panel.index, name="regime_state")
        proba_df = pd.DataFrame(
            np.tile([0, 0, 1, 0, 0], (len(panel), 1)),
            index=panel.index,
            columns=[f"p_state_{i}" for i in range(5)],
            dtype=float,
        )

    # --- Signals ---
    signals = decide_series(
        panel, state_series, proba_df,
        tau=tau, alpha=alpha, metric_names=metric_names
    )

    # --- Position series: 1 = long TQQQ, 0 = flat ---
    # Signal at close[t] → position filled at open[t+1]
    # We shift signals by 1 (next day open) and hold until next signal change
    raw_position = (signals["action"] == "buy").astype(int)
    # Position is determined by YESTERDAY's signal (no-lookahead at open fill)
    position = raw_position.shift(1).fillna(0).astype(int)

    # --- Prices ---
    tqqq_open  = panel.get("TQQQ_open")
    tqqq_close = panel.get("TQQQ_close")
    qqq_close  = panel.get("QQQ_close")

    if tqqq_open is None or tqqq_close is None:
        raise ValueError("Panel must contain TQQQ_open and TQQQ_close.")

    # --- PnL: daily returns at t+1 open (entry) to t+2 open (exit) ---
    # More precisely: we compute equity from open-to-open returns while in position
    tqqq_open_ret = tqqq_open.pct_change(fill_method=None).fillna(0)

    # Apply costs on position changes
    pos_change = position.diff().abs().fillna(0)
    cost_series = pos_change * (COST_BPS / 10_000)

    strategy_ret = position * tqqq_open_ret - cost_series

    equity = (1 + strategy_ret).cumprod()
    equity.name = "strategy"

    # Benchmarks (buy-and-hold from day 1)
    bah_tqqq = (1 + tqqq_open_ret).cumprod()
    bah_tqqq.name = "tqqq_bah"

    if qqq_close is not None:
        qqq_ret = qqq_close.pct_change(fill_method=None).fillna(0)
        bah_qqq = (1 + qqq_ret).cumprod()
    else:
        bah_qqq = pd.Series(np.nan, index=panel.index, name="qqq_bah")
    bah_qqq.name = "qqq_bah"

    # --- Trade ledger ---
    trades = _build_trade_ledger(position, tqqq_open, tqqq_close)

    # --- Perf stats ---
    perf = _perf_stats(equity)
    perf["hit_rate"] = (trades["pnl_pct"] > 0).mean() if len(trades) > 0 else np.nan
    perf["turnover"] = float(pos_change.mean())
    perf["exposure_pct"] = float(position.mean() * 100)
    tqqq_cagr = _perf_stats(bah_tqqq)["cagr"]
    perf["vs_tqqq_bh_excess_cagr"] = round(perf["cagr"] - tqqq_cagr, 6)

    # --- Gap vs intraday ---
    gap, intraday = _gap_intraday_decomp(tqqq_open, tqqq_close, position)

    return BacktestResult(
        equity=equity,
        benchmark_tqqq=bah_tqqq,
        benchmark_qqq=bah_qqq,
        positions=position,
        signals=signals,
        trade_ledger=trades,
        perf=perf,
        gap_return_pct=round(gap * 100, 4),
        intraday_return_pct=round(intraday * 100, 4),
        regime_states=state_series if use_regime else None,
        regime_probs=proba_df if use_regime else None,
        split=split,
    )


def _build_trade_ledger(
    position: pd.Series,
    open_prices: pd.Series,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """Build per-round-trip trade ledger."""
    records = []
    in_trade = False
    entry_date = None
    entry_price = None

    for date, pos in position.items():
        if pos == 1 and not in_trade:
            in_trade = True
            entry_date = date
            entry_price = open_prices.loc[date] if date in open_prices.index else np.nan
        elif pos == 0 and in_trade:
            exit_date = date
            exit_price = open_prices.loc[date] if date in open_prices.index else np.nan
            pnl_pct = (exit_price / entry_price - 1 - 0.0012) * 100 if entry_price and not np.isnan(entry_price) else np.nan
            records.append({
                "entry_date": entry_date,
                "exit_date":  exit_date,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "pnl_pct":     pnl_pct,
            })
            in_trade = False

    if in_trade:
        exit_date = position.index[-1]
        exit_price = open_prices.iloc[-1] if len(open_prices) > 0 else np.nan
        pnl_pct = (exit_price / entry_price - 1 - 0.0012) * 100 if entry_price and not np.isnan(entry_price) else np.nan
        records.append({
            "entry_date": entry_date,
            "exit_date":  exit_date,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "pnl_pct":     pnl_pct,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# MASTER_LOG append
# ---------------------------------------------------------------------------

_LOG_FIELDS = [
    "run_id", "strategy_version", "metrics_voting", "metrics_watch",
    "tau", "alpha_state_0", "alpha_state_1", "alpha_state_2", "alpha_state_3", "alpha_state_4",
    "regime_model", "train_window", "val_window", "test_window", "split_evaluated",
    "cagr", "sharpe", "sortino", "calmar",
    "maxdd_pct", "maxdd_duration_days", "current_dd_duration_days",
    "hit_rate", "turnover", "exposure_pct",
    "gap_return_pct", "intraday_return_pct",
    "brier_buy", "brier_sell", "log_loss", "ic_5d",
    "vs_tqqq_bh_excess_cagr", "notes",
]


def append_master_log(
    result: BacktestResult,
    *,
    strategy_version: str = "v1",
    metric_names: list[str] | None = None,
    tau: float = _DEFAULT_TAU,
    alpha: np.ndarray | None = None,
    train_window: str = "≤2017",
    val_window: str = "2018–2021",
    test_window: str = "2022→",
    regime_model: str = "hsmm5",
    notes: str = "",
) -> None:
    """Append one row to MASTER_LOG.csv. Never overwrites existing rows."""
    if alpha is None:
        alpha = _DEFAULT_ALPHA

    if metric_names is None:
        metric_names = [n for n, m in REGISTRY.items() if m.status == "voting"]
    watch_names = [n for n, m in REGISTRY.items() if m.status == "watch"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]

    row = {
        "run_id":           run_id,
        "strategy_version": strategy_version,
        "metrics_voting":   ";".join(metric_names),
        "metrics_watch":    ";".join(watch_names),
        "tau":              tau,
        **{f"alpha_state_{i}": alpha[i] for i in range(5)},
        "regime_model":     regime_model,
        "train_window":     train_window,
        "val_window":       val_window,
        "test_window":      test_window,
        "split_evaluated":  result.split,
        **{k: result.perf.get(k, "") for k in [
            "cagr", "sharpe", "sortino", "calmar",
            "maxdd_pct", "maxdd_duration_days", "current_dd_duration_days",
            "hit_rate", "turnover", "exposure_pct", "vs_tqqq_bh_excess_cagr",
        ]},
        "gap_return_pct":     result.gap_return_pct,
        "intraday_return_pct": result.intraday_return_pct,
        "brier_buy":  "",
        "brier_sell": "",
        "log_loss":   "",
        "ic_5d":      "",
        "notes":      notes,
    }

    write_header = not MASTER_LOG_PATH.exists() or MASTER_LOG_PATH.stat().st_size == 0
    with open(MASTER_LOG_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run TQQQ walk-forward backtest.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--tau",   default=_DEFAULT_TAU, type=float)
    parser.add_argument("--report", action="store_true", help="Print perf summary.")
    parser.add_argument("--no-regime", action="store_true")
    parser.add_argument("--notes", default="", help="Free-text note for MASTER_LOG.")
    parser.add_argument("--data-dir", default=None, help="Path to data/ directory.")
    args = parser.parse_args()

    from data import SYMBOLS_ALL
    try:
        if args.data_dir:
            panel = load_panel(SYMBOLS_ALL, data_dir=Path(args.data_dir), warn_missing=True)
        else:
            panel = load_panel(SYMBOLS_ALL, warn_missing=True)
    except ValueError as e:
        print(f"Error loading panel: {e}")
        return

    result = run_backtest(
        panel,
        tau=args.tau,
        use_regime=not args.no_regime,
        split=args.split,
    )

    if args.report:
        p = result.perf
        print(f"\n{'='*50}")
        print(f"  Split: {result.split}  |  CAGR: {p['cagr']:.2%}  |  Sharpe: {p['sharpe']:.2f}")
        print(f"  MaxDD: {p['maxdd_pct']:.1f}%  |  DD Duration: {p['maxdd_duration_days']}d")
        print(f"  Exposure: {p['exposure_pct']:.1f}%  |  vs TQQQ BaH: {p['vs_tqqq_bh_excess_cagr']:.2%}")
        print(f"  Gap: {result.gap_return_pct:.2f}%  |  Intraday: {result.intraday_return_pct:.2f}%")
        print(f"{'='*50}\n")

    append_master_log(result, tau=args.tau, notes=args.notes)
    print("Appended to MASTER_LOG.csv")


if __name__ == "__main__":
    _cli()
