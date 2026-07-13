"""Headless driver: execute the logic of each notebook (00-04) and print outputs.

This mirrors the cells of notebooks/*.py without invoking marimo's server.
Run: /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python run_all_notebooks.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data import load_panel, SYMBOLS_ALL
from metrics import REGISTRY
from regime import walk_forward_states
from backtest import run_backtest, _perf_stats

from quantcore import config as _qc_config


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Shared: load panel
# ---------------------------------------------------------------------------
banner("LOAD PANEL")
DATA_DIR = _qc_config.data_dir()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
print(f"Panel shape: {panel.shape}")
print(f"Date range : {panel.index[0].date()} → {panel.index[-1].date()}")
print(f"Columns    : {len(panel.columns)} columns across {len(set(c.split('_')[0] for c in panel.columns))} symbols")


# ---------------------------------------------------------------------------
# 01_data_sanity.py
# ---------------------------------------------------------------------------
banner("01 — DATA SANITY")

na_pct = panel.isna().mean() * 100
print("Missing values (% of rows) per column where >0:")
print(na_pct[na_pct > 0].round(2).to_string() if (na_pct > 0).any() else "  (none)")

# Coverage: per symbol first/last valid date
close_cols = [c for c in panel.columns if c.endswith("_close")]
print("\nPer-symbol coverage:")
for col in close_cols:
    sym = col.replace("_close", "")
    valid = panel[col].notna()
    if valid.any():
        first = panel.index[valid].min().date()
        last = panel.index[valid].max().date()
        n = int(valid.sum())
        print(f"  {sym:8s}: {first} → {last}  ({n} rows)")

# Split-adjustment sanity: >50% single-day jumps on equity ETFs
print("\nSplit-adjustment sanity (equity/ETF symbols only):")
EQUITY_SYMS = ["TQQQ", "QQQ", "SPY", "SPXL", "HYG", "LQD"]
for sym in EQUITY_SYMS:
    col = f"{sym}_close"
    if col not in panel.columns:
        continue
    series = panel[col]
    bad = series.pct_change(fill_method=None).abs()
    bad = bad[bad > 0.5]
    if len(bad) == 0:
        print(f"  PASS {sym:6s}: no >50% single-day jumps")
    else:
        print(f"  WARN {sym:6s}: {len(bad)} jump(s) >50%")
        for d, v in bad.items():
            print(f"        {d.date()}: {v*100:.1f}%")

# Correlation matrix
print("\nReturn correlation matrix (close-to-close):")
rets = panel[close_cols].pct_change(fill_method=None).dropna()
corr = rets.corr().round(2)
print(corr.to_string())


# ---------------------------------------------------------------------------
# 00_v0_proof.py
# ---------------------------------------------------------------------------
banner("00 — V0 PROOF (3 metrics, majority vote vs TQQQ BaH)")
V0_METRICS = ["qqq_sma50_200_regime", "qqq_rsi2", "qqq_yz_vol_20d"]
votes = {}
for name in V0_METRICS:
    m = REGISTRY[name]
    s = m.compute(panel)
    votes[name] = m.vote(s)

vote_df = pd.DataFrame(votes)
score = vote_df.mean(axis=1)
signal = (score > 0).astype(int)

print("Per-metric vote distribution:")
for name in V0_METRICS:
    vc = vote_df[name].value_counts().sort_index()
    print(f"  {name:30s}: {dict(vc)}")

print(f"\nSignal=1 (long) days: {int(signal.sum())} / {len(signal)} ({signal.mean()*100:.1f}%)")

COST_BPS = 5.0
tqqq_open = panel.get("TQQQ_open")
if tqqq_open is not None:
    position = signal.shift(1).fillna(0).astype(int)
    pos_change = position.diff().abs().fillna(0)
    cost = pos_change * (COST_BPS / 10_000)
    daily_ret = tqqq_open.pct_change(fill_method=None).fillna(0)
    strat_ret = position * daily_ret - cost
    equity = (1 + strat_ret).cumprod()
    bah = (1 + daily_ret).cumprod()
    p = _perf_stats(equity)
    p_bah = _perf_stats(bah)

    print(f"\n  V0 Strategy | CAGR: {p['cagr']:.2%}  Sharpe: {p['sharpe']:.2f}  "
          f"MaxDD: {p['maxdd_pct']:.1f}%  DDdur: {p['maxdd_duration_days']}d  "
          f"Exposure: {position.mean()*100:.1f}%")
    print(f"  TQQQ BaH    | CAGR: {p_bah['cagr']:.2%}  Sharpe: {p_bah['sharpe']:.2f}  "
          f"MaxDD: {p_bah['maxdd_pct']:.1f}%  DDdur: {p_bah['maxdd_duration_days']}d")

    # Save equity series for cross-checking
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"v0_equity": equity, "tqqq_bah": bah}).to_csv(out_dir / "v0_equity.csv")
    print(f"\n  Saved → outputs/v0_equity.csv")


# ---------------------------------------------------------------------------
# 02_metric_inspection.py — IC table for all metrics
# ---------------------------------------------------------------------------
banner("02 — METRIC INSPECTION (IC table)")
from scipy.stats import spearmanr

close = panel.get("QQQ_close")
fwd_rets = {
    "ic_1d":  close.pct_change(1, fill_method=None).shift(-1),
    "ic_5d":  close.pct_change(5, fill_method=None).shift(-5),
    "ic_20d": close.pct_change(20, fill_method=None).shift(-20),
}

rows = []
for name, m in REGISTRY.items():
    s = m.compute(panel).dropna()
    row = {"metric": name, "family": m.family, "status": m.status}
    for ic_name, fwd in fwd_rets.items():
        common = s.index.intersection(fwd.dropna().index)
        if len(common) < 30:
            row[ic_name] = np.nan
            row[ic_name + "_p"] = np.nan
        else:
            ic, pval = spearmanr(s.loc[common].values, fwd.loc[common].values)
            row[ic_name] = round(ic, 4)
            row[ic_name + "_p"] = round(pval, 4)
    rows.append(row)

ic_df = pd.DataFrame(rows).set_index("metric")
print(ic_df.to_string())

# Benjamini-Hochberg correction at FDR=0.10 across voting metrics on 5d IC
voting = ic_df[ic_df["status"] == "voting"].copy()
pvals = voting["ic_5d_p"].dropna().sort_values()
m_tests = len(pvals)
alpha = 0.10
bh_thresholds = (np.arange(1, m_tests + 1) / m_tests) * alpha
sig_mask = pvals.values <= bh_thresholds
# largest k such that pval[k] <= bh_thresholds[k]
if sig_mask.any():
    k_max = np.where(sig_mask)[0].max()
    sig_metrics = pvals.index[: k_max + 1].tolist()
else:
    sig_metrics = []
print(f"\nBH-FDR correction (alpha=0.10) across {m_tests} voting metrics on ic_5d:")
print(f"  Significant: {sig_metrics if sig_metrics else '(none)'}")

ic_df.to_csv(ROOT / "outputs" / "ic_table.csv")
print(f"  Saved → outputs/ic_table.csv")


# ---------------------------------------------------------------------------
# 03_regime.py
# ---------------------------------------------------------------------------
banner("03 — HSMM5 WALK-FORWARD REGIME")
print("Fitting HSMM5 walk-forward (min_train_years=3, refit annually)...")
state_series, proba_df = walk_forward_states(panel, min_train_years=3)

print("State distribution (out-of-sample):")
vc = state_series[state_series >= 0].value_counts().sort_index()
LABELS = ["strong_bull", "weak_bull", "sideways", "weak_bear", "strong_bear"]
total = vc.sum()
for state_id, count in vc.items():
    label = LABELS[state_id] if 0 <= state_id < 5 else f"state_{state_id}"
    print(f"  {state_id} {label:13s}: {count:5d} days ({count/total*100:5.1f}%)")
warmup = (state_series == -1).sum()
print(f"  warmup (no fit) :  {warmup} days")

print("\nConditional QQQ daily-return stats per state (out-of-sample):")
qqq_ret = panel["QQQ_close"].pct_change(fill_method=None)
print(f"  {'state':<6}{'label':<14}{'count':>7}{'mean_%':>10}{'std_%':>9}{'sharpe':>9}")
for i in range(5):
    mask = state_series == i
    r = qqq_ret[mask].dropna()
    if len(r) > 0:
        sh = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
        print(f"  {i:<6}{LABELS[i]:<14}{len(r):>7d}{r.mean()*100:>10.4f}{r.std()*100:>9.4f}{sh:>9.2f}")
    else:
        print(f"  {i:<6}{LABELS[i]:<14}{0:>7d}{'--':>10}{'--':>9}{'--':>9}")

state_series.to_csv(ROOT / "outputs" / "regime_states.csv")
proba_df.to_csv(ROOT / "outputs" / "regime_probas.csv")
print(f"\n  Saved → outputs/regime_states.csv, outputs/regime_probas.csv")


# ---------------------------------------------------------------------------
# 04_strategy.py — full backtest on train + val
# ---------------------------------------------------------------------------
banner("04 — FULL STRATEGY: walk-forward backtest")

for split in ("train", "val"):
    print(f"\n--- split: {split} (tau=1.0, use_regime=True) ---")
    # Filter panel by split window
    if split == "train":
        sub = panel.loc[:"2017-12-31"]
    elif split == "val":
        sub = panel.loc["2018-01-01":"2021-12-31"]
    else:
        sub = panel.loc["2022-01-01":]
    print(f"  Panel rows: {len(sub)}  ({sub.index[0].date()} → {sub.index[-1].date()})")

    result = run_backtest(sub, tau=1.0, use_regime=True, split=split)
    p = result.perf
    print(f"  CAGR    : {p['cagr']:.2%}")
    print(f"  Sharpe  : {p['sharpe']:.2f}")
    print(f"  Sortino : {p['sortino']:.2f}")
    print(f"  Calmar  : {p['calmar']:.2f}")
    print(f"  MaxDD   : {p['maxdd_pct']:.1f}%  (dur {p['maxdd_duration_days']}d)")
    print(f"  Hit rate: {p.get('hit_rate', float('nan')):.1%}")
    print(f"  Turnover: {p['turnover']:.4f}/day  (avg ~{p['turnover']*252:.0f}/yr)")
    print(f"  Exposure: {p['exposure_pct']:.1f}%")
    print(f"  vs TQQQ BaH excess CAGR: {p['vs_tqqq_bh_excess_cagr']:+.2%}")
    print(f"  Gap return : {result.gap_return_pct:+.2f}%   Intraday: {result.intraday_return_pct:+.2f}%")

    # Final equity vs benchmark
    eq_final = result.equity.iloc[-1]
    bm_final = result.benchmark_tqqq.iloc[-1]
    print(f"  Final NAV: strategy={eq_final:.3f}  TQQQ BaH={bm_final:.3f}")
    print(f"  Action distribution: {result.signals['action'].value_counts().to_dict()}")
    print(f"  Trades: {len(result.trade_ledger)}")
    if len(result.trade_ledger) > 0:
        print(f"  Avg trade pnl: {result.trade_ledger['pnl_pct'].mean():.2f}%  "
              f"median: {result.trade_ledger['pnl_pct'].median():.2f}%")

    # Persist equity for inspection
    pd.DataFrame({
        "equity": result.equity,
        "tqqq_bah": result.benchmark_tqqq,
        "qqq_bah":  result.benchmark_qqq,
        "position": result.positions,
    }).to_csv(ROOT / "outputs" / f"strategy_{split}.csv")
    print(f"  Saved → outputs/strategy_{split}.csv")

print("\nDONE.")
