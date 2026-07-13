"""V2 research strategy: weighted metric votes -> long/flat exposure.

This module is deliberately simpler than the v1 softmax/regime strategy. It
uses train-only signal diagnostics to choose metric direction and weight, then
reports validation results without touching the frozen test set.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import COST_BPS, _gap_intraday_decomp, _perf_stats
from data import SYMBOLS_ALL, load_panel
from metrics import REGISTRY, vote_all
from signal_diagnostics import (
    TRAIN_END,
    VAL_END,
    VAL_START,
    metric_decision_table,
    split_panel,
)


@dataclass(frozen=True)
class V2MetricConfig:
    metric: str
    direction: int
    weight: float


@dataclass
class V2BacktestResult:
    equity: pd.Series
    benchmark_tqqq: pd.Series
    benchmark_qqq: pd.Series
    fixed_25_tqqq: pd.Series
    fixed_50_tqqq: pd.Series
    exposure: pd.Series
    signals: pd.DataFrame
    perf: dict
    benchmark_perf: pd.DataFrame
    gap_return_pct: float
    intraday_return_pct: float
    split: str
    mode: str
    threshold: float
    medium_threshold: float


def configs_from_decision_table(decisions: pd.DataFrame) -> list[V2MetricConfig]:
    """Convert keep/invert rows into strategy configs."""
    usable = decisions.loc[decisions["decision"].isin(["keep", "invert"])].copy()
    usable = usable.loc[(usable["direction"] != 0) & (usable["weight"] > 0)]
    return [
        V2MetricConfig(
            metric=str(row["metric"]),
            direction=int(row["direction"]),
            weight=float(row["weight"]),
        )
        for _, row in usable.iterrows()
    ]


def weighted_vote_signals(panel: pd.DataFrame, configs: list[V2MetricConfig]) -> pd.DataFrame:
    """Return weighted score series using causal metric votes."""
    if not configs:
        raise ValueError("At least one V2MetricConfig is required.")

    names = [c.metric for c in configs]
    missing = [name for name in names if name not in REGISTRY]
    if missing:
        raise KeyError(f"Unknown metric(s): {missing}")

    votes = vote_all(panel, names=names).reindex(panel.index).fillna(0).astype(float)
    weighted = pd.DataFrame(index=panel.index)
    total_weight = 0.0
    score = pd.Series(0.0, index=panel.index)
    for cfg in configs:
        contribution = votes[cfg.metric] * cfg.direction * cfg.weight
        weighted[cfg.metric] = contribution
        score = score + contribution
        total_weight += cfg.weight

    score = score / total_weight if total_weight > 0 else score
    result = pd.DataFrame({"score": score}, index=panel.index)
    return result.join(weighted)


def exposure_from_score(
    score: pd.Series,
    *,
    mode: str,
    threshold: float,
    medium_threshold: float = 0.0,
) -> pd.Series:
    """Map score to binary or ternary target exposure."""
    if mode == "binary":
        exposure = (score >= threshold).astype(float)
    elif mode == "ternary":
        exposure = pd.Series(0.0, index=score.index)
        exposure.loc[score >= medium_threshold] = 0.5
        exposure.loc[score >= threshold] = 1.0
    else:
        raise ValueError("mode must be 'binary' or 'ternary'.")
    return exposure.rename("exposure")


def _benchmark_nav(ret: pd.Series, exposure: float = 1.0) -> pd.Series:
    return (1 + ret.fillna(0) * exposure).cumprod()


def _perf_table(series_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, equity in series_by_name.items():
        row = {"name": name}
        row.update(_perf_stats(equity))
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")


def _realized_returns(price: pd.Series) -> pd.Series:
    """Return price[t] / price[t-1] - 1, with the first NAV row anchored at 1."""
    return price.pct_change(fill_method=None).fillna(0)


def run_v2_backtest(
    panel: pd.DataFrame,
    configs: list[V2MetricConfig],
    *,
    mode: str = "binary",
    threshold: float = 0.1,
    medium_threshold: float = 0.0,
    split: str = "unknown",
    evaluation_index: pd.Index | None = None,
) -> V2BacktestResult:
    """Run v2 backtest with close[t] signal filled at open[t+1].

    If ``evaluation_index`` is provided, metrics are computed on ``panel`` to
    preserve pre-split rolling history, then performance is measured only on
    the selected dates.
    """
    if "TQQQ_open" not in panel.columns or "TQQQ_close" not in panel.columns:
        raise ValueError("Panel must contain TQQQ_open and TQQQ_close.")
    if "QQQ_close" not in panel.columns:
        raise ValueError("Panel must contain QQQ_close.")

    first_valid = panel["TQQQ_open"].first_valid_index()
    if first_valid is not None:
        panel = panel.loc[first_valid:].copy()

    signals = weighted_vote_signals(panel, configs)
    raw_exposure = exposure_from_score(
        signals["score"],
        mode=mode,
        threshold=threshold,
        medium_threshold=medium_threshold,
    )
    exposure = raw_exposure.shift(1).fillna(0.0).rename("exposure")
    signals = signals.join(raw_exposure.rename("target_exposure"))
    signals["action"] = np.where(raw_exposure > 0, "buy", "hold")

    if evaluation_index is not None:
        eval_index = panel.index.intersection(evaluation_index)
        if eval_index.empty:
            raise ValueError("evaluation_index has no overlap with panel.")
        panel = panel.loc[eval_index].copy()
        signals = signals.loc[eval_index].copy()
        exposure = exposure.loc[eval_index].copy()

    tqqq_open = panel["TQQQ_open"]
    tqqq_close = panel["TQQQ_close"]
    qqq_close = panel["QQQ_close"]

    # Exposure at date t is the position after the open[t] fill. It first earns
    # the open[t] -> open[t+1] return, recorded on t+1.
    tqqq_ret = _realized_returns(tqqq_open)
    qqq_ret = _realized_returns(qqq_close)

    pos_change = exposure.diff().abs().fillna(0)
    if len(pos_change):
        pos_change.iloc[0] = abs(float(exposure.iloc[0]))
    cost_series = pos_change * (COST_BPS / 10_000)
    interval_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = interval_exposure * tqqq_ret - cost_series
    equity = (1 + strategy_ret).cumprod().rename("strategy_v2")

    bah_tqqq = _benchmark_nav(tqqq_ret).rename("tqqq_bah")
    bah_qqq = _benchmark_nav(qqq_ret).rename("qqq_bah")
    fixed_25 = _benchmark_nav(tqqq_ret, 0.25).rename("tqqq_25")
    fixed_50 = _benchmark_nav(tqqq_ret, 0.50).rename("tqqq_50")

    perf = _perf_stats(equity)
    perf["turnover"] = float(pos_change.mean())
    perf["exposure_pct"] = float(exposure.mean() * 100)
    perf["trade_count"] = int((pos_change > 0).sum())
    perf["vs_tqqq_bh_excess_cagr"] = round(perf["cagr"] - _perf_stats(bah_tqqq)["cagr"], 6)
    perf["vs_qqq_bh_excess_cagr"] = round(perf["cagr"] - _perf_stats(bah_qqq)["cagr"], 6)

    gap, intraday = _gap_intraday_decomp(tqqq_open, tqqq_close, exposure)
    benchmark_perf = _perf_table({
        "strategy_v2": equity,
        "qqq_bah": bah_qqq,
        "tqqq_bah": bah_tqqq,
        "tqqq_25": fixed_25,
        "tqqq_50": fixed_50,
    })

    return V2BacktestResult(
        equity=equity,
        benchmark_tqqq=bah_tqqq,
        benchmark_qqq=bah_qqq,
        fixed_25_tqqq=fixed_25,
        fixed_50_tqqq=fixed_50,
        exposure=exposure,
        signals=signals,
        perf=perf,
        benchmark_perf=benchmark_perf,
        gap_return_pct=round(gap * 100, 4),
        intraday_return_pct=round(intraday * 100, 4),
        split=split,
        mode=mode,
        threshold=threshold,
        medium_threshold=medium_threshold,
    )


def select_threshold_on_train(
    panel_train: pd.DataFrame,
    configs: list[V2MetricConfig],
) -> dict[str, float | str]:
    """Select a small, predeclared exposure rule on train only."""
    candidates: list[dict[str, float | str]] = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20):
        candidates.append({"mode": "binary", "threshold": threshold, "medium_threshold": 0.0})
    for medium, high in ((0.0, 0.10), (0.05, 0.15), (0.10, 0.20)):
        candidates.append({"mode": "ternary", "threshold": high, "medium_threshold": medium})

    rows = []
    for candidate in candidates:
        result = run_v2_backtest(
            panel_train,
            configs,
            mode=str(candidate["mode"]),
            threshold=float(candidate["threshold"]),
            medium_threshold=float(candidate["medium_threshold"]),
            split="train",
        )
        row = dict(candidate)
        row.update(result.perf)
        rows.append(row)

    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["trade_count"] >= 5].copy()
    if eligible.empty:
        eligible = grid
    eligible = eligible.sort_values(
        ["sharpe", "maxdd_pct", "cagr"],
        ascending=[False, False, False],
    )
    return eligible.iloc[0][["mode", "threshold", "medium_threshold"]].to_dict()


def _result_frame(result: V2BacktestResult) -> pd.DataFrame:
    return pd.DataFrame({
        "equity": result.equity,
        "tqqq_bah": result.benchmark_tqqq,
        "qqq_bah": result.benchmark_qqq,
        "tqqq_25": result.fixed_25_tqqq,
        "tqqq_50": result.fixed_50_tqqq,
        "exposure": result.exposure,
        "score": result.signals["score"],
        "target_exposure": result.signals["target_exposure"],
    })


def _memo_text(
    decisions: pd.DataFrame,
    comparison: pd.DataFrame,
    train_result: V2BacktestResult,
    val_result: V2BacktestResult,
    selected: dict[str, float | str],
) -> str:
    kept = decisions.loc[decisions["decision"].isin(["keep", "invert"]), [
        "metric", "decision", "weight", "tqqq_edge_bps_train", "tqqq_edge_bps_val"
    ]]
    dropped = decisions.loc[decisions["decision"] == "drop", ["metric"]]

    def _perf_line(label: str, result: V2BacktestResult) -> str:
        p = result.perf
        return (
            f"| {label} | {p['cagr']:.2%} | {p['sharpe']:.2f} | "
            f"{p['maxdd_pct']:.1f}% | {p['maxdd_duration_days']} | "
            f"{p['exposure_pct']:.1f}% | {p['turnover']:.4f} | "
            f"{p['trade_count']} | {p['vs_tqqq_bh_excess_cagr']:.2%} |"
        )

    compare_text = comparison.to_string(index=False) if not comparison.empty else "(no comparison rows)"

    return f"""# ETF Strategy Research Memo

## Hypothesis

    A smaller set of train-selected metric votes, with harmful vote directions inverted, should provide a cleaner TQQQ timing signal than the current equal-weight softmax ensemble.

## Motivation

The current v1 strategy uses many votes with equal influence plus regime machinery. The visual and IC work suggests some metrics have useful information, but direction and stability vary a lot. V2 tests whether a smaller evidence-gated signal set can improve timing discipline before adding more model complexity.

## Method

    - Signal selection, direction, and weights used train only.
    - Validation is used only for evaluation/context.
    - Frozen test window remains unused: 2022-01-01 onward.
- Decisions are based on TQQQ next-open tradable forward returns.
- Selected exposure rule on train only: `{selected}`.
- HSMM regime gating is disabled for v2.
- Fill rule: close[t] signal fills at open[t+1].
- Costs: `{COST_BPS}` bps on each position change.

## Metric Decisions

Kept or inverted:

```text
{kept.to_string(index=False) if not kept.empty else '(none)'}
```

Dropped:

```text
{', '.join(dropped['metric'].tolist()) if not dropped.empty else '(none)'}
```

## Results

| Split | CAGR | Sharpe | MaxDD | DD Duration | Exposure | Turnover | Trades | vs TQQQ B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{_perf_line('Train', train_result)}
{_perf_line('Val', val_result)}

Comparison:

```text
{compare_text}
```

## Benchmark Tables

Train:

```text
{train_result.benchmark_perf.to_string()}
```

Validation:

```text
{val_result.benchmark_perf.to_string()}
```

## Failure Modes

- The selected metric set is still partly broad and may be a proxy for beta reduction.
- Validation CAGR remains weak, so this is not an investable candidate.
- Threshold choice is deliberately simple; better validation requires a clearer signal, not just more tuning.
- Several inverted metrics are plausible warnings that the current vote definitions need finance review.
- The frozen test set was not used, so there is still no final OOS claim.

## Decision

Decision: Revise.

Next action: narrow the signal set further, starting with the metrics whose train/validation direction and finance intuition agree, then retest with the same split discipline.
"""


def _existing_v1_perf(output_dir: Path, split: str) -> dict[str, float | str] | None:
    path = output_dir / f"strategy_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if "equity" not in df:
        return None
    row: dict[str, float | str] = {"split": split, "name": "v1_current_strategy"}
    row.update(_perf_stats(df["equity"]))
    if "position" in df:
        row["exposure_pct"] = float(df["position"].mean() * 100)
        row["trade_count"] = int(df["position"].diff().abs().fillna(0).gt(0).sum())
    return row


def _comparison_table(
    output_dir: Path,
    train_result: V2BacktestResult,
    val_result: V2BacktestResult,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for split, result in (("train", train_result), ("val", val_result)):
        v1 = _existing_v1_perf(output_dir, split)
        if v1 is not None:
            rows.append(v1)

        strategy_row: dict[str, float | str] = {"split": split, "name": "v2_weighted_strategy"}
        strategy_row.update(result.perf)
        rows.append(strategy_row)

        for name, equity in {
            "qqq_buy_hold": result.benchmark_qqq,
            "tqqq_buy_hold": result.benchmark_tqqq,
            "tqqq_fixed_25": result.fixed_25_tqqq,
            "tqqq_fixed_50": result.fixed_50_tqqq,
        }.items():
            row: dict[str, float | str] = {"split": split, "name": name}
            row.update(_perf_stats(equity))
            rows.append(row)

    return pd.DataFrame(rows)


def _backtest_report_text(comparison: pd.DataFrame) -> str:
    display = comparison.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].round(4)
    return f"""# V2 Backtest Report

This report compares train and validation results only. The frozen 2022-01-01 onward test window is untouched.

```text
{display.to_string(index=False)}
```
"""


def run_train_val_research(
    panel: pd.DataFrame,
    *,
    output_dir: Path = Path("outputs"),
    horizon: int = 5,
) -> dict[str, object]:
    """Generate train/validation-only v2 research artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = metric_decision_table(panel, horizon=horizon)
    decisions.to_csv(output_dir / f"signal_credibility_{horizon}d_train_val.csv", index=False)

    configs = configs_from_decision_table(decisions)
    if not configs:
        raise ValueError("No keep/invert metrics selected; cannot run v2 strategy.")

    train_panel = split_panel(panel, "train")
    val_panel = split_panel(panel, "val")
    selected = select_threshold_on_train(train_panel, configs)

    train_history = panel.loc[:TRAIN_END].copy()
    val_history = panel.loc[:VAL_END].copy()

    train_result = run_v2_backtest(
        train_history,
        configs,
        mode=str(selected["mode"]),
        threshold=float(selected["threshold"]),
        medium_threshold=float(selected["medium_threshold"]),
        split="train",
        evaluation_index=train_panel.index,
    )
    val_result = run_v2_backtest(
        val_history,
        configs,
        mode=str(selected["mode"]),
        threshold=float(selected["threshold"]),
        medium_threshold=float(selected["medium_threshold"]),
        split="val",
        evaluation_index=val_panel.index,
    )

    _result_frame(train_result).to_csv(output_dir / "strategy_v2_train.csv", index_label="date")
    _result_frame(val_result).to_csv(output_dir / "strategy_v2_val.csv", index_label="date")
    train_result.benchmark_perf.to_csv(output_dir / "strategy_v2_train_benchmarks.csv")
    val_result.benchmark_perf.to_csv(output_dir / "strategy_v2_val_benchmarks.csv")
    comparison = _comparison_table(output_dir, train_result, val_result)
    comparison.to_csv(output_dir / "strategy_v2_comparison.csv", index=False)
    (output_dir / "strategy_v2_backtest_report.md").write_text(_backtest_report_text(comparison))

    config_payload = {
        "horizon": horizon,
        "selected_rule": selected,
        "selection_basis": "train_only",
        "metrics": [cfg.__dict__ for cfg in configs],
        "train_window": f"<= {TRAIN_END.date()}",
        "val_window": f"{VAL_START.date()} to {VAL_END.date()}",
        "frozen_test_window": "2022-01-01 onward",
    }
    (output_dir / "strategy_v2_config.json").write_text(json.dumps(config_payload, indent=2))
    (output_dir / "strategy_v2_research_memo.md").write_text(
        _memo_text(decisions, comparison, train_result, val_result, selected)
    )

    return {
        "decisions": decisions,
        "configs": configs,
        "selected": selected,
        "train": train_result,
        "val": val_result,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run train/validation-only v2 TQQQ research.")
    parser.add_argument("--data-dir", default=None, help="Path to data directory.")
    parser.add_argument("--output-dir", default="outputs", help="Output artifact directory.")
    parser.add_argument("--horizon", default=5, type=int)
    args = parser.parse_args()

    from quantcore import config as _qc_config
    data_dir = Path(args.data_dir) if args.data_dir else _qc_config.data_dir()
    panel = load_panel(SYMBOLS_ALL, data_dir=data_dir, warn_missing=True)
    artifacts = run_train_val_research(panel, output_dir=Path(args.output_dir), horizon=args.horizon)
    train = artifacts["train"]
    val = artifacts["val"]
    assert isinstance(train, V2BacktestResult)
    assert isinstance(val, V2BacktestResult)
    print("Selected rule:", artifacts["selected"])
    print("Metrics:", [cfg.metric for cfg in artifacts["configs"]])
    print("Train:", train.perf)
    print("Val:", val.perf)


if __name__ == "__main__":
    _cli()
