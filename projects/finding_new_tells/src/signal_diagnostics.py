"""Signal credibility diagnostics for ETF strategy research.

The goal is to rank metrics by evidence quality, not by one lucky backtest.
Diagnostics compare vote buckets against future QQQ returns and can contrast
train vs validation behavior without touching the frozen test window.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from data import SYMBOLS_ALL, load_panel
from metrics import REGISTRY
from quantcore.validation import split_by_date as _qcv_split


TRAIN_END = pd.Timestamp("2017-12-31")
VAL_START = pd.Timestamp("2018-01-01")
VAL_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

DEFAULT_HORIZONS = (1, 5, 20)
RESEARCH_HORIZONS = (1, 2, 5, 10, 20)


@dataclass
class MetricForwardProfile:
    """Single-metric forward-return diagnostics for research notebooks."""

    metric_name: str
    split: str
    target_symbol: str
    target_kind: str
    horizons: tuple[int, ...]
    aligned: pd.DataFrame
    ic_table: pd.DataFrame
    vote_bucket_table: pd.DataFrame
    quantile_table: pd.DataFrame
    rolling_ic: pd.DataFrame
    event_paths: pd.DataFrame
    latest: dict
    warnings: list[str]


@dataclass
class PairMetricProfile:
    """Two-metric conditional diagnostics for hypothesis generation."""

    primary_metric: str
    filter_metric: str
    split: str
    target_symbol: str
    target_kind: str
    horizons: tuple[int, ...]
    condition_table: pd.DataFrame
    masks: pd.DataFrame
    warnings: list[str]


def split_panel(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return a train/val/test/all slice using the repo holdout convention."""
    if split == "all":
        return panel.copy()
    _train, _val, _test = _qcv_split(panel, train_end=TRAIN_END, val_end=VAL_END)
    mapping = {"train": _train, "val": _val, "test": _test}
    if split not in mapping:
        raise ValueError(f"Unknown split {split!r}; expected train, val, test, or all.")
    return mapping[split]


def research_split_panel(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return train/validation research slices without exposing frozen test rows."""
    normalized = split.replace("_", "+").lower()
    if normalized == "train":
        return split_panel(panel, "train")
    if normalized == "val":
        return split_panel(panel, "val")
    if normalized in {"train+val", "trainval"}:
        return panel.loc[:VAL_END].copy()
    raise ValueError("Research diagnostics only support 'train', 'val', or 'train+val'.")


def forward_returns(
    panel: pd.DataFrame,
    *,
    symbol: str = "QQQ",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[int, pd.Series]:
    """Compute forward close-to-close simple returns for one symbol."""
    close_name = f"{symbol}_close"
    if close_name not in panel.columns:
        raise ValueError(f"Panel must contain {close_name!r}.")

    close = panel[close_name]
    return {
        int(h): (close.shift(-int(h)) / close - 1).rename(f"{symbol}_fwd_{int(h)}d")
        for h in horizons
    }


def tradable_open_forward_returns(
    panel: pd.DataFrame,
    *,
    symbol: str = "TQQQ",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[int, pd.Series]:
    """Compute future open-to-open returns available after a close-time signal.

    A signal observed at close[t] can first trade at open[t+1]. For horizon h,
    the return is open[t+1+h] / open[t+1] - 1.
    """
    open_name = f"{symbol}_open"
    if open_name not in panel.columns:
        raise ValueError(f"Panel must contain {open_name!r}.")

    open_ = panel[open_name]
    return {
        int(h): (open_.shift(-(int(h) + 1)) / open_.shift(-1) - 1).rename(
            f"{symbol}_tradable_fwd_{int(h)}d"
        )
        for h in horizons
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 30:
        return np.nan, np.nan
    if x.loc[common].nunique(dropna=True) < 2 or y.loc[common].nunique(dropna=True) < 2:
        return np.nan, np.nan
    ic, pval = spearmanr(x.loc[common].values, y.loc[common].values)
    return float(ic), float(pval)


def _target_forward_returns(
    panel: pd.DataFrame,
    *,
    symbol: str,
    target_kind: str,
    horizons: Iterable[int],
) -> dict[int, pd.Series]:
    if target_kind == "close":
        return forward_returns(panel, symbol=symbol, horizons=horizons)
    if target_kind == "tradable_open":
        return tradable_open_forward_returns(panel, symbol=symbol, horizons=horizons)
    raise ValueError("target_kind must be 'close' or 'tradable_open'.")


def _vote_bucket_stats(votes: pd.Series, fwd: pd.Series) -> dict[str, float]:
    aligned = pd.concat({"vote": votes, "fwd": fwd}, axis=1).dropna()
    stats: dict[str, float] = {"n": float(len(aligned))}

    for vote, label in ((1, "bull"), (0, "neutral"), (-1, "bear")):
        bucket = aligned.loc[aligned["vote"] == vote, "fwd"]
        stats[f"n_{label}"] = float(len(bucket))
        stats[f"mean_{label}"] = float(bucket.mean()) if len(bucket) else np.nan
        stats[f"hit_{label}"] = float((bucket > 0).mean()) if len(bucket) else np.nan

    bull = stats["mean_bull"]
    bear = stats["mean_bear"]
    stats["bull_minus_bear"] = (
        float(bull - bear) if not np.isnan(bull) and not np.isnan(bear) else np.nan
    )
    return stats


def _rolling_spearman(values: pd.Series, fwd: pd.Series, *, horizon: int, window: int = 252) -> pd.DataFrame:
    aligned = pd.concat({"metric": values, "fwd": fwd}, axis=1)
    rows = []
    min_obs = min(max(window // 4, 30), window)
    for end in range(min_obs - 1, len(aligned)):
        sample = aligned.iloc[max(0, end - window + 1):end + 1].dropna()
        if len(sample) < min_obs:
            continue
        ic, _ = _safe_spearman(sample["metric"], sample["fwd"])
        rows.append({
            "date": aligned.index[end],
            "horizon": int(horizon),
            "rolling_raw_ic": ic,
            "n": len(sample),
        })
    return pd.DataFrame(rows)


def _vote_bucket_table(votes: pd.Series, fwd_by_horizon: dict[int, pd.Series]) -> pd.DataFrame:
    rows = []
    labels = {1: "bull", 0: "neutral", -1: "bear"}
    for horizon, fwd in fwd_by_horizon.items():
        aligned = pd.concat({"vote": votes, "fwd": fwd}, axis=1).dropna()
        for vote in (1, 0, -1):
            bucket = aligned.loc[aligned["vote"] == vote, "fwd"]
            rows.append({
                "horizon": int(horizon),
                "vote": vote,
                "label": labels[vote],
                "n": len(bucket),
                "mean_fwd": float(bucket.mean()) if len(bucket) else np.nan,
                "mean_fwd_bps": float(bucket.mean() * 10_000) if len(bucket) else np.nan,
                "hit_rate": float((bucket > 0).mean()) if len(bucket) else np.nan,
            })
        bull = aligned.loc[aligned["vote"] == 1, "fwd"]
        bear = aligned.loc[aligned["vote"] == -1, "fwd"]
        rows.append({
            "horizon": int(horizon),
            "vote": 99,
            "label": "bull_minus_bear",
            "n": min(len(bull), len(bear)),
            "mean_fwd": float(bull.mean() - bear.mean()) if len(bull) and len(bear) else np.nan,
            "mean_fwd_bps": float((bull.mean() - bear.mean()) * 10_000) if len(bull) and len(bear) else np.nan,
            "hit_rate": np.nan,
        })
    return pd.DataFrame(rows)


def _quantile_bucket_table(values: pd.Series, fwd_by_horizon: dict[int, pd.Series], *, bins: int) -> pd.DataFrame:
    rows = []
    for horizon, fwd in fwd_by_horizon.items():
        aligned = pd.concat({"metric": values, "fwd": fwd}, axis=1).dropna()
        if aligned["metric"].nunique(dropna=True) < 2 or len(aligned) < max(10, bins):
            rows.append({
                "horizon": int(horizon),
                "quantile": np.nan,
                "n": len(aligned),
                "metric_min": np.nan,
                "metric_max": np.nan,
                "metric_mean": np.nan,
                "mean_fwd": np.nan,
                "mean_fwd_bps": np.nan,
                "hit_rate": np.nan,
            })
            continue
        try:
            quantiles = pd.qcut(aligned["metric"], q=bins, labels=False, duplicates="drop")
        except ValueError:
            quantiles = pd.Series(np.nan, index=aligned.index)
        aligned = aligned.assign(quantile=quantiles)
        for q, bucket in aligned.dropna(subset=["quantile"]).groupby("quantile"):
            rows.append({
                "horizon": int(horizon),
                "quantile": int(q) + 1,
                "n": len(bucket),
                "metric_min": float(bucket["metric"].min()),
                "metric_max": float(bucket["metric"].max()),
                "metric_mean": float(bucket["metric"].mean()),
                "mean_fwd": float(bucket["fwd"].mean()),
                "mean_fwd_bps": float(bucket["fwd"].mean() * 10_000),
                "hit_rate": float((bucket["fwd"] > 0).mean()),
            })
    return pd.DataFrame(rows)


def _event_study_paths(
    panel: pd.DataFrame,
    votes: pd.Series,
    *,
    symbol: str,
    target_kind: str,
    max_horizon: int,
) -> pd.DataFrame:
    price_col = f"{symbol}_open" if target_kind == "tradable_open" else f"{symbol}_close"
    if price_col not in panel.columns:
        raise ValueError(f"Panel must contain {price_col!r}.")
    price = panel[price_col]
    rows = []
    for vote, label in ((1, "bull"), (-1, "bear")):
        event_dates = votes.index[votes == vote]
        paths = []
        for date in event_dates:
            loc = price.index.get_indexer([date])[0]
            entry_loc = loc + 1 if target_kind == "tradable_open" else loc
            exit_loc = entry_loc + max_horizon
            if entry_loc < 0 or exit_loc >= len(price):
                continue
            base = price.iloc[entry_loc]
            if pd.isna(base) or base == 0:
                continue
            path = [
                price.iloc[entry_loc + step] / base - 1
                for step in range(max_horizon + 1)
            ]
            paths.append(path)
        if not paths:
            for step in range(max_horizon + 1):
                rows.append({"vote": vote, "label": label, "step": step, "n_events": 0, "mean_return": np.nan, "hit_rate": np.nan})
            continue
        arr = np.asarray(paths, dtype=float)
        for step in range(max_horizon + 1):
            vals = pd.Series(arr[:, step]).dropna()
            rows.append({
                "vote": vote,
                "label": label,
                "step": step,
                "n_events": len(vals),
                "mean_return": float(vals.mean()) if len(vals) else np.nan,
                "hit_rate": float((vals > 0).mean()) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def metric_forward_profile(
    panel: pd.DataFrame,
    metric_name: str,
    *,
    split: str = "train",
    horizons: Iterable[int] = RESEARCH_HORIZONS,
    target_symbol: str = "TQQQ",
    target_kind: str = "tradable_open",
    bins: int = 10,
    rolling_window: int = 252,
) -> MetricForwardProfile:
    """Build raw, vote, quantile, rolling-IC, and event diagnostics for one metric.

    This helper is intentionally train/validation-only and refuses the frozen
    test split.
    """
    if metric_name not in REGISTRY:
        raise KeyError(f"Metric {metric_name!r} not in REGISTRY.")
    horizons_tuple = tuple(int(h) for h in horizons)
    if not horizons_tuple:
        raise ValueError("At least one horizon is required.")

    panel_split = research_split_panel(panel, split)
    if panel_split.empty:
        raise ValueError(f"No rows available for split {split!r}.")

    metric = REGISTRY[metric_name]
    values = metric.compute(panel).reindex(panel_split.index)
    votes = metric.vote(values).reindex(panel_split.index).fillna(0).astype(int)
    fwd_by_horizon = _target_forward_returns(
        panel_split,
        symbol=target_symbol,
        target_kind=target_kind,
        horizons=horizons_tuple,
    )

    aligned = pd.DataFrame({"metric": values, "vote": votes}, index=panel_split.index)
    for horizon, fwd in fwd_by_horizon.items():
        aligned[f"fwd_{horizon}d"] = fwd

    warnings: list[str] = []
    if values.dropna().empty:
        warnings.append(f"{metric_name} has no raw values in the selected split.")
    if votes.abs().sum() == 0:
        warnings.append(f"{metric_name} has no non-neutral votes in the selected split.")

    ic_rows = []
    rolling_frames = []
    for horizon, fwd in fwd_by_horizon.items():
        raw_ic, raw_ic_p = _safe_spearman(values, fwd)
        vote_ic, vote_ic_p = _safe_spearman(votes.astype(float), fwd)
        n = len(pd.concat({"metric": values, "fwd": fwd}, axis=1).dropna())
        ic_rows.append({
            "horizon": int(horizon),
            "raw_ic": raw_ic,
            "raw_ic_p": raw_ic_p,
            "vote_ic": vote_ic,
            "vote_ic_p": vote_ic_p,
            "n": n,
        })
        rolling_frames.append(
            _rolling_spearman(values, fwd, horizon=int(horizon), window=rolling_window)
        )

    rolling_ic = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    max_horizon = max(horizons_tuple)
    valid_values = values.dropna()
    latest_idx = valid_values.index.max() if not valid_values.empty else None
    latest = {
        "metric": metric_name,
        "family": metric.family,
        "status": metric.status,
        "split": split,
        "latest_date": latest_idx,
        "latest_value": float(values.loc[latest_idx]) if latest_idx is not None and pd.notna(values.loc[latest_idx]) else np.nan,
        "latest_vote": int(votes.loc[latest_idx]) if latest_idx is not None else 0,
    }

    return MetricForwardProfile(
        metric_name=metric_name,
        split=split,
        target_symbol=target_symbol,
        target_kind=target_kind,
        horizons=horizons_tuple,
        aligned=aligned,
        ic_table=pd.DataFrame(ic_rows),
        vote_bucket_table=_vote_bucket_table(votes, fwd_by_horizon),
        quantile_table=_quantile_bucket_table(values, fwd_by_horizon, bins=bins),
        rolling_ic=rolling_ic,
        event_paths=_event_study_paths(
            panel_split,
            votes,
            symbol=target_symbol,
            target_kind=target_kind,
            max_horizon=max_horizon,
        ),
        latest=latest,
        warnings=warnings,
    )


def metric_redundancy_table(
    panel: pd.DataFrame,
    metric_name: str,
    *,
    split: str = "train",
    include_watch: bool = True,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Compare one metric with the rest of the registry on raw values and votes."""
    if metric_name not in REGISTRY:
        raise KeyError(f"Metric {metric_name!r} not in REGISTRY.")
    panel_split = research_split_panel(panel, split)
    base_metric = REGISTRY[metric_name]
    base_values = base_metric.compute(panel).reindex(panel_split.index)
    base_votes = base_metric.vote(base_values).reindex(panel_split.index).fillna(0).astype(int)

    rows = []
    for other_name, other in REGISTRY.items():
        if other_name == metric_name:
            continue
        if other.status != "voting" and not include_watch:
            continue
        other_values = other.compute(panel).reindex(panel_split.index)
        other_votes = other.vote(other_values).reindex(panel_split.index).fillna(0).astype(int)
        raw_ic, _ = _safe_spearman(base_values, other_values)
        vote_ic, _ = _safe_spearman(base_votes.astype(float), other_votes.astype(float))
        rows.append({
            "metric": other_name,
            "family": other.family,
            "status": other.status,
            "raw_corr": raw_ic,
            "vote_corr": vote_ic,
            "is_redundant": (
                (pd.notna(raw_ic) and abs(raw_ic) >= threshold)
                or (pd.notna(vote_ic) and abs(vote_ic) >= threshold)
            ),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["is_redundant", "raw_corr"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def pairwise_redundancy_table(
    panel: pd.DataFrame,
    *,
    split: str = "train+val",
    include_watch: bool = True,
    metric_names: list[str] | None = None,
    kind: str = "raw",
) -> pd.DataFrame:
    """N×N Spearman correlation between every pair of metrics on one split.

    Returns a square DataFrame indexed and columned by metric name, rows/cols
    sorted by (family, metric) for visual grouping. kind="raw" uses raw
    metric values; kind="vote" uses {-1, 0, +1} votes cast to float.
    """
    panel_split = research_split_panel(panel, split)

    names = [
        name for name, m in REGISTRY.items()
        if (metric_names is None or name in metric_names)
        and (include_watch or m.status == "voting")
    ]
    # Sort by (family, name) for visual grouping
    names = sorted(names, key=lambda n: (REGISTRY[n].family, n))

    series_dict: dict[str, pd.Series] = {}
    for name in names:
        m = REGISTRY[name]
        vals = m.compute(panel).reindex(panel_split.index)
        if kind == "vote":
            vals = m.vote(vals).reindex(panel_split.index).fillna(0).astype(float)
        series_dict[name] = vals

    wide = pd.DataFrame(series_dict)
    return wide.corr(method="spearman")


def pair_metric_profile(
    panel: pd.DataFrame,
    primary_metric: str,
    filter_metric: str,
    *,
    split: str = "train",
    horizons: Iterable[int] = RESEARCH_HORIZONS,
    target_symbol: str = "TQQQ",
    target_kind: str = "tradable_open",
) -> PairMetricProfile:
    """Evaluate a small set of two-metric conditional vote hypotheses."""
    if primary_metric not in REGISTRY:
        raise KeyError(f"Metric {primary_metric!r} not in REGISTRY.")
    if filter_metric not in REGISTRY:
        raise KeyError(f"Metric {filter_metric!r} not in REGISTRY.")

    horizons_tuple = tuple(int(h) for h in horizons)
    panel_split = research_split_panel(panel, split)
    primary = REGISTRY[primary_metric]
    filter_m = REGISTRY[filter_metric]
    p_values = primary.compute(panel).reindex(panel_split.index)
    f_values = filter_m.compute(panel).reindex(panel_split.index)
    p_votes = primary.vote(p_values).reindex(panel_split.index).fillna(0).astype(int)
    f_votes = filter_m.vote(f_values).reindex(panel_split.index).fillna(0).astype(int)

    masks = pd.DataFrame({
        "primary_vote": p_votes,
        "filter_vote": f_votes,
        "primary_bull_and_filter_bull": (p_votes == 1) & (f_votes == 1),
        "primary_bull_and_filter_not_bear": (p_votes == 1) & (f_votes != -1),
        "primary_bear_and_filter_bear": (p_votes == -1) & (f_votes == -1),
    }, index=panel_split.index)

    fwd_by_horizon = _target_forward_returns(
        panel_split,
        symbol=target_symbol,
        target_kind=target_kind,
        horizons=horizons_tuple,
    )
    rows = []
    for condition in [
        "primary_bull_and_filter_bull",
        "primary_bull_and_filter_not_bear",
        "primary_bear_and_filter_bear",
    ]:
        mask = masks[condition].fillna(False)
        for horizon, fwd in fwd_by_horizon.items():
            returns = fwd.loc[mask.reindex(fwd.index).fillna(False)].dropna()
            rows.append({
                "condition": condition,
                "horizon": int(horizon),
                "n": len(returns),
                "mean_fwd": float(returns.mean()) if len(returns) else np.nan,
                "mean_fwd_bps": float(returns.mean() * 10_000) if len(returns) else np.nan,
                "hit_rate": float((returns > 0).mean()) if len(returns) else np.nan,
            })

    warnings = []
    if not masks.iloc[:, 2:].any().any():
        warnings.append("No paired conditions fired in the selected split.")
    elif masks.iloc[:, 2:].sum().min() < 10:
        warnings.append("At least one paired condition has fewer than 10 events.")

    return PairMetricProfile(
        primary_metric=primary_metric,
        filter_metric=filter_metric,
        split=split,
        target_symbol=target_symbol,
        target_kind=target_kind,
        horizons=horizons_tuple,
        condition_table=pd.DataFrame(rows),
        masks=masks,
        warnings=warnings,
    )


def signal_credibility_table(
    panel: pd.DataFrame,
    *,
    split: str = "train",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    target_symbol: str = "QQQ",
    target_kind: str = "close",
    metric_names: list[str] | None = None,
    include_watch: bool = False,
) -> pd.DataFrame:
    """Summarize per-metric vote buckets and ICs for one split.

    Returns one row per voting metric and horizon. Positive `bull_minus_bear`
    means bullish votes had higher forward returns than bearish votes.

    Metric values are computed on the full causal panel, then reindexed to the
    requested split. This preserves pre-split rolling history for validation
    while forward-return targets remain split-limited.
    """
    panel_split = split_panel(panel, split)
    if panel_split.empty:
        raise ValueError(f"No rows available for split {split!r}.")

    if target_kind == "close":
        fwd_by_horizon = forward_returns(panel_split, symbol=target_symbol, horizons=horizons)
    elif target_kind == "tradable_open":
        fwd_by_horizon = tradable_open_forward_returns(
            panel_split, symbol=target_symbol, horizons=horizons
        )
    else:
        raise ValueError("target_kind must be 'close' or 'tradable_open'.")
    if metric_names is None:
        metric_names = [
            name for name, m in REGISTRY.items()
            if include_watch or m.status == "voting"
        ]

    rows: list[dict[str, float | str | int]] = []
    for name in metric_names:
        metric = REGISTRY[name]
        if metric.status != "voting" and not include_watch:
            continue

        values = metric.compute(panel).reindex(panel_split.index)
        votes = metric.vote(values).reindex(panel_split.index).fillna(0).astype(int)

        for horizon, fwd in fwd_by_horizon.items():
            raw_ic, raw_ic_p = _safe_spearman(values, fwd)
            vote_ic, vote_ic_p = _safe_spearman(votes.astype(float), fwd)
            row = {
                "split": split,
                "metric": name,
                "family": metric.family,
                "status": metric.status,
                "horizon": int(horizon),
                "raw_ic": raw_ic,
                "raw_ic_p": raw_ic_p,
                "vote_ic": vote_ic,
                "vote_ic_p": vote_ic_p,
            }
            row.update(_vote_bucket_stats(votes, fwd))
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["bull_minus_bear_bps"] = result["bull_minus_bear"] * 10_000
    result["min_directional_obs"] = result[["n_bull", "n_bear"]].min(axis=1)
    return result.sort_values(
        ["horizon", "bull_minus_bear_bps", "raw_ic"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def train_val_credibility_report(
    panel: pd.DataFrame,
    *,
    horizon: int = 5,
    target_symbol: str = "QQQ",
    target_kind: str = "close",
    metric_names: list[str] | None = None,
    include_watch: bool = False,
) -> pd.DataFrame:
    """Compare train and validation evidence for one forward-return horizon."""
    train = signal_credibility_table(
        panel,
        split="train",
        horizons=(horizon,),
        target_symbol=target_symbol,
        target_kind=target_kind,
        metric_names=metric_names,
        include_watch=include_watch,
    )
    val = signal_credibility_table(
        panel,
        split="val",
        horizons=(horizon,),
        target_symbol=target_symbol,
        target_kind=target_kind,
        metric_names=metric_names,
        include_watch=include_watch,
    )

    keep = [
        "metric",
        "family",
        "status",
        "raw_ic",
        "vote_ic",
        "bull_minus_bear_bps",
        "min_directional_obs",
        "n",
    ]
    merged = train[keep].merge(
        val[keep], on=["metric", "family", "status"], suffixes=("_train", "_val")
    )

    edge_train = merged["bull_minus_bear_bps_train"]
    edge_val = merged["bull_minus_bear_bps_val"]
    ic_train = merged["raw_ic_train"]
    ic_val = merged["raw_ic_val"]

    edge_sign_agrees = np.sign(edge_train.fillna(0)) == np.sign(edge_val.fillna(0))
    ic_sign_agrees = np.sign(ic_train.fillna(0)) == np.sign(ic_val.fillna(0))
    enough_val_obs = merged["min_directional_obs_val"] >= 30

    merged["edge_sign_agrees"] = edge_sign_agrees
    merged["ic_sign_agrees"] = ic_sign_agrees
    merged["enough_val_directional_obs"] = enough_val_obs

    direction_multiplier = np.where(edge_sign_agrees, 1.0, -0.5)
    ic_multiplier = np.where(ic_sign_agrees, 1.0, 0.5)
    obs_multiplier = np.minimum(1.0, merged["min_directional_obs_val"].fillna(0) / 60.0)
    merged["credibility_score"] = (
        edge_val.abs().fillna(0) * direction_multiplier * ic_multiplier * obs_multiplier
    )

    labels = []
    for _, row in merged.iterrows():
        if row["credibility_score"] > 0 and row["edge_sign_agrees"] and row["enough_val_directional_obs"]:
            labels.append("promising")
        elif row["credibility_score"] < 0 or not row["edge_sign_agrees"]:
            labels.append("mixed")
        else:
            labels.append("weak")
    merged["credibility_label"] = labels

    cols = [
        "metric",
        "family",
        "credibility_label",
        "credibility_score",
        "status",
        "bull_minus_bear_bps_train",
        "bull_minus_bear_bps_val",
        "raw_ic_train",
        "raw_ic_val",
        "vote_ic_train",
        "vote_ic_val",
        "edge_sign_agrees",
        "ic_sign_agrees",
        "min_directional_obs_train",
        "min_directional_obs_val",
        "n_train",
        "n_val",
    ]
    return merged[cols].sort_values("credibility_score", ascending=False).reset_index(drop=True)


def metric_decision_table(
    panel: pd.DataFrame,
    *,
    horizon: int = 5,
    metric_names: list[str] | None = None,
    min_directional_obs: int = 30,
) -> pd.DataFrame:
    """Make train-only keep/invert/drop decisions for v2 research.

    Decisions are based on tradable TQQQ next-open returns. QQQ close-to-close
    diagnostics and validation results are included as context only because many
    metrics are defined on QQQ. Validation does not influence decision, direction,
    or weight.
    """
    tqqq = train_val_credibility_report(
        panel,
        horizon=horizon,
        target_symbol="TQQQ",
        target_kind="tradable_open",
        metric_names=metric_names,
    )
    qqq = train_val_credibility_report(
        panel,
        horizon=horizon,
        target_symbol="QQQ",
        target_kind="close",
        metric_names=metric_names,
    )

    qqq_keep = qqq[[
        "metric",
        "bull_minus_bear_bps_train",
        "bull_minus_bear_bps_val",
        "raw_ic_train",
        "raw_ic_val",
    ]].rename(columns={
        "bull_minus_bear_bps_train": "qqq_edge_bps_train",
        "bull_minus_bear_bps_val": "qqq_edge_bps_val",
        "raw_ic_train": "qqq_raw_ic_train",
        "raw_ic_val": "qqq_raw_ic_val",
    })

    result = tqqq.merge(qqq_keep, on="metric", how="left").rename(columns={
        "bull_minus_bear_bps_train": "tqqq_edge_bps_train",
        "bull_minus_bear_bps_val": "tqqq_edge_bps_val",
        "raw_ic_train": "tqqq_raw_ic_train",
        "raw_ic_val": "tqqq_raw_ic_val",
    })

    enough_obs = result["min_directional_obs_train"] >= min_directional_obs
    train_edge = result["tqqq_edge_bps_train"]
    val_edge = result["tqqq_edge_bps_val"]
    same_sign = np.sign(train_edge.fillna(0)) == np.sign(val_edge.fillna(0))

    decisions = []
    directions = []
    weights = []
    for ok_obs, tr in zip(enough_obs, train_edge):
        if not ok_obs or pd.isna(tr) or tr == 0:
            decisions.append("drop")
            directions.append(0)
            weights.append(0.0)
        elif tr > 0:
            decisions.append("keep")
            directions.append(1)
            weights.append(float(min(abs(tr), 100.0) / 100.0))
        else:
            decisions.append("invert")
            directions.append(-1)
            weights.append(float(min(abs(tr), 100.0) / 100.0))

    result["decision"] = decisions
    result["direction"] = directions
    result["weight"] = weights
    result["selection_basis"] = "train_only"
    result["decision_rank"] = result["decision"].map({"keep": 0, "invert": 1, "drop": 2})

    cols = [
        "metric",
        "family",
        "status",
        "decision",
        "direction",
        "weight",
        "selection_basis",
        "tqqq_edge_bps_train",
        "tqqq_edge_bps_val",
        "qqq_edge_bps_train",
        "qqq_edge_bps_val",
        "tqqq_raw_ic_train",
        "tqqq_raw_ic_val",
        "qqq_raw_ic_train",
        "qqq_raw_ic_val",
        "edge_sign_agrees",
        "ic_sign_agrees",
        "min_directional_obs_train",
        "min_directional_obs_val",
        "n_train",
        "n_val",
    ]
    return result.sort_values(
        ["decision_rank", "weight"], ascending=[True, False]
    )[cols].reset_index(drop=True)


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    """BH (FDR) adjusted q-values. NaN p-values pass through as NaN."""
    result = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid_mask = pvals.notna()
    if valid_mask.sum() == 0:
        return result
    _, qvals, _, _ = multipletests(pvals[valid_mask].values, method="fdr_bh")
    result.loc[valid_mask] = qvals
    return result


def quantile_monotonicity(values: pd.Series, fwd: pd.Series, *, bins: int = 10) -> float:
    """Spearman rank corr between bucket index (1..bins) and bucket mean fwd return.

    Returns float in [-1, 1]. NaN if fewer than 4 non-degenerate buckets
    or fewer than 30 aligned observations.
    """
    aligned = pd.concat({"metric": values, "fwd": fwd}, axis=1).dropna()
    if len(aligned) < 30 or aligned["metric"].nunique(dropna=True) < 2:
        return np.nan
    try:
        quantiles = pd.qcut(aligned["metric"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    aligned = aligned.assign(quantile=quantiles)
    bucket_means = (
        aligned.dropna(subset=["quantile"])
        .groupby("quantile")["fwd"]
        .mean()
    )
    if len(bucket_means) < 4:
        return np.nan
    ic, _ = spearmanr(bucket_means.index.values, bucket_means.values)
    return float(ic)


def multi_horizon_credibility_report(
    panel: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20),
    target_symbol: str = "TQQQ",
    target_kind: str = "tradable_open",
    include_watch: bool = True,
    min_directional_obs: int = 30,
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    """One row per metric, summarizing evidence across all `horizons`.

    Columns:
      metric, family, status,
      edge_train_5d, edge_val_5d,
      raw_ic_train_5d, raw_ic_val_5d,
      raw_ic_p_val_5d, raw_ic_q_val_5d,
      vote_ic_val_5d,
      monotonicity_val_5d,
      n_horizons_edge_sign_agree,
      min_directional_obs_val,
      n_val,
      credibility_score,
      credibility_label,
      passes_min_obs,
    """
    primary_h = 5 if 5 in horizons else horizons[0]

    # Collect train_val reports per horizon for edge_sign_agrees
    horizon_reports: dict[int, pd.DataFrame] = {}
    for h in horizons:
        horizon_reports[h] = train_val_credibility_report(
            panel,
            horizon=h,
            target_symbol=target_symbol,
            target_kind=target_kind,
            include_watch=include_watch,
            metric_names=metric_names,
        )

    primary_report = horizon_reports[primary_h]

    # n_horizons_edge_sign_agree: count horizons where sign agrees for each metric
    agree_counts: dict[str, int] = {}
    all_metrics = primary_report["metric"].tolist()
    for name in all_metrics:
        count = 0
        for h, rpt in horizon_reports.items():
            row = rpt.loc[rpt["metric"] == name]
            if not row.empty and bool(row.iloc[0]["edge_sign_agrees"]):
                count += 1
        agree_counts[name] = count

    # Build raw_ic_p at primary horizon from signal_credibility_table
    train_ct = signal_credibility_table(
        panel,
        split="train",
        horizons=(primary_h,),
        target_symbol=target_symbol,
        target_kind=target_kind,
        include_watch=include_watch,
        metric_names=metric_names,
    )
    val_ct = signal_credibility_table(
        panel,
        split="val",
        horizons=(primary_h,),
        target_symbol=target_symbol,
        target_kind=target_kind,
        include_watch=include_watch,
        metric_names=metric_names,
    )

    # Compute monotonicity on val split
    panel_val = split_panel(panel, "val")
    if target_kind == "tradable_open":
        fwd_val = tradable_open_forward_returns(
            panel_val, symbol=target_symbol, horizons=(primary_h,)
        )[primary_h]
    else:
        fwd_val = forward_returns(panel_val, symbol=target_symbol, horizons=(primary_h,))[primary_h]

    monotonicity_map: dict[str, float] = {}
    for name in all_metrics:
        metric = REGISTRY[name]
        values = metric.compute(panel_val).reindex(panel_val.index)
        monotonicity_map[name] = quantile_monotonicity(values, fwd_val)

    # p-values from val credibility table
    p_map = {}
    for _, row in val_ct.iterrows():
        p_map[row["metric"]] = row["raw_ic_p"]

    p_series = pd.Series(
        {name: p_map.get(name, np.nan) for name in all_metrics},
        dtype=float,
    )
    q_series = bh_qvalues(p_series)

    rows = []
    for _, pr_row in primary_report.iterrows():
        name = pr_row["metric"]
        tr_row = train_ct.loc[train_ct["metric"] == name]
        va_row = val_ct.loc[val_ct["metric"] == name]
        raw_ic_p = p_map.get(name, np.nan)
        raw_ic_q = float(q_series.loc[name]) if name in q_series.index else np.nan
        vote_ic_val = float(va_row.iloc[0]["vote_ic"]) if not va_row.empty else np.nan
        min_dir_obs_val = float(pr_row["min_directional_obs_val"])
        rows.append({
            "metric": name,
            "family": pr_row["family"],
            "status": pr_row["status"],
            "edge_train_5d": float(pr_row["bull_minus_bear_bps_train"]),
            "edge_val_5d": float(pr_row["bull_minus_bear_bps_val"]),
            "raw_ic_train_5d": float(pr_row["raw_ic_train"]),
            "raw_ic_val_5d": float(pr_row["raw_ic_val"]),
            "raw_ic_p_val_5d": raw_ic_p,
            "raw_ic_q_val_5d": raw_ic_q,
            "vote_ic_val_5d": vote_ic_val,
            "monotonicity_val_5d": monotonicity_map.get(name, np.nan),
            "n_horizons_edge_sign_agree": agree_counts.get(name, 0),
            "min_directional_obs_val": min_dir_obs_val,
            "n_val": float(pr_row["n_val"]),
            "credibility_score": float(pr_row["credibility_score"]),
            "credibility_label": pr_row["credibility_label"],
            "passes_min_obs": min_dir_obs_val >= min_directional_obs,
        })

    col_order = [
        "metric", "family", "status",
        "edge_train_5d", "edge_val_5d",
        "raw_ic_train_5d", "raw_ic_val_5d",
        "raw_ic_p_val_5d", "raw_ic_q_val_5d",
        "vote_ic_val_5d",
        "monotonicity_val_5d",
        "n_horizons_edge_sign_agree",
        "min_directional_obs_val",
        "n_val",
        "credibility_score",
        "credibility_label",
        "passes_min_obs",
    ]
    return (
        pd.DataFrame(rows)[col_order]
        .sort_values("credibility_score", ascending=False)
        .reset_index(drop=True)
    )


def vote_dynamics_report(
    panel: pd.DataFrame,
    *,
    split: str = "train+val",
    metric_names: list[str] | None = None,
    include_watch: bool = True,
) -> pd.DataFrame:
    """One row per metric: vote-persistence and trade-frequency diagnostics.

    Columns:
      metric, family, status,
      pct_bull, pct_neutral, pct_bear,
      n_flips_any, n_flips_directional,
      flips_per_year, mean_run_length, median_run_length,
      avg_long_fraction, avg_short_fraction,
      vote_autocorr_1d, vote_autocorr_5d,
    """
    panel_split = research_split_panel(panel, split)
    years = len(panel_split) / 252.0

    names = list(REGISTRY.keys()) if metric_names is None else metric_names
    rows = []
    for name in names:
        if name not in REGISTRY:
            continue
        m = REGISTRY[name]
        if not include_watch and m.status == "watch":
            continue

        votes = (
            m.vote(m.compute(panel_split))
            .reindex(panel_split.index)
            .fillna(0)
            .astype(int)
        )
        n = len(votes)
        if n == 0:
            continue

        n_bull = int((votes == 1).sum())
        n_neutral = int((votes == 0).sum())
        n_bear = int((votes == -1).sum())

        n_flips_any = int((votes != votes.shift(1)).sum()) - 1
        n_flips_any = max(n_flips_any, 0)

        directional = votes[votes != 0]
        n_flips_directional = int((directional != directional.shift(1)).sum()) - 1
        n_flips_directional = max(n_flips_directional, 0)

        groups = (votes != votes.shift()).cumsum()
        run_lengths = votes.groupby(groups).transform("count")
        unique_runs = votes.groupby(groups).size()
        mean_run = float(unique_runs.mean())
        median_run = float(unique_runs.median())

        rows.append({
            "metric": name,
            "family": m.family,
            "status": m.status,
            "pct_bull": n_bull / n,
            "pct_neutral": n_neutral / n,
            "pct_bear": n_bear / n,
            "n_flips_any": n_flips_any,
            "n_flips_directional": n_flips_directional,
            "flips_per_year": n_flips_any / years if years > 0 else np.nan,
            "mean_run_length": mean_run,
            "median_run_length": median_run,
            "avg_long_fraction": n_bull / n,
            "avg_short_fraction": n_bear / n,
            "vote_autocorr_1d": float(votes.astype(float).autocorr(lag=1)),
            "vote_autocorr_5d": float(votes.astype(float).autocorr(lag=5)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("flips_per_year").reset_index(drop=True)


def apply_cost_model(
    *,
    gross_edge_bps_per_trade: float,
    flips_per_year: float,
    long_fraction: float,
    one_way_spread_bps: float = 2.0,
    one_way_commission_bps: float = 0.0,
    annual_expense_bps: float = 86.0,
) -> dict[str, float]:
    """Translate gross per-trade edge into net annualised return after costs.

    trades_per_year ≈ flips_per_year / 2 (each round trip = 2 flips).
    gross_annual_bps = gross_edge_bps_per_trade × trades_per_year
    friction_bps_per_round_trip = 2 × (spread + commission)
    friction_annual_bps = friction_bps_per_round_trip × trades_per_year
    etf_drag_annual_bps = annual_expense_bps × long_fraction
    net_annual_bps = gross_annual_bps - friction_annual_bps - etf_drag_annual_bps
    breakeven_edge_bps_per_trade = (friction_annual_bps + etf_drag_annual_bps) / trades_per_year
    """
    trades_per_year = flips_per_year / 2.0
    gross_annual_bps = gross_edge_bps_per_trade * trades_per_year
    friction_bps_per_round_trip = 2.0 * (one_way_spread_bps + one_way_commission_bps)
    friction_annual_bps = friction_bps_per_round_trip * trades_per_year
    etf_drag_annual_bps = annual_expense_bps * long_fraction
    net_annual_bps = gross_annual_bps - friction_annual_bps - etf_drag_annual_bps
    if trades_per_year > 0:
        breakeven_edge = (friction_annual_bps + etf_drag_annual_bps) / trades_per_year
    else:
        breakeven_edge = np.nan

    return {
        "gross_edge_bps_per_trade": gross_edge_bps_per_trade,
        "trades_per_year": trades_per_year,
        "gross_annual_bps": gross_annual_bps,
        "friction_bps_per_round_trip": friction_bps_per_round_trip,
        "friction_annual_bps": friction_annual_bps,
        "etf_drag_annual_bps": etf_drag_annual_bps,
        "net_annual_bps": net_annual_bps,
        "breakeven_edge_bps_per_trade": breakeven_edge,
    }


_REGIME_LABELS = ["strong_bull", "weak_bull", "sideways", "weak_bear", "strong_bear"]


def regime_conditional_edge_table(
    panel: pd.DataFrame,
    metric_name: str,
    regime_states: pd.Series,
    *,
    split: str = "train+val",
    horizon: int = 5,
    target_symbol: str = "TQQQ",
    target_kind: str = "tradable_open",
) -> pd.DataFrame:
    """One row per regime state: vote distribution and edge in that state.

    Columns:
      regime_state, regime_label, n_days,
      pct_bull, pct_bear,
      mean_fwd_bull_bps, mean_fwd_bear_bps,
      edge_bps, raw_ic, n_directional,
    """
    panel_split = research_split_panel(panel, split)
    m = REGISTRY[metric_name]
    values = m.compute(panel_split).reindex(panel_split.index)
    votes = m.vote(values).reindex(panel_split.index).fillna(0).astype(int)

    fwd = _target_forward_returns(
        panel_split, symbol=target_symbol, target_kind=target_kind, horizons=(horizon,)
    )[horizon]

    states_aligned = regime_states.reindex(panel_split.index)
    valid_states = states_aligned[states_aligned != -1]

    rows = []
    for state in range(5):
        idx = valid_states[valid_states == state].index
        idx = idx.intersection(fwd.dropna().index)
        n_days = len(idx)

        v = votes.reindex(idx)
        f = fwd.reindex(idx)
        raw_v = values.reindex(idx)

        n_bull = int((v == 1).sum())
        n_bear = int((v == -1).sum())

        bull_fwd = f[v == 1]
        bear_fwd = f[v == -1]
        mean_bull = float(bull_fwd.mean() * 10_000) if len(bull_fwd) else np.nan
        mean_bear = float(bear_fwd.mean() * 10_000) if len(bear_fwd) else np.nan
        edge_bps = (
            float(mean_bull - mean_bear)
            if not np.isnan(mean_bull) and not np.isnan(mean_bear)
            else np.nan
        )
        raw_ic, _ = _safe_spearman(raw_v, f)

        rows.append({
            "regime_state": state,
            "regime_label": _REGIME_LABELS[state],
            "n_days": n_days,
            "pct_bull": n_bull / n_days if n_days else np.nan,
            "pct_bear": n_bear / n_days if n_days else np.nan,
            "mean_fwd_bull_bps": mean_bull,
            "mean_fwd_bear_bps": mean_bear,
            "edge_bps": edge_bps,
            "raw_ic": raw_ic,
            "n_directional": min(n_bull, n_bear),
        })

    return pd.DataFrame(rows)


def rolling_edge_decay(
    panel: pd.DataFrame,
    metric_name: str,
    *,
    split: str = "train+val",
    window: int = 504,
    step: int = 21,
    horizon: int = 5,
    target_symbol: str = "TQQQ",
    target_kind: str = "tradable_open",
) -> pd.DataFrame:
    """Rolling bull-minus-bear edge in trailing `window`-day buckets.

    Columns: window_end, edge_bps, raw_ic, n, n_directional
    """
    panel_split = research_split_panel(panel, split)
    m = REGISTRY[metric_name]
    values = m.compute(panel_split).reindex(panel_split.index)
    votes = m.vote(values).reindex(panel_split.index).fillna(0).astype(int)
    fwd = _target_forward_returns(
        panel_split, symbol=target_symbol, target_kind=target_kind, horizons=(horizon,)
    )[horizon]

    aligned = pd.concat({"metric": values, "vote": votes, "fwd": fwd}, axis=1).dropna()
    n_total = len(aligned)

    rows = []
    for end_idx in range(window - 1, n_total, step):
        chunk = aligned.iloc[max(0, end_idx - window + 1): end_idx + 1]
        v = chunk["vote"]
        f = chunk["fwd"]
        raw_v = chunk["metric"]

        bull_fwd = f[v == 1]
        bear_fwd = f[v == -1]
        mean_bull = float(bull_fwd.mean() * 10_000) if len(bull_fwd) else np.nan
        mean_bear = float(bear_fwd.mean() * 10_000) if len(bear_fwd) else np.nan
        edge_bps = (
            float(mean_bull - mean_bear)
            if not np.isnan(mean_bull) and not np.isnan(mean_bear)
            else np.nan
        )
        raw_ic, _ = _safe_spearman(raw_v, f)

        rows.append({
            "window_end": aligned.index[end_idx],
            "edge_bps": edge_bps,
            "raw_ic": raw_ic,
            "n": len(chunk),
            "n_directional": min(len(bull_fwd), len(bear_fwd)),
        })

    return pd.DataFrame(rows)


def _format_for_console(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].round(4)
    return display.to_string(index=False)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Rank ETF signals by train/validation credibility.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--horizon", default=5, type=int, help="Forward-return horizon in trading days.")
    parser.add_argument("--target-symbol", default="QQQ", help="Symbol used for forward returns.")
    parser.add_argument(
        "--target-kind",
        default="close",
        choices=["close", "tradable_open"],
        help="Forward-return target type.",
    )
    parser.add_argument(
        "--include-watch",
        action="store_true",
        help="Include watch-only metrics in raw IC diagnostics.",
    )
    parser.add_argument("--compare-train-val", action="store_true", help="Rank metrics by train/val stability.")
    parser.add_argument("--decision-table", action="store_true", help="Emit v2 keep/invert/drop decision table.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--data-dir", default=None, help="Path to data/ directory.")
    args = parser.parse_args()

    if args.data_dir:
        panel = load_panel(SYMBOLS_ALL, data_dir=Path(args.data_dir), warn_missing=True)
    else:
        panel = load_panel(SYMBOLS_ALL, warn_missing=True)

    if args.decision_table:
        df = metric_decision_table(panel, horizon=args.horizon)
    elif args.compare_train_val:
        df = train_val_credibility_report(
            panel,
            horizon=args.horizon,
            target_symbol=args.target_symbol,
            target_kind=args.target_kind,
            include_watch=args.include_watch,
        )
    else:
        df = signal_credibility_table(
            panel,
            split=args.split,
            horizons=(args.horizon,),
            target_symbol=args.target_symbol,
            target_kind=args.target_kind,
            include_watch=args.include_watch,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)

    print(_format_for_console(df))


if __name__ == "__main__":
    _cli()
