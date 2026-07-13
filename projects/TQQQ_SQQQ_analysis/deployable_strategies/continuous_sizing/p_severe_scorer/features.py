"""Feature helpers and input validation for the `p_severe` scorer.

The scorer can derive only features whose inputs are available outside the
original strategy. It cannot reconstruct strategy-internal state such as
`RSI_entry`, `BBP_entry`, `bars_since_last_stop`, or `regime_entry`; the calling
backtest must provide those fields.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .constants import (
    DAILY_CONTEXT,
    MODEL_FEATURES,
    RAW_DAILY_SYMBOLS,
    STRATEGY_INTERNAL_COLUMNS,
)


class FeatureContractError(ValueError):
    """Raised when candidate trades do not satisfy the scorer input contract."""


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def realized_vol(series: pd.Series, n: int = 20) -> pd.Series:
    return series.pct_change().rolling(n, min_periods=n).std() * np.sqrt(252)


def rolling_pctile(series: pd.Series, n: int = 252) -> pd.Series:
    return series.rolling(n, min_periods=max(60, n // 4)).rank(pct=True)


def _normalize_daily_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise FeatureContractError(f"Daily data for {name} is missing columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if "adj_close" in out.columns:
        out["px"] = pd.to_numeric(out["adj_close"], errors="coerce").fillna(
            pd.to_numeric(out["close"], errors="coerce")
        )
    else:
        out["px"] = pd.to_numeric(out["close"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def build_daily_context(daily_bars: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the 21 daily context features used by item 17.

    `daily_bars` must provide daily OHLC data for:
    `QQQ`, `SPY`, `^VIX`, `^VIX3M`, `HYG`, `LQD`, `^TNX`, and `^IRX`.
    `adj_close` is optional; when present it is preferred for return features.
    """
    missing = sorted(set(RAW_DAILY_SYMBOLS) - set(daily_bars))
    if missing:
        raise FeatureContractError(f"daily_bars is missing required symbols: {missing}")

    qqq = _normalize_daily_frame(daily_bars["QQQ"], "QQQ")
    spy = _normalize_daily_frame(daily_bars["SPY"], "SPY")
    vix = _normalize_daily_frame(daily_bars["^VIX"], "^VIX")
    vix3m = _normalize_daily_frame(daily_bars["^VIX3M"], "^VIX3M")
    hyg = _normalize_daily_frame(daily_bars["HYG"], "HYG")
    lqd = _normalize_daily_frame(daily_bars["LQD"], "LQD")
    tnx = _normalize_daily_frame(daily_bars["^TNX"], "^TNX")
    irx = _normalize_daily_frame(daily_bars["^IRX"], "^IRX")

    ctx = pd.DataFrame({"date": qqq["date"]})
    ctx["QQQ_RSI_14"] = rsi(qqq["px"], 14).values
    ctx["QQQ_dist_MA20"] = (qqq["px"] / qqq["px"].rolling(20).mean() - 1).values
    ctx["QQQ_dist_MA50"] = (qqq["px"] / qqq["px"].rolling(50).mean() - 1).values
    ctx["QQQ_dist_MA200"] = (qqq["px"] / qqq["px"].rolling(200).mean() - 1).values
    ctx["QQQ_realized_vol_20d"] = realized_vol(qqq["px"], 20).values
    ctx["QQQ_dist_high_20d"] = (qqq["px"] / qqq["px"].rolling(20).max() - 1).values
    ctx["QQQ_50d_return"] = qqq["px"].pct_change(50).values
    ctx["QQQ_50d_return_pctile_252"] = rolling_pctile(ctx["QQQ_50d_return"], 252).values
    ctx["QQQ_drawdown_5d"] = (qqq["px"] / qqq["px"].rolling(5).max() - 1).values
    ctx["QQQ_drawdown_60d"] = (qqq["px"] / qqq["px"].rolling(60).max() - 1).values
    ctx["QQQ_gap_overnight"] = (qqq["open"] / qqq["close"].shift(1) - 1).values

    ctx = ctx.merge(spy[["date", "px"]].rename(columns={"px": "spy_px"}), on="date", how="left")
    ctx["SPY_RSI_14"] = rsi(ctx["spy_px"], 14).values
    ctx["SPY_dist_MA50"] = ctx["spy_px"] / ctx["spy_px"].rolling(50).mean() - 1

    ctx = ctx.merge(vix[["date", "close"]].rename(columns={"close": "vix_lvl"}), on="date", how="left")
    ctx = ctx.merge(vix3m[["date", "close"]].rename(columns={"close": "vix3m_lvl"}), on="date", how="left")
    ctx["VIX_level"] = ctx["vix_lvl"]
    ctx["VIX_5d_change"] = ctx["vix_lvl"].pct_change(5)
    ctx["VIX_pctile_252d"] = rolling_pctile(ctx["vix_lvl"], 252)
    ctx["VIX_term_structure"] = ctx["vix_lvl"] / ctx["vix3m_lvl"]

    ctx = ctx.merge(hyg[["date", "px"]].rename(columns={"px": "hyg_px"}), on="date", how="left")
    ctx = ctx.merge(lqd[["date", "px"]].rename(columns={"px": "lqd_px"}), on="date", how="left")
    ctx["HYG_LQD_ratio"] = ctx["hyg_px"] / ctx["lqd_px"]
    ctx["HYG_5d_change"] = ctx["hyg_px"].pct_change(5)

    ctx = ctx.merge(tnx[["date", "close"]].rename(columns={"close": "tnx"}), on="date", how="left")
    ctx = ctx.merge(irx[["date", "close"]].rename(columns={"close": "irx"}), on="date", how="left")
    ctx["yield_curve_slope"] = ctx["tnx"] - ctx["irx"]
    ctx["TNX_5d_change"] = ctx["tnx"].diff(5)

    return ctx[["date"] + DAILY_CONTEXT].copy()


def _join_daily_context(df: pd.DataFrame, daily_context: pd.DataFrame) -> pd.DataFrame:
    ctx = daily_context.copy()
    if "date" not in ctx.columns:
        raise FeatureContractError("daily_context must contain a 'date' column")
    missing_context = sorted(set(DAILY_CONTEXT) - set(ctx.columns))
    if missing_context:
        raise FeatureContractError(f"daily_context is missing columns: {missing_context}")
    ctx["date"] = pd.to_datetime(ctx["date"]).dt.normalize()
    out = df.copy()
    out["entry_date"] = out["entry_time"].dt.normalize()
    out = out.sort_values("entry_date").reset_index(drop=False).rename(columns={"index": "_orig_index"})
    ctx = ctx.sort_values("date").reset_index(drop=True)
    joined = pd.merge_asof(
        out,
        ctx[["date"] + DAILY_CONTEXT],
        left_on="entry_date",
        right_on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    return joined.sort_values("_orig_index").drop(columns=["_orig_index"]).reset_index(drop=True)


def compute_required_features(
    trades: pd.DataFrame,
    daily_context: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return trades with all derivable model features added.

    Parameters
    ----------
    trades:
        Candidate trade rows. Strategy-internal fields such as `RSI_entry`,
        `BBP_entry`, `bars_since_last_stop`, and `regime_entry` must already be
        present; this project does not have the original strategy code needed to
        recreate them.
    daily_context:
        Either a precomputed context DataFrame with `date` plus the 21
        `DAILY_CONTEXT` columns, or a mapping of raw daily OHLC DataFrames keyed
        by `QQQ`, `SPY`, `^VIX`, `^VIX3M`, `HYG`, `LQD`, `^TNX`, and `^IRX`.
        Context is joined strictly from dates before the trade date.
    """
    if "entry_time" not in trades.columns:
        raise FeatureContractError("trades must include 'entry_time'")

    df = trades.copy()
    if "_features_computed" in df.columns:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    if "atr_pct" not in df.columns and {"atr", "decision_price"}.issubset(df.columns):
        df["atr_pct"] = pd.to_numeric(df["atr"], errors="coerce") / pd.to_numeric(
            df["decision_price"], errors="coerce"
        ) * 100.0
    if "log_volume_ratio" not in df.columns and "volume_ratio" in df.columns:
        df["log_volume_ratio"] = np.log(pd.to_numeric(df["volume_ratio"], errors="coerce"))
    if "hour_of_entry" not in df.columns:
        df["hour_of_entry"] = df["entry_time"].dt.hour

    if "decision_price" in df.columns:
        entry = pd.to_numeric(df["decision_price"], errors="coerce")
        for ma in ("MA20", "MA50", "MA100"):
            dist = f"dist_to_{ma}"
            if dist not in df.columns and ma in df.columns:
                df[dist] = entry / pd.to_numeric(df[ma], errors="coerce") - 1.0

    if "regime_entry" in df.columns:
        df["regime_chop_highvol"] = (df["regime_entry"] == "chop_highvol").astype(int)
        df["regime_sideways_lowvol"] = (df["regime_entry"] == "sideways_lowvol").astype(int)

    if daily_context is not None:
        ctx = build_daily_context(daily_context) if isinstance(daily_context, Mapping) else daily_context
        existing_context = set(DAILY_CONTEXT).intersection(df.columns)
        if existing_context:
            df = df.drop(columns=sorted(existing_context))
        df = _join_daily_context(df, ctx)

    df["_features_computed"] = True
    return df


def missing_model_features(df: pd.DataFrame) -> list[str]:
    return sorted(set(MODEL_FEATURES) - set(df.columns))


def validate_model_ready(df: pd.DataFrame) -> None:
    """Raise a clear error if `df` cannot be scored by the exported artifacts."""
    missing = missing_model_features(df)
    if missing:
        internal = [c for c in STRATEGY_INTERNAL_COLUMNS if c in missing or c not in df.columns]
        help_msg = ""
        if internal:
            help_msg = (
                " Strategy-internal fields cannot be reconstructed by this package; "
                f"the calling backtest must provide: {STRATEGY_INTERNAL_COLUMNS}."
            )
        raise FeatureContractError(f"Missing required model feature columns: {missing}.{help_msg}")
    bad = [c for c in MODEL_FEATURES if pd.to_numeric(df[c], errors="coerce").isna().any()]
    if bad:
        raise FeatureContractError(
            "Required model feature columns contain NaN or non-numeric values: "
            f"{bad}. Supply complete prior-day context and strategy-entry fields."
        )
