"""Tests for scripts/tca_monitor.py — parsing, idempotent ledger, and drift alerting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tca_monitor import (  # noqa: E402
    parse_new_fills, check_drift, load_state, save_state, DEFAULT_WINDOW,
)

BUY_LINE = (">>> BUY 2026-05-22 09:47:03 TQQQ qty 100.0 decision 79.50 fill 79.52")
SELL_LINE = ("SELL 2026-05-22 10:15:00 TQQQ qty 100.0 decision 79.60 fill 79.58 style market")


def _write_log(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


# --------------------------------------------------------------------------- parsing
def test_parse_new_fills_extracts_buy_and_sell(tmp_path):
    _write_log(tmp_path, "a.log", [BUY_LINE, SELL_LINE])
    df, newly = parse_new_fills(tmp_path, already_processed=set())
    assert newly == ["a.log"]
    assert len(df) == 2
    buy = df[df.side == "buy"].iloc[0]
    assert buy.symbol == "TQQQ" and buy.style == "limit"
    assert buy.slip_bps == pytest.approx((79.52 - 79.50) / 79.50 * 1e4)
    sell = df[df.side == "sell"].iloc[0]
    assert sell.style == "market"
    assert sell.slip_bps == pytest.approx((79.60 - 79.58) / 79.60 * 1e4)


def test_parse_new_fills_skips_already_processed(tmp_path):
    _write_log(tmp_path, "a.log", [BUY_LINE])
    df, newly = parse_new_fills(tmp_path, already_processed={"a.log"})
    assert newly == []
    assert df.empty


def test_parse_new_fills_only_parses_unprocessed_files(tmp_path):
    _write_log(tmp_path, "a.log", [BUY_LINE])
    _write_log(tmp_path, "b.log", [SELL_LINE])
    df, newly = parse_new_fills(tmp_path, already_processed={"a.log"})
    assert newly == ["b.log"]
    assert len(df) == 1 and df.iloc[0].side == "sell"


# --------------------------------------------------------------------------- ledger round-trip
def test_save_and_load_state_round_trips(tmp_path):
    fills = pd.DataFrame({"time": [pd.Timestamp("2026-05-22 09:47:00")], "side": ["buy"],
                          "slip_bps": [5.0]})
    save_state(tmp_path, {"a.log", "b.log"}, fills)
    processed, loaded = load_state(tmp_path)
    assert processed == {"a.log", "b.log"}
    assert len(loaded) == 1
    assert loaded.iloc[0]["side"] == "buy"


def test_load_state_empty_when_no_prior_run(tmp_path):
    processed, fills = load_state(tmp_path)
    assert processed == set()
    assert fills.empty


# --------------------------------------------------------------------------- drift check
def _synthetic_fills(n, side, style, resid_mean, sigma, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-05-22 09:45", periods=n, freq="15min")
    resid = rng.normal(resid_mean, 1.0, n)
    return pd.DataFrame({
        "time": times, "side": side, "style": style,
        "slip_bps": resid + 10.0, "predicted_mean_bps": 10.0,
        "residual_bps": resid, "timing_sigma_bps": sigma,
    })


def test_check_drift_alerts_when_residual_exceeds_2sigma():
    fills = _synthetic_fills(DEFAULT_WINDOW, "buy", "limit", resid_mean=10.0, sigma=2.0)
    drift, alert = check_drift(fills, DEFAULT_WINDOW)
    assert alert is True
    row = drift[(drift.side == "buy") & (drift.style == "limit")].iloc[0]
    assert row["breach_2sigma"]


def test_check_drift_no_alert_when_residual_within_band():
    fills = _synthetic_fills(DEFAULT_WINDOW, "buy", "limit", resid_mean=0.1, sigma=5.0)
    drift, alert = check_drift(fills, DEFAULT_WINDOW)
    assert alert is False
    row = drift[(drift.side == "buy") & (drift.style == "limit")].iloc[0]
    assert not row["breach_2sigma"]


def test_check_drift_skips_groups_below_window():
    fills = _synthetic_fills(DEFAULT_WINDOW - 1, "buy", "limit", resid_mean=100.0, sigma=1.0)
    drift, alert = check_drift(fills, DEFAULT_WINDOW)
    assert drift.empty
    assert alert is False


def test_check_drift_groups_are_independent():
    # a breach in one (side, style) group must not be masked by a clean second group.
    clean = _synthetic_fills(DEFAULT_WINDOW, "sell", "market", resid_mean=0.0, sigma=2.0)
    dirty = _synthetic_fills(DEFAULT_WINDOW, "buy", "limit", resid_mean=20.0, sigma=1.0)
    drift, alert = check_drift(pd.concat([clean, dirty], ignore_index=True), DEFAULT_WINDOW)
    assert alert is True
    assert not drift[(drift.side == "sell") & (drift.style == "market")].iloc[0]["breach_2sigma"]
    assert drift[(drift.side == "buy") & (drift.style == "limit")].iloc[0]["breach_2sigma"]
