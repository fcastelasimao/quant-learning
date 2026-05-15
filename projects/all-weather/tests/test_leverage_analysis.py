import numpy as np
import pandas as pd
import pytest

from research.leverage_analysis import (
    BASE,
    BENCHMARK,
    compute_heatmap_pivot,
    default_focus_strategies,
    derive_leverage_tables,
    fmt_num,
    fmt_pp,
    fmt_pct,
    label_selector,
    maybe_filter_benchmark,
    portfolio_view_strategies,
    presentation_table,
    slice_dates,
    strategy_label,
    ticker_from_strategy,
    visible_strategies,
)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def test_fmt_pct_normal():
    assert fmt_pct(5.1234) == "5.12%"


def test_fmt_pct_nan():
    assert fmt_pct(float("nan")) == "n/a"


def test_fmt_num_normal():
    assert fmt_num(0.4567) == "0.457"


def test_fmt_pp_positive():
    assert fmt_pp(1.5) == "+1.50 pp"


def test_fmt_pp_negative():
    assert fmt_pp(-2.0) == "-2.00 pp"


# ---------------------------------------------------------------------------
# Strategy label helpers
# ---------------------------------------------------------------------------

def test_ticker_from_strategy_base():
    assert ticker_from_strategy(BASE) == "BASE"


def test_ticker_from_strategy_overlay():
    assert ticker_from_strategy("My Strategy + GLD RSI Overlay") == "GLD"


def test_ticker_from_strategy_unknown_passthrough():
    result = ticker_from_strategy("something_else")
    assert result == "something_else"


def test_strategy_label_base():
    assert strategy_label(BASE) == "Base"


def test_strategy_label_benchmark():
    assert strategy_label(BENCHMARK) == "SPY benchmark"


def test_strategy_label_overlay():
    assert strategy_label("My Strategy + GLD RSI Overlay") == "GLD overlay"


# ---------------------------------------------------------------------------
# visible_strategies / default_focus_strategies / portfolio_view_strategies
# ---------------------------------------------------------------------------

def test_visible_strategies_excludes_benchmark():
    strategies = [BASE, "My Strategy + GLD RSI Overlay", BENCHMARK]
    result = visible_strategies(strategies)
    assert BENCHMARK not in result
    assert BASE in result


def test_visible_strategies_base_comes_first():
    strategies = [BENCHMARK, "My Strategy + GLD RSI Overlay", BASE]
    result = visible_strategies(strategies)
    assert result[0] == BASE


def test_default_focus_strategies_returns_wanted_subset():
    available = {BASE, "My Strategy + GLD RSI Overlay", "My Strategy + SPY RSI Overlay"}
    result = default_focus_strategies(available)
    assert BASE in result
    assert "My Strategy + GLD RSI Overlay" in result


def test_portfolio_view_strategies_adds_benchmark():
    available = [BASE, "My Strategy + GLD RSI Overlay", BENCHMARK]
    result = portfolio_view_strategies([BASE], include_benchmark=True, available=available)
    assert BENCHMARK in result


def test_portfolio_view_strategies_excludes_missing():
    result = portfolio_view_strategies(["missing_strat"], include_benchmark=False, available=[BASE])
    assert result == []


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def test_maybe_filter_benchmark_removes_benchmark():
    df = pd.DataFrame({"Strategy": [BASE, BENCHMARK], "Value": [1, 2]})
    result = maybe_filter_benchmark(df, include_benchmark=False)
    assert BENCHMARK not in result["Strategy"].values


def test_maybe_filter_benchmark_keeps_benchmark_when_requested():
    df = pd.DataFrame({"Strategy": [BASE, BENCHMARK], "Value": [1, 2]})
    result = maybe_filter_benchmark(df, include_benchmark=True)
    assert BENCHMARK in result["Strategy"].values


def test_slice_dates_filters_correctly():
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "Value": range(10),
    })
    result = slice_dates(df, "2020-01-03", "2020-01-07")
    assert result["Date"].min() >= pd.Timestamp("2020-01-03")
    assert result["Date"].max() <= pd.Timestamp("2020-01-07")


def test_presentation_table_filters_to_existing_columns():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = presentation_table(df, ["A", "C", "D"])
    assert list(result.columns) == ["A", "C"]


def test_label_selector_known_key():
    assert label_selector("default_30_50_20") == "Default 30/50, +20%"


def test_label_selector_unknown_key_titlecases():
    result = label_selector("some_unknown_key")
    assert result == "Some Unknown Key"


# ---------------------------------------------------------------------------
# compute_heatmap_pivot — three modes
# ---------------------------------------------------------------------------

def _grid_df() -> pd.DataFrame:
    rows = []
    for entry in [20.0, 22.0, 24.0]:
        for exit_ in [40.0, 42.0, 44.0]:
            for lev in [0.15, 0.20]:
                rows.append({
                    "Entry Threshold": entry,
                    "Exit Threshold": exit_,
                    "Overlay Weight (%)": lev * 100,
                    "Calmar": 0.3 + (entry - 20) * 0.01 + (exit_ - 40) * 0.005,
                })
    return pd.DataFrame(rows)


def test_compute_heatmap_pivot_leverage_x_exit():
    pivot, xlabel, ylabel, suffix = compute_heatmap_pivot(
        _grid_df(), mode="Leverage x Exit", entry=22.0, exit_=None, leverage=None, metric="Calmar",
    )
    assert isinstance(pivot, pd.DataFrame)
    assert "Exit" in xlabel
    assert "22" in suffix


def test_compute_heatmap_pivot_leverage_x_entry():
    pivot, xlabel, ylabel, suffix = compute_heatmap_pivot(
        _grid_df(), mode="Leverage x Entry", entry=None, exit_=42.0, leverage=None, metric="Calmar",
    )
    assert "Entry" in xlabel
    assert "42" in suffix


def test_compute_heatmap_pivot_exit_x_entry():
    pivot, xlabel, ylabel, suffix = compute_heatmap_pivot(
        _grid_df(), mode="Exit x Entry", entry=None, exit_=None, leverage=15.0, metric="Calmar",
    )
    assert "Entry" in xlabel
    assert "15" in suffix


# ---------------------------------------------------------------------------
# derive_leverage_tables — full integration smoke test
# ---------------------------------------------------------------------------

def _minimal_manifest() -> dict:
    return {
        "allocation": {"SPY": 0.50, "GLD": 0.50},
        "overlay_specs": [
            {"Ticker": "GLD", "Entry": 30, "Exit": 50, "Leverage": 0.20},
        ],
        "fees": {"base": 0.001, "overlay": 0.002},
    }


def _overlay_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": ["BASE", "GLD", "SPY", "GSG"],
        "CAGR (%)": [6.0, 7.5, 7.0, 5.5],
        "Calmar": [0.35, 0.45, 0.40, 0.30],
        "Max Drawdown (%)": [-17.0, -16.5, -17.5, -18.5],
        "RF Opportunity Cost CAGR (%)": [6.5, 7.0, 6.8, 5.8],
    })


def _pass_fail_df(gld_passes: bool = True) -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": ["GLD", "GSG"],
        "Selector": ["default_30_50_20", "best_calmar"],
        "Selector Label": ["Default 30/50, +20%", "Best Calmar"],
        "Overall Pass": [gld_passes, False],
        "Splits Passed": [3, 1],
        "Worst OOS Calmar Delta": [0.05, -0.15],
        "Low Trade Count Splits": [0, 1],
        "MaxDD Breach >3pp": [False, False],
    })


def test_derive_leverage_tables_returns_expected_keys():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    assert set(tables.keys()) == {"allocation", "overlay_specs", "default_rank", "pass_table", "verdict_table", "cards"}


def test_derive_allocation_rows_match_manifest():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    alloc = tables["allocation"]
    assert set(alloc["Asset"]) == {"SPY", "GLD"}
    assert pytest.approx(alloc.loc[alloc["Asset"] == "SPY", "Weight"].iloc[0]) == 0.50


def test_derive_default_rank_excludes_base():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    assert "BASE" not in tables["default_rank"]["Ticker"].values


def test_derive_default_rank_sorted_by_calmar():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    calmars = tables["default_rank"]["Calmar"].tolist()
    assert calmars == sorted(calmars, reverse=True)


def test_derive_rf_cost_drag_computed():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    rank = tables["default_rank"].set_index("Ticker")
    assert pytest.approx(rank.loc["GLD", "RF Cost Drag (pp)"], abs=1e-6) == 7.5 - 7.0


def test_derive_pass_table_adds_verdict_column():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    assert "Verdict" in tables["pass_table"].columns


def test_derive_pass_table_caveat_low_trade_count():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    pt = tables["pass_table"].set_index("Ticker")
    assert pt.loc["GSG", "Caveat"] == "Low trade count"


def test_derive_pass_table_caveat_maxdd_breach_wins_over_low_trade():
    pf = _pass_fail_df()
    pf.loc[pf["Ticker"] == "GSG", "Low Trade Count Splits"] = 1
    pf.loc[pf["Ticker"] == "GSG", "MaxDD Breach >3pp"] = True
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), pf)
    pt = tables["pass_table"].set_index("Ticker")
    assert pt.loc["GSG", "Caveat"] == "MaxDD breach"


def test_derive_gld_default_pass_gets_keep_researching_verdict():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df(gld_passes=True))
    vt = tables["verdict_table"].set_index("Ticker")
    assert vt.loc["GLD", "Verdict"] == "Keep Researching"


def test_derive_no_pass_gets_reject_verdict():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df(gld_passes=False))
    vt = tables["verdict_table"].set_index("Ticker")
    assert vt.loc["GLD", "Verdict"] == "Reject"


def test_derive_cards_has_five_rows():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), _pass_fail_df())
    assert len(tables["cards"]) == 5


def test_derive_empty_pass_fail_produces_empty_tables():
    tables = derive_leverage_tables(_minimal_manifest(), _overlay_summary(), pd.DataFrame())
    assert tables["pass_table"].empty
    assert tables["verdict_table"].empty
    cards = tables["cards"]
    assert cards[cards["Card"] == "Best OOS pass"]["Conclusion"].iloc[0] == "No OOS bundle loaded"
