"""
tax_backtest.py — tax-aware monthly rebalance simulation (D.16).

Why a separate engine
---------------------
``engine/backtest.py`` is share-based, multi-strategy, and golden-locked
(``tests/test_backtest_golden.py``). Bolting realized-gain accounting onto it
would risk that lock for no benefit. This module is the research counterfactual:
it simulates ONE strategy with full tax accounting, reusing the same
``RebalancePolicy`` (D.15), ``LotLedger`` (D.14), and ``compute_tax_on_event``
(D.13) so the pieces agree.

What it models
--------------
* Monthly rebalance to target weights, gated by a ``RebalancePolicy``.
* Realized capital-gains tax on every sell, via the chosen ``LotSelector``
  (FIFO = Alpaca reality; tax_optimal/HIFO = research counterfactual).
* Dividend tax at each distribution's ex-date (qualified vs ordinary by asset
  class). Prices are total-return, so the dividend cash is already reflected as
  price appreciation; we subtract only the *tax* a taxable holder would owe.
* GSG §1256 year-end mark-to-market: a deemed sale of the whole GSG position on
  the last rebalance date of each calendar year, taxed 60/40, with the basis
  reset to market (no change in share count).

How tax/cost is paid
--------------------
Tax and transaction cost are paid by scaling every lot down pro-rata
(``_debit``). Scaling preserves both the target weights and each lot's per-share
cost basis, so the ledger stays internally consistent — economically this is
"sell a thin slice of the whole book to settle the bill".

Caveat: a monthly engine taxes at month-end granularity and uses month-end
share counts for dividend sizing. Good enough for a strategy that only trades
monthly; documented in research/tax_drift_trigger/findings_tax_model.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .backtest import RebalancePolicy
from .calendar import pandas_resample_frequency
from .lot_ledger import LotLedger, LotSelector
from .tax import (
    AssetTaxClass,
    DividendEvent,
    SaleEvent,
    TaxRegime,
    compute_tax_on_event,
)
from . import config

_VALUE_TOL = 1e-12


@dataclass
class TaxBacktestResult:
    """Output of :func:`run_tax_aware_backtest`.

    Attributes
    ----------
    monthly:
        DataFrame indexed by month-end Date. Columns: ``Value`` (after-tax),
        ``Cumulative Tax Paid``, ``Period Tax``, and the per-character tax split.
    events:
        One row per rebalance (date, value, trade notional, realized gains/tax).
    tax_summary:
        Annual tax breakdown (ST/LT gain & tax, dividend income & tax, §1256 MTM).
    """

    monthly: pd.DataFrame
    events: pd.DataFrame
    tax_summary: pd.DataFrame
    regime_name: str
    selector: str
    policy_label: str


def _debit(ledger: LotLedger, value: float, amount: float) -> float:
    """Pay ``amount`` by scaling all lots pro-rata. Returns the new value.

    Scaling preserves relative weights and per-share basis. A debit larger than
    the portfolio is clamped to wipe it out (shouldn't happen with sane taxes).
    """
    if amount <= _VALUE_TOL or value <= _VALUE_TOL:
        return value
    factor = max(0.0, (value - amount) / value)
    for lots in ledger._lots.values():  # noqa: SLF001 - same package, intentional
        for lot in lots:
            lot.qty *= factor
    return value * factor


def _dividends_by_ticker(dividends: pd.DataFrame | None) -> dict[str, list[tuple[date, float]]]:
    """Index a long-form fetch_dividends frame as ticker -> [(ex_date, amount)]."""
    out: dict[str, list[tuple[date, float]]] = {}
    if dividends is None or dividends.empty:
        return out
    for ticker, grp in dividends.groupby("Ticker"):
        rows = []
        for _, r in grp.iterrows():
            ex = r["ExDate"]
            ex = ex if isinstance(ex, date) else pd.Timestamp(ex).date()
            rows.append((ex, float(r["Amount"])))
        out[str(ticker)] = sorted(rows)
    return out


def run_tax_aware_backtest(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    regime: TaxRegime | None = None,
    rebalance_policy: RebalancePolicy | None = None,
    lot_selector: "LotSelector | str" = LotSelector.FIFO,
    dividends: pd.DataFrame | None = None,
    portfolio_value: float | None = None,
    transaction_cost_pct: float = 0.0,
) -> TaxBacktestResult:
    """Simulate the strategy with full tax accounting.

    Parameters
    ----------
    prices:
        Daily total-return price DataFrame, one column per allocation ticker.
    allocation:
        ``{ticker: weight}``, weights sum to 1.0.
    regime:
        Tax regime; defaults to ``TaxRegime.us()``. Pass ``TaxRegime.none()``
        for the zero-tax baseline (ISA-style) used in the D.18 A/B sweep.
    rebalance_policy:
        Defaults to ``RebalancePolicy.monthly_unconditional()``.
    lot_selector:
        Which lots a sell consumes (FIFO default = Alpaca reality).
    dividends:
        Long-form frame from ``engine.data.fetch_dividends`` (or None for no
        dividend tax).
    transaction_cost_pct:
        Per-trade cost as a fraction of traded notional.
    """
    regime = regime or TaxRegime.us()
    rebalance_policy = rebalance_policy or RebalancePolicy.monthly_unconditional()
    selector = LotSelector.coerce(lot_selector)
    if portfolio_value is None:
        portfolio_value = config.INITIAL_PORTFOLIO_VALUE

    tickers = list(allocation)
    freq = pandas_resample_frequency(config.DATA_FREQUENCY)
    monthly = prices[tickers].resample(freq).last().dropna()
    if monthly.empty:
        raise ValueError("No overlapping monthly data found. Check date range.")

    div_map = _dividends_by_ticker(dividends)
    ledger = LotLedger(selector)

    # Seed initial lots at the first month-end.
    first_date = monthly.index[0]
    first_row = monthly.iloc[0]
    for t, w in allocation.items():
        ledger.buy(t, (portfolio_value * w) / float(first_row[t]),
                   float(first_row[t]), first_date.date())

    cum_tax = 0.0
    monthly_records: list[dict] = []
    event_records: list[dict] = []
    prev_date = first_date

    dates = list(monthly.index)
    for i, when in enumerate(dates):
        row = monthly.loc[when]
        when_d = when.date()
        price_of = {t: float(row[t]) for t in tickers}

        period_tax = {"st": 0.0, "lt": 0.0, "div": 0.0, "mtm": 0.0}
        st_gain = lt_gain = div_income = 0.0

        # --- 1. dividends with ex-date in (prev_date, when] ---
        if i > 0:
            for t in tickers:
                shares = ledger.total_quantity(t)
                if shares <= 0:
                    continue
                for ex, amt in div_map.get(t, ()):
                    if prev_date.date() < ex <= when_d:
                        cash_div = shares * amt
                        res = compute_tax_on_event(
                            DividendEvent(t, cash_div, ex), regime)
                        period_tax["div"] += res.dividend_tax + res.niit_tax
                        div_income += res.dividend_income

        value = sum(ledger.total_quantity(t) * price_of[t] for t in tickers)

        # pay dividend tax pro-rata
        if period_tax["div"] > 0:
            value = _debit(ledger, value, period_tax["div"])

        # --- 2. drift check ---
        weights = {t: (ledger.total_quantity(t) * price_of[t]) / value if value > 0 else 0.0
                   for t in tickers}
        # never rebalance on the very first month (initial buy already at target)
        do_rebalance = i > 0 and rebalance_policy.should_rebalance(weights, allocation)

        trade_notional = 0.0
        sale_tax = 0.0
        if do_rebalance:
            target_val = {t: value * w for t, w in allocation.items()}
            # sells first (realize gains/tax)
            for t in tickers:
                cur_val = ledger.total_quantity(t) * price_of[t]
                if cur_val - target_val[t] > _VALUE_TOL:
                    sell_shares = (cur_val - target_val[t]) / price_of[t]
                    event = ledger.sell_event(t, sell_shares, price_of[t], when_d)
                    res = compute_tax_on_event(event, regime)
                    sale_tax += res.total_tax
                    st_gain += res.short_term_gain
                    lt_gain += res.long_term_gain
                    trade_notional += cur_val - target_val[t]
            # buys (no tax)
            for t in tickers:
                cur_val = ledger.total_quantity(t) * price_of[t]
                if target_val[t] - cur_val > _VALUE_TOL:
                    buy_shares = (target_val[t] - cur_val) / price_of[t]
                    ledger.buy(t, buy_shares, price_of[t], when_d)
                    trade_notional += target_val[t] - cur_val

            cost = trade_notional * transaction_cost_pct
            value = _debit(ledger, value, sale_tax + cost)

        # --- 3. GSG §1256 year-end mark-to-market (last rebalance date of year) ---
        is_year_end = (i == len(dates) - 1) or (dates[i + 1].year != when.year)
        if is_year_end and "GSG" in tickers and regime.classify("GSG") is AssetTaxClass.SECTION_1256:
            gsg_shares = ledger.total_quantity("GSG")
            if gsg_shares > 0:
                event = ledger.sell_event("GSG", gsg_shares, price_of["GSG"], when_d,
                                          is_mark_to_market=True)
                res = compute_tax_on_event(event, regime)
                # rebuy at same price → basis reset, share count unchanged
                ledger.buy("GSG", gsg_shares, price_of["GSG"], when_d)
                period_tax["mtm"] += res.total_tax
                st_gain += res.short_term_gain
                lt_gain += res.long_term_gain
                value = sum(ledger.total_quantity(t) * price_of[t] for t in tickers)
                value = _debit(ledger, value, res.total_tax)

        value = sum(ledger.total_quantity(t) * price_of[t] for t in tickers)
        period_total_tax = sale_tax + period_tax["div"] + period_tax["mtm"]
        cum_tax += period_total_tax

        monthly_records.append({
            "Date": when,
            "Value": round(value, 2),
            "Period Tax": round(period_total_tax, 2),
            "Cumulative Tax Paid": round(cum_tax, 2),
            "Sale Tax": round(sale_tax, 2),
            "Dividend Tax": round(period_tax["div"], 2),
            "MTM Tax": round(period_tax["mtm"], 2),
            "ST Gain": round(st_gain, 2),
            "LT Gain": round(lt_gain, 2),
            "Dividend Income": round(div_income, 2),
            "Rebalanced": bool(do_rebalance),
            "Trade Notional": round(trade_notional, 2),
        })
        if do_rebalance or period_total_tax > 0:
            event_records.append({
                "Date": when,
                "Value": round(value, 2),
                "Rebalanced": bool(do_rebalance),
                "Trade Notional": round(trade_notional, 2),
                "Sale Tax": round(sale_tax, 2),
                "Dividend Tax": round(period_tax["div"], 2),
                "MTM Tax": round(period_tax["mtm"], 2),
                "ST Gain": round(st_gain, 2),
                "LT Gain": round(lt_gain, 2),
            })
        prev_date = when

    monthly_df = pd.DataFrame(monthly_records).set_index("Date")
    monthly_df["Monthly Ret (%)"] = monthly_df["Value"].pct_change() * 100
    events_df = pd.DataFrame(event_records)

    # annual tax summary
    m = monthly_df.copy()
    m["Year"] = m.index.year
    tax_summary = m.groupby("Year").agg(
        ST_Gain=("ST Gain", "sum"),
        LT_Gain=("LT Gain", "sum"),
        Dividend_Income=("Dividend Income", "sum"),
        Sale_Tax=("Sale Tax", "sum"),
        Dividend_Tax=("Dividend Tax", "sum"),
        MTM_Tax=("MTM Tax", "sum"),
        Total_Tax=("Period Tax", "sum"),
        Rebalances=("Rebalanced", "sum"),
    ).round(2).reset_index()

    return TaxBacktestResult(
        monthly=monthly_df,
        events=events_df,
        tax_summary=tax_summary,
        regime_name=regime.name,
        selector=selector.value,
        policy_label=rebalance_policy.label,
    )
