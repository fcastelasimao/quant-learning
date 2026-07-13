"""
daily_tax_backtest.py — daily-resolution tax-aware backtest (L.51).

Why a separate engine
---------------------
The production live system checks drift every day with a 31-day minimum gate
between rebalances. The monthly engine (engine/tax_backtest.py) only checks
drift on month-end prices. This engine validates that the monthly research
result is a valid proxy before F.26 flips the live policy.

Neither engine/backtest.py nor engine/tax_backtest.py is touched.

What it models
--------------
* Drift checked every trading day; rebalance fires when the RebalancePolicy
  triggers AND at least min_rebalance_days calendar days have passed since the
  last rebalance (mirrors the live 31-day cadence gate).
* Same tax mechanics as tax_backtest.py: per-lot LotLedger, compute_tax_on_event,
  GSG §1256 year-end MTM on the last calendar trading day of each year.
* Dividends taxed on ex-date (exact day match against the price index).

DailyTaxBacktestResult.monthly
-------------------------------
The `monthly` property resamples daily_records["Value"] to month-end so
existing helpers (e.g., `_window_calmar` in threshold sweep scripts) that
operate on monthly.Value work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .backtest import RebalancePolicy
from .calendar import pandas_resample_frequency
from .lot_ledger import LotLedger, LotSelector
from .tax import (
    AssetTaxClass,
    DividendEvent,
    TaxRegime,
    compute_tax_on_event,
)
from . import config

_VALUE_TOL = 1e-12


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DailyTaxBacktestResult:
    """Output of :func:`run_daily_tax_backtest`.

    Attributes
    ----------
    daily_records:
        DataFrame indexed by DatetimeIndex (one row per trading day). Columns:
        ``Value``, ``Rebalanced``, ``Trade Notional``, ``Period Tax``,
        ``Cumulative Tax Paid``, ``Sale Tax``, ``Dividend Tax``, ``MTM Tax``,
        ``ST Gain``, ``LT Gain``.
    rebalance_dates:
        Calendar dates on which a rebalance actually fired.
    monthly:
        Month-end resample of daily_records with columns ``Value`` and
        ``Cumulative Tax Paid``. Compatible with tax_backtest.TaxBacktestResult
        for callers that only read those two columns (e.g., _window_calmar).
    """

    daily_records: pd.DataFrame
    rebalance_dates: list[date]
    regime_name: str
    selector: str
    policy_label: str

    @property
    def monthly(self) -> pd.DataFrame:
        freq = pandas_resample_frequency("ME")
        m = self.daily_records.resample(freq).last()[["Value", "Cumulative Tax Paid"]]
        return m.dropna(subset=["Value"])


# ---------------------------------------------------------------------------
# Private helpers (mirrored from tax_backtest; re-declared to keep the
# engine modules independent — daily_tax_backtest must not import from
# tax_backtest to avoid coupling two separate engine implementations).
# ---------------------------------------------------------------------------

def _debit(ledger: LotLedger, value: float, amount: float) -> float:
    """Pay ``amount`` by scaling all lots pro-rata. Returns the new value."""
    if amount <= _VALUE_TOL or value <= _VALUE_TOL:
        return value
    factor = max(0.0, (value - amount) / value)
    for lots in ledger._lots.values():  # noqa: SLF001
        for lot in lots:
            lot.qty *= factor
    return value * factor


def _dividends_by_date(
    dividends: pd.DataFrame | None,
) -> dict[date, dict[str, float]]:
    """Index fetch_dividends output as {ex_date: {ticker: per_share_amount}}.

    The daily engine checks dividends by exact date match, so this index is
    date-keyed (vs. the monthly engine's ticker-keyed sorted list).
    """
    out: dict[date, dict[str, float]] = {}
    if dividends is None or dividends.empty:
        return out
    for _, row in dividends.iterrows():
        ex = row["ExDate"]
        ex = ex if isinstance(ex, date) else pd.Timestamp(ex).date()
        t = str(row["Ticker"])
        amt = float(row["Amount"])
        out.setdefault(ex, {})[t] = out.get(ex, {}).get(t, 0.0) + amt
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_daily_tax_backtest(
    prices_daily: pd.DataFrame,
    allocation: dict[str, float],
    *,
    regime: TaxRegime | None = None,
    rebalance_policy: RebalancePolicy | None = None,
    lot_selector: "LotSelector | str" = LotSelector.FIFO,
    dividends: pd.DataFrame | None = None,
    portfolio_value: float | None = None,
    transaction_cost_pct: float = 0.0,
    min_rebalance_days: int = 31,
) -> DailyTaxBacktestResult:
    """Simulate the strategy at daily resolution with full tax accounting.

    Parameters
    ----------
    prices_daily:
        Daily total-return price DataFrame (one column per ticker, DatetimeIndex).
        NaNs are forward-filled before the loop.
    allocation:
        ``{ticker: weight}``, weights sum to 1.0.
    regime:
        Tax regime; defaults to ``TaxRegime.us()``.
    rebalance_policy:
        Defaults to ``RebalancePolicy.monthly_unconditional()``. In the daily
        engine this fires the first day after ``min_rebalance_days`` have elapsed
        since the last rebalance (not on calendar month-end).
    lot_selector:
        FIFO default = Alpaca broker reality.
    dividends:
        Long-form frame from ``engine.data.fetch_dividends`` (columns: Ticker,
        ExDate, Amount). Pass None for no dividend tax.
    transaction_cost_pct:
        Per-trade cost as a fraction of traded notional.
    min_rebalance_days:
        Minimum calendar days between rebalances (mirrors live 31-day cadence
        gate). The initial buy on day 0 counts as the first rebalance.
    """
    regime = regime or TaxRegime.us()
    rebalance_policy = rebalance_policy or RebalancePolicy.monthly_unconditional()
    selector = LotSelector.coerce(lot_selector)
    if portfolio_value is None:
        portfolio_value = config.INITIAL_PORTFOLIO_VALUE

    tickers = list(allocation)
    prices = prices_daily[tickers].ffill().dropna()
    if prices.empty:
        raise ValueError("No valid daily price data found. Check date range and tickers.")

    div_map = _dividends_by_date(dividends)
    ledger = LotLedger(selector)

    # Seed initial lots on the first day (counts as the first rebalance).
    first_ts = prices.index[0]
    first_row = prices.iloc[0]
    for t, w in allocation.items():
        ledger.buy(
            t,
            (portfolio_value * w) / float(first_row[t]),
            float(first_row[t]),
            first_ts.date(),
        )

    cum_tax = 0.0
    last_rebalance_date: date = first_ts.date()
    daily_records: list[dict] = []
    rebalance_dates: list[date] = []

    dates = list(prices.index)
    n = len(dates)

    for i, when in enumerate(dates):
        row = prices.iloc[i]
        when_d = when.date()
        price_of = {t: float(row[t]) for t in tickers}

        period_tax_div = 0.0
        period_tax_mtm = 0.0
        sale_tax = 0.0
        st_gain = lt_gain = 0.0

        # --- 1. Dividends with ex-date == today ---
        if i > 0:
            for t, amt in div_map.get(when_d, {}).items():
                if t not in tickers:
                    continue
                shares = ledger.total_quantity(t)
                if shares <= 0:
                    continue
                cash_div = shares * amt
                res = compute_tax_on_event(DividendEvent(t, cash_div, when_d), regime)
                period_tax_div += res.dividend_tax + res.niit_tax

        value = sum(ledger.total_quantity(t) * price_of[t] for t in tickers)

        if period_tax_div > 0:
            value = _debit(ledger, value, period_tax_div)

        # --- 2. Drift check with min-days gate ---
        weights = {
            t: (ledger.total_quantity(t) * price_of[t]) / value
            if value > 0 else 0.0
            for t in tickers
        }
        days_since = (when_d - last_rebalance_date).days
        gate_ok = i > 0 and days_since >= min_rebalance_days
        do_rebalance = gate_ok and rebalance_policy.should_rebalance(weights, allocation)

        trade_notional = 0.0
        if do_rebalance:
            target_val = {t: value * w for t, w in allocation.items()}
            # sells first (realize gains → tax)
            for t in tickers:
                cur_val = ledger.total_quantity(t) * price_of[t]
                if cur_val - target_val[t] > _VALUE_TOL:
                    sell_qty = (cur_val - target_val[t]) / price_of[t]
                    event = ledger.sell_event(t, sell_qty, price_of[t], when_d)
                    res = compute_tax_on_event(event, regime)
                    sale_tax += res.total_tax
                    st_gain += res.short_term_gain
                    lt_gain += res.long_term_gain
                    trade_notional += cur_val - target_val[t]
            # buys (no tax event)
            for t in tickers:
                cur_val = ledger.total_quantity(t) * price_of[t]
                if target_val[t] - cur_val > _VALUE_TOL:
                    buy_qty = (target_val[t] - cur_val) / price_of[t]
                    ledger.buy(t, buy_qty, price_of[t], when_d)
                    trade_notional += target_val[t] - cur_val

            cost = trade_notional * transaction_cost_pct
            value = _debit(ledger, value, sale_tax + cost)
            last_rebalance_date = when_d
            rebalance_dates.append(when_d)

        # --- 3. GSG §1256 year-end mark-to-market ---
        is_year_end = (i == n - 1) or (dates[i + 1].year != when.year)
        if (
            is_year_end
            and "GSG" in tickers
            and regime.classify("GSG") is AssetTaxClass.SECTION_1256
        ):
            gsg_qty = ledger.total_quantity("GSG")
            if gsg_qty > 0:
                event = ledger.sell_event(
                    "GSG", gsg_qty, price_of["GSG"], when_d, is_mark_to_market=True
                )
                res = compute_tax_on_event(event, regime)
                ledger.buy("GSG", gsg_qty, price_of["GSG"], when_d)
                period_tax_mtm += res.total_tax
                st_gain += res.short_term_gain
                lt_gain += res.long_term_gain
                value = sum(ledger.total_quantity(t) * price_of[t] for t in tickers)
                value = _debit(ledger, value, res.total_tax)

        value = sum(ledger.total_quantity(t) * price_of[t] for t in tickers)
        period_total_tax = sale_tax + period_tax_div + period_tax_mtm
        cum_tax += period_total_tax

        daily_records.append({
            "Date": when,
            "Value": round(value, 4),
            "Rebalanced": do_rebalance,
            "Trade Notional": round(trade_notional, 4),
            "Period Tax": round(period_total_tax, 4),
            "Cumulative Tax Paid": round(cum_tax, 4),
            "Sale Tax": round(sale_tax, 4),
            "Dividend Tax": round(period_tax_div, 4),
            "MTM Tax": round(period_tax_mtm, 4),
            "ST Gain": round(st_gain, 4),
            "LT Gain": round(lt_gain, 4),
        })

    df = pd.DataFrame(daily_records).set_index("Date")
    return DailyTaxBacktestResult(
        daily_records=df,
        rebalance_dates=rebalance_dates,
        regime_name=regime.name,
        selector=selector.value,
        policy_label=rebalance_policy.label,
    )
