"""
failed_strategies/rolling_rp/rolling_rp.py
==========================================
CLOSED INVESTIGATION — Rolling Risk Parity.

Conclusion: Rolling RP converges to approximately the same weights as static
RP over every OOS split (2018, 2020, 2022). No consistent Calmar improvement.
Static weights are simpler, more interpretable, and equally good — kept for
production.

See failed_strategies/rolling_rp/README.md for the full investigation log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import config

SIXTY_FORTY_EQUITY = 0.60
SIXTY_FORTY_BOND   = 0.40


def run_backtest_rolling_rp(prices: pd.DataFrame,
                            benchmark_prices: pd.Series,
                            tickers: list[str],
                            portfolio_value: float | None = None,
                            tlt_prices: pd.Series | None = None,
                            transaction_cost_pct: float = 0.0,
                            tax_drag_pct: float = 0.0,
                            rp_lookback_years: float = 5.0,
                            rp_recompute_freq: str = "QS",
                            ) -> tuple[pd.DataFrame, list[dict]]:
    """
    Backtest with rolling risk parity: recompute RP weights every quarter
    (or other frequency) from trailing covariance, then rebalance to new weights.

    Parameters
    ----------
    prices              : daily prices for all tickers
    benchmark_prices    : daily SPY prices
    tickers             : list of tickers in the portfolio
    rp_lookback_years   : covariance estimation window (default 5yr)
    rp_recompute_freq   : how often to recompute weights
                          "QS" = quarter start, "MS" = month start, "YS" = year start

    Returns
    -------
    (backtest_df, weight_history)
    backtest_df    : same format as run_backtest output
    weight_history : list of {date, weights_dict} for each recomputation
    """
    from engine.optimiser import compute_risk_parity_weights

    if portfolio_value is None:
        portfolio_value = config.INITIAL_PORTFOLIO_VALUE

    # Resample to DATA_FREQUENCY for the backtest
    monthly = prices[tickers].resample(config.DATA_FREQUENCY).last().dropna()
    bench = benchmark_prices.resample(config.DATA_FREQUENCY).last().dropna()
    tlt_monthly = (tlt_prices.resample(config.DATA_FREQUENCY).last().dropna()
                   if tlt_prices is not None else None)

    common = monthly.index.intersection(bench.index)
    if tlt_monthly is not None:
        common = common.intersection(tlt_monthly.index)
    monthly = monthly.loc[common]
    bench = bench.loc[common]
    if tlt_monthly is not None:
        tlt_monthly = tlt_monthly.loc[common]

    # Determine recomputation dates.
    # resample().first().index returns anchor dates (e.g. 2010-01-01 for "QS"),
    # not the actual month-end dates in monthly.index.  We need the first
    # *actual* date in each period so the membership test `date in
    # recompute_dates` can match against monthly.iterrows().
    recompute_dates = set()
    for _, group in monthly.groupby(pd.Grouper(freq=rp_recompute_freq)):
        if len(group) > 0:
            recompute_dates.add(group.index[0])

    # Compute initial RP weights from the first available window
    initial_weights = compute_risk_parity_weights(
        prices, tickers, estimation_years=rp_lookback_years,
        end_date=str(monthly.index[0].date()),
        min_weight=0.02,
    )
    allocation = dict(initial_weights)
    weight_history = [{"date": monthly.index[0], **allocation}]

    # Initialise holdings
    first_row = monthly.iloc[0]
    bench_shares = portfolio_value / float(bench.iloc[0])
    aw_holdings = {t: (portfolio_value * allocation[t]) / float(first_row[t])
                   for t in tickers}
    bh_holdings = {t: (portfolio_value * allocation[t]) / float(first_row[t])
                   for t in tickers}

    sixty_forty_spy = None
    sixty_forty_tlt = None
    sixty_forty_prev_year = None
    if tlt_monthly is not None:
        sixty_forty_spy = portfolio_value * SIXTY_FORTY_EQUITY / float(bench.iloc[0])
        sixty_forty_tlt = portfolio_value * SIXTY_FORTY_BOND / float(tlt_monthly.iloc[0])
        sixty_forty_prev_year = monthly.index[0].year

    aw_prev_year = monthly.index[0].year

    # Note: this loop is intentionally iterative due to stateful monthly rebalancing
    # with rolling RP weight recomputation and 60/40 annual rebalance logic.
    records = []
    for date_idx, (date, row) in enumerate(monthly.iterrows()):
        # --- Recompute RP weights at recomputation dates ---
        if date in recompute_dates and date_idx > 0:
            new_weights = compute_risk_parity_weights(
                prices, tickers, estimation_years=rp_lookback_years,
                end_date=str(date.date()),
                min_weight=0.02,
            )
            allocation = dict(new_weights)
            weight_history.append({"date": date, **allocation})

        # --- 60/40 annual rebalance ---
        if sixty_forty_spy is not None and date.year != sixty_forty_prev_year:
            current_6040 = (sixty_forty_spy * float(bench.loc[date])
                            + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            sixty_forty_spy = current_6040 * SIXTY_FORTY_EQUITY / float(bench.loc[date])
            sixty_forty_tlt = current_6040 * SIXTY_FORTY_BOND / float(tlt_monthly.loc[date])
            sixty_forty_prev_year = date.year

        aw_value = sum(sh * float(row[t]) for t, sh in aw_holdings.items())

        if tax_drag_pct > 0 and date.year != aw_prev_year:
            aw_value *= (1 - tax_drag_pct)
            aw_prev_year = date.year

        if transaction_cost_pct > 0:
            trade_values = sum(
                abs((aw_value * allocation[t]) - (aw_holdings[t] * float(row[t])))
                for t in tickers
            )
            aw_value -= trade_values * transaction_cost_pct

        bh_value = sum(sh * float(row[t]) for t, sh in bh_holdings.items())
        spy_value = bench_shares * float(bench.loc[date])

        bh_weights = {t: (bh_holdings[t] * float(row[t])) / bh_value for t in tickers}

        record = {
            "Date": date,
            "All Weather Value": round(aw_value, 2),
            "Buy & Hold All Weather": round(bh_value, 2),
            "S&P 500 Value": round(spy_value, 2),
        }
        if sixty_forty_spy is not None:
            sixty_forty_value = (sixty_forty_spy * float(bench.loc[date])
                                 + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            record["60/40 Value"] = round(sixty_forty_value, 2)
        for t in tickers:
            record[f"B&H {t} Weight (%)"] = round(bh_weights[t] * 100, 1)
        records.append(record)

        # Rebalance to current RP allocation
        for t in tickers:
            aw_holdings[t] = (aw_value * allocation[t]) / float(row[t])

    df = pd.DataFrame(records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100
    if "60/40 Value" in df.columns:
        df["60/40 Value Monthly Ret (%)"] = df["60/40 Value"].pct_change() * 100

    return df, weight_history
