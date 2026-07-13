import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from datetime import date
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from engine import config
    from engine.data import fetch_prices_from_fmp_db, get_price_provenance
    from engine.explorers import data_quality_report
    from engine.plotting import plot_normalised_prices

    return (
        PROJECT_ROOT,
        data_quality_report,
        date,
        fetch_prices_from_fmp_db,
        get_price_provenance,
        json,
        mo,
        pd,
        plot_normalised_prices,
    )


@app.cell
def _(PROJECT_ROOT, json, mo):
    strategies_path = PROJECT_ROOT / "strategies.json"
    strategies = json.loads(strategies_path.read_text())["strategies"]
    strategy_ids = list(strategies)
    selected_strategy = mo.ui.dropdown(
        options=strategy_ids,
        value="6asset_tip_gsg_rpavg",
        label="Strategy",
    )
    selected_strategy
    return selected_strategy, strategies


@app.cell
def _(date, mo):
    start = mo.ui.text(value="2006-01-01", label="Start")
    end = mo.ui.text(value=date.today().isoformat(), label="End")
    price_column = mo.ui.dropdown(
        options=["adj_close", "close"],
        value="adj_close",
        label="FMP price column",
    )
    mo.hstack([start, end, price_column])
    return end, price_column, start


@app.cell
def _(
    end,
    fetch_prices_from_fmp_db,
    get_price_provenance,
    price_column,
    selected_strategy,
    start,
    strategies,
):
    allocation = strategies[selected_strategy.value]["allocation"]
    tickers = list(allocation)
    prices = fetch_prices_from_fmp_db(
        tickers,
        start.value,
        end.value,
        price_column=price_column.value,
    )
    provenance = get_price_provenance(prices)
    returns = prices.pct_change()
    allocation
    return allocation, prices, provenance, returns


@app.cell
def _(mo, prices, provenance):
    _basis = provenance.get("price_column", "unknown")
    _warning = (
        "\n\n> Warning: `close` is useful for raw data inspection, but production validation should use `adj_close`."
        if _basis == "close"
        else ""
    )
    mo.md(f"""
    ## FMP ETF Data

    Rows: **{len(prices):,}**<br>
    Window: **{prices.index.min().date()}** to **{prices.index.max().date()}**<br>
    Price basis: **{_basis}**<br>
    Retrieved on: **{provenance.get("retrieved_on", "unknown")}**
    {_warning}
    """)
    return


@app.cell
def _(mo, prices, provenance):
    _basis = provenance.get("price_column", "prices")
    mo.ui.table(prices.tail(20).round(2), label=f"Latest daily {_basis}")
    return


@app.cell
def _(allocation, mo, pd):
    alloc_df = pd.DataFrame(
        [{"Ticker": t, "Weight": w, "Weight (%)": f"{w:.1%}"} for t, w in allocation.items()]
    )
    mo.ui.table(alloc_df, label="Current allocation")
    return


@app.cell
def _(data_quality_report, mo, prices):
    mo.ui.table(data_quality_report(prices).reset_index(names="Ticker"), label="Data quality")
    return


@app.cell
def _(mo, pd, provenance):
    _rows = []
    for _key, _value in provenance.items():
        if _key == "missing_fraction_by_column":
            continue
        if isinstance(_value, list):
            _value = ", ".join(map(str, _value))
        _rows.append({"Field": _key, "Value": _value})
    mo.ui.table(pd.DataFrame(_rows), label="Price provenance")
    return


@app.cell
def _(plot_normalised_prices, prices, provenance):
    plot_normalised_prices(
        prices,
        f"FMP {provenance.get('price_column', 'prices')} prices, indexed to 100",
    )
    return


@app.cell
def _(mo, returns):
    summary = returns.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    mo.ui.table(summary.round(4).reset_index(names="Ticker"), label="Daily return summary")
    return


if __name__ == "__main__":
    app.run()
