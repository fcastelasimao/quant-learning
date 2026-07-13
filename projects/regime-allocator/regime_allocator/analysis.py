import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def performance_metrics(returns, name="Strategy"):
    """Compute standard performance metrics from a monthly return series."""
    ann_ret = (1 + returns).prod() ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg = returns[returns < 0]
    downside_vol = neg.std() * np.sqrt(12) if len(neg) > 0 else 0
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdowns = (cumulative - peak) / peak
    max_dd = drawdowns.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    total_ret = cumulative.iloc[-1] - 1
    years = len(returns) / 12

    return {
        "name": name,
        "total_return": total_ret,
        "cagr": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "best_month": returns.max(),
        "worst_month": returns.min(),
        "pct_positive": (returns > 0).mean(),
        "years": years,
    }


def comparison_table(metrics_list):
    """Pretty-print a comparison table from a list of metric dicts."""
    df = pd.DataFrame(metrics_list).set_index("name")
    fmt = {
        "total_return": "{:.1%}",
        "cagr": "{:.1%}",
        "annual_vol": "{:.1%}",
        "sharpe": "{:.2f}",
        "sortino": "{:.2f}",
        "max_drawdown": "{:.1%}",
        "calmar": "{:.2f}",
        "best_month": "{:.1%}",
        "worst_month": "{:.1%}",
        "pct_positive": "{:.1%}",
        "years": "{:.1f}",
    }
    for col, f in fmt.items():
        if col in df.columns:
            df[col] = df[col].map(f.format)
    return df


def plot_results(results_df, benchmarks, config, save_path=None):
    """
    Three-panel figure:
      1. Equity curves (strategy + benchmarks)
      2. Regime probabilities over time
      3. Asset allocation over time
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Panel 1: Equity curves ---
    ax = axes[0]
    ax.plot(results_df.index, results_df["portfolio_value"], linewidth=2, label="Regime Allocator")
    for name, (vals, _) in benchmarks.items():
        aligned = vals.reindex(results_df.index).dropna()
        if len(aligned) > 0:
            scaled = aligned / aligned.iloc[0] * results_df["portfolio_value"].iloc[0]
            first_common = results_df.index[results_df.index.isin(aligned.index)][0]
            scale_factor = results_df.loc[first_common, "portfolio_value"] / aligned.loc[first_common]
            scaled = aligned * scale_factor
            ax.plot(scaled.index, scaled, linewidth=1, alpha=0.7, label=name)
    ax.set_ylabel("Growth of $1")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.set_title("Regime-Adaptive Multi-Asset Allocator — Walk-Forward Backtest")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Regime probabilities ---
    ax = axes[1]
    regime_cols = [c for c in results_df.columns if c.startswith("regime_p")]
    n_regimes = len(regime_cols)
    if n_regimes == 2:
        regime_labels = {0: "Risk-Off", 1: "Risk-On"}
        colors = {"regime_p0": "#d62728", "regime_p1": "#2ca02c"}
    else:
        regime_labels = {0: "Crisis", 1: "Transition", 2: "Growth"}
        colors = {"regime_p0": "#d62728", "regime_p1": "#ff7f0e", "regime_p2": "#2ca02c"}
    bottom = np.zeros(len(results_df))
    for col in regime_cols:
        idx = int(col.replace("regime_p", ""))
        label = regime_labels.get(idx, col)
        ax.fill_between(
            results_df.index,
            bottom,
            bottom + results_df[col],
            label=label,
            alpha=0.7,
            color=colors.get(col, None),
        )
        bottom += results_df[col].values
    ax.set_ylabel("Regime Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Asset allocation ---
    ax = axes[2]
    weight_cols = [c for c in results_df.columns if c.startswith("w_")]
    weight_data = results_df[weight_cols]
    weight_data.columns = [c.replace("w_", "") for c in weight_cols]
    cash = 1 - weight_data.sum(axis=1)
    if (cash > 0.01).any():
        weight_data["Cash"] = cash
    ax.stackplot(
        weight_data.index,
        weight_data.T.values,
        labels=weight_data.columns,
        alpha=0.8,
    )
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_drawdown(results_df, save_path=None):
    """Underwater equity curve."""
    fig, ax = plt.subplots(figsize=(14, 4))
    cumulative = results_df["portfolio_value"]
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    ax.fill_between(dd.index, dd, 0, color="#d62728", alpha=0.5)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def sensitivity_analysis(daily_prices, daily_features, config_factory, param_name, param_values):
    """
    Run backtest across a range of parameter values.
    config_factory(value) should return a StrategyConfig.
    Returns a DataFrame of metrics per parameter value.
    """
    from .backtest import run_walkforward

    rows = []
    for val in param_values:
        cfg = config_factory(val)
        results = run_walkforward(daily_prices, daily_features, cfg)
        rets = results["return"]
        m = performance_metrics(rets, name=f"{param_name}={val}")
        m["param_value"] = val
        rows.append(m)
    return pd.DataFrame(rows)
