from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from filters import rolling_fft_components, rolling_fft_signal, rolling_ssa_signal


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    params: dict[str, float | int | None]


def default_method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("raw_return", "raw", {}),
        MethodSpec("ma_20", "ma", {"window": 20}),
        MethodSpec("ma_60", "ma", {"window": 60}),
        MethodSpec("ema_20", "ema", {"span": 20}),
        MethodSpec("ema_60", "ema", {"span": 60}),
        MethodSpec("fft_e90_w128", "fft", {"window": 128, "retained_energy": 0.90, "top_k": None}),
        MethodSpec("fft_e95_w128", "fft", {"window": 128, "retained_energy": 0.95, "top_k": None}),
        MethodSpec("fft_noise_e95_w128", "fft_noise", {"window": 128, "retained_energy": 0.95, "top_k": None}),
        MethodSpec("fft_e98_w128", "fft", {"window": 128, "retained_energy": 0.98, "top_k": None}),
        MethodSpec("fft_noise_e98_w128", "fft_noise", {"window": 128, "retained_energy": 0.98, "top_k": None}),
        MethodSpec("fft_top8_w128", "fft", {"window": 128, "retained_energy": None, "top_k": 8}),
        MethodSpec("fft_noise_top8_w128", "fft_noise", {"window": 128, "retained_energy": None, "top_k": 8}),
        MethodSpec("fft_e95_w252", "fft", {"window": 252, "retained_energy": 0.95, "top_k": None}),
        MethodSpec("ssa_e90_w128_l40", "ssa", {"window": 128, "window_length": 40, "retained_energy": 0.90}),
        MethodSpec("ssa_e95_w128_l40", "ssa", {"window": 128, "window_length": 40, "retained_energy": 0.95}),
    ]


def get_method_spec(name: str, specs: list[MethodSpec] | None = None) -> MethodSpec:
    specs = specs or default_method_specs()
    for spec in specs:
        if spec.name == name:
            return spec
    valid = ", ".join(spec.name for spec in specs)
    raise ValueError(f"Unknown method {name!r}. Valid methods: {valid}")


def best_method_by_family(summary: pd.DataFrame, family: str) -> str | None:
    family_rows = summary[summary["family"] == family].copy()
    if family_rows.empty:
        return None
    return str(family_rows.sort_values(["sharpe", "correlation"], ascending=False).iloc[0]["name"])


def compute_smoothed_log_price(log_price: pd.Series, spec: MethodSpec) -> pd.Series:
    if spec.family == "raw":
        return log_price.copy()
    if spec.family == "ma":
        return log_price.rolling(int(spec.params["window"])).mean()
    if spec.family == "ema":
        return log_price.ewm(span=int(spec.params["span"]), adjust=False).mean()
    if spec.family == "fft":
        return rolling_fft_signal(
            log_price,
            window=int(spec.params["window"]),
            retained_energy=spec.params.get("retained_energy"),
            top_k=spec.params.get("top_k"),
        )
    if spec.family == "fft_noise":
        _, residual = rolling_fft_components(
            log_price,
            window=int(spec.params["window"]),
            retained_energy=spec.params.get("retained_energy"),
            top_k=spec.params.get("top_k"),
        )
        return residual
    if spec.family == "ssa":
        return rolling_ssa_signal(
            log_price,
            window=int(spec.params["window"]),
            window_length=int(spec.params["window_length"]),
            retained_energy=spec.params.get("retained_energy"),
        )
    raise ValueError(f"Unknown method family: {spec.family}")


def _sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() <= 1e-12:
        return 0.0
    return float(clean.mean() / clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _cagr(equity: pd.Series) -> float:
    clean = equity.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return 0.0
    years = (len(clean) - 1) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1)


def evaluate_method(
    prices: pd.Series,
    spec: MethodSpec,
    *,
    cost_bps: float = 1.0,
) -> tuple[dict[str, float | str | int | None], pd.Series]:
    prices = prices.astype(float).sort_index().dropna()
    log_price = np.log(prices)
    log_returns = log_price.diff()
    simple_returns = prices.pct_change(fill_method=None)

    smoothed = compute_smoothed_log_price(log_price, spec)
    predicted_next_log_return = smoothed.diff()
    target_next_log_return = log_returns.shift(-1)

    prediction_frame = pd.DataFrame(
        {
            "prediction": predicted_next_log_return,
            "target_next": target_next_log_return,
        }
    ).dropna()

    if prediction_frame.empty:
        corr = np.nan
        directional_accuracy = np.nan
        rmse = np.nan
        observations = 0
    else:
        corr = float(prediction_frame["prediction"].corr(prediction_frame["target_next"]))
        directional_accuracy = float(
            (np.sign(prediction_frame["prediction"]) == np.sign(prediction_frame["target_next"])).mean()
        )
        rmse = float(np.sqrt(((prediction_frame["prediction"] - prediction_frame["target_next"]) ** 2).mean()))
        observations = int(len(prediction_frame))

    raw_signal = (predicted_next_log_return > 0).astype(float)
    position = raw_signal.shift(1).reindex(simple_returns.index).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = (position * simple_returns.fillna(0.0)) - (turnover * cost_bps / 10_000)
    equity = (1.0 + strategy_returns).cumprod()

    row: dict[str, float | str | int | None] = {
        "name": spec.name,
        "family": spec.family,
        "observations": observations,
        "correlation": corr,
        "directional_accuracy": directional_accuracy,
        "rmse": rmse,
        "sharpe": _sharpe(strategy_returns),
        "cagr": _cagr(equity),
        "max_drawdown": _max_drawdown(equity),
        "turnover": float(turnover.mean() * TRADING_DAYS_PER_YEAR),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else np.nan,
    }
    row.update(spec.params)
    return row, equity.rename(spec.name)


def buy_and_hold_summary(prices: pd.Series) -> tuple[dict[str, float | str], pd.Series]:
    prices = prices.astype(float).sort_index().dropna()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    equity = (1.0 + returns).cumprod().rename("buy_and_hold")
    return (
        {
            "name": "buy_and_hold",
            "family": "benchmark",
            "observations": int(returns.notna().sum()),
            "correlation": np.nan,
            "directional_accuracy": np.nan,
            "rmse": np.nan,
            "sharpe": _sharpe(returns),
            "cagr": _cagr(equity),
            "max_drawdown": _max_drawdown(equity),
            "turnover": 0.0,
            "final_equity": float(equity.iloc[-1]) if not equity.empty else np.nan,
        },
        equity,
    )


def run_parameter_sweep(
    prices: pd.Series,
    specs: list[MethodSpec] | None = None,
    *,
    cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = specs or default_method_specs()
    rows = []
    equities = []
    for spec in specs:
        row, equity = evaluate_method(prices, spec, cost_bps=cost_bps)
        rows.append(row)
        equities.append(equity)

    benchmark_row, benchmark_equity = buy_and_hold_summary(prices)
    summary = pd.DataFrame([benchmark_row, *rows])
    equity_curves = pd.concat([benchmark_equity, *equities], axis=1).dropna(how="all")
    return summary, equity_curves


def best_research_method(summary: pd.DataFrame) -> pd.Series:
    candidates = summary[summary["family"] != "benchmark"].copy()
    if candidates.empty:
        raise ValueError("No candidate methods found")
    return candidates.sort_values(["sharpe", "correlation"], ascending=False).iloc[0]
