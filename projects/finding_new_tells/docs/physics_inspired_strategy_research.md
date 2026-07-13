# Physics-Inspired QQQ/TQQQ Strategy Research

This document is the research inventory for physics-inspired indicators. The goal is practical: improve QQQ/TQQQ allocation, risk control, and validation stability. A metric is only useful if it improves investing behavior after costs and out-of-sample checks.

## Research Rules

- Start every new idea as `watch` unless there is clear train/validation support.
- Prefer risk-state filters over standalone directional predictors.
- Use existing local data first: QQQ, TQQQ, SPY, SPXL, VIX, VIX3M, rates, HYG, LQD, and other local ETF databases when available.
- Keep the test period frozen until the final strategy is selected.
- Reject ideas that only improve CAGR by increasing drawdown duration, turnover, or crisis dependence.

## First Implemented Batch

| Family | Metric | Formula / Proxy | Data | Frequency | Vote Rule | Decision |
|---|---|---|---|---|---|---|
| Entropy | `qqq_sign_entropy_20d` | Shannon entropy of QQQ return signs over 20 days | QQQ | Daily | Watch only | test |
| Entropy | `qqq_return_entropy_60d` | Normalized histogram entropy of QQQ log returns over 60 days | QQQ | Daily | Watch only | test |
| Entropy | `qqq_sample_entropy_60d` | Sample entropy of QQQ log returns, m=2, r=0.2 sigma | QQQ | Daily | Watch only | test |
| Entropy | `qqq_entropy_return_ratio_60d` | 60-day QQQ return divided by return entropy | QQQ | Daily | Watch only | test |
| Entropy | `vix_to_qqq_transfer_entropy_252d` | Conditional mutual information proxy: lagged VIX sign to QQQ sign, conditioned on lagged QQQ sign | QQQ, VIX | Daily | Watch only | watch |
| Scaling | `qqq_hurst_126d` | Rolling Hurst exponent from log-price difference scaling | QQQ | Daily | Watch only | test |
| Criticality | `qqq_lppl_curvature_126d` | Quadratic log-price curvature proxy over 126 days | QQQ | Daily | Watch only | watch |
| RMT | `rmt_market_mode_126d` | Largest eigenvalue share of rolling cross-asset correlation matrix | Cross-asset | Daily | Watch only | test |
| RMT | `rmt_mean_abs_corr_126d` | Mean absolute off-diagonal rolling correlation | Cross-asset | Daily | Watch only | test |
| Black-Scholes / IV | `vix_implied_realized_gap_20d` | VIX/100 minus QQQ realized volatility | QQQ, VIX | Daily | Watch only | test |
| Black-Scholes / IV | `vix_vol_of_vol_20d` | Annualized volatility of VIX log changes | VIX | Daily | Watch only | test |
| Herding | `cross_asset_herding_alignment_20d` | Share of available assets moving in QQQ's direction, averaged over 20 days | Cross-asset | Daily | Watch only | watch |
| Kelly sizing | `qqq_kelly_fraction_252d` | Rolling daily mean return divided by rolling variance, clipped to +/-3 | QQQ | Daily | Watch only | watch |

## Candidate Families

### Entropy / Information Theory

Mechanism: entropy measures disorder, predictability, or regime instability. For TQQQ, disorder matters because high-volatility sideways markets can destroy leveraged compounding.

Sources:
- Entropy predictability: Maasoumi and Racine, 2002, https://www.sciencedirect.com/science/article/pii/S0304407601001257
- Entropy trading: Efremidze et al., 2021, https://www.sciencedirect.com/science/article/pii/S1059056021000861
- Intraday entropy selection: Nedela and Kresta, 2026, https://link.springer.com/article/10.1007/s10614-026-11347-2

Failure modes: entropy can become a noisy proxy for volatility, thresholds are easy to overfit, and transfer entropy needs much more data than simple daily returns provide.

### Scaling / Fractal / Multifractal

Mechanism: financial markets show scaling, persistence, and regime-dependent roughness. A Hurst or scaling signal may help distinguish trend-friendly periods from noisy mean-reverting periods.

Source:
- Hurst estimation: Gomez-Aguila et al., 2022, https://link.springer.com/article/10.1186/s40854-022-00394-x

Failure modes: rolling Hurst estimates are unstable, window length matters, and apparent persistence may disappear after costs.

### Random Matrix Theory

Mechanism: when the market mode dominates the correlation matrix, diversification falls and crash risk rises. This is better framed as a risk filter than a buy signal.

Source:
- Random matrix filtering: Daly, Crane and Ruskin, 2008, https://www.sciencedirect.com/science/article/pii/S0378437108001829

Failure modes: the local asset universe is small, strongly duplicated assets can exaggerate the first eigenvalue, and results may mostly reflect equity beta.

### LPPL / Critical Bubble Models

Mechanism: log-periodic power law models try to detect bubble-like faster-than-exponential price acceleration before regime breaks.

Source:
- LPPL bubble detection: Shu and Song, 2024, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4734944

Failure modes: full LPPL fitting is fragile and easy to overfit. The first pass uses only a simple curvature proxy as watch-only.

### Black-Scholes / Option-Implied Information

Mechanism: Black-Scholes itself is not a return predictor. Its practical value here is the option-implied volatility worldview: implied volatility, volatility term structure, skew, and implied-versus-realized gaps.

Source:
- Black-Scholes / IV surfaces: Ulrich, Zimmer and Merbecks, 2023, https://link.springer.com/article/10.1007/s11147-023-09195-5

Failure modes: VIX is an S&P 500 proxy rather than QQQ option IV, and option-surface modeling should be deferred until reliable QQQ option-chain data exists.

### Ising / Herding / Criticality

Mechanism: herding models describe phase transitions where many assets align and market behavior becomes less diversified.

Source:
- Ising/herding models: Zhou and Sornette, 2007, https://epjb.epj.org/articles/epjb/abs/2007/02/b06183/b06183.html

Failure modes: full agent-based models are not needed for this repo. Observable proxies like cross-asset sign alignment and correlation concentration are preferred.

### Kelly / Entropy Portfolio Sizing

Mechanism: Kelly-style sizing links expected edge and variance. It is a sizing diagnostic, not permission to lever aggressively.

Failure modes: rolling expected returns are very noisy; full Kelly is dangerous for TQQQ because drawdown and path risk dominate the math.

## Promotion Criteria

Before any watch metric becomes a voting metric:

- It must show same-direction evidence on train and validation.
- It must have enough directional observations.
- It must improve validation risk-adjusted metrics or drawdown control.
- It must survive higher transaction-cost assumptions.
- It must not rely on one crisis window.
