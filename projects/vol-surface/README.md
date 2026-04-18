# Volatility Surface & Options Pricing Engine

A from-scratch implementation of volatility surface construction, stochastic volatility calibration, and exotic option pricing. Built as a learning project and interview portfolio piece.

> **This is not a trading system.** It is a quantitative finance toolkit that demonstrates mastery of the mathematical foundations of options pricing.

---

## What This Project Covers

### 1. Implied Volatility Surface Construction
- Fetch live options chains (SPY, AAPL) from market data
- Compute implied volatility from market prices via Black-Scholes inversion
- Build and interpolate the (strike, expiry) -> IV surface
- Detect and handle arbitrage violations (butterfly, calendar spread)

### 2. Stochastic Volatility Models
- **Black-Scholes** — the baseline (constant vol, geometric Brownian motion)
- **Heston** — stochastic variance with mean reversion and vol-of-vol
- **SABR** — stochastic alpha-beta-rho, the industry standard for rates/FX smiles

### 3. Model Calibration
- Calibrate Heston parameters (kappa, theta, xi, rho, v0) to the observed smile
- Calibrate SABR (alpha, beta, rho, nu) per expiry slice
- Objective: minimise squared IV error between model and market
- Explore the calibration landscape (local minima, parameter degeneracies)

### 4. Option Pricing
- **Analytical**: Black-Scholes closed form, Heston semi-analytical (characteristic function + FFT)
- **Monte Carlo**: GBM paths, Heston paths (full truncation scheme), variance reduction (antithetic, control variate)
- **PDE / Finite Differences**: Black-Scholes PDE solved via Crank-Nicolson, optional Heston PDE (ADI scheme)
- **Exotic payoffs**: barrier options, Asian options, lookbacks — priced via MC on calibrated dynamics

### 5. Greeks
- Analytical Greeks (Black-Scholes): delta, gamma, vega, theta, rho
- Numerical Greeks via bump-and-revalue (works for any model)
- Greek surfaces: how delta/gamma/vega vary across (K, T)

---

## Project Structure

```
vol-surface/
├── data/
│   ├── fetcher.py            Download options chains and spot prices
│   └── cleaning.py           Filter, validate, handle missing data
├── models/
│   ├── black_scholes.py      BS formula, implied vol inversion, Greeks
│   ├── heston.py             Heston characteristic function, MC paths
│   └── sabr.py               SABR implied vol approximation (Hagan et al.)
├── calibration/
│   ├── iv_surface.py         Build implied vol surface from market data
│   ├── heston_calibrator.py  Calibrate Heston to market smile
│   └── sabr_calibrator.py    Calibrate SABR per expiry slice
├── pricing/
│   ├── monte_carlo.py        MC engine (GBM + Heston paths, variance reduction)
│   ├── finite_diff.py        PDE solver (Crank-Nicolson for BS PDE)
│   └── payoffs.py            Payoff functions (European, barrier, Asian, lookback)
├── tests/
│   ├── test_black_scholes.py
│   ├── test_heston.py
│   ├── test_monte_carlo.py
│   └── test_greeks.py
├── notebooks/                Exploration and visualisation (not production code)
├── results/                  Generated plots and calibration outputs (.gitignore'd)
├── config.py                 Shared constants (day count, rate conventions)
├── plan.md                   Phase-by-phase implementation plan
├── concepts.md               Mathematical foundations and learning guide
├── requirements.txt
└── README.md
```

## Installation

```bash
cd quant-learning/projects/vol-surface
conda create -n volsurf python=3.11
conda activate volsurf
pip install -r requirements.txt
```

## Known Limitations

- Market data via yfinance (delayed, not real-time)
- No dividend handling (assumes continuous dividend yield approximation)
- Heston PDE (ADI) is a stretch goal, not required for the core project
- No transaction costs or bid-ask spread modelling

## References

See `concepts.md` for the full mathematical treatment and reading list.
