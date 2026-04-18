# Implementation Plan

---

## Phase 1 — Black-Scholes Foundation (Week 1)

**Goal:** Implement the baseline model that everything else builds on.

### 1A: Black-Scholes formula
- Implement `bs_call_price(S, K, T, r, sigma)` and `bs_put_price()`
- Use the standard closed-form: `C = S*N(d1) - K*exp(-rT)*N(d2)`
- Test against known values (e.g. Hull textbook examples)

### 1B: Implied volatility inversion
- Given a market price, recover sigma via root-finding
- Use Brent's method (`scipy.optimize.brentq`) — robust and fast
- Handle edge cases: deep ITM/OTM, near-zero time to expiry
- Optional: implement Jaeckel's rational approximation (much faster, shows off)

### 1C: Analytical Greeks
- Delta, gamma, vega, theta, rho — all have closed-form in BS
- Verify numerically: bump S by epsilon, recompute price, compare to analytical delta
- Plot Greek surfaces: delta as function of (S, T), gamma smile, etc.

### 1D: Tests
- Put-call parity: `C - P = S - K*exp(-rT)` must hold exactly
- Boundary conditions: `C(S=0) = 0`, `C(S->inf) ~ S`, `P(K->inf) ~ K*exp(-rT)`
- Vega > 0 always, gamma > 0 always, delta in [0,1] for calls
- IV round-trip: price -> IV -> price must recover original

**Deliverable:** `models/black_scholes.py` + `tests/test_black_scholes.py`

---

## Phase 2 — Market Data & Implied Vol Surface (Week 1-2)

**Goal:** Fetch real options data, compute the IV surface, visualise the smile.

### 2A: Data fetcher
- Download SPY options chain via `yfinance` (all strikes, all expiries)
- Extract: strike, expiry, bid, ask, last price, open interest, volume
- Compute mid price = (bid + ask) / 2
- Get spot price and risk-free rate (use Treasury yield or Fed funds)

### 2B: Data cleaning
- Filter illiquid options: drop if OI < 100 or volume < 10
- Drop deep OTM (delta < 0.05) and deep ITM (delta > 0.95)
- Handle zero bids, stale prices, negative time values
- Use OTM options only (calls for K > S, puts for K < S) — standard practice

### 2C: IV surface construction
- For each (K, T) pair, invert BS to get implied vol
- Organise into a grid: moneyness (K/S or log(K/F)) vs time to expiry
- Interpolate missing points (cubic spline or RBF)

### 2D: Visualisation
- 3D surface plot: IV as function of (moneyness, T)
- 2D smile slices: IV vs strike for fixed expiry
- Term structure: ATM IV vs expiry

### 2E: Arbitrage checks
- Butterfly: d^2C/dK^2 >= 0 (call prices must be convex in K)
- Calendar: longer-dated option must cost more than shorter-dated (same strike)
- Flag violations in the data

**Deliverable:** `data/fetcher.py`, `data/cleaning.py`, `calibration/iv_surface.py`, surface plots

---

## Phase 3 — Heston Model (Week 2-3)

**Goal:** Implement the industry-standard stochastic volatility model.

### 3A: Heston dynamics
The model:
```
dS = r*S*dt + sqrt(v)*S*dW_1
dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_2
corr(dW_1, dW_2) = rho
```

Parameters:
- `kappa`: mean reversion speed of variance
- `theta`: long-run variance level
- `xi`: vol-of-vol (how volatile is the volatility itself)
- `rho`: correlation between spot and vol (typically negative — "leverage effect")
- `v0`: initial variance

### 3B: Heston characteristic function
- The magic of Heston: European option prices have a semi-analytical form
  via the characteristic function phi(u) of log(S_T)
- Implement the Heston characteristic function (use the "good" formulation
  from Albrecher et al. to avoid branch cut issues)
- Price European calls via Fourier inversion or the Carr-Madan FFT method

### 3C: Heston Monte Carlo
- Simulate (S, v) paths jointly using correlated Brownian motions
- Use the **full truncation scheme** for the variance process:
  `v_{t+1} = v_t + kappa*(theta - max(v_t, 0))*dt + xi*sqrt(max(v_t, 0))*dW`
  (prevents negative variance without bias)
- Price European calls via MC; compare to semi-analytical — they must agree

### 3D: Tests
- Heston with xi=0 (zero vol-of-vol) must recover Black-Scholes
- MC price must converge to semi-analytical as N_paths -> infinity
- Characteristic function must satisfy phi(0) = 1

**Deliverable:** `models/heston.py`, `pricing/monte_carlo.py` (GBM + Heston paths)

---

## Phase 4 — Calibration (Week 3)

**Goal:** Fit Heston to the real market smile. This is the hard part.

### 4A: Objective function
- For each observed (K_i, T_i, IV_market_i), compute IV_model_i from Heston
- Minimise: `sum_i (IV_model_i - IV_market_i)^2`
- Weight by vega or by 1/bid-ask spread (liquid options count more)

### 4B: Optimisation
- Use `scipy.optimize.differential_evolution` or `minimize` (L-BFGS-B)
- Parameter bounds: kappa in [0.1, 10], theta in [0.01, 1], xi in [0.1, 3],
  rho in [-0.99, 0], v0 in [0.01, 1]
- Feller condition: 2*kappa*theta > xi^2 (ensures v stays positive in continuous time)
- Start from multiple initial points to avoid local minima

### 4C: Diagnostics
- Plot fitted vs market smile for each expiry
- Print calibrated parameters and residual RMSE
- Show parameter sensitivity: how does the smile change if you bump rho by 0.1?

### 4D: SABR calibration (alternative/complement)
- SABR is calibrated per expiry slice (not globally like Heston)
- Hagan's approximation gives IV as a function of (K, F, T, alpha, beta, rho, nu)
- Fix beta = 0.5 or 1.0, calibrate (alpha, rho, nu) per slice
- Much faster than Heston, widely used in rates/FX

**Deliverable:** `calibration/heston_calibrator.py`, `calibration/sabr_calibrator.py`

---

## Phase 5 — Exotic Pricing (Week 4)

**Goal:** Price path-dependent options on calibrated Heston dynamics.

### 5A: Payoff library
- European: `max(S_T - K, 0)`
- Barrier (knock-out): `max(S_T - K, 0) * 1(min(S_t) > B)` (down-and-out call)
- Asian: `max(mean(S_t) - K, 0)` (arithmetic average)
- Lookback: `S_T - min(S_t)` (floating strike)

### 5B: MC pricing engine
- Use calibrated Heston parameters from Phase 4
- Generate N paths, evaluate payoff on each, discount and average
- Variance reduction:
  - **Antithetic**: for each path Z, also simulate -Z, average the two payoffs
  - **Control variate**: use the known European BS price as control

### 5C: PDE solver (stretch goal)
- Solve the Black-Scholes PDE via Crank-Nicolson finite differences
- Set up the grid: S from 0 to 4*K, T from 0 to maturity
- Boundary conditions: C(0, t) = 0, C(S_max, t) = S_max - K*exp(-r*(T-t))
- Verify against BS analytical — should match to machine precision

### 5D: Convergence analysis
- Plot MC price vs N_paths — show 1/sqrt(N) convergence
- Show variance reduction factor (antithetic vs naive)
- Compare MC, PDE, and analytical for European (must all agree)

**Deliverable:** `pricing/payoffs.py`, `pricing/finite_diff.py`, convergence plots

---

## Phase 6 — Polish & Interview Prep (Week 4-5)

- Clean up all code: type hints, docstrings, tests passing
- Write a 1-page summary of results (calibrated params, pricing accuracy)
- Prepare to explain:
  - Why Heston? (captures smile, leverage effect, mean-reverting vol)
  - Why not local vol? (fits today's smile perfectly but has poor dynamics)
  - What goes wrong with calibration? (local minima, Feller condition, overfitting near-term)
  - How would you hedge a barrier option? (delta + vega + pin risk near barrier)
