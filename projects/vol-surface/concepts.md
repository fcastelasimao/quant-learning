# Mathematical Foundations

This document covers the theory you need before writing code. Read each section,
make sure you could explain it at a whiteboard, then implement.

---

## 1. Black-Scholes: The Starting Point

### The model
A stock follows geometric Brownian motion under the risk-neutral measure:

```
dS = r*S*dt + sigma*S*dW
```

- `r`: risk-free rate (constant)
- `sigma`: volatility (constant) — this is the assumption we'll break later
- `dW`: standard Brownian increment, dW ~ N(0, dt)

The solution: `S_T = S_0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)` where Z ~ N(0,1).

### The formula
European call price:

```
C = S*N(d1) - K*exp(-rT)*N(d2)

d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
```

N() is the standard normal CDF. Put price follows from put-call parity:
`P = K*exp(-rT)*N(-d2) - S*N(-d1)`.

### Why this matters
Every other model in this project is defined by how it **deviates** from Black-Scholes.
If BS were perfect, implied vol would be flat across all strikes and expiries.
It isn't — and the pattern of deviation (the "smile" or "skew") is what we're modelling.

### What you need to implement
- `bs_price(S, K, T, r, sigma, option_type)` — the formula above
- `bs_implied_vol(price, S, K, T, r, option_type)` — invert numerically
- `bs_delta`, `bs_gamma`, `bs_vega`, `bs_theta`, `bs_rho` — partial derivatives

### Key insight for interviews
Black-Scholes is wrong but useful. It gives us a **language** (implied volatility)
to quote option prices in units that are comparable across strikes and expiries.
When traders say "the 25-delta put is at 22 vol", they're using BS as a translation layer.

---

## 2. Implied Volatility and the Smile

### What is implied volatility?
Given a market price C_market for a European call, the implied vol sigma_imp is the
value of sigma that, when plugged into the BS formula, reproduces C_market:

```
BS(S, K, T, r, sigma_imp) = C_market
```

This is a root-finding problem. The BS price is monotonically increasing in sigma
(vega > 0), so a unique solution always exists (for arbitrage-free prices).

### The volatility smile
If BS were correct, sigma_imp would be the same for all (K, T).
In reality:
- **Equity index** (SPY): strong negative skew — OTM puts have higher IV than OTM calls.
  This reflects crash risk: the market prices left-tail protection expensively.
- **Single stocks**: similar skew but less pronounced.
- **FX**: more symmetric smile (both tails priced up).
- **Rates**: depends on the rate environment.

### Moneyness conventions
Instead of raw strike K, we parameterise the smile by:
- **Simple moneyness**: K/S (or S/K)
- **Log moneyness**: ln(K/F) where F = S*exp(rT) is the forward price
- **Delta**: the BS delta of the option (e.g. "25-delta put")

Log moneyness is standard in academic work. Delta is standard on trading desks.

### The volatility surface
The full object is a 2D surface: `sigma_imp(K, T)` or equivalently `sigma_imp(m, T)`
where m is moneyness.

Key features:
- **Smile/skew**: shape of sigma_imp vs K at fixed T
- **Term structure**: ATM sigma_imp vs T
- **Wings**: behavior at extreme strikes (far OTM puts and calls)

### Arbitrage constraints
Not every surface is arbitrage-free. The constraints are:
1. **Butterfly**: `d^2C/dK^2 >= 0` (call prices convex in strike)
2. **Calendar**: `C(K, T2) >= C(K, T1)` for T2 > T1 (more time = more value)
3. **No negative time value**: option price >= intrinsic value

Violation of these means you can construct a riskless profit. In practice,
violations appear in illiquid options and should be filtered out.

---

## 3. The Heston Model

### Why go beyond BS?
BS assumes constant volatility. But we observe:
- Volatility clusters (high-vol days follow high-vol days)
- Leverage effect (vol goes up when prices go down)
- Fat tails (more extreme moves than a normal distribution predicts)

Heston captures all three by making volatility itself a random process.

### The dynamics
Under the risk-neutral measure:

```
dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW_1
dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW_2
corr(dW_1, dW_2) = rho
```

Five parameters:
- **v0**: initial variance (observable from current ATM IV: v0 ~ sigma_ATM^2)
- **kappa**: mean reversion speed of variance. High kappa = vol shocks die quickly.
- **theta**: long-run variance. The variance process is pulled toward theta.
- **xi** (vol-of-vol): how noisy the variance process is. Controls smile curvature.
- **rho** (spot-vol correlation): typically -0.7 to -0.3 for equities. Controls skew direction.

### Parameter intuition (this will be asked in interviews)

| Parameter | Effect on smile |
|-----------|----------------|
| rho < 0 | Negative skew (OTM puts more expensive). More negative = steeper skew. |
| xi large | Fatter smile (both wings up). More vol-of-vol = more curvature. |
| kappa large | Flatter term structure (vol mean-reverts quickly, short-term = long-term). |
| theta | Shifts the overall level of the surface (long-run vol). |
| v0 | Shifts the near-term level (current vol). |

### The Feller condition
```
2 * kappa * theta > xi^2
```
If this holds, the variance process v_t never hits zero (in continuous time).
If violated, v_t can touch zero, which complicates simulation.
In practice, calibrated parameters often violate Feller — the market smile
sometimes requires it. This is fine; just use the full truncation scheme in MC.

### The characteristic function
The key mathematical result: the characteristic function of ln(S_T) under Heston
has a known closed form involving complex exponentials. This means European option
prices can be computed via Fourier inversion:

```
C = S*P1 - K*exp(-rT)*P2

P_j = 1/2 + (1/pi) * integral_0^inf Re[exp(-iu*ln(K)) * phi_j(u)] / (iu) du
```

where phi_j(u) is the Heston characteristic function (different for j=1,2).

You don't need to memorise the formula — you need to understand:
1. It exists and is semi-analytical (fast, no MC needed for Europeans)
2. It involves numerical integration (use scipy.integrate.quad or FFT)
3. Branch cut issues exist — use the Albrecher/Kahl formulation to avoid them

### References
- Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
- Gatheral (2006): "The Volatility Surface" — Chapter 2 (best practical treatment)
- Albrecher et al. (2007): "The Little Heston Trap" — fixes numerical issues

---

## 4. SABR Model

### What it is
SABR (Stochastic Alpha Beta Rho) is an alternative to Heston, widely used
in rates and FX. The dynamics:

```
dF = alpha * F^beta * dW_1
dalpha = nu * alpha * dW_2
corr(dW_1, dW_2) = rho
```

F is the forward price, alpha is the stochastic volatility.

### Why SABR vs Heston?
- SABR is calibrated **per expiry slice** (not globally). This gives a perfect
  fit to each smile but doesn't guarantee consistency across expiries.
- Heston is calibrated **globally** across all expiries simultaneously. Harder
  to fit but gives a consistent dynamic model.
- In practice: SABR for interpolation and quoting, Heston for risk management and exotics.

### Hagan's approximation
The key result: Hagan et al. (2002) derived an approximate closed-form for
the implied vol under SABR:

```
sigma_BS(K) = alpha / ((F*K)^((1-beta)/2)) * (z / x(z)) * correction_terms
```

This is an approximation (breaks down for long-dated or far OTM), but it's
fast and widely used. You implement this formula, then calibrate (alpha, rho, nu)
to minimise the error between model IV and market IV for a given expiry.

### Typical calibration
Fix beta (usually 0.5 for rates, 1.0 for equities/FX). Then for each expiry:
- Objective: minimise `sum_i (sigma_SABR(K_i) - sigma_market(K_i))^2`
- Free parameters: alpha, rho, nu (3 parameters)
- Constraints: alpha > 0, -1 < rho < 1, nu > 0

### References
- Hagan et al. (2002): "Managing Smile Risk" — the original SABR paper
- Obloj (2008): "Fine-Tune Your Smile" — corrections to Hagan's formula

---

## 5. Monte Carlo Methods

### The idea
Can't solve the PDE? Simulate many paths of the underlying, compute the payoff
on each path, take the discounted average. By the law of large numbers, this
converges to the true price.

```
C = exp(-rT) * E[payoff(S_T)]
  ~ exp(-rT) * (1/N) * sum_{i=1}^N payoff(S_T^(i))
```

### Simulating GBM paths (Black-Scholes dynamics)
```
S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
```
where Z ~ N(0,1). This is the exact solution (not Euler — no discretisation error).

### Simulating Heston paths
No exact solution for the joint (S, v) process. Use Euler discretisation:
```
v_{t+dt} = v_t + kappa*(theta - v_t)*dt + xi*sqrt(v_t)*sqrt(dt)*Z_2
S_{t+dt} = S_t * exp((r - v_t/2)*dt + sqrt(v_t)*sqrt(dt)*Z_1)
```
where `Z_1 = rho*Z_2 + sqrt(1-rho^2)*Z_ind` (correlated normals).

**Problem:** v_t can go negative in the Euler scheme. Solutions:
- **Full truncation**: replace v_t with max(v_t, 0) in drift and diffusion
- **Reflection**: if v < 0, set v = |v|
- **QE scheme** (Andersen 2008): more sophisticated, better convergence

Use full truncation — it's simple and well-studied.

### Variance reduction
MC converges as 1/sqrt(N). To get one more digit of accuracy, you need 100x
more paths. Variance reduction techniques give you accuracy for free:

- **Antithetic variates**: for each random draw Z, also compute the payoff with -Z.
  The average of the two has lower variance because upside and downside partially cancel.

- **Control variates**: use a quantity whose expectation you know analytically.
  E.g., use the BS European price as control:
  ```
  C_adjusted = C_MC + beta * (C_BS_analytical - C_BS_MC)
  ```
  where beta is estimated from the paths. Dramatically reduces variance.

### What you need to implement
- `simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)` — returns (n_paths, n_steps+1)
- `simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths)` — returns (S, v) arrays
- `price_european_mc(paths, K, r, T, option_type)` — with antithetic and control variate
- `price_barrier_mc(paths, K, B, r, T, barrier_type)` — down-and-out, up-and-out, etc.

---

## 6. PDE / Finite Differences (Stretch Goal)

### The Black-Scholes PDE
The price V(S, t) of any European derivative satisfies:

```
dV/dt + (1/2)*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V = 0
```

with terminal condition V(S, T) = payoff(S).

### Crank-Nicolson
Discretise S on a grid [0, S_max] with M points, and t on [0, T] with N steps.
Crank-Nicolson averages the explicit and implicit Euler schemes:

```
(V^{n+1} - V^n) / dt = 0.5 * L[V^{n+1}] + 0.5 * L[V^n]
```

where L is the spatial differential operator. This gives a tridiagonal system
at each time step, solved in O(M) via Thomas algorithm.

Advantages: unconditionally stable, second-order in both time and space.
For BS this gives machine-precision results with modest grids (200 x 200).

---

## 7. Greeks: The Sensitivities

### What are Greeks?
Partial derivatives of the option price with respect to model inputs.
They measure risk exposure and determine hedging ratios.

| Greek | Definition | What it measures |
|-------|-----------|-----------------|
| Delta | dV/dS | Exposure to spot moves. Hedge ratio. |
| Gamma | d^2V/dS^2 | Convexity. How fast delta changes. |
| Vega | dV/dsigma | Exposure to volatility changes. |
| Theta | dV/dT | Time decay. How much value bleeds per day. |
| Rho | dV/dr | Exposure to interest rate changes. |

### Analytical vs numerical
BS Greeks have closed-form formulas. For Heston/SABR, use bump-and-revalue:

```
delta ~ (V(S + dS) - V(S - dS)) / (2 * dS)
```

Central differences (as above) are second-order accurate. Forward differences
are first-order. Always use central.

### Interview gotcha
"What's the gamma of a digital option near expiry?" Answer: it blows up.
The payoff has a discontinuity at K, so the second derivative is a delta function.
This is why exotic hedging is hard — smooth payoffs have smooth Greeks,
discontinuous payoffs have Greeks that spike near barriers/strikes at expiry.

---

## Reading List

### Essential (read before or during implementation)
1. **Hull, "Options, Futures, and Other Derivatives"** — Chapters 13-15, 19-21, 27.
   The standard reference. Clear, practical, well-structured.

2. **Gatheral, "The Volatility Surface"** — Chapters 1-4.
   The best treatment of implied vol, local vol, and stochastic vol for practitioners.
   Assumes you know BS already.

3. **Heston (1993), "A Closed-Form Solution for Options with Stochastic Volatility"**
   Read the original. It's short (15 pages) and the maths is accessible with your background.

### Recommended (deepen understanding)
4. **Hagan et al. (2002), "Managing Smile Risk"** — the SABR paper. Short and practical.

5. **Andersen (2008), "Simple and Efficient Simulation of the Heston Model"**
   The QE scheme for Heston MC. Read if you want the best simulation method.

6. **Glasserman, "Monte Carlo Methods in Financial Engineering"** — Chapters 1-4, 7.
   The definitive MC reference. Covers variance reduction rigorously.

### Nice to have (interview depth)
7. **Dupire (1994), "Pricing with a Smile"** — local volatility model. Read to understand
   why Heston is preferred over local vol for risk management.

8. **Rebonato, "Volatility and Correlation"** — deep dive on smile dynamics.
