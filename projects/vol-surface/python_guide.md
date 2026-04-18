# Python Toolkit for Vol Surface

This guide covers the specific Python/numpy/scipy tools you'll need for each phase.
Not a general Python tutorial — assumes you know the basics (functions, classes, loops,
list comprehensions). Focuses on the numerical and scientific computing patterns
that quant code relies on.

---

## 1. NumPy Essentials

### Arrays and vectorisation
Everything in this project operates on arrays, not scalars. Get used to writing
functions that work on arrays without explicit loops.

```python
import numpy as np

# Scalar version (slow, un-Pythonic)
def bs_d1_scalar(S, K, T, r, sigma):
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

# Vectorised version (fast, idiomatic)
# Same code — but now S, K, T can be arrays and it just works
def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

# Use it on arrays
strikes = np.linspace(80, 120, 100)
d1_values = bs_d1(S=100, K=strikes, T=1.0, r=0.05, sigma=0.2)  # returns array of 100
```

**Key rule:** if you're writing a `for` loop over array elements, there's almost
certainly a vectorised way to do it. `np.log`, `np.exp`, `np.sqrt`, `np.maximum`
all operate element-wise on arrays.

### Broadcasting
When arrays have different shapes, numpy "broadcasts" the smaller one:

```python
# S is a scalar, K is (100,), T is (5,)
# To compute d1 for all (K, T) pairs, reshape:
K = np.linspace(80, 120, 100)[:, None]   # shape (100, 1)
T = np.array([0.1, 0.25, 0.5, 1.0, 2.0])[None, :]  # shape (1, 5)
d1 = bs_d1(S=100, K=K, T=T, r=0.05, sigma=0.2)  # shape (100, 5)
```

You'll use this for computing the full IV surface grid in one call.

### Random number generation
For Monte Carlo, always use the modern `Generator` API:

```python
rng = np.random.default_rng(seed=42)  # reproducible

# Correlated normals for Heston (rho = spot-vol correlation)
Z1 = rng.standard_normal((n_paths, n_steps))
Z_ind = rng.standard_normal((n_paths, n_steps))
Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_ind  # corr(Z1, Z2) = rho
```

### Cumulative products (for path simulation)
```python
# GBM: S_T = S_0 * prod(1 + daily_return_t)
daily_rets = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
log_increments = daily_rets  # already log returns
paths = S0 * np.exp(np.cumsum(log_increments, axis=1))  # (n_paths, n_steps)
```

`np.cumsum` along an axis is how you build paths without loops.

### Useful functions you'll need
```python
np.maximum(x, 0)          # element-wise max (for payoffs and truncation)
np.cumprod(x, axis=1)     # cumulative product along rows (path building)
np.cumsum(x, axis=1)      # cumulative sum (log-price paths)
np.maximum.accumulate(x)  # running maximum (for barrier detection)
np.minimum.accumulate(x)  # running minimum (for lookback payoffs)
np.mean(x, axis=0)        # average across paths (MC price estimate)
np.std(x, axis=0, ddof=1) # standard deviation (MC standard error)
np.where(condition, x, y) # conditional element-wise selection
```

---

## 2. SciPy: Optimisation and Root-Finding

### Root-finding (for implied vol inversion)
Given market price, find sigma such that BS(sigma) = price.

```python
from scipy.optimize import brentq

def implied_vol(market_price, S, K, T, r, option_type='call'):
    """Invert BS to find implied vol."""
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    # Brent's method: guaranteed to converge if root is bracketed
    return brentq(objective, a=1e-6, b=5.0)  # sigma between 0.0001% and 500%
```

**Why Brent's?** It's bracketed — you give it an interval [a, b] where the function
changes sign, and it guarantees finding the root. Newton's method is faster but
can diverge if the initial guess is bad. For IV inversion, Brent's is standard.

For vectorised IV computation over many strikes, you'll still need a loop (brentq
doesn't vectorise), but the loop is over strikes, not over iterations:

```python
ivs = np.array([implied_vol(p, S, K_i, T, r) for K_i, p in zip(strikes, prices)])
```

### Optimisation (for calibration)
Heston calibration: minimise sum of squared IV errors.

```python
from scipy.optimize import minimize, differential_evolution

def calibration_objective(params, market_data):
    """Sum of squared IV errors between model and market."""
    kappa, theta, xi, rho, v0 = params
    model_ivs = heston_implied_vols(market_data.strikes, market_data.expiries,
                                      kappa, theta, xi, rho, v0)
    return np.sum((model_ivs - market_data.ivs)**2)

# Option 1: L-BFGS-B (gradient-based, fast, but needs good starting point)
result = minimize(
    calibration_objective,
    x0=[2.0, 0.04, 0.3, -0.7, 0.04],   # initial guess
    args=(market_data,),
    method='L-BFGS-B',
    bounds=[(0.1, 10), (0.01, 1), (0.1, 3), (-0.99, 0), (0.01, 1)],
)

# Option 2: Differential evolution (global, slower, but finds global minimum)
result = differential_evolution(
    calibration_objective,
    bounds=[(0.1, 10), (0.01, 1), (0.1, 3), (-0.99, 0), (0.01, 1)],
    args=(market_data,),
    seed=42,
    maxiter=500,
)

best_params = result.x
```

**When to use which:**
- `minimize` (L-BFGS-B): fast (seconds), but can get stuck in local minima. Use for SABR (3 params, smooth landscape).
- `differential_evolution`: slow (minutes), but finds global minimum. Use for Heston (5 params, nasty landscape with local minima).

### Numerical integration (for Heston characteristic function)
```python
from scipy.integrate import quad

def heston_call_price(S, K, T, r, kappa, theta, xi, rho, v0):
    """Heston European call via Fourier inversion."""
    def integrand(u):
        phi = heston_char_func(u, T, r, kappa, theta, xi, rho, v0)
        return np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u))

    integral, _ = quad(integrand, 0, 200, limit=200)  # upper limit = 200 is enough
    return S - K * np.exp(-r * T) * (0.5 + integral / np.pi)
```

`quad` does adaptive Gaussian quadrature. The `limit` parameter controls the
number of subintervals. For Heston, the integrand oscillates and decays — setting
the upper bound to ~200 is standard.

---

## 3. SciPy: Linear Algebra (for PDE solver)

### Tridiagonal systems (Crank-Nicolson)
The PDE solver produces a tridiagonal matrix equation at each time step: `A @ V = b`.

```python
from scipy.linalg import solve_banded

# Tridiagonal matrix stored in banded form:
# ab[0, :] = upper diagonal
# ab[1, :] = main diagonal
# ab[2, :] = lower diagonal
ab = np.zeros((3, M))
ab[0, 1:] = upper_diag    # above main diagonal
ab[1, :]  = main_diag     # main diagonal
ab[2, :-1] = lower_diag   # below main diagonal

V_next = solve_banded((1, 1), ab, rhs)  # O(M) solve
```

This is the Thomas algorithm under the hood. Runs in O(M) instead of O(M^3).

### Sparse matrices (if you need larger grids)
```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Build tridiagonal sparse matrix
A = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], shape=(M, M))
V_next = spsolve(A, rhs)
```

---

## 4. Complex Numbers (for characteristic functions)

Python handles complex numbers natively. NumPy operations work on complex arrays:

```python
# Heston characteristic function involves complex exponentials
u = np.linspace(0.01, 200, 1000)  # integration variable (real)
i = 1j  # imaginary unit

d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

# All operations are element-wise on complex arrays
C = r * i * u * T + (kappa * theta / xi**2) * (
    (kappa - rho * xi * i * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
)
D = ((kappa - rho * xi * i * u - d) / xi**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

phi = np.exp(C + D * v0 + i * u * np.log(S))
```

**Key point:** `np.exp`, `np.log`, `np.sqrt` all work correctly on complex arrays.
The only thing that doesn't is comparison operators (`<`, `>`) — complex numbers
aren't ordered.

---

## 5. Pandas (for market data handling)

Minimal use in this project compared to all-weather, but needed for fetching
and organising options data:

```python
import yfinance as yf

# Fetch options chain
ticker = yf.Ticker("SPY")
expiries = ticker.options  # list of expiry date strings

# Get one expiry's chain
chain = ticker.option_chain(expiries[0])
calls = chain.calls  # DataFrame with: strike, lastPrice, bid, ask, volume, OI, IV
puts = chain.puts

# Filter
liquid_calls = calls[(calls['volume'] > 10) & (calls['openInterest'] > 100)]
mid_price = (liquid_calls['bid'] + liquid_calls['ask']) / 2
```

### MultiIndex for the vol surface
```python
# Organise IV data as (strike, expiry) -> IV
iv_data = pd.DataFrame({
    'strike': strikes,
    'expiry': expiries,
    'iv': implied_vols,
}).set_index(['expiry', 'strike'])

# Pivot to a grid for plotting
iv_grid = iv_data['iv'].unstack(level='strike')  # rows=expiry, cols=strike
```

---

## 6. Matplotlib (for surface and smile plots)

### 3D surface plot (the money shot)
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# moneyness and T are 1D arrays; IV_grid is 2D
M, T_grid = np.meshgrid(moneyness, expiries_years)

ax.plot_surface(M, T_grid, IV_grid, cmap='viridis', alpha=0.8)
ax.set_xlabel('Moneyness (K/S)')
ax.set_ylabel('Time to Expiry (years)')
ax.set_zlabel('Implied Volatility')
ax.set_title('SPY Implied Volatility Surface')
```

### 2D smile plot (per expiry slice)
```python
fig, ax = plt.subplots(figsize=(10, 6))
for T_label, iv_slice in iv_by_expiry.items():
    ax.plot(strikes, iv_slice, label=f'T = {T_label}')
ax.set_xlabel('Strike')
ax.set_ylabel('Implied Vol')
ax.legend()
```

### Convergence plot (MC)
```python
n_paths_range = [100, 500, 1000, 5000, 10000, 50000, 100000]
mc_prices = [price_european_mc(n) for n in n_paths_range]
analytical_price = bs_price(...)

fig, ax = plt.subplots()
ax.semilogx(n_paths_range, mc_prices, 'o-', label='MC estimate')
ax.axhline(analytical_price, color='red', linestyle='--', label='Analytical')
ax.fill_between(n_paths_range, [p - 2*se for p, se in results],
                [p + 2*se for p, se in results], alpha=0.2)
ax.set_xlabel('Number of paths')
ax.set_ylabel('Option price')
ax.legend()
```

---

## 7. Typing and Dataclasses (for clean code)

Use the patterns from your all-weather project:

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HestonParams:
    """Heston model parameters."""
    kappa: float   # mean reversion speed
    theta: float   # long-run variance
    xi: float      # vol-of-vol
    rho: float     # spot-vol correlation
    v0: float      # initial variance

    @property
    def feller_satisfied(self) -> bool:
        """Check if the Feller condition holds (variance stays positive)."""
        return 2 * self.kappa * self.theta > self.xi**2

    def to_array(self) -> np.ndarray:
        """For passing to scipy optimisers."""
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> HestonParams:
        """Reconstruct from scipy optimiser output."""
        return cls(*arr)
```

### Type hints for array functions
```python
def simulate_heston_paths(
    params: HestonParams,
    S0: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston paths using full truncation.

    Returns
    -------
    S : np.ndarray, shape (n_paths, n_steps + 1) — spot price paths
    v : np.ndarray, shape (n_paths, n_steps + 1) — variance paths
    """
    ...
```

---

## 8. Testing Patterns

```python
import pytest
import numpy as np

def test_bs_put_call_parity():
    """C - P = S - K*exp(-rT) must hold for any valid inputs."""
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    C = bs_price(S, K, T, r, sigma, 'call')
    P = bs_price(S, K, T, r, sigma, 'put')
    assert abs((C - P) - (S - K * np.exp(-r * T))) < 1e-10

def test_iv_round_trip():
    """price -> IV -> price must recover the original."""
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.25
    price = bs_price(S, K, T, r, sigma, 'call')
    recovered_sigma = implied_vol(price, S, K, T, r, 'call')
    assert abs(recovered_sigma - sigma) < 1e-8

def test_heston_reduces_to_bs():
    """Heston with xi=0 (no vol-of-vol) must give BS price."""
    params = HestonParams(kappa=1.0, theta=0.04, xi=0.0001, rho=0.0, v0=0.04)
    heston_price = heston_call_price(S=100, K=100, T=1, r=0.05, params=params)
    bs_price_val = bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2)  # sigma = sqrt(v0)
    assert abs(heston_price - bs_price_val) < 0.05  # within 5 cents

@pytest.mark.parametrize("n_paths", [10_000, 100_000])
def test_mc_convergence(n_paths):
    """MC European price must be within 3 standard errors of analytical."""
    ...
```

**Key testing principles for numerical code:**
- Test against known analytical solutions (BS formulas)
- Test limiting cases (Heston -> BS when xi=0)
- Test symmetries (put-call parity, round trips)
- Use `pytest.approx` or explicit tolerance checks, never exact equality
- Parametrise over inputs to catch edge cases
