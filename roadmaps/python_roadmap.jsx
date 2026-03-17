
const pythonRoadmap = [
  {
    month: "Month 1",
    title: "Python Fundamentals & Data Manipulation",
    period: "Weeks 1-4",
    color: "#3b82f6",
    integrationNote: "Building Phase 1-2 of Quant Roadmap",
    weeks: [
      {
        week: "Week 1-2",
        title: "Data Structures & Flow Control",
        difficulty: "Beginner",
        timeCommitment: "10-15 hours/week",
        goal: "Download data, basic manipulation, plotting",
        quantTrigger: "Phase 1, Topic 1: Download SPY and VIX data",
        topics: [
          {
            title: "Core Data Structures",
            concepts: ["Lists & list comprehensions", "Dictionaries & tuples", "When to use each type", "Iterating efficiently"],
            practice: [
              "Store 100 stock tickers in a list",
              "Create dict mapping tickers to sectors",
              "Use list comprehension to filter by sector"
            ],
            checkYourself: "Can you explain when to use list vs dict vs tuple?",
            ankiCards: [
              "Q: List vs Tuple? | A: List mutable, Tuple immutable. Use tuple for fixed data.",
              "Q: Dict lookup time complexity? | A: O(1) average case",
              "Q: List comprehension syntax? | A: [expr for item in iterable if condition]"
            ]
          },
          {
            title: "Pandas Basics",
            concepts: ["DataFrame creation", "Reading CSV/Excel", "Column/row selection", "Basic filtering", ".head(), .tail(), .info()"],
            practice: [
              "Load SPY data with yfinance",
              "Extract 'Adj Close' column",
              "Filter rows where Close > 400",
              "Handle missing data with .fillna()"
            ],
            checkYourself: "Can you load data and extract specific columns without looking it up?",
            commonPitfalls: [
              "df['Close'] returns Series, df[['Close']] returns DataFrame",
              "df.Close works but df['Adj Close'] needed for spaces",
              "Always use 'Adj Close' for price analysis, not 'Close'"
            ],
            ankiCards: [
              "Q: Read CSV in pandas? | A: pd.read_csv('file.csv')",
              "Q: Select rows 10-20? | A: df.iloc[10:20] or df.loc[start:end]",
              "Q: Filter by condition? | A: df[df['column'] > value]"
            ]
          },
          {
            title: "Basic Matplotlib",
            concepts: ["plt.plot() basics", "Multiple series on one chart", "Labels, titles, legends", "Saving figures"],
            practice: [
              "Plot SPY price over time",
              "Overlay SPY and VIX on dual y-axis",
              "Add recession shading (2008, 2020)",
              "Export as PNG for reports"
            ],
            checkYourself: "Can you create a publication-quality chart in <10 lines?",
            codeTemplate: `import matplotlib.pyplot as plt
import yfinance as yf

# Download data
spy = yf.download('SPY', start='2020-01-01')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(spy.index, spy['Adj Close'])
plt.title('SPY Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(alpha=0.3)
plt.savefig('spy_price.png', dpi=150, bbox_inches='tight')
plt.show()`,
            ankiCards: [
              "Q: Create figure with size? | A: plt.figure(figsize=(width, height))",
              "Q: Save high-res figure? | A: plt.savefig('file.png', dpi=150)",
              "Q: Twin y-axis? | A: ax2 = ax1.twinx()"
            ]
          }
        ],
        milestone: {
          task: "Build Data Pipeline",
          description: "Create reusable data downloader that handles multiple tickers, date ranges, and missing data",
          codeChallenge: `def get_clean_data(tickers, start, end):
    """
    Download and clean OHLC data for multiple tickers.
    
    Args:
        tickers: list of ticker symbols
        start: start date string 'YYYY-MM-DD'
        end: end date string 'YYYY-MM-DD'
    
    Returns:
        dict of DataFrames, one per ticker
    """
    # Your implementation here
    pass

# Test it
data = get_clean_data(['SPY', 'TLT', 'GLD'], '2020-01-01', '2024-01-01')
assert len(data) == 3
assert 'SPY' in data
assert data['SPY'].isna().sum().sum() == 0  # No missing data`,
          successCriteria: ["No missing data", "Handles errors gracefully", "Returns consistent format", "Takes <1 min for 5 years of data"]
        },
        resources: [
          { type: "📚", name: "Python Crash Course (Ch 2-4)", url: "Book" },
          { type: "🎥", name: "Corey Schafer - Pandas Tutorial", url: "YouTube" },
          { type: "📄", name: "Pandas official Getting Started", url: "pandas.pydata.org" }
        ]
      },
      {
        week: "Week 3-4",
        title: "NumPy & Vectorization",
        difficulty: "Intermediate",
        timeCommitment: "12-18 hours/week",
        goal: "Think in arrays, eliminate loops, 10-100x speedups",
        quantTrigger: "Phase 1, Topic 2: Rolling linear regression channel",
        topics: [
          {
            title: "NumPy Arrays",
            concepts: ["Array creation methods", "Indexing (basic, boolean, fancy)", "Shape manipulation", "Broadcasting rules"],
            practice: [
              "Create price array from DataFrame",
              "Boolean indexing: prices[prices > 100]",
              "Fancy indexing: prices[[0, 10, 20, 30]]",
              "Reshape 1D to 2D for batch operations"
            ],
            checkYourself: "Can you manipulate array shapes without trial-and-error?",
            commonPitfalls: [
              "Array indexing returns view (modifying changes original)",
              "Boolean indexing returns copy",
              "reshape(-1, 1) vs reshape(1, -1) - row vs column vector",
              "Always use .copy() when you want independent array"
            ],
            ankiCards: [
              "Q: Create 3x4 zeros? | A: np.zeros((3, 4))",
              "Q: Last element? | A: arr[-1]",
              "Q: Boolean indexing returns? | A: Copy, not view",
              "Q: arr[arr > 0] = 0 works? | A: Yes, boolean indexing can assign"
            ]
          },
          {
            title: "Vectorization Mindset",
            concepts: ["Why loops are slow", "Broadcasting for element-wise ops", "Universal functions (ufuncs)", "When vectorization isn't possible"],
            practice: [
              "Compute log returns: np.log(prices[1:] / prices[:-1])",
              "Z-score normalization without loop",
              "Moving average using np.convolve",
              "Benchmark loop vs vectorized (use %timeit)"
            ],
            checkYourself: "Can you remove 90% of for-loops from your code?",
            beforeAfter: {
              bad: `# DON'T DO THIS
returns = []
for i in range(1, len(prices)):
    ret = (prices[i] - prices[i-1]) / prices[i-1]
    returns.append(ret)
returns = np.array(returns)`,
              good: `# DO THIS
returns = np.diff(prices) / prices[:-1]
# Or even better (log returns):
returns = np.diff(np.log(prices))`
            },
            ankiCards: [
              "Q: Compute log returns? | A: np.diff(np.log(prices))",
              "Q: Why vectorize? | A: 10-100x faster, NumPy uses C loops",
              "Q: Element-wise multiply? | A: arr1 * arr2 (broadcasts)",
              "Q: Matrix multiply? | A: arr1 @ arr2 or np.dot(arr1, arr2)"
            ]
          },
          {
            title: "Rolling Operations",
            concepts: ["Convolution for moving average", "Stride tricks for windows", "Pandas rolling vs NumPy", "Memory efficiency"],
            practice: [
              "20-day moving average (3 methods)",
              "Rolling std using np.lib.stride_tricks",
              "Exponential moving average",
              "Rolling correlation of two series"
            ],
            checkYourself: "Can you implement rolling operations without pandas?",
            codeTemplate: `# Fast rolling mean using cumsum
def rolling_mean(arr, window):
    cumsum = np.cumsum(arr)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = cumsum[window - 1:] / window
    return np.concatenate([np.full(window - 1, np.nan), result])

# Stride tricks for rolling windows (advanced)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(arr, window)
rolling_std = np.std(windows, axis=1)`,
            ankiCards: [
              "Q: Fast rolling sum? | A: Use cumsum, subtract offset",
              "Q: pandas rolling slow? | A: Yes for small windows, use NumPy",
              "Q: Get all rolling windows? | A: sliding_window_view(arr, window)"
            ]
          }
        ],
        milestone: {
          task: "Channel Detection Functions",
          description: "Implement rolling regression channel calculator from scratch",
          codeChallenge: `def compute_channel(prices, window=60):
    """
    Compute linear regression channel boundaries.
    
    Args:
        prices: 1D numpy array
        window: rolling window size
    
    Returns:
        slope, intercept, upper_band, lower_band (all arrays)
    """
    # Your vectorized implementation here
    # No for-loops allowed!
    pass

# Test on SPY
spy = yf.download('SPY', start='2020-01-01')['Adj Close'].values
slope, intercept, upper, lower = compute_channel(spy, 60)

# Verify
assert len(slope) == len(spy)
assert np.all(upper >= spy)  # Upper band above prices
assert np.all(lower <= spy)  # Lower band below prices`,
          successCriteria: ["Fully vectorized (no loops)", "Handles edge cases (first 59 values)", "Runs in <100ms for 5 years data", "Numerically stable"]
        },
        resources: [
          { type: "📚", name: "Python for Data Analysis (Ch 4-5)", url: "Book" },
          { type: "🌐", name: "NumPy Quickstart Tutorial", url: "numpy.org" },
          { type: "🎥", name: "Jake VanderPlas - Losing Your Loops", url: "YouTube" }
        ]
      }
    ]
  },
  {
    month: "Month 2",
    title: "Intermediate Python & OOP",
    period: "Weeks 5-8",
    color: "#8b5cf6",
    integrationNote: "Building Phase 3-4 of Quant Roadmap",
    weeks: [
      {
        week: "Week 5-6",
        title: "Object-Oriented Programming",
        difficulty: "Intermediate",
        timeCommitment: "12-15 hours/week",
        goal: "Design clean, reusable classes for quant models",
        quantTrigger: "Phase 4: Build ChannelRegimeDetector class",
        topics: [
          {
            title: "Classes & Objects",
            concepts: ["Class definition", "__init__ constructor", "Instance vs class attributes", "self parameter", "Methods"],
            practice: [
              "Create Stock class with ticker, price attributes",
              "Add method to compute returns",
              "Create Portfolio class managing multiple stocks",
              "Implement __repr__ for nice printing"
            ],
            checkYourself: "Can you design a class without looking up syntax?",
            codeTemplate: `class Stock:
    """Represents a single stock with price history."""
    
    def __init__(self, ticker, prices):
        self.ticker = ticker
        self.prices = np.array(prices)
        self._returns = None  # Cached
    
    @property
    def returns(self):
        """Lazy-compute returns."""
        if self._returns is None:
            self._returns = np.diff(np.log(self.prices))
        return self._returns
    
    def sharpe_ratio(self, rfr=0.02):
        """Compute annualized Sharpe ratio."""
        excess = self.returns.mean() * 252 - rfr
        vol = self.returns.std() * np.sqrt(252)
        return excess / vol
    
    def __repr__(self):
        return f"Stock({self.ticker}, {len(self.prices)} days)"`,
            ankiCards: [
              "Q: Define class? | A: class ClassName:",
              "Q: Constructor method? | A: def __init__(self, args):",
              "Q: Access attribute? | A: self.attribute_name",
              "Q: Make property? | A: @property decorator"
            ]
          },
          {
            title: "Design Patterns for Quant",
            concepts: ["Builder pattern for models", "Strategy pattern for signals", "Observer for live data", "When NOT to use OOP"],
            practice: [
              "ChannelDetector class with fit/predict",
              "BaseStrategy abstract class",
              "Multiple strategy implementations",
              "Backtester class managing strategies"
            ],
            checkYourself: "Does your class have a single, clear purpose?",
            designPrinciples: [
              "Single Responsibility: One class, one job",
              "Composition > Inheritance: Prefer has-a over is-a",
              "Explicit > Implicit: Clear method names",
              "Don't over-engineer: Start simple, refactor later"
            ],
            ankiCards: [
              "Q: When to use class? | A: When state + behavior together make sense",
              "Q: Inheritance vs composition? | A: Prefer composition (has-a)",
              "Q: Private attribute convention? | A: _private (single underscore)"
            ]
          }
        ],
        milestone: {
          task: "Build ChannelDetector Class",
          description: "Production-quality regime detector with clean API",
          codeChallenge: `class ChannelDetector:
    """Detect bull/bear/sideways channel regimes using HMM."""
    
    def __init__(self, n_states=3, window=60, features=None):
        self.n_states = n_states
        self.window = window
        self.features = features or ['slope', 'width', 'position']
        self.model = None
        self.scaler = None
    
    def fit(self, prices):
        """Fit model to price history."""
        # Extract features
        # Normalize
        # Fit HMM
        # Return self for chaining
        return self
    
    def predict(self, prices=None):
        """Predict regime labels."""
        pass
    
    def predict_proba(self, prices=None):
        """Get regime probabilities."""
        pass
    
    def plot(self, prices, save_path=None):
        """4-panel diagnostic plot."""
        pass

# Usage
detector = ChannelDetector(n_states=3)
detector.fit(spy_prices)
probs = detector.predict_proba()
detector.plot(spy_prices, 'channel_regimes.png')`,
          successCriteria: ["Clean API (fit/predict pattern)", "Type hints on all methods", "Comprehensive docstrings", "Handles edge cases", "Chainable methods"]
        },
        resources: [
          { type: "🎥", name: "Corey Schafer - OOP Series", url: "YouTube (6 videos)" },
          { type: "📚", name: "Python Tricks (Ch 3)", url: "Book" },
          { type: "🌐", name: "Real Python - OOP in Python 3", url: "realpython.com" }
        ]
      },
      {
        week: "Week 7-8",
        title: "Advanced NumPy & SciPy",
        difficulty: "Advanced",
        timeCommitment: "15-20 hours/week",
        goal: "Master matrix operations, implement Kalman filter",
        quantTrigger: "Phase 2: Kalman Filter from scratch",
        topics: [
          {
            title: "Linear Algebra in NumPy",
            concepts: ["Matrix multiplication (@)", "Transpose, inverse", "Eigenvalues/vectors", "Solving linear systems", "Numerical stability"],
            practice: [
              "Implement Kalman predict step",
              "Implement Kalman update step",
              "Compare np.linalg.inv vs scipy.linalg.inv",
              "Check condition numbers for stability"
            ],
            checkYourself: "Can you implement matrix recursions without bugs?",
            codeTemplate: `# Kalman filter predict step
def kalman_predict(x, P, F, Q, B=None, u=None):
    """
    Predict step of Kalman filter.
    
    x: state vector (n,)
    P: covariance matrix (n, n)
    F: state transition matrix (n, n)
    Q: process noise covariance (n, n)
    """
    # State prediction
    x_pred = F @ x
    if B is not None and u is not None:
        x_pred += B @ u
    
    # Covariance prediction
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred`,
            commonPitfalls: [
              "Don't use np.linalg.inv - use scipy.linalg.inv (more stable)",
              "Check matrix is positive definite before inverting",
              "Use @ for matrix multiply, not np.dot (clearer)",
              "Always check shapes: (n,) vs (n, 1) matters!"
            ],
            ankiCards: [
              "Q: Matrix multiply? | A: A @ B",
              "Q: Transpose? | A: A.T",
              "Q: Solve Ax = b? | A: np.linalg.solve(A, b) not inv(A) @ b",
              "Q: Eigenvalues? | A: np.linalg.eig(A) or eigh for symmetric"
            ]
          },
          {
            title: "SciPy Statistics",
            concepts: ["Probability distributions", "Statistical tests", "Maximum likelihood", "Optimization"],
            practice: [
              "Fit normal distribution to returns",
              "ADF test for stationarity",
              "Likelihood ratio test",
              "MLE parameter estimation"
            ],
            checkYourself: "Can you implement statistical tests from scratch?",
            ankiCards: [
              "Q: Normal distribution? | A: scipy.stats.norm(loc=μ, scale=σ)",
              "Q: ADF test? | A: from statsmodels.tsa.stattools import adfuller",
              "Q: Minimize function? | A: scipy.optimize.minimize(func, x0)"
            ]
          }
        ],
        milestone: {
          task: "Kalman Filter Implementation",
          description: "Full predict-update cycle, compare against pykalman",
          codeChallenge: `class KalmanFilter:
    """1D Kalman filter for tracking price trend."""
    
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.x = 0  # Initial state
        self.P = 1  # Initial uncertainty
    
    def predict(self):
        """Predict step."""
        # x_pred = F * x + B * u
        # P_pred = F * P * F.T + Q
        pass
    
    def update(self, measurement):
        """Update step."""
        # Kalman gain: K = P * H.T / (H * P * H.T + R)
        # x_update = x + K * (z - H * x)
        # P_update = (1 - K * H) * P
        pass
    
    def filter(self, measurements):
        """Filter entire series."""
        estimates = []
        for z in measurements:
            self.predict()
            self.update(z)
            estimates.append(self.x)
        return np.array(estimates)`,
          successCriteria: ["Matches pykalman output", "Numerically stable", "Handles edge cases", "Documented with math"]
        },
        resources: [
          { type: "📄", name: "Welch & Bishop - Kalman Filter Tutorial", url: "PDF (free)" },
          { type: "🌐", name: "SciPy Lecture Notes", url: "scipy-lectures.org" },
          { type: "📚", name: "Elegant SciPy", url: "Book" }
        ]
      }
    ]
  },
  {
    month: "Month 3",
    title: "Production Python & Optimization",
    period: "Weeks 9-12",
    color: "#f59e0b",
    integrationNote: "Building Phase 5-6 of Quant Roadmap",
    weeks: [
      {
        week: "Week 9-10",
        title: "Performance Optimization",
        difficulty: "Advanced",
        timeCommitment: "15-20 hours/week",
        goal: "10x speedups with profiling and Numba",
        quantTrigger: "HMM forward pass too slow on 100k time steps",
        topics: [
          {
            title: "Profiling & Benchmarking",
            concepts: ["%timeit vs %time", "line_profiler", "memory_profiler", "cProfile", "Finding bottlenecks"],
            practice: [
              "Profile your HMM implementation",
              "Identify the 20% of code taking 80% time",
              "Benchmark before/after optimization",
              "Track memory usage"
            ],
            checkYourself: "Can you identify bottlenecks without guessing?",
            codeTemplate: `# In Jupyter
%load_ext line_profiler

def slow_function(data):
    # Your code
    pass

%lprun -f slow_function slow_function(data)

# This shows time per line:
# Line #  Hits   Time    Per Hit  % Time  Line Contents
# ======  ====   ====    =======  ======  =============
#     12    1    150.0    150.0    75.0    result = [expensive_op(x) for x in data]
#     13    1     50.0     50.0    25.0    return np.array(result)`,
            ankiCards: [
              "Q: Quick benchmark? | A: %timeit function(args)",
              "Q: Profile line-by-line? | A: %lprun -f function function(args)",
              "Q: Memory usage? | A: %memit or memory_profiler",
              "Q: Profile full program? | A: python -m cProfile script.py"
            ]
          },
          {
            title: "Numba JIT Compilation",
            concepts: ["@jit decorator", "nopython mode", "Parallel loops with prange", "Limitations", "When JIT helps"],
            practice: [
              "JIT-compile forward algorithm",
              "Parallelize feature computation",
              "Compare: Python loop vs NumPy vs Numba",
              "Measure compilation overhead"
            ],
            checkYourself: "Can you get 10x speedup on hot loops?",
            codeTemplate: `from numba import jit, prange
import numpy as np

# Sequential
@jit(nopython=True)
def forward_pass(obs, A, B, pi):
    T, N = len(obs), len(pi)
    alpha = np.zeros((T, N))
    
    # Initialize
    alpha[0] = pi * B[:, obs[0]]
    
    # Forward recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, obs[t]] * np.sum(alpha[t-1] * A[:, j])
    
    return alpha

# Parallel
@jit(nopython=True, parallel=True)
def compute_features(prices, window):
    n = len(prices)
    features = np.zeros((n, 3))  # slope, width, position
    
    for i in prange(window, n):  # prange = parallel range
        window_prices = prices[i-window:i]
        # Compute features (operations must be Numba-compatible)
        features[i, 0] = compute_slope(window_prices)
        features[i, 1] = compute_width(window_prices)
        features[i, 2] = compute_position(window_prices)
    
    return features`,
            commonPitfalls: [
              "Not all NumPy functions work in nopython mode",
              "String operations don't work",
              "List comprehensions compile, but loops often faster",
              "First call is slow (compilation), subsequent fast"
            ],
            ankiCards: [
              "Q: Basic JIT? | A: @jit or @jit(nopython=True)",
              "Q: Parallel loop? | A: from numba import prange, use in loop",
              "Q: When Numba helps? | A: Tight loops with NumPy arrays",
              "Q: Numba limitation? | A: No pandas, limited NumPy functions"
            ]
          }
        ],
        milestone: {
          task: "Optimize HMM Forward Pass",
          description: "Achieve 10x+ speedup on realistic data",
          codeChallenge: `# Baseline (pure Python)
def forward_python(obs, A, B, pi):
    # Implementation (slow)
    pass

# Target (Numba)
@jit(nopython=True)
def forward_numba(obs, A, B, pi):
    # Same logic, compiled
    pass

# Benchmark
obs = np.random.randint(0, 100, 100000)
A = np.random.rand(3, 3)
B = np.random.rand(3, 100)
pi = np.array([0.33, 0.33, 0.34])

%timeit forward_python(obs, A, B, pi)
# Target: ~10 seconds

%timeit forward_numba(obs, A, B, pi)
# Target: ~1 second (10x speedup)`,
          successCriteria: ["10x faster than baseline", "Correct output (verify against hmmlearn)", "Handles large T (100k+)", "Documented performance"]
        },
        resources: [
          { type: "🌐", name: "Numba Documentation", url: "numba.pydata.org" },
          { type: "📚", name: "High Performance Python", url: "Book by Gorelick & Ozsvald" },
          { type: "🎥", name: "Jake VanderPlas - Numba", url: "YouTube" }
        ]
      },
      {
        week: "Week 11-12",
        title: "Software Engineering Practices",
        difficulty: "Intermediate",
        timeCommitment: "12-15 hours/week",
        goal: "Production-quality code ready for GitHub",
        quantTrigger: "Polish full codebase for job applications",
        topics: [
          {
            title: "Type Hints & Documentation",
            concepts: ["Type annotations", "numpy.typing", "Docstring formats", "Type checking with mypy"],
            practice: [
              "Add type hints to all functions",
              "Write NumPy-style docstrings",
              "Use mypy to catch type errors",
              "Document edge cases"
            ],
            checkYourself: "Can someone use your code without asking questions?",
            codeTemplate: `from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

def compute_channel(
    prices: NDArray[np.float64],
    window: int = 60,
    n_std: float = 2.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute linear regression channel boundaries.
    
    Fits a rolling linear regression to price data and computes
    bands at ±n_std standard deviations around the trend line.
    
    Parameters
    ----------
    prices : ndarray of shape (n_samples,)
        Historical price data.
    window : int, default=60
        Rolling window size in periods.
    n_std : float, default=2.0
        Number of standard deviations for bands.
    
    Returns
    -------
    upper_band : ndarray of shape (n_samples,)
        Upper channel boundary.
    lower_band : ndarray of shape (n_samples,)
        Lower channel boundary.
    
    Raises
    ------
    ValueError
        If window > len(prices).
    
    Examples
    --------
    >>> prices = np.array([100, 101, 102, 103, 104])
    >>> upper, lower = compute_channel(prices, window=3)
    >>> assert upper.shape == prices.shape
    
    Notes
    -----
    First (window-1) values are NaN since regression undefined.
    Uses OLS regression: y = mx + b
    """
    if window > len(prices):
        raise ValueError(f"window ({window}) > len(prices) ({len(prices)})")
    
    # Implementation
    pass`,
            ankiCards: [
              "Q: Type hint for array? | A: NDArray[np.float64]",
              "Q: Optional type? | A: Optional[Type] or Type | None",
              "Q: Docstring format? | A: NumPy or Google style (be consistent)",
              "Q: Check types? | A: mypy filename.py"
            ]
          },
          {
            title: "Testing with pytest",
            concepts: ["Writing test functions", "Fixtures", "Parametrize", "Coverage", "Test-driven development"],
            practice: [
              "Test channel detector on synthetic data",
              "Test edge cases (empty data, NaNs)",
              "Parametrize tests for multiple inputs",
              "Aim for 80%+ coverage"
            ],
            checkYourself: "Do you trust your code won't break?",
            codeTemplate: `import pytest
import numpy as np
from channel_detector import ChannelDetector

@pytest.fixture
def synthetic_bull_channel():
    """Generate synthetic bull channel data."""
    t = np.arange(100)
    trend = 100 + 0.1 * t  # Upward trend
    noise = np.random.randn(100) * 0.5
    return trend + noise

def test_detector_initialization():
    """Test detector can be created with valid params."""
    detector = ChannelDetector(n_states=3, window=60)
    assert detector.n_states == 3
    assert detector.window == 60
    assert detector.model is None  # Not fitted yet

def test_detector_fit(synthetic_bull_channel):
    """Test detector can fit to data."""
    detector = ChannelDetector(n_states=3)
    detector.fit(synthetic_bull_channel)
    
    assert detector.model is not None
    assert detector.feature_matrix is not None
    assert detector.feature_matrix.shape[0] == len(synthetic_bull_channel)

@pytest.mark.parametrize("n_states", [2, 3, 4, 5])
def test_detector_different_states(n_states, synthetic_bull_channel):
    """Test detector works with different state counts."""
    detector = ChannelDetector(n_states=n_states)
    detector.fit(synthetic_bull_channel)
    
    probs = detector.predict_proba()
    assert probs.shape == (len(synthetic_bull_channel), n_states)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1

def test_detector_handles_nans():
    """Test detector handles missing data gracefully."""
    prices = np.array([100, 101, np.nan, 103, 104])
    detector = ChannelDetector()
    
    with pytest.raises(ValueError, match="contains NaN"):
        detector.fit(prices)

# Run tests:
# pytest tests/ -v --cov=channel_detector`,
            ankiCards: [
              "Q: Test function name? | A: def test_function_name():",
              "Q: Assert equal? | A: assert a == b or np.allclose(a, b)",
              "Q: Test raises error? | A: with pytest.raises(ErrorType):",
              "Q: Run tests? | A: pytest or pytest tests/ -v"
            ]
          },
          {
            title: "Git Workflow",
            concepts: ["Atomic commits", "Commit messages", "Branching", "Pull requests", "GitHub portfolio"],
            practice: [
              "Commit working code daily",
              "Write descriptive commit messages",
              "Use feature branches for experiments",
              "Create clean Git history"
            ],
            checkYourself: "Can someone understand your project history?",
            gitWorkflow: `# Daily workflow
git status                          # What changed?
git add channel_detector/models.py  # Stage specific file
git commit -m "Add Viterbi decoding to HMM class

- Implement Viterbi algorithm for most likely state sequence
- Add unit tests for synthetic data
- Benchmark against hmmlearn (matches within 1e-6)"

git push origin main

# Feature branch for experiments
git checkout -b experiment/bocpd-integration
# Make changes, commit
git checkout main
git merge experiment/bocpd-integration

# Good commit message structure:
# Line 1: Summary (50 chars, imperative mood)
# Line 2: Blank
# Lines 3+: Details (what, why, how)`,
            ankiCards: [
              "Q: Stage file? | A: git add filename",
              "Q: Commit? | A: git commit -m 'message'",
              "Q: Create branch? | A: git checkout -b branch-name",
              "Q: Good commit message? | A: Imperative mood, 50 char summary"
            ]
          }
        ],
        milestone: {
          task: "Production-Ready Codebase",
          description: "Full channel-regime-detection package on GitHub",
          structure: `channel-regime-detection/
├── README.md                    # Project overview, examples, installation
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
├── .gitignore                   # Ignore __pycache__, .pyc, data/
├── channel_detector/
│   ├── __init__.py
│   ├── features.py              # Feature engineering functions
│   ├── models.py                # HMM, Kalman classes
│   ├── visualization.py         # Plotting functions
│   └── utils.py                 # Data loading, preprocessing
├── tests/
│   ├── __init__.py
│   ├── test_features.py         # 10+ tests
│   ├── test_models.py           # 15+ tests
│   └── test_utils.py            # 5+ tests
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_hmm_baseline.ipynb
│   └── 04_full_pipeline.ipynb
├── examples/
│   └── quickstart.py            # Runnable example
└── docs/
    └── methodology.md           # Math & approach`,
          successCriteria: [
            "README with badges (tests passing, coverage)",
            "80%+ test coverage",
            "Type hints on all public functions",
            "Passes mypy type checking",
            "Clean Git history (atomic commits)",
            "Example notebooks run without errors",
            "Published on GitHub with stars"
          ]
        },
        resources: [
          { type: "🌐", name: "Real Python - Testing Guide", url: "realpython.com" },
          { type: "📚", name: "Git Pro Book (Free)", url: "git-scm.com" },
          { type: "🎥", name: "Corey Schafer - Git Tutorial", url: "YouTube" }
        ]
      }
    ]
  }
];

const knowledgeSystem = {
  title: "4-Layer Knowledge Organization",
  layers: [
    {
      name: "Layer 1: Digital Notes",
      tool: "Obsidian / Notion",
      purpose: "Quick reference, searchable, linked",
      updateFrequency: "Daily",
      structure: [
        "Quick-Reference/ (cheatsheets)",
        "Concepts/ (deep dives)",
        "Snippets/ (code patterns)",
        "Projects/ (what you're building)"
      ],
      example: "NumPy-Broadcasting.md with examples, pitfalls, links to related concepts"
    },
    {
      name: "Layer 2: Code Repository",
      tool: "GitHub",
      purpose: "Reusable patterns, portfolio, version control",
      updateFrequency: "Every working session",
      structure: [
        "python-quant-patterns/",
        "  array_operations/",
        "  time_series/",
        "  hmm/",
        "  utilities/"
      ],
      example: "rolling_operations.py with fast implementations, tests, benchmarks"
    },
    {
      name: "Layer 3: Physical Notebook",
      tool: "Paper notebook / iPad",
      purpose: "Math, debugging, sketching, thinking",
      updateFrequency: "While coding",
      structure: [
        "Date each page",
        "Math derivations",
        "Algorithm sketches",
        "Debugging traces"
      ],
      example: "Hand-drawn HMM state diagram with transition probabilities annotated"
    },
    {
      name: "Layer 4: Spaced Repetition",
      tool: "Anki",
      purpose: "Long-term retention, interview prep",
      updateFrequency: "Add when you make mistakes, review daily",
      structure: [
        "Python-Syntax deck",
        "NumPy-Operations deck",
        "Common-Pitfalls deck",
        "Interview-Prep deck"
      ],
      example: "Q: Fast rolling sum? | A: Use cumsum, subtract offset cumsum"
    }
  ],
  workflow: {
    daily: [
      "Morning: 15min Anki review",
      "Coding: Physical notes for math/debugging",
      "Evening: Transfer insights to digital notes",
      "Commit working code to GitHub"
    ],
    weekly: [
      "Sunday: Review & consolidate digital notes",
      "Sunday: Refactor & document code",
      "Sunday: Add new Anki cards from week's mistakes"
    ]
  }
};

function PythonRoadmap() {
  const [selectedMonth, setSelectedMonth] = useState(0);
  const [expandedWeek, setExpandedWeek] = useState(null);
  const [expandedTopic, setExpandedTopic] = useState(null);
  const [completedItems, setCompletedItems] = useState(new Set());
  const [showKnowledge, setShowKnowledge] = useState(false);

  const toggleComplete = (id) => {
    setCompletedItems(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const totalItems = pythonRoadmap.reduce((acc, month) => 
    acc + month.weeks.reduce((a, w) => a + w.topics.length, 0), 0
  );
  const progress = Math.round((completedItems.size / totalItems) * 100);

  return (
    <div style={{
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      background: "linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%)",
      minHeight: "100vh",
      color: "#e0e0e0",
      padding: "2rem 1rem"
    }}>
      <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
        
        {/* Header */}
        <div style={{
          background: "linear-gradient(135deg, #1e293b 0%, #334155 100%)",
          borderRadius: "16px",
          padding: "2rem",
          marginBottom: "2rem",
          border: "1px solid #475569"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
            <div style={{
              width: "48px",
              height: "48px",
              background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.5rem"
            }}>🐍</div>
            <div>
              <h1 style={{ margin: 0, fontSize: "2rem", color: "#f1f5f9" }}>Python for Quant Finance</h1>
              <p style={{ margin: "0.25rem 0 0", color: "#94a3b8", fontSize: "0.95rem" }}>
                Integrated 12-week learning plan
              </p>
            </div>
          </div>
          
          <div style={{
            background: "#1e293b",
            borderRadius: "12px",
            padding: "1rem",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "1rem"
          }}>
            <div>
              <div style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "0.25rem" }}>
                Progress
              </div>
              <div style={{ fontSize: "1.75rem", fontWeight: "600", color: "#3b82f6" }}>
                {progress}%
              </div>
            </div>
            <div>
              <div style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "0.25rem" }}>
                Completed
              </div>
              <div style={{ fontSize: "1.75rem", fontWeight: "600", color: "#10b981" }}>
                {completedItems.size}/{totalItems}
              </div>
            </div>
            <div style={{ flex: 1, minWidth: "200px" }}>
              <div style={{
                height: "8px",
                background: "#334155",
                borderRadius: "999px",
                overflow: "hidden"
              }}>
                <div style={{
                  height: "100%",
                  width: `${progress}%`,
                  background: "linear-gradient(90deg, #3b82f6, #8b5cf6, #f59e0b)",
                  transition: "width 0.3s ease"
                }} />
              </div>
            </div>
          </div>
          
          <button
            onClick={() => setShowKnowledge(!showKnowledge)}
            style={{
              marginTop: "1rem",
              background: showKnowledge ? "#8b5cf6" : "#475569",
              color: "white",
              border: "none",
              borderRadius: "8px",
              padding: "0.75rem 1.25rem",
              fontSize: "0.9rem",
              cursor: "pointer",
              fontWeight: "500",
              transition: "all 0.2s"
            }}
          >
            {showKnowledge ? "Hide" : "Show"} Knowledge Organization System
          </button>
        </div>

        {/* Knowledge System */}
        {showKnowledge && (
          <div style={{
            background: "#1e293b",
            borderRadius: "16px",
            padding: "2rem",
            marginBottom: "2rem",
            border: "1px solid #475569"
          }}>
            <h2 style={{ margin: "0 0 1.5rem", color: "#f1f5f9", fontSize: "1.5rem" }}>
              📚 {knowledgeSystem.title}
            </h2>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
              {knowledgeSystem.layers.map((layer, i) => (
                <div key={i} style={{
                  background: "#0f172a",
                  borderRadius: "12px",
                  padding: "1.25rem",
                  border: "1px solid #334155"
                }}>
                  <h3 style={{ margin: "0 0 0.5rem", color: "#3b82f6", fontSize: "1rem" }}>
                    {layer.name}
                  </h3>
                  <div style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "0.75rem" }}>
                    <strong>Tool:</strong> {layer.tool}
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "#cbd5e1", marginBottom: "0.75rem" }}>
                    {layer.purpose}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "#64748b", fontStyle: "italic" }}>
                    Update: {layer.updateFrequency}
                  </div>
                </div>
              ))}
            </div>
            
            <div style={{
              background: "#0f172a",
              borderRadius: "12px",
              padding: "1.5rem",
              border: "1px solid #334155"
            }}>
              <h3 style={{ margin: "0 0 1rem", color: "#f1f5f9", fontSize: "1.1rem" }}>
                Daily Workflow
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
                {knowledgeSystem.workflow.daily.map((step, i) => (
                  <div key={i} style={{
                    fontSize: "0.85rem",
                    color: "#cbd5e1",
                    padding: "0.5rem",
                    background: "#1e293b",
                    borderRadius: "6px",
                    borderLeft: "3px solid #3b82f6"
                  }}>
                    {step}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Month Selector */}
        <div style={{ display: "flex", gap: "0.75rem", marginBottom: "2rem", flexWrap: "wrap" }}>
          {pythonRoadmap.map((month, i) => (
            <button
              key={i}
              onClick={() => {
                setSelectedMonth(i);
                setExpandedWeek(null);
                setExpandedTopic(null);
              }}
              style={{
                background: selectedMonth === i 
                  ? `linear-gradient(135deg, ${month.color}, ${month.color}cc)`
                  : "#334155",
                color: "white",
                border: "none",
                borderRadius: "12px",
                padding: "1rem 1.5rem",
                cursor: "pointer",
                fontSize: "0.95rem",
                fontWeight: "500",
                transition: "all 0.2s",
                flex: "1",
                minWidth: "150px",
                boxShadow: selectedMonth === i ? "0 4px 12px rgba(0,0,0,0.3)" : "none"
              }}
            >
              <div style={{ fontSize: "0.75rem", opacity: 0.9, marginBottom: "0.25rem" }}>
                {month.month}
              </div>
              <div>{month.title}</div>
            </button>
          ))}
        </div>

        {/* Selected Month Content */}
        {pythonRoadmap.map((month, monthIdx) => {
          if (monthIdx !== selectedMonth) return null;
          
          return (
            <div key={monthIdx}>
              <div style={{
                background: `linear-gradient(135deg, ${month.color}20, ${month.color}10)`,
                borderRadius: "12px",
                padding: "1.5rem",
                marginBottom: "2rem",
                border: `1px solid ${month.color}40`
              }}>
                <div style={{ fontSize: "0.85rem", color: month.color, marginBottom: "0.5rem", fontWeight: "500" }}>
                  {month.period} • {month.integrationNote}
                </div>
                <h2 style={{ margin: 0, color: "#f1f5f9", fontSize: "1.75rem" }}>
                  {month.title}
                </h2>
              </div>

              {/* Weeks */}
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {month.weeks.map((week, weekIdx) => {
                  const weekId = `${monthIdx}-${weekIdx}`;
                  const isExpanded = expandedWeek === weekId;
                  const weekCompleted = week.topics.every((_, ti) => 
                    completedItems.has(`${weekId}-${ti}`)
                  );

                  return (
                    <div
                      key={weekIdx}
                      style={{
                        background: "#1e293b",
                        borderRadius: "12px",
                        border: `1px solid ${weekCompleted ? "#10b981" : "#475569"}`,
                        overflow: "hidden"
                      }}
                    >
                      {/* Week Header */}
                      <div
                        onClick={() => setExpandedWeek(isExpanded ? null : weekId)}
                        style={{
                          padding: "1.5rem",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          gap: "1rem",
                          background: isExpanded ? "#334155" : "transparent",
                          transition: "background 0.2s"
                        }}
                      >
                        <div style={{
                          width: "40px",
                          height: "40px",
                          borderRadius: "10px",
                          background: weekCompleted ? "#10b981" : month.color,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "1.25rem",
                          flexShrink: 0
                        }}>
                          {weekCompleted ? "✓" : weekIdx + 1}
                        </div>
                        
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginBottom: "0.25rem" }}>
                            {week.week} • {week.difficulty} • {week.timeCommitment}
                          </div>
                          <div style={{ fontSize: "1.2rem", color: "#f1f5f9", fontWeight: "500", marginBottom: "0.25rem" }}>
                            {week.title}
                          </div>
                          <div style={{ fontSize: "0.85rem", color: "#cbd5e1" }}>
                            Goal: {week.goal}
                          </div>
                          <div style={{
                            fontSize: "0.75rem",
                            color: month.color,
                            marginTop: "0.5rem",
                            padding: "0.5rem",
                            background: `${month.color}15`,
                            borderRadius: "6px",
                            borderLeft: `3px solid ${month.color}`
                          }}>
                            🎯 Quant Trigger: {week.quantTrigger}
                          </div>
                        </div>
                        
                        <div style={{
                          color: "#94a3b8",
                          fontSize: "1.5rem",
                          transform: isExpanded ? "rotate(180deg)" : "rotate(0)",
                          transition: "transform 0.2s"
                        }}>
                          ▼
                        </div>
                      </div>

                      {/* Week Content */}
                      {isExpanded && (
                        <div style={{ padding: "0 1.5rem 1.5rem" }}>
                          
                          {/* Topics */}
                          {week.topics.map((topic, topicIdx) => {
                            const topicId = `${weekId}-${topicIdx}`;
                            const isTopicExpanded = expandedTopic === topicId;
                            const isCompleted = completedItems.has(topicId);

                            return (
                              <div
                                key={topicIdx}
                                style={{
                                  background: "#0f172a",
                                  borderRadius: "8px",
                                  marginBottom: "0.75rem",
                                  border: `1px solid ${isCompleted ? "#10b98150" : "#334155"}`,
                                  overflow: "hidden"
                                }}
                              >
                                <div
                                  onClick={() => setExpandedTopic(isTopicExpanded ? null : topicId)}
                                  style={{
                                    padding: "1rem",
                                    cursor: "pointer",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "0.75rem"
                                  }}
                                >
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleComplete(topicId);
                                    }}
                                    style={{
                                      width: "24px",
                                      height: "24px",
                                      borderRadius: "6px",
                                      background: isCompleted ? "#10b981" : "transparent",
                                      border: `2px solid ${isCompleted ? "#10b981" : "#475569"}`,
                                      color: "white",
                                      cursor: "pointer",
                                      fontSize: "0.85rem",
                                      display: "flex",
                                      alignItems: "center",
                                      justifyContent: "center",
                                      flexShrink: 0
                                    }}
                                  >
                                    {isCompleted ? "✓" : ""}
                                  </button>
                                  
                                  <div style={{ flex: 1 }}>
                                    <div style={{
                                      fontSize: "1rem",
                                      color: isCompleted ? "#94a3b8" : "#f1f5f9",
                                      fontWeight: "500",
                                      textDecoration: isCompleted ? "line-through" : "none"
                                    }}>
                                      {topic.title}
                                    </div>
                                  </div>
                                  
                                  <div style={{
                                    color: "#64748b",
                                    fontSize: "1rem",
                                    transform: isTopicExpanded ? "rotate(180deg)" : "rotate(0)",
                                    transition: "transform 0.2s"
                                  }}>
                                    ▼
                                  </div>
                                </div>

                                {isTopicExpanded && (
                                  <div style={{ padding: "0 1rem 1rem" }}>
                                    <div style={{
                                      fontSize: "0.85rem",
                                      color: "#cbd5e1",
                                      marginBottom: "1rem",
                                      lineHeight: "1.6"
                                    }}>
                                      <strong>Concepts:</strong>
                                      <ul style={{ margin: "0.5rem 0", paddingLeft: "1.5rem" }}>
                                        {topic.concepts.map((c, i) => (
                                          <li key={i}>{c}</li>
                                        ))}
                                      </ul>
                                    </div>

                                    <div style={{
                                      fontSize: "0.85rem",
                                      color: "#cbd5e1",
                                      marginBottom: "1rem",
                                      lineHeight: "1.6"
                                    }}>
                                      <strong>Practice:</strong>
                                      <ul style={{ margin: "0.5rem 0", paddingLeft: "1.5rem" }}>
                                        {topic.practice.map((p, i) => (
                                          <li key={i}>{p}</li>
                                        ))}
                                      </ul>
                                    </div>

                                    <div style={{
                                      background: "#1e293b",
                                      borderRadius: "6px",
                                      padding: "0.75rem",
                                      fontSize: "0.85rem",
                                      color: "#fbbf24",
                                      borderLeft: "3px solid #fbbf24"
                                    }}>
                                      <strong>Check yourself:</strong> {topic.checkYourself}
                                    </div>

                                    {topic.codeTemplate && (
                                      <div style={{ marginTop: "1rem" }}>
                                        <div style={{
                                          fontSize: "0.75rem",
                                          color: "#94a3b8",
                                          marginBottom: "0.5rem",
                                          fontFamily: "monospace"
                                        }}>
                                          CODE TEMPLATE:
                                        </div>
                                        <pre style={{
                                          background: "#0f172a",
                                          border: "1px solid #334155",
                                          borderRadius: "6px",
                                          padding: "1rem",
                                          overflow: "auto",
                                          fontSize: "0.8rem",
                                          color: "#e0e0e0",
                                          fontFamily: "monospace"
                                        }}>
                                          {topic.codeTemplate}
                                        </pre>
                                      </div>
                                    )}

                                    {topic.ankiCards && (
                                      <div style={{ marginTop: "1rem" }}>
                                        <div style={{
                                          fontSize: "0.75rem",
                                          color: "#94a3b8",
                                          marginBottom: "0.5rem",
                                          fontFamily: "monospace"
                                        }}>
                                          ANKI CARDS TO CREATE:
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                                          {topic.ankiCards.map((card, i) => (
                                            <div key={i} style={{
                                              background: "#8b5cf620",
                                              border: "1px solid #8b5cf650",
                                              borderRadius: "6px",
                                              padding: "0.75rem",
                                              fontSize: "0.8rem",
                                              color: "#cbd5e1",
                                              fontFamily: "monospace"
                                            }}>
                                              {card}
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}

                          {/* Milestone */}
                          {week.milestone && (
                            <div style={{
                              background: `linear-gradient(135deg, ${month.color}20, ${month.color}10)`,
                              border: `2px solid ${month.color}`,
                              borderRadius: "12px",
                              padding: "1.5rem",
                              marginTop: "1rem"
                            }}>
                              <div style={{
                                fontSize: "0.9rem",
                                color: month.color,
                                fontWeight: "600",
                                marginBottom: "0.5rem",
                                display: "flex",
                                alignItems: "center",
                                gap: "0.5rem"
                              }}>
                                🎯 WEEK MILESTONE: {week.milestone.task}
                              </div>
                              <div style={{
                                fontSize: "0.9rem",
                                color: "#cbd5e1",
                                marginBottom: "1rem",
                                lineHeight: "1.6"
                              }}>
                                {week.milestone.description}
                              </div>
                              {week.milestone.codeChallenge && (
                                <pre style={{
                                  background: "#0f172a",
                                  border: "1px solid #334155",
                                  borderRadius: "8px",
                                  padding: "1rem",
                                  overflow: "auto",
                                  fontSize: "0.8rem",
                                  color: "#e0e0e0",
                                  fontFamily: "monospace",
                                  marginBottom: "1rem"
                                }}>
                                  {week.milestone.codeChallenge}
                                </pre>
                              )}
                              <div style={{
                                fontSize: "0.85rem",
                                color: "#94a3b8"
                              }}>
                                <strong>Success criteria:</strong>
                                <ul style={{ margin: "0.5rem 0", paddingLeft: "1.5rem" }}>
                                  {week.milestone.successCriteria.map((c, i) => (
                                    <li key={i}>{c}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}

                          {/* Resources */}
                          <div style={{
                            marginTop: "1rem",
                            padding: "1rem",
                            background: "#0f172a",
                            borderRadius: "8px",
                            border: "1px solid #334155"
                          }}>
                            <div style={{
                              fontSize: "0.85rem",
                              color: "#94a3b8",
                              marginBottom: "0.75rem",
                              fontWeight: "600"
                            }}>
                              📚 Resources
                            </div>
                            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                              {week.resources.map((res, i) => (
                                <div key={i} style={{
                                  fontSize: "0.85rem",
                                  color: "#cbd5e1",
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "0.5rem"
                                }}>
                                  <span>{res.type}</span>
                                  <span>{res.name}</span>
                                  {res.url !== "Book" && res.url !== "YouTube" && (
                                    <span style={{ color: "#64748b", fontSize: "0.75rem" }}>
                                      ({res.url})
                                    </span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}