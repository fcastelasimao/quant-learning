"""
Implementation agent — translates a factor hypothesis into Python code
using Qwen2.5-Coder:14b via Ollama.
"""
from __future__ import annotations

import ollama

from qframe.pipeline.models import HypothesisSpec

_OLLAMA_MODEL = "qwen2.5-coder:14b"

_PROMPT_TEMPLATE = """\
You are a quantitative finance engineer implementing a cross-sectional equity factor.

Factor name:  {name}
Description:  {description}
Rationale:    {rationale}

Write a Python function with EXACTLY this signature:

def factor(prices: pd.DataFrame) -> pd.DataFrame:
    ...

=== INPUT CONTRACT ===
- `prices`: daily adjusted CLOSE prices ONLY (no Open, High, Low, Volume)
- Shape: (dates, tickers) — rows are trading days, columns are ticker symbols
- Index: pd.DatetimeIndex sorted ascending. Example shape: (3770, 412)

=== OUTPUT CONTRACT ===
- Return pd.DataFrame of EXACTLY the same shape AND index AND columns as `prices`
- Values = factor signal for each (date, ticker); NaN where insufficient history

=== AVAILABLE TOOLS ===
pandas (pd), numpy (np), and scipy.stats (stats) are already imported.
DO NOT import anything else (no from scipy import stats, no import math, etc.).

=== MANDATORY RULES ===
1. NO row-by-row Python loops (no for/while, no iterrows, no itertuples)
2. No look-ahead bias — at date t use only rows up to row t (.shift(), .rolling(), .ewm())
3. No print statements, no side effects
4. If numpy returns an ndarray, wrap it: pd.DataFrame(arr, index=prices.index, columns=prices.columns)

=== CRITICAL BUGS TO AVOID ===

BUG 1 — DataFrame groupby(DataFrame): NEVER do this — raises "not 1-dimensional"
  WRONG:  prices.groupby(other_df)
  WRONG:  direction.groupby(direction.cumsum())
  CORRECT for run-length: use .apply() column-by-column — see example below.

BUG 2 — Tilde on floats: .shift() introduces NaN which makes bool columns float.
  WRONG:  ~direction.shift(1)           # TypeError: bad operand type for unary ~
  CORRECT: direction.shift(1).fillna(False).astype(bool)   # safe

BUG 3 — Unaligned operands: never compare/subtract two DataFrames unless you know
  their columns align. Always use prices.index and prices.columns for output.

BUG 4 — Using .diff() then comparing without handling NaN:
  WRONG:  prices.diff() > 0             # first row is NaN → mixed bool/float
  CORRECT: prices.diff().fillna(0) > 0  # explicit NaN handling

BUG 5 — Boolean DataFrame item assignment creates object dtype (breaks np.isnan):
  WRONG:  result = pd.DataFrame(index=prices.index, columns=prices.columns)
          result[mask] = some_values   # creates mixed-type object array
  CORRECT: use np.where or direct arithmetic:
          result = pd.DataFrame(np.where(mask, some_values, np.nan),
                                index=prices.index, columns=prices.columns)

BUG 6 — ANY secondary groupby after building run_id still fails (run_id is a DataFrame):
  WRONG:  avg = returns.where(mask).groupby(run_id).transform('mean')  # run_id = DataFrame → error
  WRONG:  (returns != 0).groupby(run_id).cumsum()
  CORRECT: for per-run statistics, use rolling windows masked by the condition:
          avg = returns.where(mask).rolling(window, min_periods=1).mean()
  RULE: NEVER call .groupby() on a 2D DataFrame key in ANY step.

BUG 7 — .astype(int) on a column that contains NaN raises IntCastingNaNError:
  WRONG:  change = (direction != prev).fillna(True).astype(int)  # NaN sneaks through .ne()
  CORRECT: explicitly replace NaN first:
          diff = direction.diff()
          change = (diff.abs() > 0).astype(int)  # diff of numeric → NaN only at row 0
          change.iloc[0] = 0  # handle first row explicitly

BUG 8 — df.apply(scalar_fn) collapses to a Series (one value per column), NOT a DataFrame:
  WRONG:  hurst_series = rs_values.apply(scalar_fn)   # shape: (n_tickers,) — WRONG
          pd.DataFrame(hurst_series.values, ...)       # shape mismatch error!
  CORRECT: use .rolling().apply() directly — it returns a DataFrame automatically:
          result = returns.rolling(window=252).apply(my_scalar_fn, raw=True)
          # result has shape (n_dates, n_tickers) ✓ — return this directly
  RULE: NEVER call df.apply(fn) where fn returns a scalar — it reduces shape.

BUG 10 — np.where() on a DataFrame returns a plain ndarray, NOT a DataFrame:
  WRONG:  direction = np.where(returns > 0, 1, -1)  # returns ndarray
          prev = direction.shift(1)                  # AttributeError: ndarray has no shift!
  CORRECT option A — use numpy ufunc that preserves DataFrame type:
          direction = np.sign(returns)               # np.sign keeps DataFrame type
  CORRECT option B — wrap np.where result immediately:
          direction = pd.DataFrame(np.where(returns > 0, 1, np.where(returns < 0, -1, 0)),
                                   index=prices.index, columns=prices.columns)

=== PERFORMANCE CONSTRAINT ===
The universe is ~450 stocks × ~3800 dates. Any factor that runs in O(n_stocks × n_dates) with
a pure-Python scalar function will timeout. FORBIDDEN patterns (will cause TimeoutError):
- Rolling OLS regression per stock (stats.linregress, np.polyfit inside .rolling().apply())
- R/S Hurst exponent (rolling().apply() with a multi-step Python function)
- Any .rolling().apply() where the applied function has more than ~5 numpy operations

FAST alternatives: use only native pandas/numpy vectorized operations (.rolling().mean(),
.rolling().std(), .rank(), .pct_change(), .diff(), .ewm()). These run in compiled C code and
will NOT timeout.

=== APPROVED PATTERNS ===

# Rolling z-score:
mu = prices.rolling(252).mean()
sigma = prices.rolling(252).std()
result = (prices - mu) / sigma.replace(0, np.nan)

# Momentum (skip last month):
result = prices.shift(21) / prices.shift(252) - 1

# Vectorized run-length (consecutive same-direction days):
returns = prices.pct_change()
up = (returns.fillna(0) > 0).astype(int)
# safe change-point detection (no ~float bug):
prev_up = up.shift(1).fillna(-1).astype(int)
change = (up != prev_up).astype(int)
run_id = change.cumsum()
# apply column-by-column to avoid 2D groupby error:
run_len = up.apply(lambda col: col.groupby(run_id[col.name]).cumcount() + 1)
result = run_len.where(returns.notna())  # NaN where returns are undefined

# EWMA:
result = prices.ewm(span=20, min_periods=20).mean()

# Rolling autocorrelation (lag-1):
returns = prices.pct_change()
result = returns.rolling(60).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False)

# Rolling skewness (use raw=True for speed):
from scipy.stats import skew as _skew  # only inside approved pattern comment — stats is pre-imported
result = returns.rolling(252, min_periods=60).apply(lambda x: stats.skew(x), raw=True)

# Rolling Sharpe ratio (trend quality proxy — FAST: only pandas rolling ops):
returns = prices.pct_change()
roll_mean = returns.rolling(252, min_periods=63).mean()
roll_std  = returns.rolling(252, min_periods=63).std()
result = roll_mean / roll_std.replace(0, np.nan)

Return ONLY the function definition — no imports, no example usage, no explanation.\
"""


_FIX_PROMPT_TEMPLATE = """\
The following Python factor function raised an error when executed:

ERROR: {error}

ORIGINAL CODE:
{code}

Fix the bug. Common causes:
- "Grouper for DataFrame not 1-dimensional" → you called .groupby(DataFrame_key). Use .apply(lambda col: col.groupby(run_id[col.name]).cumcount()) instead, OR use rolling windows.
- "bad operand type for unary ~" → shift introduced NaN making bool column float. Use .fillna(False).astype(bool).
- "Cannot convert non-finite values to integer" → NaN present before .astype(int). Use .fillna(0).astype(int).
- "ndarray has no attribute shift/rolling" → np.where() returned ndarray. Use np.sign() or wrap: pd.DataFrame(arr, index=prices.index, columns=prices.columns).
- "factor contains infinite values" → division by zero. Replace denominator zeros: denom.replace(0, np.nan).
- "Operands are not aligned" → two DataFrames with different column sets. Use .reindex(columns=prices.columns) or align first.
- "'<' not supported between Timestamp and str" → comparing DatetimeIndex to a string. Use pd.Timestamp('2018-01-01') for comparisons.
- "Shape of passed values is (M, 1), indices imply (N, M)" → df.apply(scalar_fn) where scalar_fn returns a SCALAR per column, collapsing to a 1-D Series.
  WRONG:  result = df.apply(lambda col: skewness(col))
  CORRECT: result = df.rolling(window=252, min_periods=50).apply(lambda x: skewness(x), raw=True)

Return ONLY the corrected function definition. No explanation.\
"""


class ImplementationAgent:
    """
    Calls Qwen2.5-Coder:14b (Ollama) to generate factor code.
    On execution errors, attempts a one-shot self-healing fix.

    Args:
        model:   Ollama model tag. Default: qwen2.5-coder:14b.
        timeout: seconds to wait for Ollama response. Default: 120.
    """

    def __init__(self, model: str = _OLLAMA_MODEL, timeout: int = 120):
        self._model = model
        self._timeout = timeout

    def _call_ollama(self, prompt: str, temperature: float) -> dict:
        """
        Call ollama.generate with backward-compatible timeout handling.

        Some ollama Python SDK versions don't accept the `timeout` keyword on
        `generate()`. In that case we retry once without timeout to avoid
        breaking the pipeline.
        """
        kwargs = {
            "model": self._model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": 768},
            "timeout": self._timeout,
        }
        try:
            return ollama.generate(**kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument 'timeout'" not in msg:
                raise
            kwargs.pop("timeout", None)
            return ollama.generate(**kwargs)

    def generate(self, hypothesis: HypothesisSpec) -> str:
        """
        Generate Python factor code for the given hypothesis.

        Args:
            hypothesis: HypothesisSpec from the Synthesis agent.

        Returns:
            String containing the Python function definition.
        """
        prompt = _PROMPT_TEMPLATE.format(
            name=hypothesis.name,
            description=hypothesis.description,
            rationale=hypothesis.rationale,
        )

        response = self._call_ollama(prompt, temperature=0.2)

        return response["response"]

    def fix(self, code: str, error: str) -> str:
        """
        Attempt a one-shot fix of broken factor code given the error message.

        Args:
            code:  The original generated code string.
            error: The error message / last line of the traceback.

        Returns:
            String containing the fixed Python function definition.
        """
        prompt = _FIX_PROMPT_TEMPLATE.format(code=code, error=error)

        response = self._call_ollama(prompt, temperature=0.1)

        return response["response"]
