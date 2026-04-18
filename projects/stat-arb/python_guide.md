# Python Toolkit for Stat Arb

This guide covers the specific Python/pandas/numpy/statsmodels tools you'll need.
The stat-arb project is much more pandas-heavy than vol-surface — the core challenge
is manipulating large DataFrames efficiently, not solving PDEs.

---

## 1. Pandas: The Core Tool

### The data shape
Your main object is a panel: (dates × stocks). Everything flows from this.

```python
import pandas as pd
import numpy as np

# prices: DataFrame with DatetimeIndex, columns = tickers
# shape: (~5000 trading days, ~500 stocks)
prices = pd.DataFrame(...)  # dates as rows, tickers as columns

# Daily returns (percentage change)
returns = prices.pct_change()

# Monthly returns (resample to month-end, compute return)
monthly_prices = prices.resample('ME').last()
monthly_returns = monthly_prices.pct_change()
```

### Cross-sectional operations
The key pattern: apply an operation **across stocks** at each date.

```python
# Cross-sectional rank at each date
# rank(axis=1) ranks across columns (stocks) for each row (date)
momentum_ranks = momentum_signal.rank(axis=1, pct=True)

# Cross-sectional z-score at each date
def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise each row to zero mean, unit std."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

z_scores = cross_sectional_zscore(momentum_signal)
```

**Critical:** `axis=1` means "across columns" (across stocks at each date).
`axis=0` means "across rows" (across dates for each stock). Getting this wrong
is the #1 source of bugs in factor code.

### Rolling windows
```python
# 252-day trailing momentum (cumulative return over past year)
momentum = prices.pct_change(252)

# 21-day skip (exclude most recent month)
# Shift by 21 days so we look at t-252 to t-21
momentum_12_1 = prices.shift(21).pct_change(252 - 21)

# 60-day rolling volatility
rolling_vol = returns.rolling(60).std() * np.sqrt(252)

# 20-day average volume
avg_volume = prices_volume.rolling(20).mean()
```

### GroupBy for sector-neutral construction
```python
# sector_map: Series mapping ticker -> sector
sector_map = pd.Series({'AAPL': 'Tech', 'JPM': 'Financials', 'XOM': 'Energy', ...})

# Sector-neutral z-score: rank within each sector
def sector_neutral_zscore(signal_row: pd.Series, sectors: pd.Series) -> pd.Series:
    """Z-score within each sector for a single date."""
    return signal_row.groupby(sectors).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

# Apply to each date
sector_neutral = signal.apply(lambda row: sector_neutral_zscore(row, sector_map), axis=1)
```

### Pivoting and reshaping
```python
# Long format: (date, ticker, signal_value)
long_df = signal.stack().reset_index()
long_df.columns = ['date', 'ticker', 'signal']

# Back to wide: dates as rows, tickers as columns
wide_df = long_df.pivot(index='date', columns='ticker', values='signal')

# Merge factor signals with returns for regression
merged = pd.concat({
    'momentum': z_momentum,
    'value': z_value,
    'return': forward_returns,
}, axis=1)
```

---

## 2. NumPy: Vectorised Factor Computation

### Quintile portfolio formation
```python
def quintile_portfolio_returns(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    At each date, sort stocks into quantiles by signal,
    compute equal-weight return of each quantile.

    Returns DataFrame: (dates, quantiles) -> average return.
    """
    quantile_returns = {}

    for q in range(1, n_quantiles + 1):
        # pd.qcut assigns quantile labels cross-sectionally
        in_quantile = signal.apply(
            lambda row: pd.qcut(row.dropna(), n_quantiles, labels=False) == (q - 1),
            axis=1,
        )
        # Equal-weight average return of stocks in this quantile
        quantile_returns[q] = forward_returns[in_quantile].mean(axis=1)

    return pd.DataFrame(quantile_returns)
```

A faster vectorised approach using ranks directly:

```python
def fast_quintile_returns(signal: pd.DataFrame, fwd_ret: pd.DataFrame, n_q: int = 5):
    """Vectorised quintile sort — no row-level loop."""
    # Rank cross-sectionally, scale to [0, 1]
    pct_ranks = signal.rank(axis=1, pct=True)

    results = {}
    for q in range(n_q):
        lo, hi = q / n_q, (q + 1) / n_q
        mask = (pct_ranks > lo) & (pct_ranks <= hi)
        # count per row to compute mean
        counts = mask.sum(axis=1).replace(0, np.nan)
        results[q + 1] = (fwd_ret * mask).sum(axis=1) / counts

    return pd.DataFrame(results)
```

### Information Coefficient
```python
from scipy.stats import spearmanr

def compute_ic_series(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Spearman rank correlation between signal and next-period return,
    computed cross-sectionally at each date.
    """
    ic_values = {}
    for date in signal.index:
        s = signal.loc[date].dropna()
        r = forward_returns.loc[date].reindex(s.index).dropna()
        common = s.index.intersection(r.index)
        if len(common) < 30:  # need minimum stocks for meaningful correlation
            continue
        corr, _ = spearmanr(s[common], r[common])
        ic_values[date] = corr

    return pd.Series(ic_values)
```

Faster with `DataFrame.corrwith`:
```python
def compute_ic_series_fast(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """Vectorised IC using rank correlation row by row."""
    ranked_signal = signal.rank(axis=1)
    ranked_returns = forward_returns.rank(axis=1)

    # Pearson correlation of ranks = Spearman correlation
    # Demean each row, then correlate
    sig_dm = ranked_signal.sub(ranked_signal.mean(axis=1), axis=0)
    ret_dm = ranked_returns.sub(ranked_returns.mean(axis=1), axis=0)

    numerator = (sig_dm * ret_dm).sum(axis=1)
    denominator = np.sqrt((sig_dm**2).sum(axis=1) * (ret_dm**2).sum(axis=1))

    return numerator / denominator
```

---

## 3. Statsmodels: Regressions

### Fama-MacBeth regression
```python
import statsmodels.api as sm

def fama_macbeth(
    signals: dict[str, pd.DataFrame],   # {'momentum': df, 'value': df, ...}
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run cross-sectional regression at each date, collect slopes.

    Returns DataFrame of slopes: (dates × factor_names).
    """
    factor_names = list(signals.keys())
    slopes = []

    for date in forward_returns.index:
        y = forward_returns.loc[date].dropna()

        # Build X matrix: each column is one factor signal
        X_dict = {}
        for name in factor_names:
            X_dict[name] = signals[name].loc[date].reindex(y.index)

        X = pd.DataFrame(X_dict).dropna()
        y = y.reindex(X.index)

        if len(y) < 50:
            continue

        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        row = {name: model.params[name] for name in factor_names}
        row['date'] = date
        row['r_squared'] = model.rsquared
        slopes.append(row)

    result = pd.DataFrame(slopes).set_index('date')

    # Summary statistics
    print("Fama-MacBeth Results:")
    print(f"{'Factor':<12} {'Mean':>8} {'Std':>8} {'t-stat':>8}")
    print("-" * 40)
    for name in factor_names:
        mean = result[name].mean()
        std = result[name].std()
        t = mean / (std / np.sqrt(len(result)))
        print(f"{name:<12} {mean:>8.4f} {std:>8.4f} {t:>8.2f}")

    return result
```

### Factor return regression (attribution)
```python
def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,  # e.g. Fama-French factors from Ken French's site
) -> None:
    """Regress portfolio returns on known factor returns."""
    common = portfolio_returns.index.intersection(factor_returns.index)
    y = portfolio_returns.loc[common]
    X = sm.add_constant(factor_returns.loc[common])

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Alpha is the intercept — the unexplained return
    alpha = model.params['const']
    alpha_tstat = model.tvalues['const']
    print(f"\nAlpha: {alpha:.4f} (t={alpha_tstat:.2f})")
```

---

## 4. Data Fetching at Scale

### Batch download with yfinance
```python
import yfinance as yf

def fetch_sp500_prices(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download daily prices for many tickers in batches.
    yfinance struggles with >50 tickers at once.
    """
    all_frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Fetching batch {i // batch_size + 1}: {len(batch)} tickers")

        data = yf.download(batch, start=start, end=end,
                           progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']].rename(columns={'Close': batch[0]})

        all_frames.append(prices)

    combined = pd.concat(all_frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]  # drop duplicate cols

    print(f"  Total: {combined.shape[1]} tickers, {combined.shape[0]} trading days")
    return combined
```

### S&P 500 constituents from Wikipedia
```python
def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    return sorted(sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist())
```

Note: this gives you **today's** constituents, not historical. For proper
survivorship-bias handling, you'd need historical membership data (see plan.md Phase 1A).

---

## 5. Portfolio Backtest Engine

### The core loop
```python
def backtest_long_short(
    signal: pd.DataFrame,        # (dates, tickers) -> z-score
    returns: pd.DataFrame,       # (dates, tickers) -> next-period return
    n_long: int = 100,           # number of stocks in long leg
    n_short: int = 100,          # number of stocks in short leg
    cost_per_trade: float = 0.001,  # 10 bps per side
) -> pd.DataFrame:
    """
    Equal-weight long/short portfolio backtest.

    At each date:
    1. Rank stocks by signal
    2. Go long top n_long, short bottom n_short (equal weight)
    3. Compute portfolio return = mean(long returns) - mean(short returns)
    4. Subtract transaction costs based on turnover
    """
    portfolio_returns = []
    prev_longs = set()
    prev_shorts = set()

    for date in signal.index:
        sig = signal.loc[date].dropna()
        ret = returns.loc[date].reindex(sig.index).dropna()
        common = sig.index.intersection(ret.index)

        if len(common) < n_long + n_short:
            continue

        sig = sig[common]
        ret = ret[common]

        # Sort and select
        ranked = sig.sort_values()
        shorts = set(ranked.head(n_short).index)
        longs = set(ranked.tail(n_long).index)

        # Portfolio return
        long_ret = ret[list(longs)].mean()
        short_ret = ret[list(shorts)].mean()
        gross_return = long_ret - short_ret

        # Turnover cost
        long_turnover = len(longs - prev_longs) / n_long if prev_longs else 0
        short_turnover = len(shorts - prev_shorts) / n_short if prev_shorts else 0
        turnover = (long_turnover + short_turnover) / 2
        cost = turnover * cost_per_trade * 2  # buy + sell

        portfolio_returns.append({
            'date': date,
            'gross_return': gross_return,
            'net_return': gross_return - cost,
            'turnover': turnover,
        })

        prev_longs = longs
        prev_shorts = shorts

    return pd.DataFrame(portfolio_returns).set_index('date')
```

### Performance metrics (reuse patterns from all-weather)
```python
def compute_strategy_stats(returns: pd.Series) -> dict:
    """Compute standard performance metrics from a return series."""
    cumulative = (1 + returns).cumprod()
    total_days = (returns.index[-1] - returns.index[0]).days
    years = total_days / 365.25

    cagr = (cumulative.iloc[-1] ** (1 / years) - 1)

    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = drawdowns.min()

    sharpe = returns.mean() / returns.std() * np.sqrt(12)  # annualised from monthly
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'hit_rate': (returns > 0).mean(),
        'avg_turnover': None,  # fill in from backtest
    }
```

---

## 6. Scikit-Learn (for PCA and optional ML)

### PCA on factor signals
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def factor_pca(signals: dict[str, pd.DataFrame], date: str, n_components: int = 3):
    """
    Run PCA on cross-section of factor signals at a given date.
    Shows how many independent sources of variation exist.
    """
    # Build (n_stocks, n_factors) matrix
    X_dict = {name: df.loc[date].dropna() for name, df in signals.items()}
    X = pd.DataFrame(X_dict).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    return pca
```

---

## 7. Fetching Fama-French Factor Data

```python
def fetch_ff_factors(start: str = '2006-01-01') -> pd.DataFrame:
    """
    Download Fama-French 5 factors + momentum from Ken French's data library.
    Returns monthly factor returns as percentages.
    """
    import pandas_datareader.data as web

    # Fama-French 5 factors
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start)
    ff5_monthly = ff5[0] / 100  # convert from percentage to decimal

    # Momentum factor
    mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=start)
    mom_monthly = mom[0] / 100

    return ff5_monthly.join(mom_monthly)
```

If `pandas_datareader` isn't available, download the CSV directly:
```python
def fetch_ff_factors_csv() -> pd.DataFrame:
    """Download FF factors directly from Ken French's website."""
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
           "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
    df = pd.read_csv(url, skiprows=3, index_col=0)
    # Parse the YYYYMM index
    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m')
    df = df.apply(pd.to_numeric, errors='coerce') / 100
    return df.dropna()
```

---

## 8. Testing Patterns for Factor Code

```python
def test_cross_sectional_zscore_properties():
    """Z-scores must have mean~0 and std~1 at each date."""
    signal = pd.DataFrame(
        np.random.randn(100, 50),
        index=pd.date_range('2020-01-01', periods=100, freq='B'),
    )
    z = cross_sectional_zscore(signal)
    # Mean of each row should be ~0
    assert z.mean(axis=1).abs().max() < 1e-10
    # Std of each row should be ~1
    assert (z.std(axis=1) - 1.0).abs().max() < 1e-10

def test_long_short_is_dollar_neutral():
    """Long and short legs should have equal total weight."""
    ...

def test_quintile_returns_cover_all_stocks():
    """Every stock should appear in exactly one quintile."""
    ...

def test_ic_bounded():
    """IC must be in [-1, 1]."""
    ic = compute_ic_series(signal, returns)
    assert ic.min() >= -1.0
    assert ic.max() <= 1.0

def test_sector_neutral_zero_sector_exposure():
    """Sector-neutral portfolio should have zero net weight in each sector."""
    ...
```

**Key testing principles for factor code:**
- Test statistical properties (means, bounds, coverage)
- Test invariances (dollar neutrality, sector neutrality)
- Use synthetic data with known properties for unit tests
- Use real data for integration/smoke tests
- Factor returns are noisy — don't test for specific return levels
