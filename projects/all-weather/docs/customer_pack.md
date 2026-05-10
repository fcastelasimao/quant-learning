# All Weather Strategy Customer Pack

## Plain-English Thesis

This is a low-cost, rules-based ETF portfolio designed for investors who care
more about avoiding severe drawdowns than maximizing equity-like upside.

The portfolio spreads capital across US equities, long-term Treasuries,
inflation-linked bonds, gold, and broad commodities. It is rebalanced monthly
to a fixed risk-balanced allocation. It does not use leverage, forecasts, short
positions, derivatives, or discretionary market calls.

## Current Production Allocation

| Role | ETF used in backtest | Live ETF | Weight |
|---|---:|---:|---:|
| Broad US equity | SPY | SPY | 13.4% |
| US growth equity | QQQ | QQQ | 10.3% |
| Deflation/risk-off hedge | TLT | TLT | 17.5% |
| Inflation-linked bonds | TIP | TIP | 34.8% |
| Gold hedge | GLD | GLDM | 14.2% |
| Commodity inflation hedge | GSG | GSG | 9.8% |

GLDM is used live instead of GLD for lower fees while preserving similar gold
exposure. Other lower-fee substitutions are not used unless the strategy is
re-validated with those assets.

## Who This Is For

- Investors who want a transparent defensive allocation.
- Investors comfortable with ETFs and monthly rebalancing.
- Investors who accept lower upside in strong equity bull markets in exchange
  for lower expected drawdowns.
- Investors who understand that bonds, commodities, and gold can all lose money.

## Who This Is Not For

- Investors seeking guaranteed capital protection.
- Investors seeking high-growth equity-like returns.
- Investors who need personalized tax, currency, or regulatory advice.
- Investors who cannot tolerate multi-year underperformance versus the S&P 500.

## Expected Behavior

- In equity selloffs, the strategy should usually fall less than an all-equity
  portfolio, but it can still lose money.
- In strong equity rallies, it will usually trail the S&P 500 because less than
  one quarter of capital is in equity ETFs.
- In inflation or commodity shocks, TIP, GLD/GLDM, and GSG are intended to help,
  but they are not guaranteed hedges.
- In sharp rate-hiking cycles, TLT and TIP can both lose money at the same time.

## Important Limitations

- Backtests are not live trading results.
- The ALLW ETF comparison starts only from ALLW's March 2025 launch, so it is a
  short marketing comparison, not a full-cycle proof.
- The 2018, 2020, and 2022 validation windows overlap and should be described as
  stress windows, not fully independent samples.
- Current results are USD-based. GBP/EUR currency effects are not modeled.
- This material is educational and research-oriented. It is not financial advice.
