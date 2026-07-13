# Metrics Guide

This file explains every metric currently registered in `src/metrics.py`.

The project is trying to answer a practical question: **when should we be long TQQQ, and when should we avoid it?** Most signals are computed on **QQQ**, not TQQQ, because QQQ is the underlying Nasdaq-100 ETF and is cleaner for measuring market state. TQQQ is a 3x leveraged ETF; its price is strongly affected by leverage, volatility drag, and path dependence.

Each metric produces a time series. Voting metrics then map that time series into:

| Vote | Meaning |
|---|---|
| `+1` | Bullish / favorable for being long TQQQ |
| `0` | Neutral / no opinion |
| `-1` | Bearish / unfavorable for being long TQQQ |

Most percentile rules use a rolling 252-trading-day window, roughly one market year. For example, `> p80` means “higher than this metric has been 80% of the time over the last year.” This is useful because many financial quantities do not have stable absolute levels; a “high” volatility value in 2017 may be very different from a “high” volatility value in 2020.

## Useful Finance Concepts

**QQQ vs TQQQ**

QQQ tracks the Nasdaq-100. TQQQ targets roughly 3x the daily return of QQQ. If QQQ returns `+1%` today, TQQQ aims for about `+3%` today. But over many days, TQQQ is not simply `3 x QQQ` because returns compound.

**Return**

The simple return from price `P0` to `P1` is:

```text
return = P1 / P0 - 1
```

The log return is:

```text
log_return = log(P1 / P0)
```

Log returns are mathematically convenient because multi-day log returns add approximately over time.

**Trend / Momentum**

Trend means price has been moving persistently in one direction. Momentum strategies assume recent strength may continue.

**Mean Reversion**

Mean reversion means price has moved too far in one direction and may snap back. For example, a very short-term oversold condition may be bullish if you expect a rebound.

**Volatility**

Volatility measures how violently price moves. For a leveraged ETF like TQQQ, high volatility can be dangerous even if the long-term trend is up, because compounding can damage returns.

**Percentile Vote**

Many metrics use this structure:

```text
if metric is very high relative to its last 252 trading days -> vote one way
if metric is very low relative to its last 252 trading days -> vote the other way
otherwise -> vote 0
```

That makes the rules adaptive to changing market regimes.

## Trend / Momentum Metrics

These ask: **is the Nasdaq-100 trending upward or downward?**

### `qqq_sma50_200_regime`

**What it measures:** whether the medium-term trend is above or below the long-term trend.

Formula:

```text
SMA50(QQQ close) - SMA200(QQQ close)
```

`SMA50` is the average closing price over the last 50 trading days. `SMA200` is the average closing price over the last 200 trading days.

Interpretation:

- If the 50-day average is above the 200-day average, the market is usually considered to be in an uptrend.
- If the 50-day average is below the 200-day average, the market is usually considered to be in a downtrend.

Vote rule:

| Condition | Vote |
|---|---|
| `SMA50 - SMA200 > 0` (golden cross) | `-1` |
| `SMA50 - SMA200 < 0` (death cross) | `+1` |

Finance intuition: **inverted from the classic trend-filter interpretation.** Validation data showed that being long TQQQ *below* the golden cross (death-cross regime) produced higher forward returns than being long above it. This is a mean-reversion / contrarian read: the market may be too extended when above the 200-day trend, and recovering from oversold conditions below it.

**Research results (train+val, TQQQ 5d tradable open):**

| Metric | Val edge 5d | BH q-val | Horizons agree | Net ann bps (2bps spread) | Regime robust | Last-window edge | Overall |
|---|---|---|---|---|---|---|---|
| `qqq_sma50_200_regime` | +122 bps | 0.17 | 5/5 | +65 | 3/5 — fails strong_bear | +408 bps, **improving** | Borderline |

Regime breakdown (HSMM states): strong_bull +394, weak_bull −29, sideways +361, weak_bear +74, **strong_bear −455**. Works in most regimes but fails in crashes, which is its biggest risk. Rolling edge has been improving (first window +118 bps → last window +408 bps). Cost filter barely passes at only 0.7 trades/year.

### `qqq_mom_12_1`

**What it measures:** long-term momentum, excluding the most recent month.

Formula:

```text
QQQ_close[t - 21] / QQQ_close[t - 252] - 1
```

This is roughly the return from 12 months ago to 1 month ago. The most recent month is skipped because very recent moves can sometimes reverse.

Interpretation:

- High value: QQQ was strong over the prior year.
- Low value: QQQ was weak over the prior year.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `+1` |
| Below rolling 252-day `p20` | `-1` |
| Otherwise | `0` |

Finance intuition: markets with strong intermediate/long-term momentum often continue to do well, especially equity indices.

Status: `watch` — sub-window analysis of train+val showed sign instability across periods (+16, −47, +26, −183 bps across rolling sub-windows). The aggregate 5/5-horizon agreement is a statistical artefact of pooling; the signal is not reliable enough to include in production voting.

### `qqq_20d_slope`

**What it measures:** short-term trend strength over the last 20 trading days.

Formula:

```text
OLS slope of log(QQQ close) over 20 trading days, annualized
```

In more mathematical terms, for each 20-day window it fits:

```text
log(price_i) = a + b * i
```

and reports `b * 252`.

Interpretation:

- Positive slope: QQQ has been rising.
- Negative slope: QQQ has been falling.
- Larger absolute value: stronger short-term trend.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `+1` |
| Below rolling 252-day `p20` | `-1` |
| Otherwise | `0` |

Finance intuition: this is a cleaner trend estimate than simply comparing today’s price with price 20 days ago, because it uses the whole window.

### `qqq_mom_term_structure`

**What it measures:** whether momentum agrees across short, medium, and long horizons.

Formula:

```text
sign(1-month return) + sign(3-month return) + sign(12-month return)
```

Each component is:

```text
sign(QQQ_close[t] / QQQ_close[t - lookback] - 1)
```

with lookbacks of approximately 21, 63, and 252 trading days.

Possible values are `-3`, `-1`, `1`, and `3`.

Interpretation:

- `3`: all horizons are positive.
- `-3`: all horizons are negative.
- Mixed values: time horizons disagree.

Vote rule:

| Condition | Vote |
|---|---|
| Value >= `2.5` | `+1` |
| Value <= `-2.5` | `-1` |
| Otherwise | `0` |

Finance intuition: a trend is more convincing when multiple horizons point the same way.

## Mean Reversion Metrics

These ask: **has QQQ moved too far too quickly?**

Mean reversion signals are different from trend signals. A trend signal says “strength may continue.” A mean reversion signal says “this move may be stretched.”

### `qqq_rsi2`

**What it measures:** very short-term overbought/oversold pressure.

RSI means Relative Strength Index. This project uses a 2-day RSI, so it reacts very quickly.

Formula idea:

```text
RSI = 100 - 100 / (1 + smoothed_up_moves / smoothed_down_moves)
```

Interpretation:

- Near `0`: recent price action has been extremely weak.
- Near `100`: recent price action has been extremely strong.
- Around `50`: balanced.

Vote rule:

| Condition | Vote |
|---|---|
| RSI <= `20` | `+1` |
| RSI >= `80` | `-1` |
| Otherwise | `0` |

Finance intuition: a very low 2-day RSI means QQQ may be short-term oversold and due for a bounce. A very high 2-day RSI may mean it is stretched and vulnerable to a pullback.

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (2bps spread) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| +52 bps | 0.01 | 5/5 | 107 | +2567 | 4/5 ✓ | +118 bps, stable |

Regime breakdown: strong_bull +162, weak_bull +89, **sideways −7**, weak_bear +267, strong_bear +973. Nearly regime-agnostic — only fails marginally in sideways. Despite the very high flip rate (107/yr), the large per-trade gross edge (52 bps) absorbs friction comfortably. Best-rounded candidate across all three diagnostic dimensions.

### `qqq_bb_z20`

**What it measures:** how far QQQ is from its 20-day moving average.

Formula:

```text
(QQQ_close - SMA20) / (2 * rolling_std20)
```

This is related to Bollinger Bands. A value of `+1` means price is about two standard deviations above its 20-day average. A value of `-1` means price is about two standard deviations below.

Interpretation:

- High positive value: price is stretched above its recent average.
- Low negative value: price is stretched below its recent average.

Vote rule:

| Condition | Vote |
|---|---|
| Value <= `-1.0` | `+1` |
| Value >= `1.0` | `-1` |
| Otherwise | `0` |

Finance intuition: this assumes unusually large deviations from the recent average may partially reverse.

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (2bps spread) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| +174 bps | 0.03 | 5/5 | 27.5 | +2334 | 4/5 ✓ | **−95 bps ⚠️ decayed** |

Regime breakdown: strong_bull +211, weak_bull +354, **sideways −595**, weak_bear +1108, strong_bear +291. Works strongly in volatile bear regimes (weak_bear +1108 bps) but craters in sideways (−595 bps). The rolling 2-year edge trended from +407 bps (early train) to **−95 bps** in the most recent 2-year window (roughly 2020–2021 low-vol bull). The aggregate val average overstates the current edge — this is a real signal but one that starves in calm, trending markets. Likely cyclical rather than structural decay since sideways regimes suppress its fire-rate.

## Volatility Metrics

These ask: **how risky or unstable is the market right now?**

This is especially important for TQQQ. A leveraged ETF can suffer in high-volatility sideways markets because daily compounding can erode returns.

### `qqq_rv_20d`

**What it measures:** annualized realized volatility over the last 20 trading days.

Formula:

```text
sqrt(252 * mean(log_return^2 over 20 days))
```

`252` is used because there are about 252 trading days in a year.

Interpretation:

- High value: QQQ has been moving violently.
- Low value: QQQ has been calm.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `-1` |
| Below rolling 252-day `p20` | `+1` |
| Otherwise | `0` |

Finance intuition: lower volatility is usually friendlier for leveraged long exposure. High volatility increases risk and volatility drag.

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (raw direction) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| −23 bps | 0.00 | 5/5 | 16.6 | −242 | 1/5 ✗ | +103 bps, stable |

Val edge is negative because the *bear* vote outperforms the *bull* vote (high-RV periods are actually bullish for TQQQ in the short term — mean reversion after volatility spikes). Regime breakdown: strong_bull −78, weak_bull −203, **sideways +552**, weak_bear −171. This is an almost pure **sideways-regime signal**: works only when the HSMM is in the sideways state. Not suitable as a standalone signal, but worth revisiting as a regime-conditional filter. Marked as finalist via negative edge (inverted direction) with consistent q=0.00.

### `qqq_rv_60d`

**What it measures:** annualized realized volatility over the last 60 trading days.

Formula:

```text
sqrt(252 * mean(log_return^2 over 60 days))
```

Interpretation is the same as `qqq_rv_20d`, but over a slower window.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `-1` |
| Below rolling 252-day `p20` | `+1` |
| Otherwise | `0` |

Finance intuition: 60-day volatility measures the broader risk regime, not just a recent spike.

### `qqq_rv_ratio`

**What it measures:** whether short-term volatility is high or low compared with medium-term volatility.

Formula:

```text
rv_20d / rv_60d
```

Interpretation:

- Above `1`: 20-day volatility is higher than 60-day volatility.
- Below `1`: 20-day volatility is lower than 60-day volatility.

Vote rule:

| Condition | Vote |
|---|---|
| Ratio <= `0.8` | `+1` |
| Ratio >= `1.3` | `-1` |
| Otherwise | `0` |

Finance intuition: if short-term volatility is much higher than medium-term volatility, risk may be accelerating. That is usually bad for TQQQ.

### `qqq_yz_vol_20d`

**What it measures:** a more complete 20-day volatility estimate using open, high, low, and close prices.

The Yang-Zhang volatility estimator combines:

- overnight moves: previous close to today’s open;
- intraday moves: today’s open to today’s close;
- high/low range information.

Interpretation:

- It captures more information than close-to-close volatility.
- It can notice intraday instability even if close-to-close returns look calm.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `-1` |
| Below rolling 252-day `p20` | `+1` |
| Otherwise | `0` |

Finance intuition: high realized volatility from any source is generally less friendly for a leveraged long ETF.

### `qqq_williams_vix_fix`

**What it measures:** how far the current low is below the recent high close.

Formula:

```text
(highest_close_22d - QQQ_low) / highest_close_22d * 100
```

Interpretation:

- High value: price has dropped far from a recent high, a stress/fear-like condition.
- Low value: price is close to recent highs.

Vote rule implemented in code:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p90` (large drop from high) | `+1` |
| Below rolling 252-day `p10` (price near recent high) | `-1` |
| Otherwise | `0` |

Finance intuition: **inverted from a naive bearish-volatility read.** A large WVF value (price far below recent high) is treated as a contrarian buy signal — a fear spike that historically rebounds. Validation data confirmed 5/5 horizon agreement in the bullish direction. Raw vote correlation with `bb_z20` is only 0.26 (safe to hold both despite raw-value correlation of −0.82).

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (2bps spread) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| +129 bps | 0.01 | 5/5 | 47.8 | +2964 | 4/5 ✓ | **−309 bps ⚠️ decayed** |

Regime breakdown: strong_bull +164, weak_bull +433, **sideways −963**, weak_bear +151, strong_bear **+1885**. The strongest signal during crash regimes (+1885 bps in strong_bear). Useless or harmful in sideways (−963 bps). Rolling edge decayed from +271 bps → **−309 bps** in the most recent 2-year window — same pattern as `bb_z20`: the 2020–2021 low-volatility bull suppressed volatility spikes and starved this signal. Likely cyclical.

### `vix_term_structure`

**What it measures:** whether near-term implied volatility is high relative to medium-term implied volatility.

Preferred formula:

```text
VIX / VIX3M
```

VIX roughly measures expected S&P 500 volatility over the next 30 days. VIX3M measures expected volatility over roughly 3 months.

Interpretation:

- Ratio above `1`: near-term fear is higher than medium-term fear.
- Ratio below `1`: market is calmer in the near term than over the medium term.

Vote rule:

| Condition | Vote |
|---|---|
| Ratio <= `0.90` | `+1` |
| Ratio >= `1.05` | `-1` |
| Otherwise | `0` |

Finance intuition: when near-term volatility jumps above medium-term volatility, markets are often stressed. That is usually a bad environment for TQQQ.

## Leveraged ETF Metrics

These ask: **what special risks come from using TQQQ instead of QQQ?**

### `tqqq_vol_drag_est`

**What it measures:** an approximation of the volatility drag from 3x leverage.

Formula:

```text
0.5 * L * (L - 1) * rv_20d^2
```

where `L = 3`.

So:

```text
0.5 * 3 * 2 * rv_20d^2 = 3 * rv_20d^2
```

Interpretation:

- Higher value means volatility is expected to be more damaging to a 3x leveraged ETF.
- Lower value means the environment is more forgiving for leverage.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p75` | `-1` |
| Below rolling 252-day `p25` | `+1` |
| Otherwise | `0` |

Finance intuition: TQQQ likes strong trends with controlled volatility. It dislikes choppy high-volatility markets.

### `tqqq_path_residual`

**What it measures:** how TQQQ performed over 60 days versus a simple `3 x QQQ` approximation.

Formula:

```text
60d cumulative TQQQ return - 3 * 60d cumulative QQQ return
```

Interpretation:

- Positive value: TQQQ did better than the simple 3x approximation.
- Negative value: TQQQ did worse than the simple 3x approximation.

Vote rule implemented in code:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p90` | `+1` |
| Below rolling 252-day `p10` | `-1` |
| Otherwise | `0` |

Finance intuition: this checks whether the leveraged ETF’s realized path has recently been favorable or unfavorable relative to the underlying index. Positive residuals usually occur when the path has been friendly to leveraged compounding.

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (raw direction) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| −91 bps | 0.10 | 3/5 | 18.0 | −865 | 3/5 ✗ | **−248 bps ⚠️** |

Val edge is negative (bear votes outperform bull votes); the vote fires bull only 9% of the time. Regime breakdown: strong_bull +13, weak_bull −523, sideways +211, weak_bear +277, strong_bear insufficient obs. Fails the cost filter in raw direction, has only 3/5 horizon agreement and q=0.10. Rolling edge decayed from +210 → −248 bps. Does not pass minimum thresholds for strategy inclusion without further investigation.

### `qqq_dd_from_high`

**What it measures:** QQQ’s drawdown from its 252-day high.

Formula:

```text
(QQQ_close - rolling_252d_max(QQQ_close)) / rolling_252d_max(QQQ_close)
```

Interpretation:

- `0`: QQQ is at a 1-year high.
- `-0.10`: QQQ is 10% below its 1-year high.
- More negative values mean deeper drawdown.

Status: `watch`

Finance intuition: this is useful for visual context. It does not vote in the current strategy.

## Cross-Asset / Macro Metrics

These ask: **what are other markets saying about risk appetite, rates, and credit?**

### `qqq_spy_ratio_slope`

**What it measures:** whether QQQ is outperforming or underperforming SPY.

Formula:

```text
OLS slope of log(QQQ / SPY) over 20 trading days
```

SPY tracks the S&P 500. QQQ is more growth/technology-heavy. The ratio `QQQ / SPY` rises when QQQ is outperforming SPY.

Interpretation:

- Positive slope: QQQ is outperforming the broad market.
- Negative slope: QQQ is underperforming.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `+1` |
| Below rolling 252-day `p20` | `-1` |
| Otherwise | `0` |

Finance intuition: TQQQ tends to do better when Nasdaq leadership is strong.

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (2bps spread) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| +17 bps | 0.43 | 4/5 | 17.9 | +102 | 3/5 ✗ | **−14 bps ✗** |

Regime breakdown: strong_bull +290, weak_bull −158, sideways +134, weak_bear −314, strong_bear +1036. Works at the extremes (crashes and strong bull runs) but unreliable in the middle states. Thin margin: net +102 bps after costs and q=0.43 (not statistically significant). Last rolling window is slightly negative. Weak candidate on its own — may have value as a secondary filter in combination with stronger signals.

### `yield_curve_10y3m`

**What it measures:** the difference between long-term and short-term interest rates.

Preferred formula:

```text
10-year Treasury yield - 3-month Treasury yield
```

In the data, this is:

```text
^TNX_close - ^IRX_close
```

Interpretation:

- Positive/steep curve: long rates are above short rates.
- Flat or inverted curve: short rates are close to or above long rates.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `+1` |
| Below rolling 252-day `p20` | `-1` |
| Otherwise | `0` |

Finance intuition: a steep curve is often associated with easier growth conditions. A flat/inverted curve can signal tight monetary policy or recession risk.

Status: `watch` — the val-split bear bucket had zero observations (no times the metric voted bear in the 2018–2021 val window). Without a populated bear bucket there is no meaningful bull-minus-bear edge to measure.

### `tnx_20d_chg`

**What it measures:** how quickly the 10-year Treasury yield has changed over 20 trading days.

Formula:

```text
^TNX_close[t] - ^TNX_close[t - 20]
```

Interpretation:

- Positive value: 10-year yields rose over the last month.
- Negative value: 10-year yields fell.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `-1` |
| Below rolling 252-day `p20` | `+1` |
| Otherwise | `0` |

Finance intuition: fast-rising rates can pressure growth stocks because higher discount rates reduce the present value of future earnings. QQQ is growth-heavy, so this matters.

### `hyg_lqd_ratio_chg`

**What it measures:** whether riskier corporate bonds are outperforming safer corporate bonds.

Formula:

```text
20-day percent change of HYG / LQD
```

HYG is high-yield corporate bonds. LQD is investment-grade corporate bonds.

Interpretation:

- Rising `HYG / LQD`: credit markets are showing risk appetite.
- Falling `HYG / LQD`: investors prefer safer credit.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p80` | `+1` |
| Below rolling 252-day `p20` | `-1` |
| Otherwise | `0` |

Finance intuition: when credit markets are willing to own riskier debt, equity risk appetite is often healthier too.

**Research results (train+val, TQQQ 5d tradable open):**

| Val edge 5d | BH q-val | Horizons agree | Flips/yr | Net ann bps (raw direction) | Regime robust | Last-window edge |
|---|---|---|---|---|---|---|
| −32 bps | 0.10 | 2/5 | 43.8 | −813 | 3/5 ✗ | +19 bps, improving |

Val edge is negative (bear votes outperform), and only 2/5 horizons agree. High flip rate (43.8/yr) makes friction very costly. Regime breakdown: strong_bull +49, weak_bull −309, sideways +374, weak_bear +147, strong_bear −646. Mixed regime profile with no robust pattern. Not a viable standalone signal at this stage.

## Microstructure / Trading Activity

### `tqqq_volume_z`

**What it measures:** unusual dollar trading volume in TQQQ.

Formula:

```text
dollar_volume = TQQQ_volume * TQQQ_close
z = (dollar_volume - rolling_20d_mean) / rolling_20d_std
```

Interpretation:

- `z = 0`: volume is normal relative to the last 20 days.
- `z = 2`: volume is about two standard deviations above normal.
- Negative values: volume is below normal.

Vote rule:

| Condition | Vote |
|---|---|
| Above rolling 252-day `p90` | `-1` |
| Below rolling 252-day `p10` | `+1` |
| Otherwise | `0` |

Finance intuition: large volume spikes in leveraged ETFs can occur during stress or speculative extremes. This implementation treats unusually high volume as a risk warning.

## Calendar Metrics

### `fomc_drift`

**What it measures:** whether today is within two trading days before an FOMC decision date.

Formula:

```text
1 if today is 1 or 2 trading days before a listed FOMC date
0 otherwise
```

FOMC means Federal Open Market Committee, the part of the Federal Reserve that sets monetary policy.

Vote rule:

| Condition | Vote |
|---|---|
| `fomc_drift = 1` | `+1` |
| `fomc_drift = 0` | `0` |

Finance intuition: there is a market pattern often called “pre-FOMC drift,” where equities have historically tended to perform well before Fed announcements. This is a calendar effect, not a price/volatility signal.

## Watch-Only Metrics

Watch-only metrics are plotted and inspected but do not contribute to the strategy vote.

### `qqq_skew_60d`

**What it measures:** asymmetry of QQQ daily log returns over the last 60 trading days.

Formula:

```text
skewness(log_returns over 60 days)
```

Interpretation:

- Positive skew: upside outliers have been larger/more common.
- Negative skew: downside outliers have been larger/more common.

Status: `watch`

Finance intuition: negative skew can indicate crash-like behavior or downside tail risk.

### `qqq_kurt_60d`

**What it measures:** tail heaviness of QQQ daily log returns over the last 60 trading days.

Formula:

```text
kurtosis(log_returns over 60 days)
```

Interpretation:

- Higher kurtosis: returns have more extreme outliers.
- Lower kurtosis: returns are closer to a normal-looking distribution.

Status: `watch`

Finance intuition: high kurtosis means the market has recently produced unusually large moves, which is relevant for leveraged products.

## Important Caveats

These metrics are **hypothesis generators**, not proof of alpha.

Common traps:

- A visually convincing pattern may disappear out of sample.
- Many metrics are correlated with each other, so combining them can double-count the same idea.
- TQQQ can lose money even when QQQ is not down much, especially during volatile sideways markets.
- A metric that worked in one regime may fail in another.
- You should avoid tuning thresholds on the test period.

The intended workflow is:

1. Use `notebooks/05_indicator_workbench.py` to visually inspect relationships.
2. Use `notebooks/02_metric_inspection.py` to examine individual metrics statistically.
3. Turn only the most plausible ideas into explicit rules.
4. Test rules on train and validation data.
5. Touch the test set only when the strategy is finalized.

The authoritative implementation is `REGISTRY` in `src/metrics.py`. This file is documentation for humans; if there is a mismatch, the code wins.

---

## Pre-Strategy Diagnostics Summary

Run by `notebooks/07_strategy_readiness.py` on train+val data (2006–2021), TQQQ 5d tradable-open forward returns, 2 bps one-way spread, 86 bps annual ETF expense, HSMM 5-state regime model. Test set (2022+) not used.

### Master diagnostic table

| Metric | Val edge 5d | q-val | Horiz agree | Net ann bps | Cost pass | Regime robust | Last window | Verdict |
|---|---|---|---|---|---|---|---|---|
| `qqq_rsi2` | +52 | 0.01 | 5/5 | +2567 | ✓ | ✓ 4/5 | +118 stable | **Best candidate** |
| `qqq_bb_z20` | +174 | 0.03 | 5/5 | +2334 | ✓ | ✓ 4/5 | −95 ⚠️ decayed | Strong but decayed recently |
| `qqq_williams_vix_fix` | +129 | 0.01 | 5/5 | +2964 | ✓ | ✓ 4/5 | −309 ⚠️ decayed | Strongest in crashes, decayed in calm |
| `qqq_sma50_200_regime` | +122 | 0.17 | 5/5 | +65 | ✓ (thin) | ✗ 3/5 | +408 improving | Improving; fails in strong_bear |
| `qqq_spy_ratio_slope` | +17 | 0.43 | 4/5 | +102 | ✓ (thin) | ✗ 3/5 | −14 ✗ | Weak; secondary filter only |
| `qqq_rv_20d` | −23 | 0.00 | 5/5 | −242 | ✗ | ✗ 1/5 | +103 stable | Sideways-only; not standalone |
| `hyg_lqd_ratio_chg` | −32 | 0.10 | 2/5 | −813 | ✗ | ✗ 3/5 | +19 improving | Not viable standalone |
| `tqqq_path_residual` | −91 | 0.10 | 3/5 | −865 | ✗ | ✗ 3/5 | −248 ✗ | Below threshold |

### Key findings

**Edge decay in the top two signals.** `bb_z20` and `williams_vix_fix` show rolling 2-year edge declining from ~270–400 bps (early train) to **negative** values in the most recent 2-year window (approximately 2019–2021). This is likely cyclical: both are volatility-spike / dip-buying signals that starve during sustained low-volatility bull markets. Their regime profiles confirm this — both collapse in the HSMM sideways state (−600 to −960 bps) and perform strongly in bear states.

**`qqq_rsi2` passes all three filters** (cost, regime robustness, rolling stability). It is the only metric that does so cleanly. High flip rate (107/yr) is compensated by large per-trade gross edge.

**Regime profiles matter more than aggregate edges.** `bb_z20` and `wvf` are regime-conditional signals: their aggregate val edges are real but concentrated in bear and volatile regimes. A strategy using them must either accept regime dependency or filter them with a regime-awareness condition.

**Next decisions required before strategy construction:**
- Aggregation rule (majority / weighted / AND-logic)
- Position sizing (binary / scaled / vol-target)
- `bb_z20` ↔ `williams_vix_fix` redundancy resolution (vote-level Spearman corr = 0.26, safe to hold both, but their decay patterns are correlated)
- Selection-bias correction (permutation test on metric survival pipeline)
- Benchmark definition (CAGR / Sharpe / MaxDD targets vs TQQQ buy-and-hold)
