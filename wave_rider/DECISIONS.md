# Wave Rider Decisions

## 2026-03-31

### Default Research Universe

Decision:
- use `us_research_core` as the default preset

Why:
- safer Yahoo Finance coverage
- longer and cleaner daily histories
- better for research before broker-specific implementation details

Consequence:
- research results will be easier to generate now
- later we still need to choose or validate the live implementation universe

### Baseline Strategy Style

Decision:
- build a long-only cross-asset trend allocator with cash fallback

Why:
- simpler to understand and validate
- avoids early complexity from shorting and options
- still fits the goal of surviving rocky markets better than passive equity

Consequence:
- the baseline aims for robust participation and defense, not maximum possible
  bear-market profit

### Trend Definition

Decision:
- use a 126-day absolute trend filter
- use blended 3/6/12 month relative strength

Why:
- medium-term trend is more stable than very short windows
- blended ranking reduces dependence on one lookback

Consequence:
- this is a strong starting point, but still needs robustness testing

### Diversification Rule

Decision:
- cap selection at one asset per bucket in the baseline

Why:
- prevents the portfolio from becoming three versions of the same idea
- easier to reason about than a fragile correlation optimizer

Consequence:
- portfolio may hold more cash if leadership is concentrated in one bucket

### Defense Mode

Decision:
- use trend breadth to scale gross exposure

Why:
- simple, interpretable, and easy to test
- avoids adding a complex regime model too early

Consequence:
- if breadth logic adds no value in testing, we should remove or simplify it

### Python Environment

Decision:
- use a dedicated `wave_rider` environment
- seed it by cloning `allweather` when possible

Why:
- both projects use a very similar Python stack
- cloning is faster than rebuilding from scratch
- a separate environment avoids hidden dependency drift between projects

Consequence:
- we keep the speed benefit of the existing environment
- we still preserve project isolation

### Validation Approach

Decision:
- add a small built-in validation sweep before changing the baseline further

Why:
- a single backtest result is too easy to overinterpret
- nearby variants reveal whether the baseline is robust or just lucky

Consequence:
- strategy changes should now be judged against both benchmarks and local
  sensitivity checks

### Promoted Baseline

Decision:
- promote the longer-trend version to the default baseline
- promote monthly rebalance and a looser bucket cap
- remove defense scaling from the default configuration

Why:
- the first validation pass showed longer trend windows had the best Calmar
- monthly rebalance reduced whipsaw and cost drag
- tighter bucket caps were suppressing return too much
- the breadth defense overlay was not clearly improving the strategy

Consequence:
- the new baseline should capture upside more cleanly without pretending to be
  a balanced all-weather mix
- defense mode remains available as a tested variant rather than a default rule

### Latest Baseline Update

Decision:
- keep weekly rebalance
- move from tight to moderate bucket clustering

Why:
- the latest validation pass showed weekly rebalance improved Calmar
- the strongest candidate used a looser cap than the tight default
- we want a trend allocator that can lean into leadership without becoming
  unconstrained

Consequence:
- the next research question is whether weekly still wins after stronger
  transaction-cost stress tests and where the clustering sweet spot really sits

### Dual Profile Structure

Decision:
- keep both an aggressive and a defensive Wave Rider profile

Why:
- the aggressive profile is currently the best return-seeking version
- the defensive profile gives us a simpler risk-managed alternative without
  turning the whole project into an all-weather clone

Consequence:
- future research should compare both profiles directly rather than forcing one
  configuration to solve every objective

### Result Storage Pattern

Decision:
- keep overwritable latest outputs and immutable archived runs

Why:
- latest files are convenient for immediate analysis
- archived runs protect the research trail and make comparisons reproducible

Consequence:
- `results/` is the working view
- `results/runs/` is the research record

### Parking Sleeve And No-Trade Band

Decision:
- promote `INT_BOND` (`IEF`) as the default parking sleeve
- add a 2% no-trade band to the default baseline

Why:
- validation showed `IEF` parking was the clearest way to improve CAGR
- the 2% no-trade band modestly reduced churn without changing the strategy's
  character
- this kept the strategy dynamic and trend-led rather than turning it into an
  all-weather balance engine

Consequence:
- the promoted aggressive baseline now earns more while accepting somewhat
  deeper drawdowns than the legacy cash-fallback version
- the defensive overlay needs to be reconsidered because it no longer improves
  both return and drawdown once the parking sleeve is active

### Position Cap Update

Decision:
- raise the promoted max position cap from 25% to 30%

Why:
- the CAGR-focused validation pass showed this was the cleanest way to lift
  return without materially worsening drawdown
- pushing target volatility higher helped CAGR too, but with a noticeably worse
  drawdown trade-off

Consequence:
- the aggressive baseline is now slightly more concentrated in strong leaders
- future CAGR work should first look for new opportunity sets before taking a
  much larger risk step
