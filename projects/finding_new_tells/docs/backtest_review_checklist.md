# Backtest Review Checklist

Use this before trusting any train/validation result.

## Causality

- Signals at time `t` use only data available at or before `t`.
- Positions are shifted so a close `t` decision fills at open `t+1`.
- Rolling statistics do not use centered windows.
- Thresholds and selected metrics were not tuned on validation or test.

## Costs And Execution

- Position changes include commission and slippage assumptions.
- Turnover is reported.
- The assumed fill price is explicit.
- Results are checked for sensitivity to higher costs.

## Risk

- Report CAGR, Sharpe, Sortino, Calmar, max drawdown, drawdown duration, exposure, turnover, and trade count.
- Inspect the equity curve, not just summary metrics.
- Check whether one market period explains most of the return.
- Compare against QQQ and TQQQ buy-and-hold.

## Statistical Evidence

- Forward-return distributions are inspected by vote bucket.
- Train and validation signal directions agree.
- Small samples are treated as weak evidence.
- Multiple testing is acknowledged when many metrics or thresholds were tried.

## Decision

- Keep only changes with a plausible mechanism and validation support.
- Prefer simpler rules when performance is similar.
- Do not inspect test until the strategy is final.

