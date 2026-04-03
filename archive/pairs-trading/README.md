# Crypto Pairs Trading

Statistical arbitrage strategy for cryptocurrency pairs using cointegration-based mean reversion. Runs fully on public Binance data — no API key required.

## Strategy Overview

Pairs trading exploits the statistical relationship between two assets that tend to move together over time. When one asset diverges from its historical relationship with the other, the strategy bets on reversion to the mean.

This is **not** risk-free arbitrage. It is a probabilistic bet that the historical relationship holds.

### How it works

1. **Cointegration test** — Engle-Granger two-step method checks if the log-price spread between two tokens is stationary (mean-reverting). A p-value below 0.10 and a half-life between 4–120 hours is required.
2. **Hedge ratio** — OLS regression fits `log(base) = α + β·log(quote)` on in-sample data. The coefficient β is the hedge ratio.
3. **Z-score signal** — The spread is normalised to a rolling z-score over 120 hours. Entries trigger at |z| > 1.6; exits at |z| < 0.3.
4. **Two-leg position** — Both legs are sized to be dollar-neutral using the hedge ratio, so directional market risk is minimised.

### Confirmed pairs (as of April 2026)

| Pair | P-value | Half-life | OOS P&L | Win rate | Sharpe |
|---|---|---|---|---|---|
| RENDER/FET | 0.024 | 24h | +11.62% | 69.6% | 3.36 |
| ONDO/POLYX | 0.035 | 25h | +14.04% | 60.0% | 4.49 |

Both pairs share the same investor narrative (AI compute and RWA tokenisation respectively), which drives the cointegration.

> These results are from an 83-day lookback with a 60/40 in-sample/out-of-sample split. Past performance does not guarantee future results. Always re-run the backtest before paper trading.

---

## Project Structure

```
pairs-trading/
├── config.py          # All strategy parameters — edit this to tune
├── models.py          # Data classes (Candle, Signal, Position, etc.)
├── data_fetcher.py    # Binance public OHLCV API + disk cache
├── cointegration.py   # Engle-Granger test, hedge ratio, half-life
├── spread_tracker.py  # Rolling z-score and signal classification
├── backtester.py      # Walk-forward simulation engine
├── paper_trader.py    # Live paper trading engine + portfolio tracker
├── run_backtest.py    # Entry point: screen pairs and run backtest
├── run_paper.py       # Entry point: live dashboard + paper trading loop
├── analyse.py         # Offline log analyser and performance report
├── data/              # Cached candle data (auto-populated, git-ignored)
└── logs/              # Trade logs and paper trader state (git-ignored)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the backtest

Screens all candidate pairs, tests cointegration, and simulates the strategy out-of-sample.

```bash
python3 run_backtest.py
```

Output includes a KEEP / WATCH / DROP verdict for each pair and a full trade-by-trade breakdown.

### 3. Start the paper trader

```bash
python3 run_paper.py
```

Refreshes a live dashboard every 5 minutes showing current z-scores, open positions, and running P&L. All trades are logged to `logs/paper_trades.jsonl`.

Press `Ctrl+C` to stop cleanly.

### 4. Review performance

```bash
python3 analyse.py                               # full report
python3 analyse.py --pair RENDERUSDT/FETUSDT     # single pair
python3 analyse.py --since 2026-05-01            # from a date
```

---

## Configuration

All parameters are in [config.py](config.py). Key settings:

| Parameter | Default | Description |
|---|---|---|
| `candidate_pairs` | RENDER/FET, ONDO/POLYX | Pairs to screen and trade |
| `lookback_candles` | 2000 | ~83 days of 1h data for fitting |
| `entry_z_score` | 1.6 | Open a position when \|z\| exceeds this |
| `exit_z_score` | 0.3 | Close when \|z\| falls back inside this |
| `stop_loss_z_score` | 3.0 | Hard stop — spread widened too far |
| `zscore_window` | 120 | Rolling window for z-score (hours) |
| `position_size_pct` | 0.80 | Fraction of bankroll per trade |
| `max_holding_hours` | 120 | Force-exit after 5 days |
| `commission` | 0.001 | 0.10% taker fee per leg (Binance) |
| `in_sample_fraction` | 0.60 | 60% of data used to fit hedge ratio |

### Adding new pairs

Edit `candidate_pairs` in `config.py` and re-run the backtest. The screener will tell you which pass:

```python
candidate_pairs: list = field(default_factory=lambda: [
    ("RENDERUSDT", "FETUSDT"),
    ("ONDOUSDT",   "POLYXUSDT"),
    ("NEWTOKEN1",  "NEWTOKEN2"),   # add candidates here
])
```

Good candidate pairs share: same sector, same investor base, similar launch dates, and no unique tokenomics (e.g. avoid mining/staking emission differences).

---

## Methodology Notes

### Why Engle-Granger and not Johansen?

Engle-Granger is simpler to implement and interpret for two-asset pairs. Johansen is preferred for portfolios of 3+ assets. For pairs trading, Engle-Granger is the standard.

### Why 1-hour candles and not 5-minute?

At 5-minute resolution, algorithmic market makers close spread divergences faster than a REST-polling strategy can react. At 1-hour resolution, you compete with human traders and slower systematic funds rather than co-located HFT.

### Why not CEX-DEX arbitrage?

CEX-DEX arbitrage on Uniswap v3 / Arbitrum was tested first. The combined commission drag (Binance 0.10% + Uniswap pool fee 0.05% + slippage) required a raw spread above 0.45% to be profitable. In practice the spread never exceeded 0.15% due to MEV bots rebalancing at block speed (~250ms). Pairs trading avoids this problem entirely by operating at hour-level timeframes.

### Risk factors

- **Regime shift** — the cointegration relationship can break permanently (e.g. one token gets a protocol-specific catalyst)
- **Thin liquidity** — ONDO and POLYX have lower liquidity than BTC/ETH; slippage in live trading will exceed the 0.10% commission assumption
- **Refit lag** — hedge ratio is refit every 24 hours; a fast regime shift can cause losses before the next refit

---

## Legal

This project is for educational and research purposes. It does not constitute financial advice. Crypto trading involves substantial risk of loss. All tokens listed are traded on Binance; verify availability and legality in your jurisdiction before live trading.
