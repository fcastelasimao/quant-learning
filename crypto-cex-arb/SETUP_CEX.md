# Crypto CEX Arbitrage Bot - Setup Guide

> Note: the runnable paper-trading path in this folder now targets **Binance + Kraken + Bitstamp on USD pairs**.

## Overview

This is a cross-exchange arbitrage bot designed for **Binance + Kraken** on **BTC, ETH, and other pairs**. It operates entirely legally in the UK and performs paper trading (simulated execution) before risking real money.

**Status**: Ready for paper trading setup. You'll set up accounts in the next step.

---

## What Changed

Your original Betfair/Smarkets sports betting arbitrage bot has been completely adapted for crypto CEX arbitrage:

| Aspect | Before | Now |
|--------|--------|-----|
| **Exchanges** | Betfair + Smarkets | Binance + Kraken + Bitstamp |
| **Commissions** | 2-5% | 0.1-0.25% |
| **Trading pairs** | Soccer, Tennis, etc. | BTC/USDT, ETH/USDT, etc. |
| **Scan interval** | 5 seconds | 1 second |
| **Settlement** | Days (sports event) | Seconds (blockchain) |
| **Simulation** | Synthetic (8% lag) | Real prices (live API) |

---

## Step 1: Create Exchange Accounts

### **Binance Setup (5 minutes)**

1. Go to [binance.com](https://binance.com)
2. Click **Register**
3. Enter email, password, complete CAPTCHA
4. Verify email
5. Complete optional KYC (recommended for higher limits)
6. Go to **Settings** → **API Management**
7. Click **Create API Key** (No phone verification required for read-only)
8. Choose **API restrictions**:
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Disable Withdrawals (safety first)
   - ❌ Disable IP Whitelist (for now, set later when deploying)
9. **Save** your:
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET` (NEVER share this)

### **Kraken Setup (5 minutes)**

1. Go to [kraken.com](https://kraken.com)
2. Click **Create Account**
3. Enter email, password, complete verification
4. Verify email + phone number (required by Kraken)
5. Go to **Settings** → **API**
6. Click **Generate New Key**
7. **Name**: "Arb Bot"
8. **Nonce window**: 0 (default)
9. **Permissions**:
   - ✅ Query Funds
   - ✅ Query Open Orders & Trades
   - ✅ Query Closed Orders & Trades
   - ✅ Access Create & Modify Orders
   - ❌ Access Cancel/Close Orders (disable for safety)
10. **Save** your:
    - `KRAKEN_API_KEY`
    - `KRAKEN_API_SECRET` (NEVER share this)

---

## Step 2: Store Your API Keys

**Option A: Environment Variables (Recommended for testing)**

```bash
export BINANCE_API_KEY="your_actual_key_here"
export BINANCE_API_SECRET="your_actual_secret_here"
export KRAKEN_API_KEY="your_actual_key_here"
export KRAKEN_API_SECRET="your_actual_secret_here"
```

Add these to your shell profile (`~/.zshrc` or `~/.bash_profile`) so they persist:

```bash
# ~/.zshrc
export BINANCE_API_KEY="your_actual_key_here"
export BINANCE_API_SECRET="your_actual_secret_here"
export KRAKEN_API_KEY="your_actual_key_here"
export KRAKEN_API_SECRET="your_actual_secret_here"
```

Then reload:
```bash
source ~/.zshrc
```

**Option B: `.env` file (Requires `python-dotenv` package)**

Create a `.env` file in the Betfair folder:

```
BINANCE_API_KEY=your_actual_key_here
BINANCE_API_SECRET=your_actual_secret_here
KRAKEN_API_KEY=your_actual_key_here
KRAKEN_API_SECRET=your_actual_secret_here
```

Then install dotenv:
```bash
pip install python-dotenv
```

And add to the top of any script that imports config:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Step 3: Install Required Packages

```bash
cd /Users/franciscosimao/Documents/quant-learning/Betfair

# Core dependencies
pip install ccxt requests flask

# Optional but recommended
pip install python-dotenv pandas
```

**What each does:**
- `ccxt`: Unified crypto exchange API library (handles Binance + Kraken)
- `requests`: HTTP client for fallback API calls
- `flask`: For the optional web dashboard
- `python-dotenv`: For `.env` file configuration

---

## Step 4: Run Paper Trading Mode

Once you have your API keys set up:

```bash
cd /Users/franciscosimao/Documents/quant-learning/Betfair

# Test with real market data (NOT simulated)
python simulate_cex.py
```

**What happens:**
- ✅ Connects to Binance API (real data)
- ✅ Connects to Kraken API (real data)
- ✅ Connects to Bitstamp API (real data)
- ✅ Fetches live BTC/ETH/etc prices from all three exchanges
- ✅ Detects arbitrage opportunities
- ✅ **Simulates trades** (logs them, no real money at risk)
- ✅ Prints live dashboard every 5 scans
- ✅ Logs all results to `data/cex_trades.jsonl`

**Example output:**
```
══════════════════════════════════════════════════════════════════════════════════════
  CRYPTO CEX ARBITRAGE SCANNER — PAPER TRADING MODE
══════════════════════════════════════════════════════════════════════════════════════
  Time: 2026-03-31 14:35:22 UTC
  Scan #147 | Trading pairs: 6
────────────────────────────────────────────────────────────────────────────────────────
  PORTFOLIO
  Bankroll: $1,024.50  (start: $1,000.00)
  P&L: +$24.50  (+2.5%)
  Trades: 8  (W:7 / L:1)  Win rate: 87%
  Open: 0  Peak: $1,024.50  Max DD: 2.3%
────────────────────────────────────────────────────────────────────────────────────────
  LIVE PRICES
  Pair        Binance Bid   Kraken Ask         Spread%  Direction
  BTC/USDT      67210.50     67225.00         0.02%   → Kraken
  ETH/USDT       3520.40      3518.50        -0.05%   ← Binance
  ...
```

---

## Step 5: Review Paper Trading Results

After running for at least 1 week with 200+ simulated trades:

```bash
python analyse.py
```

This will show you:
- Total P&L
- Win rate
- Average profit per trade
- Performance by pair
- Equity curve stats

**Target metrics before going live:**
- ✅ Win rate > 60%
- ✅ Consistent profitability (not just lucky streaks)
- ✅ Drawdown < 15%
- ✅ At least 200 trades

---

## Step 6: Optional - Live Trading (Much Later)

Only after 1+ weeks of successful paper trading and if you feel confident:

```bash
python main.py --mode live --position-size 10
```

**With safeguards:**
- 🛑 Max $10 per trade (small stakes)
- 🛑 Daily loss limit: -20% (auto-stop for the day)
- 🛑 Max drawdown: -40% (hard emergency halt)
- 🛑 Telegram alerts (optional setup)

---

## Configuration Reference

Edit `config.py` to customize behavior:

```python
min_edge_pct: 0.3         # Profit threshold (0.3% = need to earn 0.3% after commissions)
scan_interval_seconds: 1.0    # How often to check prices
max_stake_per_trade: 20.0     # Max USD per trade
max_position_pct: 0.08        # Max % of bankroll per trade
daily_loss_limit_pct: 0.20    # Stop trading if down 20% today
max_drawdown_pct: 0.40        # Hard stop at 40% cumulative drawdown

trading_pairs: [
    "BTC/USD",      # Bitcoin
    "ETH/USD",      # Ethereum
    "XRP/USD",      # Ripple
    "LTC/USD",      # Litecoin
    "LINK/USD",     # Chainlink
    "BCH/USD",      # Bitcoin Cash
    "ADA/USD",      # Cardano
    "SOL/USD",      # Solana
    "DOGE/USD",     # Dogecoin
    "DOT/USD",      # Polkadot
    "MATIC/USD",    # Polygon
]
```

---

## Files You Now Have

**Core Files:**
- `config.py` - Central configuration (commissions, pairs, risk limits)
- `binance_client.py` - Binance API client
- `kraken_client.py` - Kraken API client
- `models.py` - Data classes (PriceSnapshot, ArbOpportunity, PaperTrade)
- `arb_engine.py` - Arbitrage detection logic
- `paper_trader.py` - Paper trading engine
- `simulate_cex.py` - **Main script for paper trading with live prices**
- `main.py` - (To be updated for live trading)
- `cooldown.py` - Prevents duplicate trades on same pair
- `analyse.py` - Post-session analysis

**Supporting:**
- `dashboard.py` - Optional web dashboard (Flask)
- `data/` - Logs JSON files with all trades/snapshots

---

## Troubleshooting

### "Connection refused" / "API key not found"

```bash
# Check environment variables are set
echo $BINANCE_API_KEY
echo $BINANCE_API_SECRET

# If empty, set them:
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### "CCXT not installed"

```bash
pip install ccxt
```

### No arbitrage opportunities found

**This is normal!** Real arbitrage on major pairs (BTC, ETH) is rare and tiny:

- Binance BTC: $67,210.00 bid / $67,211.00 ask
- Kraken BTC: $67,209.50 bid / $67,212.00 ask
- Spread after commissions: -0.04% (no arb exists)

The strategy finds opportunities when:
1. Price divergence > 0.5% (after commissions)
2. Both exchanges have sufficient liquidity
3. Timing window is open (usually after major news/price moves)

**In practice:** You might see 2-10 tradeable arbs per day, not per hour.

### "ModuleNotFoundError: No module named 'ccxt'"

```bash
pip install ccxt
```

### Paper trades show extreme profits

If you see $100+ profits on small positions:
- Check that `max_edge_pct` hasn't been set too low
- Check that `min_liquidity_usd` filtering is working
- Verify the spreads are realistic (not synthetic)

---

## Next: Tell Me Your API Keys

When you have your Binance and Kraken accounts set up, provide:

**DO NOT paste them here or in any messages** — handle them as secrets. Instead:

1. Set them as environment variables on your machine
2. Test the connection by running:
   ```bash
   python -c "from binance_client import BinanceClient; BinanceClient().test_connection()"
   ```
3. Let me know if it works or show me any error message

---

## Legal & Tax Notes

✅ **Legal**: Personal crypto trading on Binance/Kraken is fully legal in UK
✅ **Tax**: Capital gains on profits are taxable (report to HMRC via self-assessment)
✅ **Threshold**: First £1,000 profit per tax year is tax-free (trading allowance)

If you make £5,000 profit:
- Tax-free: £1,000
- Taxable: £4,000
- Tax owed: ~£800 (at 20% CGT rate)

---

## Questions?

Let me know if you run into issues or want to adjust the strategy parameters once you see live data.
