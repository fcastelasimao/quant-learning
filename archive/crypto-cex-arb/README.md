# Crypto Cross-CEX Arbitrage Scanner & Paper Trader (USD Edition)

A Python system that monitors **USD-denominated crypto pairs** across **Binance**, **Kraken**, and **Bitstamp** in real-time, detects pricing discrepancies, and executes paper trades to track simulated P&L.

**Why USD?** It opens larger liquidity pools, more exchange overlap, and lower fees on major crypto pairs.

## What It Does

1. **Fetches real-time prices** from Binance, Kraken, and Bitstamp APIs (USD pairs)
2. **Scans for arbitrage** where Buy-on-Exchange-A + Sell-on-Exchange-B creates profit after commissions
3. **Polls every 1 second** for fast opportunity detection
4. **Paper trades** opportunities with instant settlement, tracking bankroll, P&L, and drawdown
5. **Enforces risk limits**: max 10% per trade, 20% daily loss limit, 40% drawdown kill switch
6. **Logs everything** to JSONL files (`data/cex_trades.jsonl`, `data/snapshots_cex.jsonl`)
7. **Auto-analyzes** performance with `analyse.py` after trading session

## Architecture

```
config.py          — Strategy params, commissions, risk limits, USD trading pairs
models.py          — Data classes: Exchange, PriceSnapshot, ArbOpportunity, PaperTrade, PortfolioState
kraken_client.py   — Real-time CCXT client for Kraken (BTC/GBP, ETH/GBP, XRP/GBP)
bitstamp_client.py — Real-time CCXT client for Bitstamp (BTC/GBP, ETH/GBP, XRP/GBP)
arb_engine.py      — Arbitrage detection: buy/sell math with commission adjustments
paper_trader.py    — Paper trading engine: execution, settlement, portfolio tracking
cooldown.py        — Prevents duplicate trades on same pair within cooldown window
simulate_cex.py    — REST polling simulator (simpler, easier to debug)
simulate_cex_ws.py — WebSocket simulator (lower-overhead for long runs)
analyse.py         — Post-session trade analysis: P&L breakdown, win rate, edge distribution
```

## Quick Start (GBP Pairs Edition)

###

### 1. Setup Environment (GBP Strategy)

**Key difference: This system now trades GBP-denominated crypto pairs**, eliminating foreign exchange conversion costs (2-4% savings).

```bash
# Create Kraken account and get API keys:
# - Kraken has excellent GBP liquidity (BTC/GBP, ETH/GBP, etc.)
# - https://www.kraken.com/en-us/settings/api

# Set environment variables:
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export KRAKEN_API_KEY="your_key"
export KRAKEN_API_SECRET="your_secret"
export BITSTAMP_API_KEY="your_key"
export BITSTAMP_API_SECRET="your_secret"

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
# Requires: ccxt, requests, websockets, python-dotenv
```

### 3. Run Paper Trading (Real Live Prices)

```bash
python simulate_cex.py
```

For longer-running sessions with lower polling overhead:

```bash
python simulate_cex_ws.py
```

**What happens:**
- Connects to live Bitstamp and Kraken APIs (uses credentials from environment variables when present, public endpoints otherwise)
- Fetches real GBP order books every 1 second for BTC, ETH, and XRP
- Detects arbitrage opportunities where the **net edge** exceeds the configured threshold after commissions
- Paper trades with **£1,000 virtual bankroll** (adjustable in `config.py`)
- Instantly simulates settlement (crypto settles in seconds)
- Displays live dashboard with: current positions, P&L, win rate, max drawdown
- Logs all trades to `data/cex_trades.jsonl` and snapshots to `data/cex_snapshots.jsonl`

**Stop gracefully:**
- Press `Ctrl+C` at any time
- System prints final summary: total trades, win rate, P&L, drawdown
- All data already logged (no data loss on shutdown)

### 4. Analyse Results

```bash
python analyse.py
```

Generates summary statistics from the JSONL trade logs:
- Total P&L and daily P&L distribution  
- Win rate and average profit per trade
- Edge distribution (opportunities detected vs traded)
- By-pair and by-direction breakdown

## How the Arbitrage Maths Works

**Strategy**: Buy on exchange A at ask price, sell on exchange B at bid price, same pair, GBP-denominated

**Profit Per Unit (after commissions)**:
```
profit = (sell_bid_price × (1 - sell_commission)) - (buy_ask_price × (1 + buy_commission))
```

**Example**: BTC/GBP
- Kraken ask: £38,500 (buy here)  
- Bitstamp bid: £38,700 (sell here)
- Raw spread: £200
- Commissions: ~0.3% Kraken + ~0.3% Bitstamp = ~£232 total
- **Net profit: -£32** (loss at this spread)

The simulator now uses a **net edge threshold**. With 0.30% Bitstamp fees and 0.40% Kraken fees, raw spreads generally need to be materially wider than 0.20% before a trade is taken.

**Position Sizing**:
- Kelly Fraction: 0.25X Kelly (conservative, accounts for variance)
- Max Stake: 10% of bankroll per trade (max £100 with default £1,000)
- Min Quantity: must meet exchange minimums (prevent dust trades)

## Configuration & Risk Management (GBP Strategy)

Edit `config.py` to customize strategy:

```python
# Core strategy threshold
min_edge_pct = 0.20             # Net edge after commissions

# Position sizing & Kelly
kelly_fraction = 0.25           # Quarter-Kelly (conservative)
initial_bankroll = 1000.0       # Paper trading bankroll (£1,000 GBP equivalent)
max_position_pct = 0.08         # Never risk >8% per trade

# Risk limits
max_open_positions = 5          # Max 5 simultaneous trades
daily_loss_limit_pct = 0.2      # Stop for the day at -20% loss
max_drawdown_pct = 0.4          # Hard kill switch at -40% drawdown

# Execution
scan_interval_seconds = 1.0     # Poll exchanges every 1 second
trading_pairs = [
    "BTC/GBP",      # Best liquidity on Kraken
    "ETH/GBP",      # Excellent Kraken/Bitstamp spreads
    "XRP/GBP",
]

# Commission structure (taker-style defaults for conservative paper trading)
kraken_commission = 0.004
bitstamp_commission = 0.003
```

**Why use a net edge threshold?**
- It keeps the trigger aligned with fee-adjusted profitability
- It is easier to tune once you start measuring live slippage and rebalancing costs

## Paper Trading Mode (Default)

**Virtual Bankroll**: £1,000 GBP
- No real money moves
- All trades are simulated with instant settlement
- Trades in GBP-denominated pairs (BTC/GBP, ETH/GBP, etc.)
- Perfect for testing strategy without risk
- Data all logged for analysis

**Why GBP pairs?**
- ✅ **No FX conversion costs** (2-4% savings vs USD path)
- ✅ **Direct GBP → Crypto → GBP** (no unnecessary conversions)
- ✅ **Cleaner fee-adjusted edge calculation**
- ✅ **Better for UK-based traders** (settle back to GBP bank directly)

**Configured overlapping pairs:**
- BTC/GBP, ETH/GBP, XRP/GBP
- These are the pairs currently configured for both Kraken and Bitstamp

**To switch bankroll**, edit `initial_bankroll: 1000.0` in `config.py` (internal calculations still use float values, you convert mentally from GBP).

**To change traded pairs**, edit the `trading_pairs` list in `config.py`.

## Transitioning to Real Trading (GBP Strategy)

When you're confident in the system (after 1-2 weeks paper trading):

1. Review `analyse.py` output: win rate >50%, average edge >0.3%, no drawdown limit hits
2. Open real accounts:
   - **Kraken** (primary): Has excellent GBP pairs (BTC/GBP, ETH/GBP, etc.)
   - **Bitstamp** (secondary): Also has GBP pairs for cross-exchange arbitrage
   - Get API keys for both with Trading permission enabled
3. Deposit GBP from your bank to Kraken (instant, no forex fees)
4. Load real credentials into environment variables
5. Modify `execute_paper_trade()` in `paper_trader.py` to call real order placement
6. Start with **tiny stakes** (£20-£50 first trades)
7. Validate:
   - Order placement succeeds on both exchanges
   - GBP settlement works end-to-end
   - No unexpected conversion fees
8. Gradually increase stake size over 1-2 weeks

**Advantage of GBP strategy**: Eliminate 2-4% FX costs, so your 0.4% edge is meaningful profit, not fee-coverage.

## Important Notes (GBP Strategy)

- **Legal in the UK**: Cross-exchange crypto arbitrage on CEXs is legal; not subject to betting regulations
- **0.3% commissions on GBP pairs** (Kraken/Bitstamp tier 2-3) are tight but manageable with 0.4% edge
- **No FX costs**: Direct GBP → Crypto → GBP eliminates 2-4% conversion friction
- **Instant settlement**: Crypto trades settle in seconds
- **GBP liquidity**: Kraken's GBP pairs have excellent depth (BTC/GBP and ETH/GBP in particular)
- **API rate limits**: 1-second scan interval is safe for both Kraken and Bitstamp
- **Slippage**: Real execution fills may differ from snapshot prices by microseconds to seconds
- **Network latency**: Minimal in paper mode; becomes significant in live trading (consider clock sync)
- **Bank settlement**: Withdraw GBP profits directly to UK bank account (clean, no forex re-entry)
