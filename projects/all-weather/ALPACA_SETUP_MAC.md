# Alpaca Monthly Rebalancer — Mac Setup Guide

This guide walks you through setting up and running `alpaca_monthly_rebalance.py` on macOS.

---

## Step 1: Get Your Alpaca Paper Trading Credentials

### 1.1 Log into Alpaca Dashboard

1. Go to [https://app.alpaca.markets](https://app.alpaca.markets)
2. Log in with your account credentials
3. You should see your paper trading account dashboard (not live trading)

### 1.2 Find Your API Keys

**For Paper Trading:**

1. Click on your account icon (top-right)
2. Select **"Paper API"** or **"API Keys"**
3. You'll see two keys:
   - **API Key ID** — starts with `PK...`
   - **Secret Key** — a longer string (keep it secret!)

**Copy both values.** You'll need them in Step 2.

### 1.3 Verify Your Paper Account Has Cash

- Your paper account starts with $100,000 USD in cash
- You can see this on the dashboard under "Account Equity"
- If it's depleted, you can reset it in account settings

---

## Step 2: Set Environment Variables (One-Time Setup)

Environment variables tell the script where to find your Alpaca credentials.

### Option A: Permanent Setup (Recommended)

Add these lines to your shell profile so they persist across terminal sessions:

**For zsh (default in modern macOS):**

```bash
nano ~/.zshrc
```

You'll see conda's initialization block at the top. **Scroll to the very END of the file** (press **Ctrl+X**), then add:

```bash
# Alpaca API credentials
export APCA_API_KEY_ID="your_api_key_id_here"
export APCA_API_SECRET_KEY="your_secret_key_here"
```

Replace:
- `your_api_key_id_here` with your actual **API Key ID** (e.g., `PKxyz123...`)
- `your_secret_key_here` with your actual **Secret Key**

**Save and exit nano:**
- Press **Ctrl+O** (write/save)
- Press **Enter** (confirm)
- Press **Ctrl+X** (exit)

**Reload the profile:**

```bash
source ~/.zshrc
```

**Verify it worked:**

```bash
echo $APCA_API_KEY_ID
```

You should see your API key ID printed. If you see nothing, the setup didn't work—check that you added the export lines at the END of the file.

**For bash (if using older macOS):**

```bash
nano ~/.bash_profile
```

Same steps as above, but save to `~/.bash_profile` instead.

### Option B: Session-Only Setup (Temporary)

If you only want the credentials for this one terminal session:

```bash
export APCA_API_KEY_ID="your_api_key_id_here"
export APCA_API_SECRET_KEY="your_secret_key_here"
```

These will disappear when you close the terminal.

---

## Step 3: Run the Script in Preview Mode

Preview mode shows what *would* trade without actually submitting any orders.

```bash
cd ~/Documents/quant-learning/All_weather_portfolio
conda activate allweather
python3 alpaca_monthly_rebalance.py
```

### Expected Output

You'll see something like:

```
Logging session started. Full log: logs/2026-03-31_14-23-15_alpaca_rebalance.log
========================================================================
ALPACA MONTHLY PAPER REBALANCER
========================================================================
Now (ET):             2026-03-31 14:23:15 ET
Strategy:             6asset_tip_gsg_rp
Paper account status: market open
Last trading day:     2026-03-31
Today is month-end:   yes
Execution mode:       PREVIEW ONLY
Equity:               $100,000.00
Cash:                 $100,000.00
Buying power:         $100,000.00

Rebalance plan:
  Symbol  Action  Target %  Current %  Target $  Current $   Delta $  Qty  Notional $  Reason
     SPY    BUY0.13       0.0        13000.0           0.0    13000.0  NaN      13000.0      
     QQQ    BUY0.11       0.0        11000.0           0.0    11000.0  NaN      11000.0      
     TLT    BUY0.19       0.0        19000.0           0.0    19000.0  NaN      19000.0      
     TIP    BUY0.33       0.0        33000.0           0.0    33000.0  NaN      33000.0      
     GLD    BUY0.14       0.0        14000.0           0.0    14000.0  NaN      14000.0      
     GSG    BUY0.10       0.0        10000.0           0.0    10000.0  NaN      10000.0      

Preview complete. Re-run with --execute to submit paper orders.
```

**This tells you:**
- Your account has $100k equity
- You have no positions yet (all current % = 0)
- The script wants to buy 6 assets to reach target weights
- No actual orders were submitted (preview mode)

### Check the Log File

The script saved a detailed log in `logs/`:

```bash
tail logs/2026-03-31_14-23-15_alpaca_rebalance.log
```

This log contains debug-level information (more detail than console) for troubleshooting.

---

## Step 4: Execute the Rebalance (On Last Trading Day of Month)

When you're ready to actually trade, add the `--execute` flag:

```bash
python3 alpaca_monthly_rebalance.py --execute
```

### Safety Guards

The script will **refuse to execute** if:

1. **Today is NOT the last trading day of the month**
   - Override with: `--force`
   - Example: `python3 alpaca_monthly_rebalance.py --execute --force`

2. **The market is closed**
   - No override possible (intentional safety measure)
   - Run during market hours only

3. **There are unresolved warnings**
   - Fix the issues first, or add `--liquidate-other-positions` if you have non-strategy positions

### What Happens When You Execute

1. **Sells first** — liquidate positions that don't match target weights
2. **Waits for fills** — polls Alpaca until orders complete (or timeout after 60 seconds)
3. **Refreshes account** — recalculates available cash
4. **Buys next** — purchases underweight positions with fresh cash
5. **Waits for fills** — again polls until orders complete
6. **Logs everything** — writes full execution details to log file

Example output:

```
Submitting sell orders first...
  SELL SPY    qty=50.0
Waiting for 1 sell orders to fill (timeout: 60s)...
  Order a1b2c3d4-e5f6-7890-abcd-ef1234567890 -> filled

Account refreshed: equity=$102,341.52, cash=$50,234.89

Submitting buy orders after refresh...
  BUY  SPY    notional=$13,000.00
  BUY  TLT    notional=$19,000.00
Waiting for 2 buy orders to fill (timeout: 60s)...
  Order xyz789...  -> filled
  Order abc456...  -> filled

Execution complete. ✓
```

---

## Step 5: Monitor the Results

### Check Your Account in Alpaca Dashboard

1. Go to [https://app.alpaca.markets](https://app.alpaca.markets)
2. Look at **Holdings** tab
3. You should see your 6 assets with the target weights:
   - SPY: ~13%
   - QQQ: ~11%
   - TLT: ~19%
   - TIP: ~33%
   - GLD: ~14%
   - GSG: ~10%

### Review Performance Tracking

The script automatically records performance metrics to `logs/performance_tracking.csv`:

```bash
cat logs/performance_tracking.csv
```

This CSV contains:
- **Date**: Date of the snapshot
- **Portfolio_Equity**: Your account value
- **SPY_Price, ALLW_Price, TLT_Price**: Benchmark prices
- **{Asset}_Weight%**: Current allocation percentage
- **{Asset}_Drift%**: How far from target allocation (shows rebalancing need)
- **Portfolio_Return%**: Your portfolio's monthly return
- **SPY_Return%, ALLW_Return%, 60_40_Return%**: Benchmark returns for comparison

Example row:
```
2026-04-30,$101950.00,$460.75,$498.10,$95.30,15.2,+2.2,10.5,-0.5,17.1,-1.9,32.8,-0.2,14.1,+0.1,10.3,+0.3,1.95,7.05,2.65,4.50
```

This tells you:
- Portfolio: $101,950 (+1.95% this month)
- SPY: 15.2% actual (+2.2% from 13% target) — needs rebalancing next month
- TLT: 17.1% actual (-1.9% from 19% target) — lagging bonds
- Your return: 1.95% vs SPY 7.05% vs ALLW 2.65% vs 60/40 4.50%

You can import this CSV into Excel/Google Sheets to track performance over time.

### Review Your Log

```bash
cat logs/2026-03-31_14-23-15_alpaca_rebalance.log
```

This contains the full execution trail:
- Timestamps for each operation
- Order IDs and statuses
- Error messages (if any)
- Account state before/after

---

## Common Scenarios

### Scenario 1: You Want to Preview on a Non-Last Trading Day

```bash
python3 alpaca_monthly_rebalance.py
```

Works fine. Shows what *would* trade if it were the last trading day.

### Scenario 2: It's the Last Trading Day but Market is Closed

```bash
python3 alpaca_monthly_rebalance.py --execute
```

**Error:** "Refusing to submit market orders while the regular session is closed."

**Solution:** Wait until 9:30 AM – 4:00 PM ET (normal market hours).

### Scenario 3: You Have Non-Strategy Positions (e.g., old trades)

```bash
python3 alpaca_monthly_rebalance.py --execute
```

**Error:** "Refusing to execute with unresolved warnings. Review the preview output first."

**Solution (Option A):** First run preview and manually liquidate the extra position in Alpaca, then re-run.

**Solution (Option B):** Let the script liquidate them automatically:

```bash
python3 alpaca_monthly_rebalance.py --execute --liquidate-other-positions
```

### Scenario 4: You Want to Test on a Non-Month-End Day

```bash
python3 alpaca_monthly_rebalance.py --execute --force
```

(Only works if market is open. The market-close guard remains.)

---

## Checking Logs

All execution logs are saved to `logs/` with timestamps:

```bash
ls logs/
```

Shows all past rebalance sessions.

**View the most recent log:**

```bash
ls -t logs/ | head -1 | xargs -I {} cat logs/{}
```

**Follow a log in real-time (if script is running):**

```bash
tail -f logs/2026-03-31_14-23-15_alpaca_rebalance.log
```

---

## Troubleshooting

### Error: "Missing Alpaca credentials"

**Cause:** Environment variables not set.

**Fix:** Check Step 2. Run:

```bash
echo $APCA_API_KEY_ID
```

If blank, your environment variables aren't loaded. Try:

```bash
export APCA_API_KEY_ID="your_key"
export APCA_API_SECRET_KEY="your_secret"
```

### Error: "Strategy 'xyz' not found"

**Cause:** Strategy ID doesn't exist in strategies.json.

**Fix:** Check available strategies:

```bash
grep '"allocation"' strategies.json | head -5
```

Or just list all strategy IDs:

```bash
python3 -c "import json; print(list(json.load(open('strategies.json'))['strategies'].keys()))"
```

Use one of those strategy IDs:

```bash
python3 alpaca_monthly_rebalance.py --strategy-id 6asset_tip_gsg_rp --execute
```

### Orders Stuck in Pending State

**Cause:** Market volatility, connection issues, or timeout too short.

**Check the log:**

```bash
grep "WARNING" logs/newest_log.log
```

**Solution:** Check Alpaca dashboard manually. Orders might fill later. If stuck too long, you can cancel them in the Alpaca UI.

### Rebalance Plan is Empty

**Cause:** All positions already match target weights within drift threshold.

**Normal behavior.** The script correctly identified that no rebalancing is needed.

---

## Next Steps

### Monthly Rebalancing Workflow (Manual)

1. First trading day of next month, run:
   ```bash
   python3 alpaca_monthly_rebalance.py
   ```
   Review the preview.

2. Last trading day of the month, run:
   ```bash
   python3 alpaca_monthly_rebalance.py --execute
   ```

3. Check your holdings in Alpaca dashboard. They should be back to target weights.

4. Review the log for any issues.

### To Automate (Later)

See the main README for cron scheduling or cloud function setup.

---

## Support

If the script fails:

1. **Check the log file** — most errors are logged there first
2. **Review the preview output** — does the plan look right?
3. **Verify credentials** — is `echo $APCA_API_KEY_ID` returning your key?
4. **Check market hours** — is the market currently open?

Good luck! 🚀
