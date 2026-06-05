# All Weather Rebalancer — Quick Start

Get a fully automated ETF rebalancer running on an AWS EC2 instance.
The system checks your portfolio daily, rebalances when drift exceeds
thresholds, and logs every action.

---

## What the rebalancer does

1. Connects to your Alpaca brokerage account
2. Reads target allocation from `strategies.json`
3. Compares current portfolio weights to targets
4. If drift exceeds the threshold, places market orders to rebalance
5. Enforces a 31-day minimum holding period per lot (FIFO)
6. Logs everything to `live/logs/`

No historical data is needed. No research code runs. The rebalancer
only uses live prices from the broker API.

---

## 1. Launch an EC2 instance

- **AMI:** Amazon Linux 2023 or Ubuntu 22.04
- **Instance type:** `t4g.nano` (ARM, ~$3/month) or `t3.micro` (x86, free tier)
- **Storage:** 8 GB default is fine
- **Security group:** outbound HTTPS only (the rebalancer calls broker APIs)

SSH into the instance.

## 2. Install Python 3.12

```bash
# Amazon Linux 2023
sudo dnf install python3.12 python3.12-pip git -y

# Ubuntu 22.04
sudo apt update && sudo apt install python3.12 python3.12-venv git -y
```

## 3. Clone and set up

```bash
git clone <your-repo-url> allweather
cd allweather

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Configure credentials

Copy the example and fill in your Alpaca keys:

```bash
cp api_keys.env.example api_keys.env
nano api_keys.env
```

At minimum, set these two lines:

```
ALPACA_API_KEY="PKXXXXXXXXXXXXXXXX"
ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

For a named account (e.g. `--account boss`), use:

```
BROKER_ALPACA_BOSS_KEY="PKXXXXXXXXXXXXXXXX"
BROKER_ALPACA_BOSS_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

The rebalancer auto-loads `api_keys.env` from the project root.

## 5. Verify the connection

```bash
source .venv/bin/activate
python -m live.healthcheck --broker alpaca --paper
```

All checks should show a green checkmark.

## 6. Preview a rebalance (no orders placed)

```bash
python -m live.rebalance --paper --broker alpaca
```

This shows what the rebalancer *would* do without placing any orders.
Review the output to confirm the allocation and trade sizes look right.

## 7. Dry-execute (simulates fills, writes logs, no real orders)

```bash
python -m live.rebalance --paper --broker alpaca --dry-execute
```

Check the output in `live/logs/runs/`.

## 8. First real execution

When ready to place real orders:

```bash
# Initialize the lot ledger (first time only)
python -m live.rebalance --live --broker alpaca --account boss --initialize-lots

# Execute
python -m live.rebalance --live --broker alpaca --account boss --execute
```

After execution, verify:
- `live/logs/runs/<latest>.json` — full run detail
- Your Alpaca dashboard — positions match the strategy

## 9. Set up automated daily checks

Install the cron template:

```bash
bash live/scheduler/install_cron.sh \
    --broker alpaca \
    --account boss \
    --mode live
```

This schedules:
- **Daily dry-execute** at 09:30 ET on weekdays (logs what it *would* trade)
- The cadence gate (31-day minimum) prevents actual over-trading

To place real orders, always run manually:

```bash
python -m live.rebalance --live --broker alpaca --account boss --execute
```

Or to auto-execute (after you trust the system):

```bash
bash live/scheduler/install_cron.sh \
    --broker alpaca \
    --account boss \
    --mode live \
    --auto-execute
```

## 10. Monitor

```bash
# Recent run summaries
tail -5 live/logs/run_summary.jsonl | python -m json.tool

# Monthly overview
cat live/logs/monthly_runs.csv

# Last run detail
ls -t live/logs/runs/*.json | head -1 | xargs cat | python -m json.tool
```

---

## Strategy configuration

The strategy is defined in `strategies.json`. The production strategy
`6asset_tip_gsg_rpavg` allocates across 6 ETFs:

| ETF | Backtest | Live | Weight |
|-----|----------|------|--------|
| SPY | SPY | SPY | 13.4% |
| QQQ | QQQ | QQQ | 10.3% |
| TLT | TLT | TLT | 17.5% |
| TIP | TIP | TIP | 34.8% |
| GLD | GLD | GLDM | 14.2% |
| GSG | GSG | GSG | 9.8% |

The only live ticker substitution is GLD to GLDM (same gold exposure,
lower expense ratio).

---

## CLI reference

```
python -m live.rebalance [OPTIONS]

Mode:
  --paper               Paper trading (default)
  --live                Live trading

Broker:
  --broker alpaca       Broker to use (alpaca or tastytrade)
  --account LABEL       Named account label

Execution:
  (no flag)             Preview only — shows plan, no orders
  --dry-execute         Simulate fills, write logs, no real orders
  --execute             Place real orders

Budget:
  --budget AMOUNT       Cap strategy to a fixed dollar amount
  --initialize-budget   Seed budget state (first run only)

Lots:
  --initialize-lots     Seed lot ledger from current positions (first run)

Cadence:
  --min-rebalance-interval-days N   Minimum days between executions (default: 31)
  --force-cadence       Bypass the interval check
```

---

## Folder structure (what matters for live trading)

```
allweather/
├── live/                     The rebalancer
│   ├── rebalance.py          Main entry point
│   ├── healthcheck.py        Pre-flight checks
│   ├── brokers/              Alpaca + Tastytrade adapters
│   ├── scheduler/            Cron/launchd templates
│   └── logs/                 All output (gitignored)
│       ├── run_summary.jsonl
│       ├── monthly_runs.csv
│       └── runs/*.json
├── engine/
│   └── config.py             Reads strategies.json
├── strategies.json           Target allocation
├── api_keys.env              Your credentials (gitignored)
└── requirements.txt          Python dependencies
```

Everything else in the repo (research/, notebooks/, etc.) is research
tooling. You never need to touch it.
