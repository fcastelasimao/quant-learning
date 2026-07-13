# All-Weather — Server Deployment Guide

Deploy the rebalancer on an AWS EC2 instance so it runs automatically every
31+ days without requiring your laptop to be on.

---

## What gets deployed

| Component | What it does |
|---|---|
| `live/rebalance.py` | The broker-agnostic rebalancer. Runs weekdays at 09:35 ET. Default: `--dry-execute`. Switch to `--execute` when ready. |
| `live/daily_snapshot.py` | Price + portfolio-drift logger. Runs weekdays at 16:10 ET after market close. No orders. Builds `live/logs/daily_snapshots.csv`. |
| `live/logs/` | All run logs, cadence state, lot ledger, budget state, snapshots. Lives on the server. |

## Excluded from server deployment (not needed for live execution)

- `quantcore` (SQLite historical data store — research/backtest only)
- `scipy`, `matplotlib`, `marimo`, `openpyxl` (research tools)
- `pytest` (test suite)

---

## EC2 instance recommendation

| Attribute | Recommendation |
|---|---|
| Instance type | **t4g.nano** ($3/month) or **t3.micro** (free-tier eligible) |
| OS | Amazon Linux 2023 or Ubuntu 22.04 LTS |
| Storage | 8 GB gp3 (default) — logs are tiny |
| Security group | SSH (port 22) from your IP only. No inbound HTTP needed. |
| Key pair | Create or reuse an existing EC2 key pair |

The rebalancer makes outbound HTTPS calls to Alpaca and Yahoo Finance. No
inbound ports other than SSH are needed.

---

## Step-by-step setup

### 1. Launch the EC2 instance

In the AWS Console or via CLI:

```bash
# Example: t3.micro, Amazon Linux 2023, us-east-1
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \    # Amazon Linux 2023, us-east-1 — check current AMI
  --instance-type t3.micro \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=allweather-rebalancer}]'
```

Or just click through the console. Nothing special needed.

### 2. SSH in

```bash
ssh -i ~/.ssh/your-key.pem ec2-user@<instance-public-ip>
# Ubuntu: ssh -i ~/.ssh/your-key.pem ubuntu@<instance-public-ip>
```

### 3. Clone the repo and run setup

```bash
# Install git if not present (Amazon Linux 2023)
sudo dnf install -y git    # or: sudo apt-get install -y git

# Clone the repo (replace with your actual repo URL)
git clone git@github.com:youruser/all-weather.git ~/all-weather

# Run the setup script
bash ~/all-weather/server/setup_ec2.sh \
  --repo-dir ~/all-weather \
  --broker alpaca \
  --account default \
  --mode paper
```

What this does:
- Installs Python 3.12 if needed
- Creates `.venv` with only the server dependencies
- Creates `strategies.json` from the example template
- Creates `~/api_keys.env` skeleton (chmod 600)
- Installs two cron jobs (rebalancer + daily snapshot)

### 4. Add your API credentials

```bash
nano ~/api_keys.env
```

Fill in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper or live credentials).

For a named account (e.g. `--account retirement`):
```bash
BROKER_ALPACA_RETIREMENT_KEY="PKXXXXXXXXXXXXXXXX"
BROKER_ALPACA_RETIREMENT_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 5. Verify the connection

```bash
cd ~/all-weather
.venv/bin/python -m live.healthcheck --broker alpaca
```

### 6. Preview a rebalance

```bash
.venv/bin/python -m live.rebalance --paper --broker alpaca --account default
```

Review the output: positions, target weights, proposed trades.

### 7. Run a dry-execute to test logging

```bash
.venv/bin/python -m live.rebalance --paper --broker alpaca --account default --dry-execute
```

Check `live/logs/run_summary.jsonl` to confirm the run was logged.

### 8. Enable real order placement (when ready)

Re-run setup with `--auto-execute`. The cron schedule will be updated:

```bash
bash ~/all-weather/server/setup_ec2.sh \
  --broker alpaca --account default --mode paper --auto-execute
```

The **31-day cadence gate** prevents double-execution — even if cron fires
every weekday, real orders only happen when 31+ days have passed since the
last executed run.

---

## Updating the server

Pull the latest code and the venv will use the same requirements:

```bash
cd ~/all-weather
git pull --ff-only
.venv/bin/pip install -r server/requirements.txt --quiet
```

No restart needed — cron picks up the new code on the next scheduled run.

---

## Monitoring

### Check cron is installed

```bash
crontab -l | grep allweather
```

### View the latest rebalancer run

```bash
tail -50 ~/all-weather/live/logs/cron_rebalance.log
```

### View structured run summaries

```bash
# All runs
cat ~/all-weather/live/logs/run_summary.jsonl | python3 -m json.tool | grep -E '"outcome"|"equity_before"|"n_buy"|"n_sell"'

# Latest run
tail -1 ~/all-weather/live/logs/run_summary.jsonl | python3 -m json.tool
```

### View daily snapshots (price + drift)

```bash
tail -10 ~/all-weather/live/logs/daily_snapshots.csv
```

### Check cadence state (when was the last real execute?)

```bash
cat ~/all-weather/live/logs/cadence_alpaca_paper_default_*.json
```

---

## Secrets management

The simplest approach is `~/api_keys.env` (chmod 600, never committed).

For stronger isolation, use **AWS Systems Manager Parameter Store**:

```bash
# Store a secret (do this once from your laptop, AWS CLI needed)
aws ssm put-parameter \
  --name "/allweather/alpaca/api_key" \
  --value "PKXXXXXXXXXXXXXXXX" \
  --type SecureString

# Retrieve in a wrapper script instead of using api_keys.env
export ALPACA_API_KEY=$(aws ssm get-parameter --name "/allweather/alpaca/api_key" \
  --with-decryption --query Parameter.Value --output text)
```

This requires an IAM role with `ssm:GetParameter` attached to the EC2 instance.
Simpler than Secrets Manager for a single-user setup.

---

## Switching from macOS launchd to EC2 cron

If you were previously using `install_launchd.sh` on your Mac:

1. The Mac schedule can stay running if your Mac is always on
2. EC2 cron uses the same `live/rebalance.py` script and the same cadence gate
3. The two deployments should NOT point at the same Alpaca account simultaneously
   — the cadence state files live locally, so they can't coordinate

Pick one and disable the other.

---

## The cron schedule explained

| Time (UTC) | ET equivalent | What runs |
|---|---|---|
| 14:35 Mon–Fri | 09:35 ET | `live/rebalance.py --dry-execute` (or `--execute` if auto) |
| 21:10 Mon–Fri | 16:10 ET | `live/daily_snapshot.py --no-broker` (price + drift log) |

- **14:35 UTC** = 5 minutes after US market open. Prices are fresh; orders fill immediately.
- **21:10 UTC** = 10 minutes after market close. Captures official closing prices.
- The cadence gate (31-day minimum) means the rebalancer fires at most once per month.

---

## File locations on the server

```
~/all-weather/
├── strategies.json             Production strategy (not in git — copy from .example)
├── live/logs/
│   ├── cron_rebalance.log      stdout/stderr from the cron rebalancer
│   ├── cron_snapshot.log       stdout/stderr from the snapshot job
│   ├── run_summary.jsonl       Structured JSON log (one line per run)
│   ├── monthly_runs.csv        Aggregated monthly view
│   ├── runs/                   Per-run JSON detail (auto-pruned at 200 files)
│   ├── daily_snapshots.csv     Daily price + drift data
│   ├── cadence_*.json          Cadence gate state
│   ├── lots_*.json             Lot ledger (holding-period enforcement)
│   └── budget_*.json           Budget state (if --budget is used)
└── server/
    ├── README.md               This file
    ├── requirements.txt        Minimal server dependencies
    ├── setup_ec2.sh            Bootstrap script
    └── strategies.json.example Template allocation file
~/api_keys.env                  Credentials (outside the repo, chmod 600)
```
