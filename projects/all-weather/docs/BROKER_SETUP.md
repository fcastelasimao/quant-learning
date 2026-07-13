# Broker Setup Guide

This guide explains how to configure the All-Weather rebalancer for each
supported broker. The rebalancer (`live/rebalance.py`) is fully
broker-agnostic; the only broker-specific step is setting environment
variables for credentials.

---

## Secrets file

All API keys should live in:

```bash
/Users/franciscosimao/Documents/QuantFinance/api_keys.env
```

The live modules load this file automatically. Set `ALLW_API_KEYS_ENV` only if
you need to point a run at a different key file.

The file already supports placeholders for broker keys, Tastytrade OAuth,
notifications, and optional market-data providers. Prefer filling those
placeholders over exporting secrets directly in shell profiles or launchd
plists.

---

## Alpaca

### Create an account

Paper trading is free. Live trading requires a funded brokerage account at
[alpaca.markets](https://alpaca.markets).

### Install the SDK

```bash
conda run -n allweather pip install alpaca-py
```

### api_keys.env variables

| Variable | Description |
|---|---|
| `ALPACA_API_KEY` | Default Alpaca API key |
| `ALPACA_SECRET_KEY` | Default Alpaca API secret |
| `BROKER_ALPACA_<LABEL>_KEY` | API key for account labelled `<LABEL>` |
| `BROKER_ALPACA_<LABEL>_SECRET` | API secret |

`<LABEL>` is the `--account` argument in UPPER CASE. For the default
account you can use the short variables:

```bash
# Default account (--account default)
ALPACA_API_KEY="PKXXXXXXXXXXXXXXXX"
ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Named account (--account retirement)
BROKER_ALPACA_RETIREMENT_KEY="PKXXXXXXXXXXXXXXXX"
BROKER_ALPACA_RETIREMENT_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Test the connection

```bash
conda run -n allweather python -m live.healthcheck --broker alpaca
```

### Preview a rebalance

```bash
conda run -n allweather python -m live.rebalance --paper --broker alpaca
```

### Switching from paper to live

Pass `--live` instead of `--paper`. Ensure you have live credentials set:

```bash
# Uses BROKER_ALPACA_LIVE_KEY / BROKER_ALPACA_LIVE_SECRET
conda run -n allweather python -m live.rebalance --live --broker alpaca \
    --account live --dry-execute
```

**Never use `--execute` on a live account until you've reviewed the
`--dry-execute` output in full.**

---

## Tastytrade

### Requirements

- An active Tastytrade brokerage account
- The `tastytrade` community SDK (Python 3.9+)

### Install the SDK

```bash
conda run -n allweather pip install tastytrade
```

### api_keys.env variables

| Variable | Description |
|---|---|
| `TASTYTRADE_PROVIDER_SECRET` | Default account OAuth application provider secret |
| `TASTYTRADE_REFRESH_TOKEN` | Default account OAuth refresh token |
| `TASTYTRADE_ACCOUNT_NUMBER` | Default account number, if needed |
| `BROKER_TASTYTRADE_<LABEL>_PROVIDER_SECRET` | OAuth application provider secret |
| `BROKER_TASTYTRADE_<LABEL>_REFRESH_TOKEN` | OAuth refresh token |
| `BROKER_TASTYTRADE_<LABEL>_ACCOUNT_NUMBER` | (optional) Account number if you have multiple accounts |

```bash
# Default account (--account default)
TASTYTRADE_PROVIDER_SECRET="provider_secret"
TASTYTRADE_REFRESH_TOKEN="refresh_token"

# Named account (--account main)
BROKER_TASTYTRADE_MAIN_PROVIDER_SECRET="provider_secret"
BROKER_TASTYTRADE_MAIN_REFRESH_TOKEN="refresh_token"
BROKER_TASTYTRADE_MAIN_ACCOUNT_NUMBER="5XX12345"
```

### OAuth setup

The pinned community SDK (`tastytrade==12.4.1`) authenticates via OAuth.
Create an OAuth application/token pair per the SDK documentation, then validate
the credentials locally:

```bash
conda run -n allweather python -m live.brokers.tastytrade login --account default
```

### Limitations

- **Notional orders are not supported.** Buys are converted to whole-share
  quantities automatically: `floor(target_notional / current_price)`.
- **Paper trading:** Tastytrade does not expose a separate paper-trading
  environment via the API. Use `--paper` only as a local state flag; the
  credentials still point to your real account.
- **Calendar API:** Market hours are determined via a wall-clock check
  (09:30–16:00 ET on weekdays), not a Tastytrade calendar endpoint.

### Test the connection

```bash
conda run -n allweather python -m live.healthcheck --broker tastytrade
```

### Preview a rebalance

```bash
conda run -n allweather python -m live.rebalance --paper --broker tastytrade --account default
```

---

## Budget cap (`--budget`)

If your brokerage account holds more money than you want to manage with
this strategy, use the budget cap:

```bash
# First run: initialise the budget at $10,000
conda run -n allweather python -m live.rebalance --paper \
    --broker tastytrade --account main \
    --budget 10000 --initialize-budget

# Subsequent runs: budget grows automatically from dividends and PnL
conda run -n allweather python -m live.rebalance --paper \
    --broker tastytrade --account main \
    --budget 10000 --execute
```

The state file is stored at:
```
live/logs/budget_tastytrade_paper_main_<strategy_id>.json
```

---

## Holding-period gate (`--min-rebalance-interval-days`)

US investors who need to hold ETF lots for at least 31 days use this flag.
The rebalancer will:
1. Check the lot ledger before building a sell plan.
2. Mark any symbol whose youngest lot is < 31 days as `HOLD-BLOCKED`.
3. Skip those sells and log the reason.

```bash
# First run: seed the lot ledger from current positions
conda run -n allweather python -m live.rebalance --paper \
    --broker alpaca --account default \
    --initialize-lots --execute

# Subsequent runs: 31-day gate is enforced automatically
conda run -n allweather python -m live.rebalance --paper \
    --broker alpaca --account default \
    --min-rebalance-interval-days 31 --execute
```

---

## Notifications

Set any of these variables to receive run notifications:

```bash
# Slack (incoming webhook)
export ALLW_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Email (SMTP)
export ALLW_SMTP_HOST="smtp.gmail.com"
export ALLW_SMTP_PORT="587"
export ALLW_SMTP_USER="you@gmail.com"
export ALLW_SMTP_PASSWORD="your_app_password"
export ALLW_NOTIFY_EMAIL="you@gmail.com,partner@gmail.com"
```

---

## Automated scheduling (macOS launchd)

The `live/scheduler/install_launchd.sh` script installs a launchd agent that
runs `--dry-execute` every weekday morning at 09:05 UTC (market open).

```bash
# Preview (dry-execute) — installed by default
bash live/scheduler/install_launchd.sh --broker alpaca --account default

# Load it
launchctl load ~/Library/LaunchAgents/com.allweather.rebalance.plist
```

**The scheduled job runs `--dry-execute` only.** Real execution must be
triggered manually after reviewing the output:

```bash
conda run -n allweather python -m live.rebalance --paper \
    --broker alpaca --account default --execute
```

---

## Switching from `live._legacy.alpaca_rebalance` (legacy)

The old `live/_legacy/alpaca_rebalance.py` is preserved unchanged for backward
compatibility. All existing `make rebalance-*` targets continue to work.

To migrate to the new broker-agnostic rebalancer:

```bash
# Old way (Alpaca only)
make rebalance-preview ACCOUNT=default

# New way (any broker)
make rebalance-new-preview BROKER=alpaca ACCOUNT=default MODE=--paper
make rebalance-dry-execute BROKER=tastytrade ACCOUNT=main MODE=--live
```

The new rebalancer writes additional state files:
- `live/logs/cadence_*.json` — minimum-interval tracking
- `live/logs/lots_*.json` — lot ledger for 31-day hold enforcement
- `live/logs/budget_*.json` — strategy budget state
- `live/logs/run_summary.jsonl` — structured run log (one JSON line per run)
- `live/logs/monthly_runs.csv` — aggregate monthly view
- `live/logs/runs/*.json` — per-run detail archive
