#!/usr/bin/env bash
# server/setup_ec2.sh
# ====================
# Bootstrap a fresh EC2 instance (Amazon Linux 2023 / Ubuntu 22.04+) to run
# the All-Weather rebalancer on a cron schedule.
#
# What this script does
# ---------------------
# 1. Install Python 3.12 and pip
# 2. Clone the all-weather repository (or update an existing clone)
# 3. Create a Python virtualenv and install server/requirements.txt
# 4. Create the live/logs/ directory
# 5. Copy strategies.json.example → strategies.json (if not already present)
# 6. Copy api_keys.env.example → api_keys.env (if not already present)
# 7. Install the cron job (weekdays, 14:35 UTC / 09:35 ET)
#
# What this script does NOT do
# ----------------------------
# - Fill in your API keys (you must do this after setup: edit api_keys.env)
# - Execute any trades
# - Install the quantcore research package (not needed for live execution)
#
# Usage
# -----
# On a fresh EC2 instance (after SSH in):
#   curl -fsSL https://raw.githubusercontent.com/<your-repo>/main/server/setup_ec2.sh | bash
#
# Or after cloning:
#   bash server/setup_ec2.sh [--repo-dir /path/to/all-weather] [--auto-execute]
#
# Options
#   --repo-dir PATH      Where the repo is / should be cloned  (default: ~/all-weather)
#   --repo-url URL       Git remote to clone from              (default: must be set)
#   --broker BROKER      alpaca | tastytrade                   (default: alpaca)
#   --account LABEL      Account label                         (default: default)
#   --mode MODE          paper | live                          (default: paper)
#   --strategy-id ID     Strategy id                           (default: 6asset_tip_gsg_rpavg)
#   --auto-execute       Schedule real order placement (default: dry-execute only)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────

REPO_DIR="$HOME/all-weather"
REPO_URL=""          # must be set if cloning from scratch
BROKER="alpaca"
ACCOUNT="default"
MODE="paper"
STRATEGY_ID="6asset_tip_gsg_rpavg"
AUTO_EXECUTE=false

# ── Arg parsing ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-dir)     REPO_DIR="$2";      shift 2;;
    --repo-url)     REPO_URL="$2";      shift 2;;
    --broker)       BROKER="$2";        shift 2;;
    --account)      ACCOUNT="$2";       shift 2;;
    --mode)         MODE="$2";          shift 2;;
    --strategy-id)  STRATEGY_ID="$2";   shift 2;;
    --auto-execute) AUTO_EXECUTE=true;  shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

echo "====================================="
echo "All-Weather EC2 setup"
echo "  Repo dir:    $REPO_DIR"
echo "  Broker:      $BROKER"
echo "  Mode:        $MODE"
echo "  Strategy:    $STRATEGY_ID"
echo "  Auto-exec:   $AUTO_EXECUTE"
echo "====================================="
echo ""

# ── Step 1: Python 3.12 ──────────────────────────────────────────────────

echo "[1/7] Checking Python 3.12..."
if command -v python3.12 &>/dev/null; then
  PYTHON="$(command -v python3.12)"
  echo "  Found: $PYTHON"
elif command -v python3 &>/dev/null && python3 --version 2>&1 | grep -q "3\.1[2-9]"; then
  PYTHON="$(command -v python3)"
  echo "  Found: $PYTHON ($(python3 --version))"
else
  echo "  Installing Python 3.12..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-pip
    PYTHON="$(command -v python3.12)"
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y python3.12 python3.12-pip
    PYTHON="$(command -v python3.12)"
  else
    echo "ERROR: Cannot install Python — unsupported OS. Install Python 3.12 manually."
    exit 1
  fi
fi

# ── Step 2: Clone / update repo ───────────────────────────────────────────

echo ""
echo "[2/7] Setting up repository at $REPO_DIR..."
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "  Existing clone found — pulling latest..."
  git -C "$REPO_DIR" pull --ff-only
else
  if [[ -z "$REPO_URL" ]]; then
    echo "ERROR: --repo-url is required when $REPO_DIR does not exist."
    echo "  Example: --repo-url git@github.com:youruser/all-weather.git"
    exit 1
  fi
  echo "  Cloning from $REPO_URL..."
  git clone "$REPO_URL" "$REPO_DIR"
fi

# ── Step 3: Virtualenv + dependencies ────────────────────────────────────

echo ""
echo "[3/7] Creating virtualenv and installing dependencies..."
VENV="$REPO_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
  echo "  Created: $VENV"
fi

VENV_PIP="$VENV/bin/pip"
VENV_PYTHON="$VENV/bin/python"

"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -r "$REPO_DIR/server/requirements.txt" --quiet
echo "  Dependencies installed."

# ── Step 4: Log directory ─────────────────────────────────────────────────

echo ""
echo "[4/7] Creating log directory..."
mkdir -p "$REPO_DIR/live/logs"
echo "  Logs: $REPO_DIR/live/logs"

# ── Step 5: strategies.json ───────────────────────────────────────────────

echo ""
echo "[5/7] Checking strategies.json..."
STRAT_FILE="$REPO_DIR/strategies.json"
if [[ -f "$STRAT_FILE" ]]; then
  echo "  Already exists: $STRAT_FILE"
else
  cp "$REPO_DIR/server/strategies.json.example" "$STRAT_FILE"
  echo "  Created from example: $STRAT_FILE"
  echo "  IMPORTANT: Review and edit strategies.json to confirm allocations."
fi

# ── Step 6: api_keys.env ──────────────────────────────────────────────────

echo ""
echo "[6/7] Checking api_keys.env..."
KEYS_FILE="$HOME/api_keys.env"
if [[ -f "$KEYS_FILE" ]]; then
  echo "  Already exists: $KEYS_FILE"
else
  cat > "$KEYS_FILE" << 'KEYS_TEMPLATE'
# All-Weather API Keys
# ====================
# Fill in your credentials and keep this file secure (chmod 600).
# This file is loaded automatically by the rebalancer.

# ── Alpaca ────────────────────────────────────────────────────────────────
# Default account (--account default)
ALPACA_API_KEY=""
ALPACA_SECRET_KEY=""

# Named account (--account live)
# BROKER_ALPACA_LIVE_KEY=""
# BROKER_ALPACA_LIVE_SECRET=""

# ── Tastytrade (optional) ─────────────────────────────────────────────────
# TASTYTRADE_PROVIDER_SECRET=""
# TASTYTRADE_REFRESH_TOKEN=""

# ── Notifications (optional) ──────────────────────────────────────────────
# ALLW_SLACK_WEBHOOK_URL=""
# ALLW_SMTP_HOST="smtp.gmail.com"
# ALLW_SMTP_PORT="587"
# ALLW_SMTP_USER=""
# ALLW_SMTP_PASSWORD=""
# ALLW_NOTIFY_EMAIL=""
KEYS_TEMPLATE

  chmod 600 "$KEYS_FILE"
  echo "  Created: $KEYS_FILE (chmod 600)"
  echo "  IMPORTANT: Edit $KEYS_FILE and add your API credentials."
fi

# Point the rebalancer at the keys file
export ALLW_API_KEYS_ENV="$KEYS_FILE"

# ── Step 7: Install cron job ──────────────────────────────────────────────

echo ""
echo "[7/7] Installing cron job..."

EXEC_FLAG="--dry-execute"
CRON_TAG="# allweather-rebalance"
LOG_FILE="$REPO_DIR/live/logs/cron_rebalance.log"

if $AUTO_EXECUTE; then
  EXEC_FLAG="--execute"
  echo ""
  echo "  ⚠  AUTO-EXECUTE: real orders will be placed on schedule."
  echo "     The 31-day cadence gate prevents double-execution."
  echo ""
fi

MODE_FLAG="--$MODE"
CMD="cd $REPO_DIR && ALLW_API_KEYS_ENV=$KEYS_FILE $VENV_PYTHON -m live.rebalance $MODE_FLAG --broker $BROKER --account $ACCOUNT --strategy-id $STRATEGY_ID --use-live-tickers $EXEC_FLAG >> $LOG_FILE 2>&1"
CRON_LINE="35 14 * * 1-5 $CMD $CRON_TAG"

# Snapshot job (no broker auth needed — uses yfinance only)
SNAP_LOG="$REPO_DIR/live/logs/cron_snapshot.log"
SNAP_CMD="cd $REPO_DIR && $VENV_PYTHON -m live.daily_snapshot $MODE_FLAG --broker $BROKER --account $ACCOUNT --strategy-id $STRATEGY_ID --no-broker >> $SNAP_LOG 2>&1"
SNAP_TAG="# allweather-snapshot"
SNAP_LINE="10 21 * * 1-5 $SNAP_CMD $SNAP_TAG"

# Remove old allweather entries, add new ones
( crontab -l 2>/dev/null | grep -v "$CRON_TAG" | grep -v "$SNAP_TAG"
  echo "$CRON_LINE"
  echo "$SNAP_LINE"
) | crontab -

echo "  Installed two cron jobs:"
echo "    14:35 UTC Mon–Fri — rebalancer ($EXEC_FLAG)"
echo "    21:10 UTC Mon–Fri — daily snapshot (--no-broker, price+drift only)"
echo "  Verify:  crontab -l | grep allweather"

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "====================================="
echo "Setup complete!"
echo "====================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Edit your API keys:"
echo "   nano $KEYS_FILE"
echo ""
echo "2. Verify the connection and strategy:"
echo "   $VENV_PYTHON -m live.healthcheck --broker $BROKER"
echo ""
echo "3. Preview a rebalance (no orders):"
echo "   cd $REPO_DIR"
echo "   $VENV_PYTHON -m live.rebalance --$MODE --broker $BROKER --account $ACCOUNT"
echo ""
echo "4. Run a dry-execute to test logging:"
echo "   $VENV_PYTHON -m live.rebalance --$MODE --broker $BROKER --account $ACCOUNT --dry-execute"
echo ""
echo "5. Check cron is scheduled:"
echo "   crontab -l | grep allweather"
echo ""
if ! $AUTO_EXECUTE; then
  echo "6. When ready for real orders, re-run with --auto-execute:"
  echo "   bash $REPO_DIR/server/setup_ec2.sh --mode $MODE --broker $BROKER --account $ACCOUNT --auto-execute"
  echo ""
fi
echo "Logs:"
echo "   $LOG_FILE"
echo "   $SNAP_LOG"
