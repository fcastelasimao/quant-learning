#!/usr/bin/env bash
# live/scheduler/install_cron.sh
# ==============================
# Install a cron job for the All-Weather rebalancer on Linux.
#
# By default installs a daily dry-execute (no real orders).
# Pass --auto-execute to schedule real order placement.
#
# Usage:
#   bash live/scheduler/install_cron.sh [OPTIONS]
#
# Options:
#   --broker      alpaca | tastytrade   (default: alpaca)
#   --account     account label         (default: default)
#   --mode        paper | live          (default: paper)
#   --strategy-id STRATEGY_ID           (default: 6asset_tip_gsg_rpavg)
#   --auto-execute  Schedule --execute instead of --dry-execute
#   --uninstall     Remove the cron job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BROKER="alpaca"
ACCOUNT="default"
MODE="--paper"
STRATEGY_ID="6asset_tip_gsg_rpavg"
AUTO_EXECUTE=false
UNINSTALL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --broker)       BROKER="$2";       shift 2;;
    --account)      ACCOUNT="$2";      shift 2;;
    --mode)
      if [[ "$2" == "live" ]]; then MODE="--live"; else MODE="--paper"; fi
      shift 2;;
    --strategy-id)  STRATEGY_ID="$2";  shift 2;;
    --auto-execute) AUTO_EXECUTE=true; shift;;
    --uninstall)    UNINSTALL=true;    shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

CRON_TAG="# allweather-rebalance"

if $UNINSTALL; then
  crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab -
  echo "Removed allweather cron jobs."
  exit 0
fi

# Find Python
PYTHON=""
for candidate in \
  "$PROJECT_ROOT/.venv/bin/python" \
  "$HOME/opt/anaconda3/envs/allweather/bin/python" \
  "$HOME/anaconda3/envs/allweather/bin/python" \
  "$HOME/miniforge3/envs/allweather/bin/python" \
  "$HOME/miniconda3/envs/allweather/bin/python"
do
  if [[ -x "$candidate" ]]; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Could not find Python. Activate your venv or set the path manually."
  exit 1
fi

echo "Python: $PYTHON"
echo "Project: $PROJECT_ROOT"

EXEC_FLAG="--dry-execute"
if $AUTO_EXECUTE; then
  EXEC_FLAG="--execute"
  echo ""
  echo "WARNING: --auto-execute will place REAL ORDERS on schedule."
  echo "The cadence gate (31-day minimum) still applies."
  echo ""
fi

LOG_DIR="$PROJECT_ROOT/live/logs"
mkdir -p "$LOG_DIR"

# Build the cron command
# Run weekdays at 14:35 UTC (09:35 ET during EDT, 10:35 during EST)
# This is 5 minutes after US market open.
CMD="cd $PROJECT_ROOT && $PYTHON -m live.rebalance $MODE --broker $BROKER --account $ACCOUNT --strategy-id $STRATEGY_ID --use-live-tickers $EXEC_FLAG >> $LOG_DIR/cron_rebalance.log 2>&1"

CRON_LINE="35 14 * * 1-5 $CMD $CRON_TAG"

# Remove old allweather entries, add new one
( crontab -l 2>/dev/null | grep -v "$CRON_TAG"; echo "$CRON_LINE" ) | crontab -

echo ""
echo "Installed cron job:"
echo "  Schedule: weekdays at 14:35 UTC (09:35 ET)"
echo "  Mode:     $MODE"
echo "  Broker:   $BROKER"
echo "  Account:  $ACCOUNT"
echo "  Action:   $EXEC_FLAG"
echo ""
echo "Verify with:"
echo "  crontab -l | grep allweather"
echo ""
echo "Logs go to:"
echo "  $LOG_DIR/cron_rebalance.log"
echo "  $LOG_DIR/runs/*.json"
echo ""
if ! $AUTO_EXECUTE; then
  echo "This runs --dry-execute only (no real orders)."
  echo "To place real orders, either:"
  echo "  1. Run manually: $PYTHON -m live.rebalance $MODE --broker $BROKER --account $ACCOUNT --execute"
  echo "  2. Rerun this script with --auto-execute"
fi
