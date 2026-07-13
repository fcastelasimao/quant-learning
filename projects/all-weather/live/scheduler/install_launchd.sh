#!/usr/bin/env bash
# live/scheduler/install_launchd.sh
# ==================================
# Install one or more launchd agents for the All-Weather rebalancer.
#
# Three agents are available (you can install all at once with --all):
#
#   rebalance   Weekdays at 09:05 ET — runs --dry-execute so you can review
#               the rebalance plan without placing real orders.
#               Pass --auto-execute to schedule real order placement instead.
#               The 31-day cadence gate prevents over-trading.
#
#   snapshot    Weekdays at 16:10 ET (after market close) — records a price
#               and portfolio-drift row to live/logs/daily_snapshots.csv.
#               No orders. No broker auth required if --no-broker is passed.
#
# Usage:
#   bash live/scheduler/install_launchd.sh [OPTIONS]
#
# Options:
#   --broker      alpaca | tastytrade   (default: alpaca)
#   --account     account label         (default: default)
#   --mode        paper | live          (default: paper)
#   --strategy-id STRATEGY_ID           (default: 6asset_tip_gsg_rpavg)
#   --auto-execute  Schedule --execute for the rebalance agent (REAL ORDERS)
#   --snapshot-only Install only the snapshot agent (no rebalance agent)
#   --rebalance-only Install only the rebalance agent (no snapshot agent)
#   --all           Install both agents (default behaviour)
#   --uninstall     Remove all installed allweather launchd agents
#
# Prerequisites:
#   - macOS with launchd
#   - allweather conda environment at a standard path (auto-detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# install_launchd.sh lives in live/scheduler/; PROJECT_ROOT is two levels up
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_ROOT/live/logs"

BROKER="alpaca"
ACCOUNT="default"
MODE="--paper"
STRATEGY_ID="6asset_tip_gsg_rpavg"
AUTO_EXECUTE=false
SNAPSHOT_ONLY=false
REBALANCE_ONLY=false
UNINSTALL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --broker)         BROKER="$2";                     shift 2;;
    --account)        ACCOUNT="$2";                    shift 2;;
    --mode)
      if [[ "$2" == "live" ]]; then MODE="--live"; else MODE="--paper"; fi
      shift 2;;
    --strategy-id)    STRATEGY_ID="$2";                shift 2;;
    --auto-execute)   AUTO_EXECUTE=true;               shift;;
    --snapshot-only)  SNAPSHOT_ONLY=true;              shift;;
    --rebalance-only) REBALANCE_ONLY=true;             shift;;
    --all)                                             shift;;   # default, no-op
    --uninstall)      UNINSTALL=true;                  shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# ── Uninstall ──────────────────────────────────────────────────────────────

if $UNINSTALL; then
  for label in com.allweather.rebalance com.allweather.snapshot; do
    plist="$LAUNCH_AGENTS/$label.plist"
    if [[ -f "$plist" ]]; then
      launchctl unload "$plist" 2>/dev/null || true
      rm "$plist"
      echo "Removed: $plist"
    else
      echo "Not installed: $plist"
    fi
  done
  exit 0
fi

# ── Detect conda Python ────────────────────────────────────────────────────

CONDA_PYTHON=""
for candidate in \
  "$HOME/opt/anaconda3/envs/allweather/bin/python" \
  "$HOME/anaconda3/envs/allweather/bin/python" \
  "$HOME/miniforge3/envs/allweather/bin/python" \
  "$HOME/miniconda3/envs/allweather/bin/python"
do
  if [[ -x "$candidate" ]]; then
    CONDA_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$CONDA_PYTHON" ]]; then
  echo "ERROR: Could not find allweather conda Python. Set CONDA_PYTHON manually."
  exit 1
fi

mkdir -p "$LAUNCH_AGENTS"
mkdir -p "$LOG_DIR"

echo "Python:  $CONDA_PYTHON"
echo "Project: $PROJECT_ROOT"
echo ""

# ── Helper: substitute placeholders ───────────────────────────────────────

_install_plist() {
  local template="$1"
  local dest="$2"
  sed \
    -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__CONDA_PYTHON__|$CONDA_PYTHON|g" \
    -e "s|__BROKER__|$BROKER|g" \
    -e "s|__ACCOUNT__|$ACCOUNT|g" \
    -e "s|__MODE__|$MODE|g" \
    -e "s|__STRATEGY_ID__|$STRATEGY_ID|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$template" > "$dest"
  echo "Installed: $dest"
}

# ── Install rebalance agent ────────────────────────────────────────────────

if ! $SNAPSHOT_ONLY; then
  REBALANCE_TEMPLATE="$SCRIPT_DIR/com.allweather.rebalance.plist.template"
  REBALANCE_DEST="$LAUNCH_AGENTS/com.allweather.rebalance.plist"

  _install_plist "$REBALANCE_TEMPLATE" "$REBALANCE_DEST"

  if $AUTO_EXECUTE; then
    # Replace --dry-execute with --execute in the installed plist
    sed -i.bak 's|<string>--dry-execute</string>|<string>--execute</string>|g' "$REBALANCE_DEST"
    rm -f "$REBALANCE_DEST.bak"
    echo ""
    echo "  ⚠  AUTO-EXECUTE MODE: this schedule will place REAL ORDERS."
    echo "     The 31-day cadence gate prevents double-execution."
    echo "     To revert to dry-execute, re-run without --auto-execute."
  fi
fi

# ── Install snapshot agent ─────────────────────────────────────────────────

if ! $REBALANCE_ONLY; then
  SNAPSHOT_TEMPLATE="$SCRIPT_DIR/com.allweather.snapshot.plist.template"
  SNAPSHOT_DEST="$LAUNCH_AGENTS/com.allweather.snapshot.plist"
  _install_plist "$SNAPSHOT_TEMPLATE" "$SNAPSHOT_DEST"
fi

# ── Instructions ──────────────────────────────────────────────────────────

echo ""
echo "Review the installed plist(s) before loading:"
if ! $SNAPSHOT_ONLY; then
  echo "  cat $LAUNCH_AGENTS/com.allweather.rebalance.plist"
fi
if ! $REBALANCE_ONLY; then
  echo "  cat $LAUNCH_AGENTS/com.allweather.snapshot.plist"
fi
echo ""
echo "Load both agents:"
if ! $SNAPSHOT_ONLY; then
  echo "  launchctl load $LAUNCH_AGENTS/com.allweather.rebalance.plist"
fi
if ! $REBALANCE_ONLY; then
  echo "  launchctl load $LAUNCH_AGENTS/com.allweather.snapshot.plist"
fi
echo ""
echo "Verify they are scheduled:"
echo "  launchctl list | grep allweather"
echo ""
echo "Test immediately:"
if ! $SNAPSHOT_ONLY; then
  echo "  launchctl start com.allweather.rebalance   # runs dry-execute (or --execute if --auto-execute)"
fi
if ! $REBALANCE_ONLY; then
  echo "  launchctl start com.allweather.snapshot    # records today's snapshot"
fi
echo ""
if ! $SNAPSHOT_ONLY && ! $AUTO_EXECUTE; then
  echo "IMPORTANT: The rebalance agent runs --dry-execute only."
  echo "To place real orders, run manually:"
  echo "  $CONDA_PYTHON -m live.rebalance $MODE --broker $BROKER --account $ACCOUNT --execute"
  echo ""
  echo "Or reinstall with --auto-execute to schedule real orders automatically."
fi
