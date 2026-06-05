#!/usr/bin/env bash
# scripts/install_launchd.sh
# ==========================
# Fill placeholders in the launchd plist template and install it to
# ~/Library/LaunchAgents/.
#
# IMPORTANT: This installs a --dry-execute schedule only.  Real execution
# is always triggered manually.  Review the plist after installation.
#
# Usage:
#   bash scripts/install_launchd.sh [OPTIONS]
#
# Options:
#   --broker      alpaca | tastytrade   (default: alpaca)
#   --account     account label         (default: default)
#   --mode        paper | live          (default: paper)
#   --strategy-id STRATEGY_ID           (default: 6asset_tip_gsg_rpavg)
#   --uninstall   remove the plist instead of installing
#
# Prerequisites:
#   - macOS with launchd
#   - allweather conda environment: /Users/$USER/opt/anaconda3/envs/allweather/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMPLATE="$PROJECT_ROOT/live/scheduler/com.allweather.rebalance.plist.template"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
DEST_PLIST="$LAUNCH_AGENTS/com.allweather.rebalance.plist"
LOG_DIR="$PROJECT_ROOT/live/logs"

BROKER="alpaca"
ACCOUNT="default"
MODE="--paper"
STRATEGY_ID="6asset_tip_gsg_rpavg"
UNINSTALL=false

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --broker)    BROKER="$2";      shift 2;;
    --account)   ACCOUNT="$2";     shift 2;;
    --mode)
      if [[ "$2" == "live" ]]; then MODE="--live"; else MODE="--paper"; fi
      shift 2;;
    --strategy-id) STRATEGY_ID="$2"; shift 2;;
    --uninstall) UNINSTALL=true;   shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# Uninstall path
if $UNINSTALL; then
  if [[ -f "$DEST_PLIST" ]]; then
    launchctl unload "$DEST_PLIST" 2>/dev/null || true
    rm "$DEST_PLIST"
    echo "Uninstalled: $DEST_PLIST"
  else
    echo "Not installed: $DEST_PLIST"
  fi
  exit 0
fi

# Detect conda Python
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
  echo "ERROR: Could not find allweather conda Python. Set it manually in the plist."
  CONDA_PYTHON="/path/to/allweather/bin/python"
fi

mkdir -p "$LAUNCH_AGENTS"
mkdir -p "$LOG_DIR"

# Substitute placeholders
sed \
  -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
  -e "s|__CONDA_PYTHON__|$CONDA_PYTHON|g" \
  -e "s|__BROKER__|$BROKER|g" \
  -e "s|__ACCOUNT__|$ACCOUNT|g" \
  -e "s|__MODE__|$MODE|g" \
  -e "s|__STRATEGY_ID__|$STRATEGY_ID|g" \
  -e "s|__LOG_DIR__|$LOG_DIR|g" \
  "$TEMPLATE" > "$DEST_PLIST"

echo "Installed: $DEST_PLIST"
echo ""
echo "Review the plist before loading:"
echo "  cat $DEST_PLIST"
echo ""
echo "Add your credentials to the EnvironmentVariables section, then load:"
echo "  launchctl load $DEST_PLIST"
echo ""
echo "Check it is scheduled:"
echo "  launchctl list | grep allweather"
echo ""
echo "To run immediately (dry-execute):"
echo "  launchctl start com.allweather.rebalance"
echo ""
echo "IMPORTANT: This schedule runs --dry-execute only."
echo "To place real orders, run manually:"
echo "  $CONDA_PYTHON -m live.rebalance $MODE --broker $BROKER --account $ACCOUNT --execute"
