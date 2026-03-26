#!/bin/bash
# cleanup.sh — Reorganise the All Weather Portfolio project
# Run from the project root: bash cleanup.sh
# Review the plan in cleanup_plan.md before running.

set -e
echo "=== Project Cleanup ==="

# ---- Create new directory structure ----
mkdir -p archive
mkdir -p results/phase11_rp_validation
mkdir -p results/phase11_allw_comparison
mkdir -p results/phase11_overlay_grid
mkdir -p results/phase11_universe_scan
mkdir -p results/archive/phase1_to_10_experiments

# ---- Move dead code to archive/ ----
echo "Moving archived code..."
mv -f run_experiment.py archive/ 2>/dev/null || true
mv -f curate_master_log.py archive/ 2>/dev/null || true
mv -f merge_master_logs.py archive/ 2>/dev/null || true
mv -f results_dashboard.py archive/ 2>/dev/null || true
mv -f experiment_plan.md archive/ 2>/dev/null || true
mv -f market_validation.md archive/ 2>/dev/null || true
mv -f visualisation_strategy.md archive/ 2>/dev/null || true
mv -f dashboard.html archive/ 2>/dev/null || true

# ---- Move loose output files from root to results/ ----
echo "Moving loose files to results/..."

# ALLW comparison files
mv -f results/allw_comparison_*.xlsx results/phase11_allw_comparison/ 2>/dev/null || true
mv -f results/allw_comparison_*.png results/phase11_allw_comparison/ 2>/dev/null || true
mv -f results/allw_fee_drag_*.png results/phase11_allw_comparison/ 2>/dev/null || true

# Overlay grid
mv -f results/overlay_grid_results.csv results/phase11_overlay_grid/ 2>/dev/null || true

# Universe scan
mv -f results/universe_scan_results.csv results/phase11_universe_scan/ 2>/dev/null || true

# Old artifacts to archive
mv -f results/scatter_calmar_mdd.png results/archive/ 2>/dev/null || true
mv -f results/experiment_summary*.txt results/archive/ 2>/dev/null || true
mv -f results/master_log_archive_*.xlsx results/archive/ 2>/dev/null || true
mv -f results/master_log_curated.xlsx results/archive/ 2>/dev/null || true
mv -f results/master_log_phase9_full.xlsx results/archive/ 2>/dev/null || true

# ---- Move Phase 11 result folders ----
echo "Organising result folders..."
for dir in results/*_manual_split* results/*_rp5yr_split*; do
    [ -d "$dir" ] && mv -f "$dir" results/phase11_rp_validation/ 2>/dev/null || true
done

# ---- Move all old result folders to archive ----
for dir in results/2026-03-1* results/2026-03-20* results/2026-03-21* results/2026-03-22* results/2026-03-23* results/2026-03-25*; do
    [ -d "$dir" ] && mv -f "$dir" results/archive/phase1_to_10_experiments/ 2>/dev/null || true
done

# Also move the March 26 full_backtest and manual runs (pre-Phase 11 cleanup)
for dir in results/2026-03-26_07-*; do
    [ -d "$dir" ] && mv -f "$dir" results/archive/phase1_to_10_experiments/ 2>/dev/null || true
done

# ---- Remove duplicate test files from root ----
echo "Removing root-level test duplicates..."
rm -f conftest.py test_data.py test_stats.py 2>/dev/null || true

# ---- Clean pycache ----
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=== Cleanup complete ==="
echo "Active code: $(wc -l config.py main.py backtest.py optimiser.py data.py export.py plotting.py portfolio.py validation.py 2>/dev/null | tail -1)"
echo "Experiment scripts: $(wc -l compare_allw.py run_rp_validation.py scan_universes.py run_overlay_grid.py 2>/dev/null | tail -1)"
echo "Archived code: $(wc -l archive/*.py 2>/dev/null | tail -1)"
echo ""
echo "Next: review the results/ structure and verify master_log.xlsx is in results/"