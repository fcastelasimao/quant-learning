# Project Cleanup Plan

## New directory structure

```
All_weather_portfolio/
├── CLAUDE.md
├── README.md
├── ToDo.md
├── research_log.md
├── session_handoff.md
├── learning_guide.md
├── requirements.txt
├── .gitignore
│
├── config.py              # parameters
├── main.py                # single-run entry point
├── backtest.py            # core engine
├── optimiser.py           # RP weights + random/SLSQP
├── data.py                # yfinance fetching
├── export.py              # results output + master log
├── plotting.py            # chart generation
├── portfolio.py           # live portfolio management
├── validation.py          # walk-forward (kept for future use)
│
├── compare_allw.py        # ALLW head-to-head comparison
├── run_rp_validation.py   # 3-split RP vs manual
├── scan_universes.py      # ETF universe scanner
├── run_overlay_grid.py    # overlay parameter grid search
│
├── strategies.json        # validated strategy registry
├── portfolio_holdings.json
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   └── test_stats.py
│
├── results/
│   ├── master_log.xlsx                    # current clean log
│   ├── phase11_rp_validation/             # 6 OOS runs (manual + RP × 3 splits)
│   ├── phase11_allw_comparison/           # ALLW comparison outputs
│   ├── phase11_overlay_grid/              # overlay grid results + CSV
│   ├── phase11_universe_scan/             # scan results CSV + correlation matrix
│   └── archive/                           # all old phase 1-10 results
│       ├── master_log_phase9_full.xlsx
│       ├── master_log_curated.xlsx
│       ├── master_log_archive_phase1.xlsx
│       ├── master_log_archive_phase9.xlsx
│       ├── master_log_archive_phase10.xlsx
│       ├── master_log_archive_phase11_manual6asset.xlsx
│       └── phase1_to_10_experiments/      # all 200+ old result folders
│
└── archive/
    ├── optimiser_de.py                    # archived DE code
    ├── run_experiment.py                  # old batch pipeline (DE-based)
    ├── curate_master_log.py               # old log curation tool
    ├── merge_master_logs.py               # old log merger
    ├── results_dashboard.py               # old dashboard generator
    ├── experiment_plan.md
    ├── market_validation.md
    ├── visualisation_strategy.md
    └── dashboard.html
```

## What to move/delete

### Move to archive/ (no longer needed for production)
- `run_experiment.py` (1,536 lines — built for DE pipeline, dead)
- `curate_master_log.py` (418 lines — old log curation)
- `merge_master_logs.py` (285 lines — old log merger)
- `results_dashboard.py` (979 lines — old dashboard)
- `experiment_plan.md`, `market_validation.md`, `visualisation_strategy.md`
- `dashboard.html`

### Move loose files from project root into results/
- All `allw_*.xlsx` → `results/phase11_allw_comparison/`
- All `allw_*.png` → `results/phase11_allw_comparison/`
- `overlay_grid_results.csv` → `results/phase11_overlay_grid/`
- `universe_scan_results.csv` → `results/phase11_universe_scan/`
- `scatter_calmar_mdd.png` → `results/archive/`
- All `experiment_summary*.txt` → `results/archive/`
- All `master_log_*.xlsx` (archives) → `results/archive/`
- `master_log_curated.xlsx` → `results/archive/`
- `master_log_phase9_full.xlsx` → `results/archive/`

### Move old result folders
- All 200+ timestamped folders in results/ → `results/archive/phase1_to_10_experiments/`
- Keep only Phase 11 results (the `_manual_split*` and `_rp5yr_split*` folders)

### Delete (truly worthless)
- `conftest.py` in project root (duplicate — tests/ has its own)
- `test_data.py` and `test_stats.py` in project root (duplicates)
- `__pycache__/` directories

## Code to keep vs archive

### Keep (core engine — ~2,800 lines)
| File | Lines | Purpose |
|---|---|---|
| backtest.py | 808 | Core engine — essential |
| optimiser.py | 482 | RP + search — essential |
| export.py | 559 | Master log — essential |
| config.py | 211 | Parameters — essential |
| main.py | 180 | Entry point — essential |
| data.py | 117 | Data fetching — essential |
| plotting.py | 202 | Charts — essential |
| portfolio.py | 178 | Live portfolio — essential |
| validation.py | 605 | Walk-forward — keep for future |

### Keep (experiment scripts — ~1,000 lines)
| File | Lines | Purpose |
|---|---|---|
| compare_allw.py | 810 | ALLW comparison — active |
| scan_universes.py | 231 | Universe scanner — done but reusable |
| run_rp_validation.py | 148 | RP validation — done but reusable |
| run_overlay_grid.py | 186 | Overlay grid — done, keep for reference |

### Archive (dead code — ~3,200 lines)
| File | Lines | Why archive |
|---|---|---|
| run_experiment.py | 1,536 | DE pipeline — Gate 1 closed |
| results_dashboard.py | 979 | Old dashboard — superseded |
| curate_master_log.py | 418 | Old log tool — one-time use |
| merge_master_logs.py | 285 | Old log merger — one-time use |

This cuts the active codebase from ~8,800 lines to ~4,800 lines.