# finding_new_tells

TQQQ daily strategy research framework. Emits `(p_buy, p_hold, p_sell)` with binary long-only positioning.

## Setup

```bash
# First time
conda env create -f environment.yml
conda activate quant
pip install -e .

# Or update existing env
conda activate quant
pip install -r requirements.txt -q
pip install -e .
```

## Backfill data (requires FMP API key in QuantFinance/api_keys.env)

```bash
conda activate quant
quantcore-ingest          # or: python -m quantcore.ingest
```

## Run tests

```bash
conda activate quant
pytest -q
```

## Interactive notebooks (marimo)

```bash
conda activate quant
marimo edit notebooks/00_v0_proof.py         # v0 sanity: 3 metrics vs buy-and-hold
marimo edit notebooks/01_data_sanity.py      # data alignment checks
marimo edit notebooks/02_metric_inspection.py # inspect each metric (4-panel plots)
marimo edit notebooks/03_regime.py           # HSMM regime states
marimo edit notebooks/04_strategy.py         # full strategy + backtest report
marimo edit notebooks/05_indicator_workbench.py # visual indicator workbench
marimo edit notebooks/06_metric_research.py  # metric-by-metric forward-return research
marimo edit notebooks/07_strategy_readiness.py  # pre-strategy diagnostics: costs, regimes, decay
marimo edit notebooks/08_progress_review.py  # narrative progress review for stakeholders
```

## Learning plan + research discipline

```bash
# One-month quant finance curriculum tailored to this repo
open docs/quant_finance_month_plan.md

# Signal credibility diagnostics for the Week 2 deliverable
cd src && python -m signal_diagnostics --compare-train-val --horizon 5
```

## Headless backtest

```bash
conda activate quant
cd src && python -m backtest --help
cd src && python -m backtest --split train     # evaluate on train set
cd src && python -m backtest --split val       # evaluate on val set
```

## Signal credibility diagnostics

```bash
conda activate quant
cd src && python -m signal_diagnostics --split train --horizon 5
cd src && python -m signal_diagnostics --split val --horizon 5
cd src && python -m signal_diagnostics --compare-train-val --horizon 5
cd src && python -m signal_diagnostics --decision-table --horizon 5 --output ../outputs/signal_credibility_5d_train_val.csv
```

## V2 train/validation research pipeline

```bash
conda activate quant
PYTHONPATH=src python -m strategy_v2 --data-dir data --output-dir outputs --horizon 5
```

This writes the train/validation-only decision table, v2 equity curves, benchmark comparison, and memo under `outputs/`. The frozen `2022-01-01+` test split is intentionally not evaluated here.

## Project layout

```
data/             SQLite DBs (one per symbol, schema: candles_1d)
src/
  data.py         DataReader — SQLite → aligned wide DataFrame
  metrics.py      ~20 voting + 3 watch metrics, pre-registered
  regime.py       5-state HSMM (Zakamulin 2023)
  strategy.py     Vote ensemble → regime-conditional softmax
  backtest.py     Walk-forward engine, MaxDD duration, MASTER_LOG writer
  viz.py          inspect / dashboard / backtest_report / indicator workbench plots
notebooks/        marimo .py notebooks
tests/            pytest suite
METRICS.md        Every metric: formula, vote rule, IC results
MASTER_LOG.csv    Append-only run ledger (auto-written by backtest.py)
```

## Holdout discipline
- **Train**: ≤ 2017-12-31
- **Val**: 2018-01-01 – 2021-12-31
- **Test**: 2022-01-01 → (frozen until strategy is final)

Tune thresholds and τ on train, validate on val, report on test exactly once.
