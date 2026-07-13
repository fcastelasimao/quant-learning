# CLAUDE.md — Session conventions for finding_new_tells

## Environment
- Conda env: `quant` (Python 3.11)
- All commands: `conda activate quant && ...` or use `/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python`

## Data
- Market data: shared SQLite DBs in `QuantFinance/data/` (source of truth), loaded via `src/data.py` (thin shim over `quantcore.data`).
- Data directory resolved via `quantcore.config.data_dir()` (env var `$QUANT_DATA_DIR`, walk-up, or fallback).
- Refresh with `quantcore-ingest` or `python -m quantcore.ingest` (reads the key from `QuantFinance/api_keys.env`).
- The shared engine (`quantcore`) is installed editable: `pip install -e ../../quantcore`.

## Core rules
- **No lookahead**: metric compute() at index t may ONLY use data at or before t. test_metrics.py enforces this.
- **No test set tuning**: train ≤ 2017, val 2018–2021, test 2022→. Test set is evaluated once at the end.
- **BH correction**: when reporting IC significance across ≥ 2 metrics, apply Benjamini-Hochberg FDR correction.
- **MASTER_LOG.csv is append-only**: backtest.py auto-appends. Never overwrite existing rows.
- **Marimo, not Jupyter**: notebooks/*.py are marimo files. Run with `marimo edit notebooks/<file>.py`.
- **Thresholds on train, validate on val**: no tuning τ or vote thresholds on val or test data.

## Symbol convention
- Column names: `{SYMBOL}_{field}` e.g. `TQQQ_close`, `QQQ_open`
- Index: pd.DatetimeIndex of ET trading days (date only, midnight)
- Missing symbols: DataReader returns panel without those columns — metrics abstain (vote = 0), no crash

## Data availability
- Always on disk: TQQQ, QQQ, SPY, SPXL, TLT (fallback only)
- Add to data manager and download: HYG, LQD, ^VIX, ^VIX3M, ^TNX, ^IRX
  (fallback ETFs if indices fail: VIXY, VXZ, SHY)
- Ignored (another project): TIP, GLD, GLDM, GSG

## Regime model
- `regime.py` implements 5-state HSMM (Zakamulin 2023 style)
- States are relabeled post-hoc by conditional mean return: 0=strong_bull → 4=strong_bear
- Walk-forward refit annually on expanding window

## Adding a new metric
1. Add `Metric` instance to `src/metrics.py` using `@register`
2. Add entry to `METRICS.md`
3. Add an entry to `MASTER_LOG.csv` with `notes` describing the rationale
4. Re-run `pytest -q` — lookahead test must pass
5. Report IC with BH correction across the full set

---

_Behavioral rules (Karpathy) now live globally in `~/.claude/CLAUDE.md` and apply automatically — no longer duplicated per project._
