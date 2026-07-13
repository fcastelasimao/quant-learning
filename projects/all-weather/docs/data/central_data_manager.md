# Central data manager — contract between projects

## TL;DR

All projects across the workspace (`personal_projects/projects/*`) consume
price and dividend data from the shared store, via the `quantcore` engine:

```
QuantFinance/
├── quantcore/                               ← shared engine (pip-installed editable)
│   └── src/quantcore/
│       ├── ingest.py                        ← single FMP fetcher (CLI: quantcore-ingest)
│       ├── data.py                          ← readers: load_prices / load_dividends / load_panel
│       └── config.py                        ← resolves data_dir() + api_keys_path()
├── data/
│   ├── DB_SPY_historical_data.db            ← per-ticker SQLite cache
│   ├── DB_QQQ_historical_data.db
│   ├── DB_JEPQ_historical_data.db
│   └── ...
└── api_keys.env                              ← FMP_API_KEY lives here
```

This document is the contract. Honour it and the All Weather project, the
TQQQ/SQQQ projects, and any future siblings all share clean data plumbing
without re-implementing FMP fetch logic. Data location is resolved by
`quantcore.config.data_dir()` (env var `$QUANT_DATA_DIR`, then walk-up, then
the workspace fallback) — no hard-coded paths in project code.

## Why a shared store

- One API key, one rate-limit budget, one cache, one bug surface.
- Backtest reproducibility: every project that reads `DB_<TICKER>_historical_data.db`
  at the same git SHA gets identical data.
- Fewer integration tests: data quality is verified at the fetcher level once.

## Schema

Each per-symbol DB contains the following tables (created on demand):

### `candles_1d`  (OHLCV daily, the bread-and-butter table)

| Column        | Type    | Notes                                                    |
|---------------|---------|----------------------------------------------------------|
| `ts`          | INTEGER | UTC epoch seconds, **primary key**                       |
| `open`        | REAL    |                                                          |
| `high`        | REAL    |                                                          |
| `low`         | REAL    |                                                          |
| `close`       | REAL    | raw close                                                |
| `adj_close`   | REAL    | dividend-adjusted close (populated via `--adjclose-backfill` or new fetches) |
| `volume`      | REAL    |                                                          |
| `utc_datetime`| TEXT    | `YYYY-MM-DD HH:MM:SS`                                    |
| `et_datetime` | TEXT    | `YYYY-MM-DD HH:MM:SS` America/New_York                   |

### `candles_<period>`  (intraday, when fetched)

`period ∈ {1min, 5min, 15min, 30min}`. Same schema as `candles_1d` minus `adj_close`.

### `dividends`  (per-symbol cash distribution history)

| Column             | Type | Notes                                                  |
|--------------------|------|--------------------------------------------------------|
| `ex_date`          | TEXT | `YYYY-MM-DD`, **primary key**                          |
| `amount`           | REAL | per-share cash distribution                            |
| `adj_amount`       | REAL | split-adjusted distribution (when FMP provides)        |
| `record_date`      | TEXT |                                                        |
| `payment_date`     | TEXT |                                                        |
| `declaration_date` | TEXT |                                                        |
| `label`            | TEXT | e.g. `"May 09, 24"`                                    |

Symbols that **don't pay cash dividends** (commodity bullion ETFs like GLD,
GLDM; commodity partnership ETFs like GSG; broad indices like ^VIX) will not
have a `dividends` table populated. That's the expected state — callers must
handle it.

## How to refresh data (CLI)

The ingestion CLI ships with `quantcore` (`quantcore-ingest`, equivalent to
`python -m quantcore.ingest`). The fetch dependencies install via the extra:
`pip install -e ../../quantcore[ingest]`.

```bash
# Refresh daily candles for the default symbol list
conda run -n allweather quantcore-ingest

# Refresh daily candles for a specific symbol set
conda run -n allweather quantcore-ingest \
    --symbols SPY QQQ TLT TIP GLD GSG GLDM JEPQ

# Refresh dividends only (the flag used by All Weather's tax model)
conda run -n allweather quantcore-ingest \
    --symbols SPY QQQ TLT TIP GLD GLDM GSG JEPQ --dividends-only

# Refresh BOTH candles and dividends in one pass
conda run -n allweather quantcore-ingest \
    --symbols SPY QQQ TLT TIP GLD GLDM GSG JEPQ --with-dividends

# Backfill adj_close into existing daily rows (legacy data integrity fix)
conda run -n allweather quantcore-ingest \
    --symbols SPY --adjclose-backfill
```

The fetcher is **idempotent** for daily data: rerunning upserts rows by `ts`
(candles) or `ex_date` (dividends). Safe to schedule daily.

## How projects consume the store

### All Weather

`engine/data.py` is a thin wrapper that reads from the central store:

```python
from engine.data import fetch_prices, fetch_dividends

# Prices — already wired since the FMP rerun
prices = fetch_prices(["SPY", "TLT", "GLD"], "2018-01-01", "2026-05-30")

# Dividends — new in C.9 / C.10, used by the tax model (D.13)
dividends = fetch_dividends(
    ["SPY", "TLT", "TIP", "JEPQ"], "2018-01-01", "2026-05-30"
)
# Returns long-form: Ticker, ExDate, Amount, AdjAmount,
#                    RecordDate, PaymentDate, DeclarationDate
```

Switch source via `config.DATA_SOURCE`:

| Setting                              | What `fetch_prices` does                           |
|--------------------------------------|----------------------------------------------------|
| `DATA_SOURCE = "yfinance"`           | Live download via yfinance (no local cache)        |
| `DATA_SOURCE = "fmp"`, `adj_close`   | Reads `candles_1d.adj_close` from the central DB   |
| `DATA_SOURCE = "fmp"`, `close`       | Reads `candles_1d.close` (price-return diagnostic) |

Both FMP modes resolve `data_dir` to `QuantFinance/data/` via
`engine.data._repo_data_dir()`, which now delegates to
`quantcore.config.data_dir()` — **no duplicate FMP fetch logic and no
hard-coded paths in `engine/data.py`**.

### TQQQ/SQQQ

Same pattern via `quantcore.data`. Reads its tickers (TQQQ, SQQQ, SPY, ^VIX,
^IRX, etc.) from the same central store. If it needs dividend data later, the
same `dividends` table is already available.

## How to add a new ticker

1. Add the ticker to `DEFAULT_SYMBOLS` in `quantcore/src/quantcore/ingest.py`
   if it should be refreshed on the default run, or pass it explicitly via
   `--symbols`.
2. Run a one-time backfill from the historical floor:
   ```bash
   conda run -n allweather quantcore-ingest \
       --symbols NEWTICKER --initial-date 2000-01-01 --with-dividends
   ```
3. Confirm the per-ticker file exists:
   ```bash
   ls /Users/franciscosimao/Documents/QuantFinance/data/DB_NEWTICKER_historical_data.db
   ```
4. No project code change required if it already calls `fetch_prices()` /
   `fetch_dividends()` with the new ticker in its list.

## What's NOT in the store (intentional)

- **Tax rates.** Year-keyed US capital-gains and qualified-dividend rates are
  small enough to hand-curate; live at `engine/tax_rates_us.yaml` (to be added
  in D.12). Do **not** try to fetch tax law from an API.
- **Per-ETF distribution composition** (qualified vs. ordinary dividend share,
  return-of-capital, K-1 / §1256 designation). These are tax-law facts about
  the ETF's structure; live in `engine/tax.py`'s classification table.
- **Live broker positions / fills.** That's `live/` territory, brokerage state.

## Schema-change protocol

If you need to add a column or table:

1. Open a small PR-style commit; update **this file** in the same change.
2. Make the SQL migration idempotent (`CREATE TABLE IF NOT EXISTS`,
   `ALTER TABLE ... ADD COLUMN` guarded by `PRAGMA table_info`).
3. Bump the per-symbol DB only on next run; no separate migration script.
4. Notify every project that reads the touched table (currently just All
   Weather; will be more once wave-rider starts reading dividends).

## Failure modes & recovery

| Symptom                                                  | Likely cause                                          | Fix                                                                 |
|----------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------|
| `RuntimeError: FMP_API_KEY not set`                       | `api_keys.env` missing or key not exported            | Add `FMP_API_KEY=...` to `QuantFinance/api_keys.env`                |
| `Missing FMP DB files for ['XXX']`                       | Ticker never fetched                                  | Run `quantcore-ingest --symbols XXX --initial-date 2000-01-01`     |
| `adj_close is X% null`                                   | Old daily rows pre-date the adj_close backfill        | Run `quantcore-ingest --symbols XXX --adjclose-backfill`           |
| `[div] XXX: no dividend history`                          | Ticker doesn't pay cash distributions (GLD, GSG, etc.) | Expected — caller should handle empty result gracefully             |
| FMP rate-limit                                            | Too many calls in a short window                      | Built-in retry/backoff handles it; reduce parallel jobs if persistent |

## Re-check date

Revisit this document **whenever a new project starts reading the store**, and
at minimum **annually** to confirm endpoints and schemas haven't drifted.

Last reviewed: 2026-05-30 (C.9 + C.10 — added `dividends` table and reader).
