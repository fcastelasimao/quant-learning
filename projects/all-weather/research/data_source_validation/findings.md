---
verdict: production
summary: "yfinance total-return and FMP adj_close are effectively identical; price-return materially understates performance"
promoted: "engine/config.py — DATA_SOURCE and FMP_PRICE_COLUMN settings"
---

# Data Source Validation

**Question:** Do yfinance total-return prices and FMP `adj_close` produce the same backtest results? Does it matter which data source we use?

**Result (2026-05-09 rerun):** yfinance total-return and FMP `adj_close` are effectively identical:
- Calmar 2018/2020/2022: 0.487/0.503/0.452 (yfinance) vs 0.488/0.504/0.453 (FMP)
- Price-return / `close` materially understates performance

This confirmed FMP as the production data source with `FMP_PRICE_COLUMN="adj_close"`.

## Scripts

- `compare_fmp_rp.py` — local FMP SQLite RP-weight comparison vs strategies.json
- `rerun_rp_validation.py` — yfinance/FMP RP rerun matrix across multiple IS/OOS splits

## Run
```bash
conda run -n allweather python3 research/data_source_validation/compare_fmp_rp.py
conda run -n allweather python3 research/data_source_validation/rerun_rp_validation.py
```
