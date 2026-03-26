# All Weather Portfolio — TODO

Last updated: 2026-03-26

---

## ✅ COMPLETED (research phase)

- [x] RP validation: beats manual on all 3 OOS splits (+11% to +18%)
- [x] Universe scan: 6-asset confirmed as near-optimal (16k subsets tested)
- [x] ALLW comparison: Calmar 2.78 vs 1.78 (RP weights, live ETFs)
- [x] Overlay grid search: 126 combos, best params +1.3% on 2/3 splits, -5.3% on hardest → closed
- [x] DE archived (26 experiments, all failed)
- [x] Data integrity audit — all OOS numbers corrected
- [x] Production weights: SPY 13%, QQQ 11%, TLT 19%, TIP 33%, GLD 14%, GSG 10%

---

## 🔴 IMMEDIATE — Before any public claims

### A1: Full-period backtest with production weights
Run RP production weights over entire 2006-2026 period. Produces the 20-year growth chart.
```bash
# Set in config.py: TARGET_ALLOCATION to RP weights, RUN_MODE="full_backtest", RUN_TAG="rp_production"
conda run -n allweather python3 main.py
```

### A2: Start paper trading
Set up RP weights in portfolio.py. Track monthly. Compare to backtest over 3+ months.
No product needed — just a spreadsheet and discipline.

### A3: Update strategies.json
Replace manual weights with averaged RP weights. Update OOS results.

---

## 🟡 DEMAND VALIDATION — Before building product

### B1: Brand name decision
Cannot use "All Weather" (Bridgewater trademark).

### B2: Write the ALLW comparison blog post
"DIY vs ALLW: Same returns, half the drawdown, 85% less fees"
Data ready. Post on Reddit (r/Bogleheads, r/portfolios), finance Twitter.
Success metric: >100 email signups.

### B3: FCA compliance consultation
30-minute call with compliance lawyer. £200-300. Must resolve before
any UK product that provides "rebalancing instructions."

---

## 🟢 PRODUCT BUILD — Only after demand validated (Gate 4)

- [ ] Landing page with email capture
- [ ] Minimal product: monthly rebalancing email with RP weights
- [ ] Rolling RP (quarterly recompute from trailing covariance)
- [ ] Fix rebalancing mismatch (backtest monthly vs live 5% threshold)
- [ ] GBP/EUR adjusted backtest for non-US customers
- [ ] Crisis analysis script for content

---

## Decision gates

| Gate | Condition | Status |
|---|---|---|
| Gate 1 | DE adds value? | CLOSED — No |
| Gate 2 | RP robust across 3 windows? | **PASSED** |
| Gate 3 | Overlay adds value? | **CLOSED — No** |
| Gate 4 | Organic demand? (>100 signups from blog) | Open |
| Gate 5 | Paper trading <1% tracking error (3 months) | Open |
| Gate 6 | FCA review passed? | Open |