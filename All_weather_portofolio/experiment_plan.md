# Experiment Plan — Phase 2

## Guiding principles

1. Full 2006-2026 history preferred. Anything missing 2008 is a spot-check only.
2. One question per experiment. Every new universe answers a specific question.
3. ETF substitution spot-checks run on a restricted date range — they do not 
   contaminate the main IS/OOS split.
4. 6asset_tip_gsg is the priority: it beat the baseline and needs robustness validation.
5. Spot-checks use a simpler 2-step pipeline (backtest only, no walk-forward, 
   no OOS) since they are equivalence tests not strategy validation.

---

## Group A — Follow-up on 6asset_tip_gsg (highest priority)

This universe beat the baseline with OOS Calmar 0.476. Before promoting it,
validate robustness with two alternative IS/OOS splits.

### A1: 6asset_tip_gsg with earlier OOS split (2018-2026)
- IS: 2006-2018, OOS: 2018-2026
- Question: does the strong OOS hold on a different test window?
- If yes: strong evidence of genuine robustness
- If no: the 2020-2026 result was regime-specific

### A2: 6asset_tip_gsg with later OOS split (2022-2026)  
- IS: 2006-2022, OOS: 2022-2026
- Question: does it hold specifically through the 2022 rate shock?
- 2022 is the hardest period — if it survives this test it is genuinely robust

---

## Group B — New asset class candidates (full 2006-2026 history)

All ETFs in this group have inception before 2006-01-01 so full backtests are valid.

### B1: Add REITs (VNQ, inception Sep 2004)
- Universe: SPY/QQQ/TLT/IEF/GLD/GSG + VNQ (7 assets)
- Manual: SPY 10%, QQQ 15%, TLT 25%, IEF 10%, GLD 15%, GSG 10%, VNQ 15%
- Question: do real assets (real estate) add a fourth uncorrelated return stream?
- Rationale: REITs are an All Weather asset class Dalio himself includes in 
  the broader framework. VNQ pays dividends and is inflation-sensitive.
- Cap: real_estate max 20%

### B2: 6asset_tip_gsg + VNQ (best universe + REITs, 7 assets)
- Manual: SPY 12%, QQQ 12%, TLT 25%, TIP 12%, GLD 13%, GSG 10%, VNQ 16%
- Question: does adding REITs to our best universe improve further?

### B3: Replace QQQ with AGG (aggregate bonds instead of tech, 8 assets)
- Universe: SPY/AGG/IWD/TLT/IEF/SHY/GLD/GSG
- AGG inception Sep 2003, fully covers backtest
- Manual: SPY 15%, AGG 15%, IWD 10%, TLT 20%, IEF 10%, SHY 5%, GLD 15%, GSG 10%
- Question: does broad bond market exposure outperform tech concentration?
- Rationale: AGG provides investment grade corporate + government bond diversification
  beyond TLT/IEF/SHY alone

### B4: Add LQD (investment grade corporate bonds) to 8-asset (9 assets)
- LQD inception Jul 2002, fully covers backtest
- Universe: SPY/QQQ/IWD/TLT/IEF/LQD/GLD/GSG (replace SHY with LQD)
- Manual: SPY 10%, QQQ 15%, IWD 8%, TLT 20%, IEF 10%, LQD 12%, GLD 15%, GSG 10%
- Question: does investment grade corporate exposure add value over pure government bonds?
- Note: LQD was previously replaced by TIP early in the project. Now testing 
  with better universe design and different role (alongside TIP rather than instead of it)

### B5: DJP replacing GSG (Bloomberg commodity index vs GSCI, 8 assets)
- DJP inception Jun 2006 — starts 5 months after GSG but still covers full backtest
- DJP tracks Bloomberg Commodity Index: more balanced (35% energy, 35% metals, 30% agriculture)
- GSG tracks GSCI: energy-heavy (70% energy)
- Universe: SPY/QQQ/IWD/TLT/IEF/SHY/GLD/DJP
- Manual: same as 8-asset baseline but DJP replacing GSG
- Question: does a more balanced commodity index outperform the energy-heavy GSCI?
- This is a direct apples-to-apples commodity index comparison

---

## Group C — ETF substitution spot-checks (restricted date ranges)

These are equivalence tests only. No walk-forward. No OOS. 
Just IS backtest to confirm the cheaper ETF tracks the same as the original.
Use a custom date range matching the shorter ETF's history.

### C1: GLD vs GLDM (2018-2026)
- GLDM inception Jun 2018
- Run parallel IS backtest: same allocation, same period, swap GLD for GLDM
- Expected: near-identical Calmar (within 0.02)
- If confirmed: safe to use GLDM in live implementation

### C2: SPY vs IVV (2006-2026, full range)
- IVV inception May 2000 — can run full backtest
- Full IS + OOS comparison is valid here unlike other substitutions
- Expected: near-identical results (IVV tracks same index)
- This one is worth the full pipeline given the full history

### C3: GSG vs PDBC (2014-2026)
- PDBC inception Nov 2014
- Run parallel IS backtest on 2014-2020 window only
- Key question: does PDBC's contango mitigation improve returns vs GSG?
- If PDBC Calmar is meaningfully higher: note this for live implementation

---

## Group D — Transaction cost sensitivity (2006-2026, current 8-asset)

These use the current validated 8-asset allocation, no new tickers.
Purpose: understand at what cost level the strategy stops beating 60/40.

### D1: TRANSACTION_COST_PCT = 0.0005 (0.05% — zero-commission broker, just spread)
### D2: TRANSACTION_COST_PCT = 0.001  (0.1% — realistic UK retail with FX)
### D3: TRANSACTION_COST_PCT = 0.005  (0.5% — traditional broker)

No optimisation or walk-forward needed here — just full_backtest mode to compare
final values and Calmars across cost levels. Quick to run.

---

## Execution order and rationale

Run in this order:

1. Group D first (transaction costs) — fastest, uses existing validated allocation,
   no new data downloads, answers an important pre-paper-trading question immediately.

2. Group C next (ETF substitution checks) — fast IS-only backtests, clears the
   ETF equivalence question before running new universes.

3. Group A (6asset_tip_gsg robustness) — highest priority follow-up on the 
   best result from Phase 1. Needs custom IS/OOS dates.

4. Group B (new asset classes) — full pipeline, longest to run, do overnight.

Total estimated runtime:
- Group D: ~5 minutes (3 full backtests, no optimisation)
- Group C: ~10 minutes (spot-checks only)
- Group A: ~2 hours (full pipeline x2 with different date splits)
- Group B: ~5 hours (full pipeline x5)