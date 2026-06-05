# Tax model — US-taxable individual (D.12 / D.13 / D.16)

**Status:** implemented, research-grade. Backed by `engine/tax_rates_us.yaml`,
`engine/tax.py`, `engine/tax_backtest.py`. Tests: `tests/test_tax_schedule.py`,
`tests/test_tax_events.py`, `tests/test_tax_backtest.py`.

## Question

How much does US federal tax actually cost the All Weather strategy, and does
the answer depend on (a) how often we rebalance and (b) which tax lots we sell?
The production backtest had only a flat `tax_drag_pct` knob — too coarse to
reason about lot selection, holding periods, or per-asset characterization.

## What is modelled

**Reference taxpayer:** US-taxable individual at the **top marginal bracket**.
This is the project's standard reference (handoff decision §3.1). ISA / PT /
cross-border treatment is explicitly out of scope.

### Rate schedule (`engine/tax_rates_us.yaml`, D.12)

Year-keyed top federal marginal rates, bisect-by-date lookup (latest
`effective ≤ date` wins; querying before the earliest entry raises rather than
extrapolate). Three regimes cover the 2006→present window:

| Effective | Ordinary / ST | LT cap-gain | Qualified div | NIIT |
|---|---|---|---|---|
| 2003-01-01 (JGTRRA) | 35.0% | 15% | 15% | 0% |
| 2013-01-01 (ATRA + ACA) | 39.6% | 20% | 20% | 3.8% |
| 2018-01-01 (TCJA) | 37.0% | 20% | 20% | 3.8% |

The 2018 entry carries through 2026 unchanged: **OBBBA (July 2025) made the
TCJA 37% top rate permanent**, verified against Tax Foundation / IRS. NIIT is
stored separately (not pre-added) so the caller decides whether the high-earner
threshold applies; the reference taxpayer always meets it (`apply_niit=True`).

### Per-asset characterization (`engine/tax.py`, D.13)

The four-field schedule is the *general* table. Two universe assets are taxed
under special-characterization rules that override it — these are encoded as
constants, not in the schedule, because they don't vary by date:

| Asset(s) | Class | Treatment |
|---|---|---|
| SPY, QQQ | equity | qualified dividends; standard ST/LT gains |
| TLT, TIP | bond | **ordinary**-income distributions (interest); standard gains |
| GLD, GLDM | collectible | LT gains capped at **28%** (IRC §408(m)), not 20%; no cash distributions |
| GSG | §1256 | gains split **60% LT / 40% ST regardless of holding period**; annual mark-to-market |

Worked example — $1,000 long-term gain, 2020, top bracket:
SPY **$238** (20% + 3.8%), GLD **$318** (28% + 3.8%), GSG **$306**
(60/40 → 0.6·20% + 0.4·37% + 3.8% on the whole), TLT **$238**.

### Realized-gain accounting (`engine/tax_backtest.py`, D.16)

`run_tax_aware_backtest` is a **separate** monthly engine (the share-based
`engine/backtest.py` is golden-locked; see `tests/test_backtest_golden.py`). It:

- rebalances per a `RebalancePolicy` (D.15), selling with a `LotSelector` (D.14);
- taxes realized gains per sale via `compute_tax_on_event`;
- taxes dividends at ex-date (qualified vs ordinary), sized on the held share
  count at the time;
- fires the **GSG §1256 year-end mark-to-market** as a deemed sale on the last
  rebalance date of each calendar year (basis reset, share count unchanged);
- pays tax + transaction cost by scaling all lots down pro-rata (preserves
  target weights and per-share basis).

## Key validation

With `TaxRegime.none()` and zero cost, the tax-aware engine reproduces the
share-based monthly engine's value path **to 1e-6** (FMP adj_close, 2006-2024:
both end at **29,684.69**). Layering US tax then strictly reduces value, and
`tax_optimal` lot selection pays **no more** tax than FIFO. These are pinned in
`tests/test_tax_backtest.py`.

## Decisions / caveats

1. **`allow_loss_offset=True`** (default): realized losses and §1256 down-year
   MTM produce tax *credits* at the marginal rate (assumes other gains absorb
   them). This means cumulative tax paid is **not monotonic** — that is correct,
   not a bug. The $3k capital-loss limit and carryforwards are not modelled.
2. **FIFO is broker reality on Alpaca.** `tax_optimal`/`HIFO` are research
   counterfactuals only — Alpaca orders cannot select lots
   (`findings_alpaca_lot_selection.md`). Every artifact is tagged with the
   selector used so results are never compared across selectors by accident.
3. **Monthly granularity.** Tax is computed at month-end; dividends are sized on
   month-end share counts. Adequate for a monthly-rebalanced strategy.
4. **Total-return prices.** Dividend cash is already in the price series as
   appreciation; the model subtracts only the *tax* a taxable holder would owe
   on the distribution, never double-counting the cash.

## Files

- `engine/tax_rates_us.yaml` — rate schedule (data)
- `engine/tax.py` — `TaxRates`, `TaxSchedule`, `AssetTaxClass`, `TaxRegime`,
  `RealizedLot`/`SaleEvent`/`DividendEvent`, `compute_tax_on_event`
- `engine/lot_ledger.py` — `LotLedger` + `LotSelector` (D.14)
- `engine/tax_backtest.py` — `run_tax_aware_backtest` (D.16)
