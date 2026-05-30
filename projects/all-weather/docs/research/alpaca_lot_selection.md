# Alpaca lot-level order support — capability check

**Question:** Can we instruct Alpaca to sell *specific* tax lots at order time so the live system can mirror the backtest's tax-optimal selector (prefer long-term lots, then highest cost basis)?

**Verified against:** `alpaca-py 0.43.2`, Alpaca Trading API + Broker API docs, as of 2026-05-29.

## Short answer

**No.** Alpaca does not accept a lot ID, cost-basis method, or tax-relief method on order placement. Order sells are matched to lots by Alpaca's internal algorithm, which is **Compressed FIFO for end-of-day positions** and Weighted Average for intraday.

## Evidence

- `alpaca.trading.requests.MarketOrderRequest` exposes `symbol`, `qty`, `notional`, `side`, `type`, `time_in_force`, `order_class`, `extended_hours`, `client_order_id`, `legs`, `take_profit`, `stop_loss`, `position_intent`. **No `lot_id`, no `tax_lot_method`, no `cost_basis_method`.** (`alpaca-py 0.43.2`)
- Alpaca Docs — Position Average Entry Price Calculation: "Weighted Average is used for intraday positions, and Compressed FIFO is used for end-of-day positions."
- GitHub Issue [alpacahq/Alpaca-API#147](https://github.com/alpacahq/Alpaca-API/issues/147) ("Feature Request: LIFO cost basis option") — opened 2020-12-23, still open.
- IRS rule: Specific Identification requires contemporaneous documentation **before** the trade executes. Even if Alpaca's API supported lot-by-lot order placement, the user (not the broker) would still need to log the intent before submission.

## Implications for the plan

| Layer | Implication |
|---|---|
| **Backtest** | Tax-optimal lot selector (`engine/lot_ledger.py` planned in D.14) is a legitimate research counterfactual. Run it as `LotSelector.tax_optimal` to quantify the upside. Compare against `LotSelector.fifo` (= broker reality on Alpaca). |
| **Live shadow** | When comparing `live` (real fills) to `engine` (simulation), the engine must be configured with `LotSelector.fifo` to be apples-to-apples with Alpaca. |
| **Live execution** | The live system cannot directly instruct lot selection. Calibrate expectations: backtests using `tax_optimal` overstate what's achievable on Alpaca. |
| **Future** | If `tax_optimal` materially beats `fifo` in OOS research, the gap is the **broker-switching argument** — Interactive Brokers, Tastytrade, and several others *do* let you specify lots on sells. Becomes a real cost-of-staying-with-Alpaca number. |

## What we still get on Alpaca

- **31-day holding-period gate** (in `live/lots.py`) is enforced client-side — we choose not to sell young lots even when the broker would let us. Independent of Alpaca's cost-basis math.
- **End-of-year cost basis on Form 1099** uses Alpaca's Compressed FIFO, which matches our client-side `live/lots.py` FIFO ledger to ~the penny once we initialise the ledger from broker reality.

## Action items

- D.14 lot ledger: support `fifo` (default) and `tax_optimal` selectors. Mark `tax_optimal` as **research-only** for Alpaca users in the docstring.
- C.10 engine/data.py refactor: nothing to change here.
- Live shadow (G.27): default the simulation to `fifo` for the live comparison; expose `tax_optimal` only as a side panel labelled "research counterfactual".
- Backtest tax model (D.13): tag every artifact with the lot selector used so we don't accidentally compare across them later.

## Re-check date

Revisit this finding **2026-12-01** (after end-of-year tax artefacts are reviewed) and again whenever `alpaca-py` ships a major bump. Issue #147 closing would be the canary.
