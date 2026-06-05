"""
lot_ledger.py — backtest tax-lot ledger with pluggable selection (D.14).

Tracks open tax lots per ticker during a backtest and, on a sale, hands the
disposed portion to a *selector* that decides which lots are consumed. The
selector emits ``engine.tax.RealizedLot`` objects, which become
``SaleEvent.lots`` for ``engine.tax.compute_tax_on_event`` (D.13).

Selectors
---------
* ``FIFO``         oldest lots first. Matches Alpaca's Compressed FIFO — i.e.
                   broker reality. **This is the default and the only one that
                   is achievable live on Alpaca.**
* ``HIFO``         highest cost-basis first → minimizes the realized gain.
* ``TAX_OPTIMAL``  long-term lots first, then highest basis. Prefers the lower
                   LT rate and the smallest gain. **Research counterfactual
                   only on Alpaca** (orders can't carry a lot id — see
                   ``research/tax_drift_trigger/findings_alpaca_lot_selection.md``).

Relationship to ``live/lots.py``
--------------------------------
The ``Lot`` shape here (``qty``, per-share ``price``, ``acquired_on``) mirrors
``live/lots.py`` so the backtest and the live ledger agree on FIFO ordering and
cost-basis bookkeeping. It is re-declared here rather than imported because the
engine must not depend on ``live`` (engine is pure backtest math). The live
ledger additionally persists to JSON and enforces a 31-day hold; this one is
in-memory per backtest run and has no holding-period gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from .tax import LONG_TERM_THRESHOLD_DAYS, RealizedLot, SaleEvent

# Quantity tolerances. ``_QTY_TOL`` treats sub-nano share counts as zero (matches
# live/lots.py). ``_OVERSELL_TOL`` is the slack allowed when selling "the whole
# position" before we treat the request as an oversell bug.
_QTY_TOL = 1e-9
_OVERSELL_TOL = 1e-6


@dataclass
class Lot:
    """One open tax lot. Mirrors ``live/lots.py`` Lot.

    ``price`` is the per-share acquisition cost basis (so total basis is
    ``qty * price``). Lots are stored oldest-first per ticker.
    """

    qty: float
    price: float          # per-share cost basis
    acquired_on: date


class LotSelector(Enum):
    """Which lots a sale consumes first."""

    FIFO = "fifo"               # oldest first (Alpaca broker reality; default)
    HIFO = "hifo"               # highest cost basis first (minimize gain)
    TAX_OPTIMAL = "tax_optimal" # long-term first, then highest basis (research-only on Alpaca)

    @classmethod
    def coerce(cls, value: "LotSelector | str") -> "LotSelector":
        """Accept an enum member or its string value (e.g. ``"fifo"``)."""
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


def _order_lots(lots: list[Lot], sale_date: date, selector: LotSelector) -> list[Lot]:
    """Return ``lots`` in the order the selector would consume them.

    Sorting is stable, so FIFO ties (identical ``acquired_on``) keep insertion
    order. HIFO and TAX_OPTIMAL add ``acquired_on`` as a deterministic tiebreak.
    """
    if selector is LotSelector.FIFO:
        return sorted(lots, key=lambda lot: lot.acquired_on)
    if selector is LotSelector.HIFO:
        return sorted(lots, key=lambda lot: (-lot.price, lot.acquired_on))
    if selector is LotSelector.TAX_OPTIMAL:
        def key(lot: Lot):
            is_long_term = (sale_date - lot.acquired_on).days > LONG_TERM_THRESHOLD_DAYS
            return (0 if is_long_term else 1, -lot.price, lot.acquired_on)
        return sorted(lots, key=key)
    raise ValueError(f"Unknown lot selector: {selector!r}")


class LotLedger:
    """In-memory tax-lot ledger for a single backtest run.

    Parameters
    ----------
    selector:
        Default selection policy for sales. Overridable per ``sell`` call.
    """

    def __init__(self, selector: "LotSelector | str" = LotSelector.FIFO) -> None:
        self._lots: dict[str, list[Lot]] = {}
        self.selector = LotSelector.coerce(selector)

    # -- mutations ----------------------------------------------------------

    def buy(self, ticker: str, qty: float, price: float, acquired_on: date) -> None:
        """Record a buy as a new lot (newest appended last, oldest-first list)."""
        if qty <= _QTY_TOL:
            return
        self._lots.setdefault(ticker, []).append(
            Lot(qty=float(qty), price=float(price), acquired_on=acquired_on)
        )

    def sell(
        self,
        ticker: str,
        qty: float,
        price: float,
        sale_date: date,
        selector: "LotSelector | str | None" = None,
    ) -> list[RealizedLot]:
        """Dispose ``qty`` shares of ``ticker`` at ``price`` on ``sale_date``.

        Consumes lots per the selector, mutating the ledger, and returns the
        ``RealizedLot`` list describing the disposition. Raises ``ValueError``
        on an oversell (selling more than is held, beyond float slack).
        """
        sel = self.selector if selector is None else LotSelector.coerce(selector)
        lots = self._lots.get(ticker, [])
        held = sum(lot.qty for lot in lots)
        if qty - held > _OVERSELL_TOL:
            raise ValueError(
                f"Oversell of {ticker}: requested {qty}, held {held}."
            )

        realized: list[RealizedLot] = []
        remaining = float(qty)
        for lot in _order_lots(lots, sale_date, sel):
            if remaining <= _QTY_TOL:
                break
            take = min(lot.qty, remaining)
            realized.append(
                RealizedLot(
                    ticker=ticker,
                    quantity=take,
                    proceeds=take * price,
                    cost_basis=take * lot.price,
                    acquired=lot.acquired_on,
                    disposed=sale_date,
                )
            )
            lot.qty -= take
            remaining -= take

        # Purge emptied lots, preserving the surviving order; drop empty tickers.
        survivors = [lot for lot in lots if lot.qty > _QTY_TOL]
        if survivors:
            self._lots[ticker] = survivors
        else:
            self._lots.pop(ticker, None)
        return realized

    def sell_event(
        self,
        ticker: str,
        qty: float,
        price: float,
        sale_date: date,
        selector: "LotSelector | str | None" = None,
        *,
        is_mark_to_market: bool = False,
    ) -> SaleEvent:
        """Like :meth:`sell` but wraps the result in a ``SaleEvent`` for D.16."""
        lots = self.sell(ticker, qty, price, sale_date, selector)
        return SaleEvent(
            ticker=ticker, lots=lots, disposed=sale_date,
            is_mark_to_market=is_mark_to_market,
        )

    # -- queries ------------------------------------------------------------

    def total_quantity(self, ticker: str) -> float:
        return sum(lot.qty for lot in self._lots.get(ticker, []))

    def total_cost_basis(self, ticker: str) -> float:
        return sum(lot.qty * lot.price for lot in self._lots.get(ticker, []))

    def average_cost(self, ticker: str) -> float | None:
        qty = self.total_quantity(ticker)
        return self.total_cost_basis(ticker) / qty if qty > _QTY_TOL else None

    def unrealized_gain(self, ticker: str, price: float) -> float:
        return self.total_quantity(ticker) * price - self.total_cost_basis(ticker)

    def lots(self, ticker: str) -> list[Lot]:
        """A copy of the open lots for ``ticker`` (oldest-first)."""
        return list(self._lots.get(ticker, []))

    def tickers(self) -> list[str]:
        return list(self._lots.keys())

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._lots

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        held = {t: round(self.total_quantity(t), 6) for t in self._lots}
        return f"LotLedger(selector={self.selector.value}, held={held})"
