"""
tax.py — US federal investment-income tax model.

Two layers live here:

D.12 — the rate *schedule* and a bisect-by-date lookup. Loads
``engine/tax_rates_us.yaml`` and answers "what were the top federal marginal
rates in effect on date X?" (``TaxRates``, ``TaxSchedule``, ``us_tax_schedule``).

D.13 — the *characterization* layer on top of the schedule: per-ETF tax class,
``TaxRegime``, taxable-event types, and ``compute_tax_on_event``. This turns a
realized disposition or a dividend into the tax owed, applying the asset-class
overrides that the four-field schedule deliberately does NOT encode:

  * Collectibles (GLD/GLDM physical-gold grantor trusts) — long-term gains are
    capped at the 28% collectibles rate (IRC §408(m)), not the 20% LT top.
  * §1256 contracts (GSG, futures-based) — gains are split 60% long-term / 40%
    short-term regardless of holding period, with annual mark-to-market. The
    60/40 split is applied here; the year-end MTM *timing* is the rebalance
    loop's job (D.16), which emits a deemed-sale event each Dec 31.

Out of scope here (lands later): lot *selection* (FIFO/HIFO/tax_optimal) is
D.14's ``engine/lot_ledger.py`` — its selector produces the ``RealizedLot``
list carried by a ``SaleEvent``. The rebalance-loop wiring is D.16.

Design notes
------------
* US FEDERAL top marginal rates only. State tax / AMT / bracket phase-ins are
  not modelled. See the YAML header for the full caveat list.
* ``short_term_top`` is the ordinary-income top rate. It covers BOTH short-term
  capital gains and non-qualified (ordinary) dividends / bond interest, so
  there is deliberately no separate "ordinary dividend" field.
* ``niit`` (Net Investment Income Tax) is stored separately, not pre-added, so
  the caller decides whether the high-earner NIIT threshold applies. The
  ``*_with_niit`` convenience properties give the additive combination.
* Lookup is bisect-by-date: the entry with the latest ``effective`` date that
  is <= the query date wins. Querying a date before the earliest entry raises,
  rather than silently extrapolating an unknown tax regime.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from functools import lru_cache
from typing import Iterable, Mapping, Sequence, Union

import yaml

DEFAULT_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "tax_rates_us.yaml")

DateLike = Union[str, date, datetime]


@dataclass(frozen=True)
class TaxRates:
    """Top US federal marginal rates applicable on a given date (decimals).

    Attributes
    ----------
    short_term_top:
        Ordinary-income top rate. Applies to short-term capital gains AND
        non-qualified (ordinary) dividends / bond interest (e.g. TLT, TIP).
    long_term_top:
        Top rate on long-term capital gains (held > 1 year).
    qualified_dividend_top:
        Top rate on qualified dividends (e.g. SPY, QQQ).
    niit:
        Net Investment Income Tax surtax (ACA, 2013+). Additive to the rate
        above for high earners; 0.0 before 2013.
    """

    short_term_top: float
    long_term_top: float
    qualified_dividend_top: float
    niit: float

    @property
    def short_term_with_niit(self) -> float:
        """Top ordinary/short-term rate including the NIIT surtax."""
        return self.short_term_top + self.niit

    @property
    def long_term_with_niit(self) -> float:
        """Top long-term capital-gains rate including the NIIT surtax."""
        return self.long_term_top + self.niit

    @property
    def qualified_dividend_with_niit(self) -> float:
        """Top qualified-dividend rate including the NIIT surtax."""
        return self.qualified_dividend_top + self.niit


def _to_date(value: DateLike) -> date:
    """Coerce a YAML date, datetime, date, or 'YYYY-MM-DD' string to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


class TaxSchedule:
    """Year-keyed US federal rate schedule with bisect-by-date lookup.

    The applicable entry for a query date is the one with the latest
    ``effective`` date <= the query date. Use :meth:`from_yaml` to build from
    ``engine/tax_rates_us.yaml`` or :func:`us_tax_schedule` for the cached
    default instance.
    """

    _REQUIRED_FIELDS = (
        "short_term_top",
        "long_term_top",
        "qualified_dividend_top",
        "niit",
    )

    def __init__(
        self,
        entries: Iterable[Mapping],
        meta: Mapping | None = None,
    ) -> None:
        ordered = sorted(
            ({"effective": _to_date(e["effective"]), **{f: float(e[f]) for f in self._REQUIRED_FIELDS}}
             for e in entries),
            key=lambda e: e["effective"],
        )
        if not ordered:
            raise ValueError("TaxSchedule requires at least one entry.")

        # Reject duplicate effective dates — they make the lookup ambiguous.
        effective = [e["effective"] for e in ordered]
        dupes = {d for d in effective if effective.count(d) > 1}
        if dupes:
            raise ValueError(
                f"Duplicate effective dates in tax schedule: "
                f"{sorted(d.isoformat() for d in dupes)}"
            )

        self._effective: list[date] = effective
        self._rates: list[TaxRates] = [
            TaxRates(
                short_term_top=e["short_term_top"],
                long_term_top=e["long_term_top"],
                qualified_dividend_top=e["qualified_dividend_top"],
                niit=e["niit"],
            )
            for e in ordered
        ]
        self.meta: dict = dict(meta or {})

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_SCHEDULE_PATH) -> "TaxSchedule":
        """Load a schedule from a ``tax_rates_us.yaml``-shaped file."""
        with open(path) as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, Mapping) or "schedule" not in doc:
            raise ValueError(f"{path}: expected a mapping with a 'schedule' key.")
        return cls(doc["schedule"], meta=doc.get("meta"))

    def rates_on(self, when: DateLike) -> TaxRates:
        """Return the :class:`TaxRates` in effect on ``when``.

        Raises
        ------
        ValueError
            If ``when`` precedes the earliest schedule entry — we do not
            extrapolate rates into an unmodelled tax regime.
        """
        q = _to_date(when)
        if q < self._effective[0]:
            raise ValueError(
                f"No tax-schedule entry on or before {q.isoformat()}; "
                f"earliest entry is {self._effective[0].isoformat()}."
            )
        # rightmost effective date <= q
        idx = bisect.bisect_right(self._effective, q) - 1
        return self._rates[idx]

    @property
    def effective_dates(self) -> list[date]:
        """Sorted effective dates of each regime entry."""
        return list(self._effective)

    def __len__(self) -> int:
        return len(self._effective)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        span = f"{self._effective[0].isoformat()}..{self._effective[-1].isoformat()}"
        return f"TaxSchedule({len(self)} entries, {span})"


@lru_cache(maxsize=1)
def us_tax_schedule() -> TaxSchedule:
    """Return the cached default US federal tax schedule from the bundled YAML."""
    return TaxSchedule.from_yaml()


# ===========================================================================
# D.13 — asset-class characterization, regime, taxable events, tax computation
# ===========================================================================

# Special-characterization constants (not date-varying across the backtest
# window, so they live as constants rather than in the schedule YAML).
COLLECTIBLES_LT_TOP = 0.28        # IRC §408(m): max LT rate on collectibles (gold)
SECTION_1256_LT_FRACTION = 0.60   # IRC §1256: 60% of gain taxed long-term ...
SECTION_1256_ST_FRACTION = 0.40   # ... and 40% short-term, regardless of holding
LONG_TERM_THRESHOLD_DAYS = 365    # IRS "more than one year" → strictly > 365 days


class AssetTaxClass(Enum):
    """How an asset's gains and distributions are characterized for US tax."""

    EQUITY = "equity"             # SPY, QQQ — qualified dividends; standard LT/ST gains
    BOND = "bond"                 # TLT, TIP — ordinary (interest) dist.; standard gains
    COLLECTIBLE = "collectible"   # GLD, GLDM — LT gains capped at 28%; no distributions
    SECTION_1256 = "section_1256" # GSG — 60/40 LT/ST split; annual mark-to-market


class DividendCharacter(Enum):
    QUALIFIED = "qualified"   # preferential rate (= qualified_dividend_top)
    ORDINARY = "ordinary"     # ordinary income rate (= short_term_top)


# Default ticker -> tax class for the All Weather universe (backtest tickers,
# plus the GLDM live substitute). Benchmarks such as JEPQ are NOT held by the
# strategy and so are not taxed by the rebalance loop; unknown tickers fall back
# to EQUITY (see ``TaxRegime.classify``).
DEFAULT_ASSET_TAX_CLASS: dict[str, AssetTaxClass] = {
    "SPY": AssetTaxClass.EQUITY,
    "QQQ": AssetTaxClass.EQUITY,
    "TLT": AssetTaxClass.BOND,
    "TIP": AssetTaxClass.BOND,
    "GLD": AssetTaxClass.COLLECTIBLE,
    "GLDM": AssetTaxClass.COLLECTIBLE,
    "GSG": AssetTaxClass.SECTION_1256,
}


def dividend_character(asset_class: AssetTaxClass) -> DividendCharacter:
    """Map an asset class to its distribution character.

    Equity dividends are treated as fully qualified (SPY/QQQ are ~100%
    qualified for a long-term holder — a documented simplification). Everything
    else (bond interest, and the no-dividend collectible / §1256 assets if they
    ever distribute) is treated as ordinary income.
    """
    return (
        DividendCharacter.QUALIFIED
        if asset_class is AssetTaxClass.EQUITY
        else DividendCharacter.ORDINARY
    )


@dataclass(frozen=True)
class RealizedLot:
    """One disposed lot — the output of lot selection (D.14) for a sale.

    ``cost_basis`` and ``proceeds`` are TOTALS for the lot (not per-share), so
    ``gain`` needs no quantity. Holding period is derived from the acquisition
    and disposal dates; for §1256 assets the holding period is ignored.
    """

    ticker: str
    quantity: float
    proceeds: float       # total sale proceeds for this lot
    cost_basis: float     # total original cost for this lot
    acquired: date
    disposed: date

    @property
    def gain(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def holding_days(self) -> int:
        return (self.disposed - self.acquired).days

    @property
    def is_long_term(self) -> bool:
        """True if held more than one year (strictly > 365 days)."""
        return self.holding_days > LONG_TERM_THRESHOLD_DAYS


@dataclass(frozen=True)
class SaleEvent:
    """A disposition of one ticker, composed of one or more selected lots."""

    ticker: str
    lots: Sequence[RealizedLot]
    disposed: date
    is_mark_to_market: bool = False  # True for §1256 year-end deemed sales (D.16)


@dataclass(frozen=True)
class DividendEvent:
    """A cash distribution received on a ticker, taxed at its ex-date."""

    ticker: str
    amount: float         # cash dividend received (always >= 0)
    ex_date: date


TaxableEvent = Union[SaleEvent, DividendEvent]


@dataclass(frozen=True)
class TaxResult:
    """Itemized tax for a single event. ``total_tax`` can be negative when a
    realized loss offsets gains (``allow_loss_offset``)."""

    ticker: str
    kind: str                 # "sale" | "dividend"
    asset_class: AssetTaxClass
    when: date
    short_term_gain: float = 0.0
    long_term_gain: float = 0.0
    short_term_tax: float = 0.0
    long_term_tax: float = 0.0
    dividend_income: float = 0.0
    dividend_tax: float = 0.0
    niit_tax: float = 0.0

    @property
    def realized_gain(self) -> float:
        return self.short_term_gain + self.long_term_gain

    @property
    def total_tax(self) -> float:
        return (
            self.short_term_tax
            + self.long_term_tax
            + self.dividend_tax
            + self.niit_tax
        )


@dataclass(frozen=True)
class TaxRegime:
    """A tax jurisdiction + policy applied to taxable events.

    ``us`` models the US-taxable individual at the top marginal bracket (the
    project's standard reference). ``none`` is the zero-tax baseline (e.g. an
    ISA) used for the A/B comparison in the D.18 sweep.
    """

    name: str
    schedule: TaxSchedule | None
    taxable: bool = True
    apply_niit: bool = True          # top-bracket reference → NIIT surtax on
    allow_loss_offset: bool = True   # realized losses offset gains at marginal rate
    asset_tax_class: Mapping[str, AssetTaxClass] = field(
        default_factory=lambda: dict(DEFAULT_ASSET_TAX_CLASS)
    )

    @classmethod
    def us(cls, *, apply_niit: bool = True, allow_loss_offset: bool = True) -> "TaxRegime":
        return cls(
            name="us",
            schedule=us_tax_schedule(),
            taxable=True,
            apply_niit=apply_niit,
            allow_loss_offset=allow_loss_offset,
        )

    @classmethod
    def none(cls) -> "TaxRegime":
        """A zero-tax regime (ISA-style). Every event yields zero tax."""
        return cls(
            name="none",
            schedule=None,
            taxable=False,
            apply_niit=False,
            allow_loss_offset=True,
        )

    def classify(self, ticker: str, default: AssetTaxClass = AssetTaxClass.EQUITY) -> AssetTaxClass:
        return self.asset_tax_class.get(ticker.upper(), default)


def _bucket_tax(gain: float, rate: float, allow_loss_offset: bool) -> float:
    """Tax on a gain bucket. Negative (loss) buckets yield a negative tax
    (a benefit) only when ``allow_loss_offset`` is set; otherwise zero."""
    if gain >= 0.0 or allow_loss_offset:
        return gain * rate
    return 0.0


def tax_on_sale(event: SaleEvent, regime: TaxRegime) -> TaxResult:
    """Compute tax on a disposition, applying the asset-class characterization.

    EQUITY/BOND: net long-term and short-term gains taxed at the schedule's LT
    and ordinary rates. COLLECTIBLE: long-term gains capped at 28%. SECTION_1256:
    holding period ignored; total gain split 60% LT / 40% ST.
    """
    asset_class = regime.classify(event.ticker)

    # Gains characterized into long-term / short-term buckets.
    if asset_class is AssetTaxClass.SECTION_1256:
        total_gain = sum(lot.gain for lot in event.lots)
        lt_gain = total_gain * SECTION_1256_LT_FRACTION
        st_gain = total_gain * SECTION_1256_ST_FRACTION
    else:
        lt_gain = sum(lot.gain for lot in event.lots if lot.is_long_term)
        st_gain = sum(lot.gain for lot in event.lots if not lot.is_long_term)

    if not regime.taxable or regime.schedule is None:
        return TaxResult(
            ticker=event.ticker, kind="sale", asset_class=asset_class,
            when=event.disposed, short_term_gain=st_gain, long_term_gain=lt_gain,
        )

    rates = regime.schedule.rates_on(event.disposed)
    niit_rate = rates.niit if regime.apply_niit else 0.0

    # Long-term rate: collectibles are capped at min(ordinary, 28%).
    if asset_class is AssetTaxClass.COLLECTIBLE:
        lt_rate = min(rates.short_term_top, COLLECTIBLES_LT_TOP)
    else:
        lt_rate = rates.long_term_top
    st_rate = rates.short_term_top

    lt_tax = _bucket_tax(lt_gain, lt_rate, regime.allow_loss_offset)
    st_tax = _bucket_tax(st_gain, st_rate, regime.allow_loss_offset)
    niit_tax = _bucket_tax(lt_gain + st_gain, niit_rate, regime.allow_loss_offset)

    return TaxResult(
        ticker=event.ticker, kind="sale", asset_class=asset_class, when=event.disposed,
        short_term_gain=st_gain, long_term_gain=lt_gain,
        short_term_tax=st_tax, long_term_tax=lt_tax, niit_tax=niit_tax,
    )


def tax_on_dividend(event: DividendEvent, regime: TaxRegime) -> TaxResult:
    """Compute tax on a cash distribution at its ex-date.

    Qualified (equity) dividends are taxed at the preferential rate; ordinary
    (bond interest) at the ordinary rate. NIIT applies to both for the
    top-bracket reference investor.
    """
    asset_class = regime.classify(event.ticker)

    if not regime.taxable or regime.schedule is None:
        return TaxResult(
            ticker=event.ticker, kind="dividend", asset_class=asset_class,
            when=event.ex_date, dividend_income=event.amount,
        )

    rates = regime.schedule.rates_on(event.ex_date)
    niit_rate = rates.niit if regime.apply_niit else 0.0

    if dividend_character(asset_class) is DividendCharacter.QUALIFIED:
        div_rate = rates.qualified_dividend_top
    else:
        div_rate = rates.short_term_top

    return TaxResult(
        ticker=event.ticker, kind="dividend", asset_class=asset_class, when=event.ex_date,
        dividend_income=event.amount,
        dividend_tax=event.amount * div_rate,
        niit_tax=event.amount * niit_rate,
    )


def compute_tax_on_event(event: TaxableEvent, regime: TaxRegime) -> TaxResult:
    """Dispatch a taxable event to the right computation.

    Note on signature: the handoff sketched ``compute_tax_on_event(event,
    lot_ledger, regime)``. Lot *selection* is D.14's job (``engine/lot_ledger``);
    its selector emits the ``RealizedLot`` list already carried by the
    ``SaleEvent``, so the ledger is not a separate argument here.
    """
    if isinstance(event, SaleEvent):
        return tax_on_sale(event, regime)
    if isinstance(event, DividendEvent):
        return tax_on_dividend(event, regime)
    raise TypeError(f"Unsupported taxable event type: {type(event).__name__}")
