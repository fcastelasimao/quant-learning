"""
test_tax_events.py
==================
Tests for the D.13 characterization layer in engine/tax.py: asset-class
classification, TaxRegime, taxable-event types, and compute_tax_on_event.

All rate facts use the TCJA regime (2018+) unless a date says otherwise:
ordinary/ST 0.37, LT 0.20, qualified-div 0.20, NIIT 0.038. The key
asset-class overrides under test:
  * GLD long-term gain capped at 28% (collectibles), not 20%.
  * GSG split 60/40 LT/ST regardless of holding period (§1256).
"""

from datetime import date, timedelta

import pytest

from engine.tax import (
    AssetTaxClass,
    DividendCharacter,
    DividendEvent,
    RealizedLot,
    SaleEvent,
    TaxRegime,
    compute_tax_on_event,
    dividend_character,
    tax_on_dividend,
    tax_on_sale,
)

US = TaxRegime.us()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lot(ticker, gain, *, long_term, disposed=date(2020, 6, 30), cost_basis=1000.0):
    """Build a one-lot disposition with a chosen gain and holding period.

    ``acquired`` is derived from ``disposed`` so the holding period is correct
    for any disposal year (400 days back = long-term; 60 days = short-term).
    """
    acquired = disposed - timedelta(days=400 if long_term else 60)
    return RealizedLot(
        ticker=ticker, quantity=1.0,
        proceeds=cost_basis + gain, cost_basis=cost_basis,
        acquired=acquired, disposed=disposed,
    )


def _sale(ticker, gain, *, long_term, disposed=date(2020, 6, 30)):
    return SaleEvent(
        ticker=ticker,
        lots=[_lot(ticker, gain, long_term=long_term, disposed=disposed)],
        disposed=disposed,
    )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ticker, klass",
    [
        ("SPY", AssetTaxClass.EQUITY),
        ("QQQ", AssetTaxClass.EQUITY),
        ("TLT", AssetTaxClass.BOND),
        ("TIP", AssetTaxClass.BOND),
        ("GLD", AssetTaxClass.COLLECTIBLE),
        ("GLDM", AssetTaxClass.COLLECTIBLE),
        ("GSG", AssetTaxClass.SECTION_1256),
        ("gld", AssetTaxClass.COLLECTIBLE),  # case-insensitive
    ],
)
def test_classify_universe(ticker, klass):
    assert US.classify(ticker) is klass


def test_classify_unknown_defaults_equity():
    assert US.classify("ZZZZ") is AssetTaxClass.EQUITY


def test_dividend_character_mapping():
    assert dividend_character(AssetTaxClass.EQUITY) is DividendCharacter.QUALIFIED
    assert dividend_character(AssetTaxClass.BOND) is DividendCharacter.ORDINARY
    assert dividend_character(AssetTaxClass.COLLECTIBLE) is DividendCharacter.ORDINARY


# ---------------------------------------------------------------------------
# equity / bond capital gains (standard rates)
# ---------------------------------------------------------------------------

def test_equity_long_term_gain():
    r = tax_on_sale(_sale("SPY", 1000.0, long_term=True), US)
    assert r.long_term_gain == pytest.approx(1000.0)
    assert r.short_term_gain == 0.0
    assert r.long_term_tax == pytest.approx(200.0)   # 20%
    assert r.niit_tax == pytest.approx(38.0)         # 3.8%
    assert r.total_tax == pytest.approx(238.0)


def test_equity_short_term_gain():
    r = tax_on_sale(_sale("SPY", 1000.0, long_term=False), US)
    assert r.short_term_tax == pytest.approx(370.0)  # 37%
    assert r.niit_tax == pytest.approx(38.0)
    assert r.total_tax == pytest.approx(408.0)


def test_bond_long_term_gain_uses_standard_lt_rate():
    """Bonds differ from equities only on dividends; LT gains use the 20% rate."""
    r = tax_on_sale(_sale("TLT", 1000.0, long_term=True), US)
    assert r.total_tax == pytest.approx(238.0)


# ---------------------------------------------------------------------------
# collectibles (GLD): 28% LT cap
# ---------------------------------------------------------------------------

def test_collectible_long_term_capped_at_28():
    r = tax_on_sale(_sale("GLD", 1000.0, long_term=True), US)
    assert r.long_term_tax == pytest.approx(280.0)   # 28%, NOT 20%
    assert r.niit_tax == pytest.approx(38.0)
    assert r.total_tax == pytest.approx(318.0)
    # explicit contrast with an equity of identical gain/holding
    assert r.total_tax > tax_on_sale(_sale("SPY", 1000.0, long_term=True), US).total_tax


def test_collectible_short_term_is_ordinary():
    """Short-term collectible gains get no special treatment — ordinary rate."""
    r = tax_on_sale(_sale("GLD", 1000.0, long_term=False), US)
    assert r.short_term_tax == pytest.approx(370.0)
    assert r.total_tax == pytest.approx(408.0)


def test_collectible_cap_binds_even_pre_2013():
    """In 2010 ordinary top = 35%; the 28% collectibles cap still binds, NIIT 0."""
    r = tax_on_sale(_sale("GLD", 1000.0, long_term=True, disposed=date(2010, 6, 30)), US)
    assert r.long_term_tax == pytest.approx(280.0)
    assert r.niit_tax == 0.0
    assert r.total_tax == pytest.approx(280.0)


# ---------------------------------------------------------------------------
# §1256 (GSG): 60/40 regardless of holding period + annual mark-to-market
# ---------------------------------------------------------------------------

def test_section_1256_sixty_forty_split():
    r = tax_on_sale(_sale("GSG", 1000.0, long_term=True), US)
    assert r.long_term_gain == pytest.approx(600.0)
    assert r.short_term_gain == pytest.approx(400.0)
    assert r.long_term_tax == pytest.approx(120.0)   # 600 * 20%
    assert r.short_term_tax == pytest.approx(148.0)  # 400 * 37%
    assert r.niit_tax == pytest.approx(38.0)
    assert r.total_tax == pytest.approx(306.0)


def test_section_1256_holding_period_irrelevant():
    """A short-held §1256 lot is taxed identically to a long-held one."""
    lt = tax_on_sale(_sale("GSG", 1000.0, long_term=True), US)
    st = tax_on_sale(_sale("GSG", 1000.0, long_term=False), US)
    assert lt.total_tax == pytest.approx(st.total_tax) == pytest.approx(306.0)


# ---------------------------------------------------------------------------
# dividends
# ---------------------------------------------------------------------------

def test_qualified_dividend_equity():
    r = tax_on_dividend(DividendEvent("SPY", 100.0, date(2020, 3, 20)), US)
    assert r.dividend_tax == pytest.approx(20.0)   # 20% qualified
    assert r.niit_tax == pytest.approx(3.8)
    assert r.total_tax == pytest.approx(23.8)


def test_ordinary_dividend_bond():
    r = tax_on_dividend(DividendEvent("TLT", 100.0, date(2020, 3, 20)), US)
    assert r.dividend_tax == pytest.approx(37.0)   # 37% ordinary
    assert r.niit_tax == pytest.approx(3.8)
    assert r.total_tax == pytest.approx(40.8)


# ---------------------------------------------------------------------------
# loss offset policy
# ---------------------------------------------------------------------------

def test_loss_offsets_when_enabled():
    r = tax_on_sale(_sale("SPY", -1000.0, long_term=True), US)
    assert r.long_term_tax == pytest.approx(-200.0)
    assert r.niit_tax == pytest.approx(-38.0)
    assert r.total_tax == pytest.approx(-238.0)


def test_loss_ignored_when_disabled():
    regime = TaxRegime.us(allow_loss_offset=False)
    r = tax_on_sale(_sale("SPY", -1000.0, long_term=True), regime)
    assert r.total_tax == 0.0
    assert r.long_term_gain == pytest.approx(-1000.0)  # gain still recorded


def test_mixed_lots_net_within_buckets():
    """A long-term gain lot and a short-term loss lot net within their buckets."""
    event = SaleEvent(
        ticker="SPY",
        lots=[
            _lot("SPY", 1000.0, long_term=True),
            _lot("SPY", -400.0, long_term=False),
        ],
        disposed=date(2020, 6, 30),
    )
    r = tax_on_sale(event, US)
    assert r.long_term_gain == pytest.approx(1000.0)
    assert r.short_term_gain == pytest.approx(-400.0)
    assert r.long_term_tax == pytest.approx(200.0)
    assert r.short_term_tax == pytest.approx(-148.0)   # -400 * 37%
    assert r.niit_tax == pytest.approx((1000.0 - 400.0) * 0.038)


# ---------------------------------------------------------------------------
# NIIT toggle and pre-2013 regime
# ---------------------------------------------------------------------------

def test_niit_can_be_disabled():
    regime = TaxRegime.us(apply_niit=False)
    r = tax_on_sale(_sale("SPY", 1000.0, long_term=True), regime)
    assert r.niit_tax == 0.0
    assert r.total_tax == pytest.approx(200.0)


def test_pre_2013_regime_has_no_niit():
    r = tax_on_sale(_sale("SPY", 1000.0, long_term=True, disposed=date(2010, 6, 30)), US)
    assert r.long_term_tax == pytest.approx(150.0)  # 15% LT in 2010
    assert r.niit_tax == 0.0


# ---------------------------------------------------------------------------
# zero-tax regime
# ---------------------------------------------------------------------------

def test_none_regime_zero_tax_but_records_gain():
    none = TaxRegime.none()
    s = tax_on_sale(_sale("GLD", 1000.0, long_term=True), none)
    assert s.total_tax == 0.0
    assert s.long_term_gain == pytest.approx(1000.0)
    d = tax_on_dividend(DividendEvent("TLT", 100.0, date(2020, 3, 20)), none)
    assert d.total_tax == 0.0
    assert d.dividend_income == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# holding-period boundary and dispatch
# ---------------------------------------------------------------------------

def test_long_term_boundary_is_strictly_more_than_365_days():
    exactly_365 = RealizedLot("SPY", 1.0, 1100.0, 1000.0, date(2019, 1, 1), date(2020, 1, 1))
    over_365 = RealizedLot("SPY", 1.0, 1100.0, 1000.0, date(2019, 1, 1), date(2020, 1, 2))
    assert exactly_365.holding_days == 365 and exactly_365.is_long_term is False
    assert over_365.holding_days == 366 and over_365.is_long_term is True


def test_compute_tax_on_event_dispatches_both_types():
    sale = _sale("SPY", 1000.0, long_term=True)
    div = DividendEvent("SPY", 100.0, date(2020, 3, 20))
    assert compute_tax_on_event(sale, US).total_tax == pytest.approx(238.0)
    assert compute_tax_on_event(div, US).total_tax == pytest.approx(23.8)


def test_compute_tax_on_event_rejects_unknown_type():
    with pytest.raises(TypeError, match="Unsupported taxable event"):
        compute_tax_on_event(object(), US)
