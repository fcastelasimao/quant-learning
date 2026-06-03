"""
test_lot_ledger.py
==================
Tests for engine/lot_ledger.py (D.14): the backtest tax-lot ledger and its
FIFO / HIFO / tax_optimal selectors. Covers lot bookkeeping, the selection
order each policy produces, proceeds/cost-basis correctness on the emitted
RealizedLots, oversell guarding, and the hand-off into the D.13 tax model.
"""

from datetime import date

import pytest

from engine.lot_ledger import Lot, LotLedger, LotSelector, _order_lots
from engine.tax import SaleEvent, TaxRegime, compute_tax_on_event


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------

def test_buy_accumulates_quantity_and_basis():
    led = LotLedger()
    led.buy("SPY", 10, 100.0, date(2020, 1, 1))
    led.buy("SPY", 5, 120.0, date(2020, 2, 1))
    assert led.total_quantity("SPY") == pytest.approx(15)
    assert led.total_cost_basis("SPY") == pytest.approx(10 * 100 + 5 * 120)
    assert led.average_cost("SPY") == pytest.approx((1000 + 600) / 15)
    assert "SPY" in led and led.tickers() == ["SPY"]


def test_buy_nonpositive_is_noop():
    led = LotLedger()
    led.buy("SPY", 0, 100.0, date(2020, 1, 1))
    led.buy("SPY", -3, 100.0, date(2020, 1, 1))
    assert "SPY" not in led
    assert led.total_quantity("SPY") == 0.0


def test_average_cost_none_when_empty():
    assert LotLedger().average_cost("SPY") is None


def test_unrealized_gain():
    led = LotLedger()
    led.buy("SPY", 10, 100.0, date(2020, 1, 1))
    assert led.unrealized_gain("SPY", 150.0) == pytest.approx(500.0)


def test_lots_returns_a_copy():
    led = LotLedger()
    led.buy("SPY", 10, 100.0, date(2020, 1, 1))
    snapshot = led.lots("SPY")
    snapshot.clear()
    assert led.total_quantity("SPY") == pytest.approx(10)  # internal state intact


# ---------------------------------------------------------------------------
# FIFO selection
# ---------------------------------------------------------------------------

def test_fifo_consumes_oldest_first():
    led = LotLedger(LotSelector.FIFO)
    led.buy("SPY", 10, 100.0, date(2019, 1, 1))  # oldest
    led.buy("SPY", 10, 130.0, date(2020, 1, 1))
    realized = led.sell("SPY", 4, price=150.0, sale_date=date(2021, 6, 1))
    assert len(realized) == 1
    r = realized[0]
    assert r.acquired == date(2019, 1, 1)
    assert r.quantity == pytest.approx(4)
    assert r.proceeds == pytest.approx(4 * 150)
    assert r.cost_basis == pytest.approx(4 * 100)   # oldest lot's basis
    assert r.gain == pytest.approx(4 * 50)
    assert led.total_quantity("SPY") == pytest.approx(16)


def test_fifo_spans_multiple_lots():
    led = LotLedger()  # default FIFO
    led.buy("SPY", 5, 100.0, date(2019, 1, 1))
    led.buy("SPY", 5, 200.0, date(2020, 1, 1))
    realized = led.sell("SPY", 8, price=300.0, sale_date=date(2021, 1, 1))
    assert [r.quantity for r in realized] == pytest.approx([5, 3])
    assert [r.cost_basis for r in realized] == pytest.approx([5 * 100, 3 * 200])
    # remaining: 2 shares of the 200-basis lot
    assert led.total_quantity("SPY") == pytest.approx(2)
    assert led.average_cost("SPY") == pytest.approx(200.0)


def test_sell_entire_position_removes_ticker():
    led = LotLedger()
    led.buy("SPY", 10, 100.0, date(2020, 1, 1))
    led.sell("SPY", 10, price=120.0, sale_date=date(2021, 1, 1))
    assert "SPY" not in led
    assert led.total_quantity("SPY") == 0.0


def test_partial_fractional_consumption():
    led = LotLedger()
    led.buy("SPY", 1.0, 100.0, date(2020, 1, 1))
    led.buy("SPY", 1.0, 100.0, date(2020, 2, 1))
    realized = led.sell("SPY", 1.5, price=100.0, sale_date=date(2021, 1, 1))
    assert [r.quantity for r in realized] == pytest.approx([1.0, 0.5])
    assert led.total_quantity("SPY") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# HIFO selection
# ---------------------------------------------------------------------------

def test_hifo_consumes_highest_basis_first():
    led = LotLedger(LotSelector.HIFO)
    led.buy("SPY", 5, 100.0, date(2019, 1, 1))
    led.buy("SPY", 5, 250.0, date(2020, 1, 1))   # highest basis
    led.buy("SPY", 5, 180.0, date(2020, 6, 1))
    realized = led.sell("SPY", 5, price=300.0, sale_date=date(2021, 1, 1))
    assert realized[0].cost_basis == pytest.approx(5 * 250)
    assert realized[0].gain == pytest.approx(5 * 50)   # smallest possible gain


# ---------------------------------------------------------------------------
# TAX_OPTIMAL selection: long-term first, then highest basis
# ---------------------------------------------------------------------------

def test_tax_optimal_prefers_long_term_then_highest_basis():
    sale_date = date(2021, 1, 1)
    led = LotLedger(LotSelector.TAX_OPTIMAL)
    # Two long-term lots (acquired > 1y before sale) and one short-term lot.
    led.buy("SPY", 5, 300.0, date(2018, 1, 1))   # LT, basis 300
    led.buy("SPY", 5, 150.0, date(2018, 6, 1))   # LT, basis 150
    led.buy("SPY", 5, 400.0, date(2020, 12, 1))  # ST, highest basis
    realized = led.sell("SPY", 7, price=350.0, sale_date=sale_date)
    # LT lots first; within LT, highest basis (300) before lower (150).
    assert realized[0].acquired == date(2018, 1, 1)
    assert realized[0].cost_basis == pytest.approx(5 * 300)
    assert realized[1].acquired == date(2018, 6, 1)
    assert realized[1].quantity == pytest.approx(2)
    # the short-term high-basis lot is untouched while LT lots remain
    assert led.total_quantity("SPY") == pytest.approx(8)


def test_order_lots_helper_directly():
    lots = [
        Lot(1, 100.0, date(2018, 1, 1)),  # LT
        Lot(1, 250.0, date(2020, 12, 1)),  # ST, high basis
        Lot(1, 180.0, date(2018, 6, 1)),  # LT, higher basis than first
    ]
    order = _order_lots(lots, date(2021, 1, 1), LotSelector.TAX_OPTIMAL)
    # LT lots first (highest basis within LT), then ST
    assert [l.acquired_on for l in order] == [
        date(2018, 6, 1), date(2018, 1, 1), date(2020, 12, 1),
    ]


# ---------------------------------------------------------------------------
# selector plumbing
# ---------------------------------------------------------------------------

def test_per_call_selector_overrides_default():
    led = LotLedger(LotSelector.FIFO)
    led.buy("SPY", 5, 100.0, date(2019, 1, 1))
    led.buy("SPY", 5, 250.0, date(2020, 1, 1))
    realized = led.sell("SPY", 5, price=300.0, sale_date=date(2021, 1, 1),
                        selector="hifo")
    assert realized[0].cost_basis == pytest.approx(5 * 250)  # HIFO, not FIFO


@pytest.mark.parametrize("value, expected", [
    ("fifo", LotSelector.FIFO),
    ("HIFO", LotSelector.HIFO),
    ("tax_optimal", LotSelector.TAX_OPTIMAL),
    (LotSelector.FIFO, LotSelector.FIFO),
])
def test_selector_coerce(value, expected):
    assert LotSelector.coerce(value) is expected


def test_selector_coerce_rejects_unknown():
    with pytest.raises(ValueError):
        LotSelector.coerce("lifo")


# ---------------------------------------------------------------------------
# oversell guard
# ---------------------------------------------------------------------------

def test_oversell_raises():
    led = LotLedger()
    led.buy("SPY", 5, 100.0, date(2020, 1, 1))
    with pytest.raises(ValueError, match="Oversell"):
        led.sell("SPY", 6, price=120.0, sale_date=date(2021, 1, 1))


def test_sell_unheld_ticker_raises():
    led = LotLedger()
    with pytest.raises(ValueError, match="Oversell"):
        led.sell("SPY", 1, price=120.0, sale_date=date(2021, 1, 1))


def test_sell_within_float_slack_is_allowed():
    led = LotLedger()
    led.buy("SPY", 5.0, 100.0, date(2020, 1, 1))
    realized = led.sell("SPY", 5.0 + 1e-9, price=120.0, sale_date=date(2021, 1, 1))
    assert sum(r.quantity for r in realized) == pytest.approx(5.0)
    assert "SPY" not in led


# ---------------------------------------------------------------------------
# hand-off into the D.13 tax model
# ---------------------------------------------------------------------------

def test_sell_event_wraps_realized_lots():
    led = LotLedger()
    led.buy("GLD", 10, 100.0, date(2018, 1, 1))
    event = led.sell_event("GLD", 4, price=200.0, sale_date=date(2021, 1, 1))
    assert isinstance(event, SaleEvent)
    assert event.ticker == "GLD"
    assert event.disposed == date(2021, 1, 1)
    assert sum(lot.gain for lot in event.lots) == pytest.approx(4 * 100)


def test_fifo_gld_sale_taxed_as_long_term_collectible():
    """End-to-end: a long-held GLD lot sold FIFO is taxed at the 28% LT cap."""
    led = LotLedger()
    led.buy("GLD", 10, 100.0, date(2017, 1, 1))   # held > 1y
    event = led.sell_event("GLD", 10, price=200.0, sale_date=date(2020, 6, 30))
    result = compute_tax_on_event(event, TaxRegime.us())
    assert result.long_term_gain == pytest.approx(1000.0)
    assert result.long_term_tax == pytest.approx(280.0)   # 28%
    assert result.total_tax == pytest.approx(318.0)       # + 3.8% NIIT


def test_selector_choice_changes_realized_gain_and_tax():
    """FIFO vs HIFO realize different basis → different tax on the same sale."""
    def build(selector):
        led = LotLedger(selector)
        led.buy("SPY", 5, 100.0, date(2017, 1, 1))   # LT, low basis
        led.buy("SPY", 5, 250.0, date(2017, 6, 1))   # LT, high basis
        event = led.sell_event("SPY", 5, price=300.0, sale_date=date(2020, 1, 1))
        return compute_tax_on_event(event, TaxRegime.us())

    fifo = build(LotSelector.FIFO)   # sells 100-basis lot → gain 1000
    hifo = build(LotSelector.HIFO)   # sells 250-basis lot → gain 250
    assert fifo.realized_gain == pytest.approx(1000.0)
    assert hifo.realized_gain == pytest.approx(250.0)
    assert hifo.total_tax < fifo.total_tax
