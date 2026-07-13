"""
test_tax_schedule.py
====================
Tests for engine/tax.py (D.12): the US federal tax-rate schedule and its
bisect-by-date lookup. These pin the rate values per regime, the boundary
behaviour of the lookup, and the guard against querying an unmodelled
(pre-schedule) date.

Rate facts pinned here (US federal top marginal, decimals):
  * 2003-2012 (JGTRRA):     ST 0.35,  LT 0.15, QDI 0.15, NIIT 0.0
  * 2013-2017 (ATRA + ACA): ST 0.396, LT 0.20, QDI 0.20, NIIT 0.038
  * 2018-present (TCJA;      ST 0.37,  LT 0.20, QDI 0.20, NIIT 0.038
    OBBBA 2025 made the 37% top rate permanent through 2026+)
"""

from datetime import date

import pytest

from engine.tax import TaxRates, TaxSchedule, us_tax_schedule


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_default_yaml_loads_three_regimes():
    sched = us_tax_schedule()
    assert isinstance(sched, TaxSchedule)
    assert sched.effective_dates == [
        date(2003, 1, 1),
        date(2013, 1, 1),
        date(2018, 1, 1),
    ]
    assert len(sched) == 3


def test_us_tax_schedule_is_cached_singleton():
    assert us_tax_schedule() is us_tax_schedule()


def test_meta_is_populated_from_yaml():
    sched = us_tax_schedule()
    assert sched.meta.get("jurisdiction", "").startswith("US federal")
    assert "last_reviewed" in sched.meta


# ---------------------------------------------------------------------------
# Per-regime rate values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "when, st, lt, qdi, niit",
    [
        # JGTRRA regime (covers the 2006 backtest start through 2012)
        ("2006-06-01", 0.35, 0.15, 0.15, 0.0),
        ("2010-01-01", 0.35, 0.15, 0.15, 0.0),
        ("2012-12-31", 0.35, 0.15, 0.15, 0.0),
        # ATRA + ACA regime
        ("2013-01-01", 0.396, 0.20, 0.20, 0.038),
        ("2015-07-04", 0.396, 0.20, 0.20, 0.038),
        ("2017-12-31", 0.396, 0.20, 0.20, 0.038),
        # TCJA regime (made permanent by OBBBA 2025)
        ("2018-01-01", 0.37, 0.20, 0.20, 0.038),
        ("2026-05-30", 0.37, 0.20, 0.20, 0.038),
    ],
)
def test_rates_on_per_regime(when, st, lt, qdi, niit):
    r = us_tax_schedule().rates_on(when)
    assert r.short_term_top == pytest.approx(st)
    assert r.long_term_top == pytest.approx(lt)
    assert r.qualified_dividend_top == pytest.approx(qdi)
    assert r.niit == pytest.approx(niit)


@pytest.mark.parametrize("year", [2018, 2020, 2022])
def test_production_calmar_windows_use_tcja_regime(year):
    """The 2018/2020/2022 OOS Calmar windows all fall under the TCJA regime."""
    r = us_tax_schedule().rates_on(date(year, 6, 30))
    assert (r.short_term_top, r.long_term_top, r.qualified_dividend_top, r.niit) == (
        0.37, 0.20, 0.20, 0.038,
    )


# ---------------------------------------------------------------------------
# Bisect boundary behaviour
# ---------------------------------------------------------------------------

def test_effective_date_boundary_is_inclusive():
    """A query exactly on an effective date picks the NEW regime."""
    sched = us_tax_schedule()
    assert sched.rates_on("2013-01-01").short_term_top == pytest.approx(0.396)
    assert sched.rates_on("2018-01-01").short_term_top == pytest.approx(0.37)


def test_day_before_boundary_picks_previous_regime():
    sched = us_tax_schedule()
    assert sched.rates_on("2012-12-31").short_term_top == pytest.approx(0.35)
    assert sched.rates_on("2017-12-31").short_term_top == pytest.approx(0.396)


def test_date_and_string_inputs_agree():
    sched = us_tax_schedule()
    assert sched.rates_on("2020-01-01") == sched.rates_on(date(2020, 1, 1))


def test_query_before_earliest_entry_raises():
    sched = us_tax_schedule()
    with pytest.raises(ValueError, match="earliest entry"):
        sched.rates_on("2002-12-31")


# ---------------------------------------------------------------------------
# NIIT convenience properties
# ---------------------------------------------------------------------------

def test_with_niit_properties_are_additive():
    r = us_tax_schedule().rates_on("2024-01-01")
    assert r.long_term_with_niit == pytest.approx(0.238)
    assert r.qualified_dividend_with_niit == pytest.approx(0.238)
    assert r.short_term_with_niit == pytest.approx(0.408)


def test_pre_niit_regime_has_zero_surtax():
    r = us_tax_schedule().rates_on("2010-01-01")
    assert r.niit == 0.0
    assert r.long_term_with_niit == pytest.approx(r.long_term_top)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def _entry(effective, st=0.30, lt=0.15, qdi=0.15, niit=0.0):
    return {
        "effective": effective,
        "short_term_top": st,
        "long_term_top": lt,
        "qualified_dividend_top": qdi,
        "niit": niit,
    }


def test_empty_schedule_raises():
    with pytest.raises(ValueError, match="at least one entry"):
        TaxSchedule([])


def test_duplicate_effective_dates_raise():
    with pytest.raises(ValueError, match="Duplicate effective dates"):
        TaxSchedule([_entry("2013-01-01"), _entry("2013-01-01", st=0.40)])


def test_entries_are_sorted_regardless_of_input_order():
    sched = TaxSchedule([_entry("2018-01-01", st=0.37), _entry("2003-01-01", st=0.35)])
    assert sched.effective_dates == [date(2003, 1, 1), date(2018, 1, 1)]
    assert sched.rates_on("2005-01-01").short_term_top == pytest.approx(0.35)


def test_taxrates_is_frozen():
    r = TaxRates(0.37, 0.20, 0.20, 0.038)
    with pytest.raises((AttributeError, TypeError)):
        r.short_term_top = 0.40  # type: ignore[misc]
