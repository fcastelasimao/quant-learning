"""
Factor identity card utility.

Usage (in any notebook)::

    from qframe.viz.identity import print_factor
    print_factor(82)

Prints a compact identity card for a given implementation ID, including
factor name, IC/ICIR/t-stat, slow_icir_63, Sharpe, MaxDD, turnover,
gate status, and retired/status flag.
"""
from __future__ import annotations

import math


def print_factor(
    impl_id: int,
    kb_path: str = "../knowledge_base/qframe.db",
    n_oos_days: int = 1762,
) -> None:
    """Print a compact identity card for a factor implementation.

    Args:
        impl_id:    The ``implementation_id`` to look up (e.g. 82).
        kb_path:    Path to the SQLite knowledge base (relative to the
                    notebooks/ directory by default).
        n_oos_days: Number of OOS trading days used for t-stat computation
                    (default 1762 ≈ 2018-01-01 to 2024-12-31).

    Example output::

        impl_82  │  trend_quality_calmar_ratio
            domain/notes      quality — calmar ratio 12-1m
            IC / ICIR         +0.0646 / +0.382
            t-stat (fast)     +10.74  [BHY threshold ~4.0]
            slow_icir_63 / t  +0.000 / +0.00  (n/a)
            Sharpe (annual)   +4.27
            MaxDD / TO        -10.3% / 883%/yr
            passed_gate       1
            status            RETIRED 2026-04-19 — look-ahead bias
    """
    from qframe.knowledge_base.db import KnowledgeBase

    kb = KnowledgeBase(kb_path)
    rows = kb.get_all_results()
    r = next((x for x in rows if x.get("implementation_id") == impl_id), None)
    if r is None:
        print(f"impl_{impl_id}: not found in knowledge base")
        return

    name = r.get("factor_name") or "(unnamed)"
    ic = r.get("ic") or 0.0
    icir = r.get("icir") or 0.0
    # Fast-signal t-stat: icir × √(n_oos_days / 252)
    t_fast = icir * math.sqrt(n_oos_days / 252)

    slow_icir = r.get("slow_icir_63") or 0.0
    n_windows = n_oos_days / 63
    t_slow = slow_icir * math.sqrt(n_windows) if slow_icir else 0.0

    sharpe = r.get("sharpe") or 0.0
    max_dd = r.get("max_drawdown") or 0.0
    turnover = (r.get("turnover") or 0.0) * 252  # daily fraction → annual

    passed = r.get("passed_gate")
    status = r.get("status") or r.get("notes") or "—"
    notes = r.get("impl_notes") or r.get("factor_type") or "—"

    print(f"impl_{impl_id:>3}  │  {name}")
    print(f"    notes/domain      {notes}")
    print(f"    IC / ICIR         {ic:+.4f} / {icir:+.3f}")
    print(f"    t-stat (fast)     {t_fast:+.2f}  [BHY threshold ~4.0 at m=84]")
    if slow_icir:
        print(f"    slow_icir_63 / t  {slow_icir:+.3f} / {t_slow:+.2f}")
    else:
        print(f"    slow_icir_63 / t  n/a")
    print(f"    Sharpe (annual)   {sharpe:+.2f}")
    print(f"    MaxDD / TO        {max_dd:+.1%} / {turnover:.0f}%/yr")
    print(f"    passed_gate       {passed}")
    print(f"    status            {status}")
