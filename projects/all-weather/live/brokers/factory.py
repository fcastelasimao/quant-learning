"""
live/brokers/factory.py
=======================
Single-function factory that maps (broker_name, trading_mode, account_label)
to a concrete Broker implementation. The rebalancer calls make_broker() and
then only uses the Broker protocol — it never imports from alpaca.py or
tastytrade.py directly.
"""

from __future__ import annotations

from .base import Broker

_KNOWN_BROKERS = ("alpaca", "tastytrade")


def make_broker(
    *,
    broker_name: str,
    trading_mode: str,
    account_label: str,
) -> Broker:
    """Construct and return a concrete Broker for the given parameters.

    Parameters
    ----------
    broker_name   : "alpaca" or "tastytrade"
    trading_mode  : "paper" or "live"
    account_label : Named label used to look up env-var credentials.
                    Typically "default" (uses the bare env vars) or a
                    descriptive name like "main" / "retirement".

    Raises
    ------
    ValueError      if broker_name is not recognised
    SystemExit      if required credentials are missing (raised by the
                    concrete broker's __init__)
    """
    name = broker_name.lower()

    if name == "alpaca":
        from .alpaca import AlpacaBroker  # local import keeps Alpaca deps optional
        return AlpacaBroker(trading_mode=trading_mode, account_label=account_label)

    if name == "tastytrade":
        from .tastytrade import TastytradeBroker  # local import keeps TT deps optional
        return TastytradeBroker(trading_mode=trading_mode, account_label=account_label)

    raise ValueError(
        f"Unknown broker {broker_name!r}. "
        f"Available brokers: {', '.join(_KNOWN_BROKERS)}"
    )
