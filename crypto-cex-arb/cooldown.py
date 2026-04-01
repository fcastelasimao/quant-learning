"""
Cooldown Manager: prevents trading the same runner/event repeatedly.

In real markets, a mispricing may persist for a few seconds across multiple
scan cycles. We don't want to stack up 10 paper trades on the same edge.
This module enforces a cooldown period per runner per event.
"""
from __future__ import annotations
import time
from collections import defaultdict


class CooldownManager:
    """Tracks recently traded runners and enforces cooldowns."""

    def __init__(self, cooldown_seconds: float = 120.0):
        """
        Args:
            cooldown_seconds: minimum seconds between trades on the same
                              runner within the same event.
        """
        self.cooldown_seconds = cooldown_seconds
        # key: (event_name, runner_name) → last trade timestamp
        self._last_traded: dict[tuple[str, str], float] = {}

    def can_trade(self, event_name: str, runner_name: str) -> bool:
        """Check if this runner is off cooldown."""
        key = (event_name, runner_name)
        last = self._last_traded.get(key, 0)
        return (time.time() - last) >= self.cooldown_seconds

    def record_trade(self, event_name: str, runner_name: str):
        """Mark this runner as just traded."""
        self._last_traded[(event_name, runner_name)] = time.time()

    def cleanup(self, max_age: float = 3600.0):
        """Remove entries older than max_age seconds."""
        now = time.time()
        self._last_traded = {
            k: v for k, v in self._last_traded.items()
            if (now - v) < max_age
        }

    @property
    def active_cooldowns(self) -> int:
        return len(self._last_traded)
