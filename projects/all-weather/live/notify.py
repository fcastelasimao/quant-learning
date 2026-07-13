"""
live/notify.py
==============
Post-rebalance notifications via Slack webhook and/or SMTP email.

This module NEVER raises exceptions.  If a notification fails, the error
is logged at WARNING level and execution continues normally.

Configuration — environment variables
--------------------------------------
Slack (webhook):
    ALLW_SLACK_WEBHOOK_URL       — incoming webhook URL
    ALLW_SLACK_CHANNEL           — optional channel override (e.g. #alerts)

Email (SMTP):
    ALLW_SMTP_HOST               — SMTP server hostname (default: localhost)
    ALLW_SMTP_PORT               — SMTP port (default: 587)
    ALLW_SMTP_USER               — username / sender address
    ALLW_SMTP_PASSWORD           — SMTP password / app password
    ALLW_SMTP_FROM               — From: address (defaults to ALLW_SMTP_USER)
    ALLW_NOTIFY_EMAIL            — comma-separated recipient addresses

Usage
-----
::

    from live.notify import notify_run_complete
    from live.runlog import RunSummary

    notify_run_complete(summary, logger=logger)

Or call Slack / email independently::

    from live.notify import send_slack, send_email
    send_slack("Portfolio rebalanced ✓", logger=logger)
    send_email(subject="AW rebalance", body="...", logger=logger)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

from live.env import load_api_keys_env

if TYPE_CHECKING:
    from live.runlog import RunSummary

load_api_keys_env()

_LOG = logging.getLogger("rebalancer.notify")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

def send_slack(
    text: str,
    *,
    webhook_url: str | None = None,
    channel: str | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Send a plain-text message to Slack via an incoming webhook.

    Returns True on success, False on any error (exception is swallowed).
    """
    log = logger or _LOG
    url = webhook_url or _env("ALLW_SLACK_WEBHOOK_URL")
    if not url:
        log.debug("Slack notification skipped: ALLW_SLACK_WEBHOOK_URL not set.")
        return False

    payload: dict = {"text": text}
    ch = channel or _env("ALLW_SLACK_CHANNEL")
    if ch:
        payload["channel"] = ch

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                log.warning(f"Slack webhook returned HTTP {resp.status}")
                return False
        log.debug("Slack notification sent.")
        return True
    except Exception as exc:
        log.warning(f"Slack notification failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(
    subject: str,
    body: str,
    *,
    recipients: list[str] | None = None,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    from_addr: str | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Send a plain-text email via SMTP.

    Returns True on success, False on any error (exception is swallowed).
    """
    log = logger or _LOG

    to_list = recipients or [r.strip() for r in _env("ALLW_NOTIFY_EMAIL").split(",") if r.strip()]
    if not to_list:
        log.debug("Email notification skipped: ALLW_NOTIFY_EMAIL not set.")
        return False

    host = smtp_host or _env("ALLW_SMTP_HOST") or "localhost"
    port = smtp_port or int(_env("ALLW_SMTP_PORT") or "587")
    user = smtp_user or _env("ALLW_SMTP_USER")
    password = smtp_password or _env("ALLW_SMTP_PASSWORD")
    sender = from_addr or _env("ALLW_SMTP_FROM") or user

    if not sender:
        log.warning("Email notification skipped: ALLW_SMTP_FROM / ALLW_SMTP_USER not set.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(to_list)
        msg.attach(MIMEText(body, "plain"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            if port != 465:
                server.starttls(context=ctx)
                server.ehlo()
            if user and password:
                server.login(user, password)
            server.sendmail(sender, to_list, msg.as_string())

        log.debug(f"Email sent to {to_list}.")
        return True
    except Exception as exc:
        log.warning(f"Email notification failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Composite: notify on run complete
# ---------------------------------------------------------------------------

def _format_run_message(summary: "RunSummary") -> str:
    """Format a short human-readable notification for a completed run."""
    from live.runlog import RunSummary  # local import avoids circular at module level

    lines = [
        f"All-Weather Rebalancer — {summary.outcome.upper()}",
        f"Broker: {summary.broker} / {summary.trading_mode} / {summary.account_label}",
        f"Strategy: {summary.strategy_id}",
        f"Started: {summary.started_at.strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if summary.duration_seconds() is not None:
        lines.append(f"Duration: {summary.duration_seconds():.0f}s")

    if summary.equity_before:
        lines.append(f"Equity before: ${summary.equity_before:,.2f}")
    if summary.equity_after:
        lines.append(f"Equity after:  ${summary.equity_after:,.2f}")
    if summary.managed_capital:
        lines.append(f"Managed capital: ${summary.managed_capital:,.2f}")

    if summary.outcome == "executed":
        lines.append(f"Trades: {summary.n_buy} buy / {summary.n_sell} sell "
                     f"/ {summary.n_holding_blocked} holding-blocked")
        if summary.total_buy_notional:
            lines.append(f"Buy notional: ${summary.total_buy_notional:,.2f}")
        if summary.total_sell_notional:
            lines.append(f"Sell notional: ${summary.total_sell_notional:,.2f}")

    if summary.warnings:
        lines.append(f"Warnings ({len(summary.warnings)}):")
        for w in summary.warnings[:5]:
            lines.append(f"  - {w}")
        if len(summary.warnings) > 5:
            lines.append(f"  ... ({len(summary.warnings) - 5} more)")

    if summary.error_message:
        lines.append(f"ERROR: {summary.error_message}")

    return "\n".join(lines)


def notify_run_complete(
    summary: "RunSummary",
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Fire all configured notification channels (never raises).

    Call this after every run, regardless of outcome.  Silent when no
    notification env vars are configured.
    """
    log = logger or _LOG
    msg = _format_run_message(summary)
    subject = (
        f"[AW {summary.broker}/{summary.trading_mode}] "
        f"{summary.outcome.upper()} — {summary.strategy_id}"
    )
    send_slack(msg, logger=log)
    send_email(subject=subject, body=msg, logger=log)
