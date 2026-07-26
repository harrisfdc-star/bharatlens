"""
Zerodha Kite Connect helpers for local Bharat Lens holdings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, TokenException

SESSION_FILE = Path(__file__).resolve().parent / ".kite_session.json"


def extract_request_token(value: str) -> str | None:
    """Pull request_token from a raw token or a full redirect URL."""
    text = (value or "").strip()
    if not text:
        return None

    if "request_token=" in text:
        parsed = urlparse(text)
        params = parse_qs(parsed.query)
        token = params.get("request_token", [None])[0]
        if token:
            return token.strip()

    # Bare token pasted by the user
    if re.fullmatch(r"[A-Za-z0-9]+", text):
        return text
    return None


def save_session(api_key: str, access_token: str, user_name: str = "") -> None:
    SESSION_FILE.write_text(
        json.dumps(
            {
                "api_key": api_key,
                "access_token": access_token,
                "user_name": user_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_session() -> dict[str, str] | None:
    if not SESSION_FILE.exists():
        return None
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        if data.get("api_key") and data.get("access_token"):
            return data
    except (json.JSONDecodeError, OSError):
        return None
    return None


def clear_session() -> None:
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def make_kite(api_key: str, access_token: str | None = None) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    if access_token:
        kite.set_access_token(access_token)
    return kite


def login_url(api_key: str) -> str:
    return make_kite(api_key).login_url()


def exchange_request_token(
    api_key: str, api_secret: str, request_token: str
) -> dict[str, Any]:
    kite = make_kite(api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    save_session(
        api_key=api_key,
        access_token=session["access_token"],
        user_name=session.get("user_name", "") or session.get("user_id", ""),
    )
    return session


def get_authenticated_kite(api_key: str, access_token: str) -> KiteConnect:
    kite = make_kite(api_key, access_token)
    # Validate token with a lightweight call
    kite.profile()
    return kite


def try_restore_session(api_key: str) -> KiteConnect | None:
    saved = load_session()
    if not saved or saved.get("api_key") != api_key:
        return None
    try:
        return get_authenticated_kite(api_key, saved["access_token"])
    except (TokenException, KiteException, Exception):
        clear_session()
        return None


def fetch_holdings(kite: KiteConnect) -> list[dict[str, Any]]:
    return kite.holdings() or []


def holdings_to_dataframe(holdings: list[dict[str, Any]]):
    import pandas as pd

    if not holdings:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Exchange",
                "Qty",
                "Avg Cost (₹)",
                "LTP (₹)",
                "Invested (₹)",
                "Current (₹)",
                "P&L (₹)",
                "P&L (%)",
            ]
        )

    rows = []
    for h in holdings:
        qty = float(h.get("quantity", 0) or 0) + float(h.get("t1_quantity", 0) or 0)
        avg = float(h.get("average_price", 0) or 0)
        ltp = float(h.get("last_price", 0) or 0)
        invested = qty * avg
        current = qty * ltp
        pnl = float(h.get("pnl", current - invested) or (current - invested))
        pnl_pct = ((ltp - avg) / avg * 100.0) if avg > 0 else 0.0
        rows.append(
            {
                "Symbol": h.get("tradingsymbol", ""),
                "Exchange": h.get("exchange", ""),
                "Qty": int(qty) if qty == int(qty) else qty,
                "Avg Cost (₹)": round(avg, 2),
                "LTP (₹)": round(ltp, 2),
                "Invested (₹)": round(invested, 2),
                "Current (₹)": round(current, 2),
                "P&L (₹)": round(pnl, 2),
                "P&L (%)": round(pnl_pct, 2),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Current (₹)", ascending=False).reset_index(drop=True)
    return df
