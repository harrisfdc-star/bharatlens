"""
BharatMarketLens — Local Nifty 50 intelligence scanner + Zerodha holdings dashboard.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from PIL import Image
from streamlit_autorefresh import st_autorefresh

import zerodha_client as zc

# ---------------------------------------------------------------------------
# Nifty 50 constituents (NSE symbols). Update if the index composition changes.
# ---------------------------------------------------------------------------
NIFTY_50: dict[str, str] = {
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS": "Bharat Electronics",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BPCL.NS": "BPCL",
    "BRITANNIA.NS": "Britannia",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DRREDDY.NS": "Dr. Reddy's",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC",
    "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid",
    "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life",
    "SBIN.NS": "State Bank of India",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SUNPHARMA.NS": "Sun Pharma",
    "TATACONSUM.NS": "Tata Consumer",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "TCS",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company",
    "TRENT.NS": "Trent",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
}

MA_WINDOW = 50
LOOKBACK_DAYS = 380
MAX_STOCK_PRICE = 1200.0

MODEL_STATE_PATH = Path(__file__).resolve().parent / "model_feedback_state.json"
TRADE_JOURNAL_PATH = Path(__file__).resolve().parent / "paper_trade_journal.json"
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "technical": 1.0,
    "fundamental": 1.0,
    "flow": 1.0,
    "resilience": 1.0,
}
COMPONENT_SCORE_SCALE: dict[str, float] = {
    "technical": 44.0,
    "fundamental": 32.0,
    "flow": 23.0,
    "resilience": 11.0,
}
MIN_COMPONENT_WEIGHT = 0.65
MAX_COMPONENT_WEIGHT = 1.55
LEARNING_RATE = 0.05
REGIME_BUCKETS = ("Bullish", "Bearish", "Sideways")

POSITIVE_SENTIMENT_TERMS = {
    "beat",
    "growth",
    "surge",
    "expands",
    "strong",
    "record",
    "wins",
    "upgrade",
    "bullish",
    "outperform",
    "profit",
    "demand",
    "order",
    "momentum",
    "higher",
}
NEGATIVE_SENTIMENT_TERMS = {
    "miss",
    "fall",
    "drop",
    "cuts",
    "weak",
    "downgrade",
    "bearish",
    "loss",
    "decline",
    "probe",
    "fraud",
    "penalty",
    "debt",
    "slump",
    "lawsuit",
}

# Toggle this to True only if you want to use an external custom logo file.
USE_CUSTOM_LOGO = False
CUSTOM_LOGO_PATH = Path(
    r"C:\Users\Dell\.cursor\projects\d-Cursor-Pro-bharat-lens\assets\c__Users_Dell_AppData_Roaming_Cursor_User_workspaceStorage_5e886754235f2dcc87b566a27c68bcf7_images_image-8ad01b6c-7999-45cf-a1e8-11d88ea20493.png"
)
DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "bharatlens-logo.svg"


def _format_signed_rupees(value: Any) -> str:
    """Render rupee delta with explicit sign."""
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    sign = "+" if numeric >= 0 else "-"
    return f"{sign}₹{abs(numeric):,.2f}"


def _format_day_change(day_change_rupees: Any, day_change_pct: Any | None = None) -> str:
    """Render day change as signed rupees and percent."""
    rupees_text = _format_signed_rupees(day_change_rupees)
    if day_change_pct is None or pd.isna(day_change_pct):
        return rupees_text
    return f"{rupees_text} ({float(day_change_pct):+.2f}%)"


def _format_since_added_change(current_price: Any, baseline_price: Any) -> str:
    """Render gain/loss since the script first appeared in dashboard."""
    if (
        current_price is None
        or baseline_price is None
        or pd.isna(current_price)
        or pd.isna(baseline_price)
        or float(baseline_price) == 0.0
    ):
        return "NA"
    delta_rupees = float(current_price) - float(baseline_price)
    delta_pct = (delta_rupees / float(baseline_price)) * 100.0
    return _format_day_change(delta_rupees, delta_pct)


def _extract_min_days(window_text: Any) -> int | None:
    """Extract first day value from a window string like '5-10'."""
    text = str(window_text or "").strip()
    if not text or text.upper() == "NA":
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _parse_day_window(window_text: Any) -> tuple[int | None, int | None]:
    """Parse a day-window string like '5-10' into (start, end)."""
    text = str(window_text or "").strip()
    if not text or text.upper() == "NA":
        return (None, None)
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if not nums:
        return (None, None)
    if len(nums) == 1:
        return (nums[0], nums[0])
    start, end = nums[0], nums[1]
    if start > end:
        start, end = end, start
    return (start, end)


def _window_overlaps(window_text: Any, start_day: int, end_day: int) -> bool:
    """Return True if parsed day window overlaps [start_day, end_day]."""
    w_start, w_end = _parse_day_window(window_text)
    if w_start is None or w_end is None:
        return False
    return not (w_end < start_day or w_start > end_day)


def _clean_text(value: Any) -> str | None:
    """Return stripped non-empty text, otherwise None."""
    text = str(value or "").strip()
    return text if text else None


def _calc_exit_gain_pct(entry_price: Any, exit_price: Any) -> float | None:
    """Calculate percentage gain from entry to exit."""
    entry = _to_float(entry_price)
    exit_val = _to_float(exit_price)
    if entry is None or exit_val is None or entry <= 0:
        return None
    return round(((exit_val - entry) / entry) * 100.0, 2)


def _format_exit_with_gain(entry_price: Any, exit_price: Any) -> str:
    """Format proposed exit as value and gain in brackets."""
    exit_val = _to_float(exit_price)
    if exit_val is None:
        return "NA"
    gain_pct = _calc_exit_gain_pct(entry_price, exit_val)
    if gain_pct is None:
        return f"₹{exit_val:,.2f}"
    return f"₹{exit_val:,.2f} ({gain_pct:+.2f}%)"


def _high_conviction_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate stricter 10% target gate for BUY/BUY ON DIP setups."""
    rec = str(row.get("Recommendation", "")).upper().strip()
    if rec not in {"BUY", "BUY ON DIP"}:
        return False, ["not-buy-signal"]

    failures: list[str] = []
    confidence = _to_float(row.get("Confidence (1-10)")) or 0.0
    promoter = _to_float(row.get("Promoter / Insider Stake (%)")) or 0.0
    cmp_value = _to_float(row.get("CMP (₹)")) or 0.0
    entry_price = _to_float(row.get("Entry Price (₹)")) or cmp_value
    exit_price = _to_float(row.get("Proposed Exit Price (₹)"))
    target_2 = _to_float(row.get("Target 2 (₹)"))
    weekly_momentum = _to_float(row.get("Weekly Momentum (%)")) or 0.0
    volume_surge = _to_float(row.get("Volume Surge (5d/20d)")) or 0.0
    buy_sell_ratio = _to_float(row.get("Buy/Sell Volume Ratio"))
    day_change = _to_float(row.get("Daily Change (%)"))
    sentiment_score = _to_float(row.get("Product Sentiment Score")) or 0.0
    earnings_days = _to_float(row.get("Results Announcement (days)"))
    sell_window = row.get("Sell Window (next 2w, days)")
    time_stop_days = _to_float(row.get("Time Stop (days)"))

    target_ref = target_2 if target_2 is not None else exit_price
    upside_pct = (((target_ref / entry_price) - 1.0) * 100.0) if target_ref and entry_price > 0 else None
    holding_ok = _window_overlaps(sell_window, 5, 10) or (
        time_stop_days is not None and 5 <= time_stop_days <= 10
    )

    if confidence < 8.0:
        failures.append("confidence<8")
    if promoter < 40.0:
        failures.append("promoter<40")
    if upside_pct is None or upside_pct < 10.0:
        failures.append("upside<10%")
    if not holding_ok:
        failures.append("hold-window-mismatch")
    if weekly_momentum < 0.8:
        failures.append("weak-momentum")
    if volume_surge < 1.05:
        failures.append("weak-volume")
    if buy_sell_ratio is not None and buy_sell_ratio < 1.0:
        failures.append("sell-dominant-flow")
    if sentiment_score < 0:
        failures.append("negative-sentiment")
    if day_change is not None and day_change > 6.0:
        failures.append("overextended-day-move")
    if earnings_days is not None and earnings_days < 2:
        failures.append("near-results-risk")

    if failures:
        return False, failures
    return True, []


def _track_since_added_change(store_key: str, symbol: Any, current_price: Any) -> str:
    """Store first-seen price and return change from that baseline."""
    symbol_key = str(symbol or "").strip()
    if not symbol_key or current_price is None or pd.isna(current_price):
        return "NA"

    current_val = float(current_price)
    baseline_store = st.session_state.setdefault(store_key, {})
    today_text = dt.datetime.now().strftime("%d %b %Y")
    existing = baseline_store.get(symbol_key)
    if existing is None:
        baseline_store[symbol_key] = {"price": current_val, "date": today_text}
        existing = baseline_store[symbol_key]
    elif not isinstance(existing, dict):
        # Backward compatibility for old sessions that stored only a float baseline.
        baseline_store[symbol_key] = {"price": float(existing), "date": "Unknown"}
        existing = baseline_store[symbol_key]

    baseline_price = existing.get("price")
    change_text = _format_since_added_change(current_val, baseline_price)
    return change_text


def _get_since_added_date(store_key: str, symbol: Any) -> str:
    """Return first-seen date for a symbol from baseline store."""
    symbol_key = str(symbol or "").strip()
    if not symbol_key:
        return "NA"

    baseline_store = st.session_state.get(store_key, {})
    existing = baseline_store.get(symbol_key)
    if isinstance(existing, dict):
        added_on = str(existing.get("date", "") or "").strip()
        return added_on if added_on else "Unknown"
    if existing is not None:
        return "Unknown"
    return "NA"


def _confidence_value(row: dict[str, Any]) -> float:
    """Safe confidence extractor for descending sort."""
    value = row.get("Confidence (1-10)")
    if value is None or pd.isna(value):
        return float("-inf")
    return float(value)


def _load_model_state() -> dict[str, Any]:
    """Load persisted adaptive weights + feedback history."""
    default_weights_by_regime = {
        regime: dict(DEFAULT_COMPONENT_WEIGHTS) for regime in REGIME_BUCKETS
    }
    default_state: dict[str, Any] = {
        "weights": dict(DEFAULT_COMPONENT_WEIGHTS),
        "weights_by_regime": default_weights_by_regime,
        "history": {},
        "metrics": {
            "feedback_updates": 0,
            "avg_return_ewma": 0.0,
            "last_feedback_date": "",
        },
    }
    if not MODEL_STATE_PATH.exists():
        return default_state

    try:
        loaded = json.loads(MODEL_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_state

    state = default_state
    if isinstance(loaded, dict):
        legacy_weights: dict[str, float] = {}
        if isinstance(loaded.get("weights"), dict):
            for k, v in loaded["weights"].items():
                if k in DEFAULT_COMPONENT_WEIGHTS and isinstance(v, (int, float)):
                    clipped = float(
                        np.clip(v, MIN_COMPONENT_WEIGHT, MAX_COMPONENT_WEIGHT)
                    )
                    state["weights"][k] = clipped
                    legacy_weights[k] = clipped
        if isinstance(loaded.get("weights_by_regime"), dict):
            for regime, weight_map in loaded["weights_by_regime"].items():
                if regime not in REGIME_BUCKETS or not isinstance(weight_map, dict):
                    continue
                for k, v in weight_map.items():
                    if k in DEFAULT_COMPONENT_WEIGHTS and isinstance(v, (int, float)):
                        state["weights_by_regime"][regime][k] = float(
                            np.clip(v, MIN_COMPONENT_WEIGHT, MAX_COMPONENT_WEIGHT)
                        )
        elif legacy_weights:
            # Migrate old global weights into every regime bucket.
            for regime in REGIME_BUCKETS:
                state["weights_by_regime"][regime].update(legacy_weights)
        if isinstance(loaded.get("history"), dict):
            state["history"] = loaded["history"]
        if isinstance(loaded.get("metrics"), dict):
            state["metrics"].update(loaded["metrics"])
    return state


def _save_model_state(state: dict[str, Any]) -> None:
    """Persist adaptive model state to local JSON."""
    try:
        MODEL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        # Non-blocking: scanning should continue even if state cannot be saved.
        return


def _get_regime_weights(state: dict[str, Any], regime: str) -> dict[str, float]:
    """Get sanitized adaptive weights for the active regime."""
    safe_regime = regime if regime in REGIME_BUCKETS else "Sideways"
    weights_by_regime = state.setdefault(
        "weights_by_regime",
        {name: dict(DEFAULT_COMPONENT_WEIGHTS) for name in REGIME_BUCKETS},
    )
    regime_weights = weights_by_regime.setdefault(safe_regime, dict(DEFAULT_COMPONENT_WEIGHTS))
    return {
        name: float(
            np.clip(
                regime_weights.get(name, DEFAULT_COMPONENT_WEIGHTS[name]),
                MIN_COMPONENT_WEIGHT,
                MAX_COMPONENT_WEIGHT,
            )
        )
        for name in DEFAULT_COMPONENT_WEIGHTS
    }


def _load_trade_journal() -> list[dict[str, Any]]:
    """Load persisted paper-trade journal rows."""
    if not TRADE_JOURNAL_PATH.exists():
        return []
    try:
        rows = json.loads(TRADE_JOURNAL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            clean_rows.append(row)
    return clean_rows


def _save_trade_journal(rows: list[dict[str, Any]]) -> None:
    """Persist paper-trade journal rows."""
    try:
        TRADE_JOURNAL_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except OSError:
        return


def _upsert_journal_in_session(rows: list[dict[str, Any]]) -> None:
    """Update in-memory + file-backed journal state."""
    st.session_state["paper_trade_journal_rows"] = rows
    _save_trade_journal(rows)


def _format_rr(entry: Any, stop: Any, target: Any) -> float | None:
    """Return risk-reward ratio for one target."""
    try:
        entry_v = float(entry)
        stop_v = float(stop)
        target_v = float(target)
    except (TypeError, ValueError):
        return None
    risk = entry_v - stop_v
    if risk <= 0:
        return None
    reward = target_v - entry_v
    return round(reward / risk, 2)


def _trade_card_score_tag(row: dict[str, Any]) -> tuple[int, str, str]:
    """Compute quick decision score and tag for trade card."""
    rec = str(row.get("Recommendation", "") or "").upper().strip()
    confidence = float(row.get("Confidence (1-10)", 0.0) or 0.0)
    rr_t1 = _to_float(row.get("RR to T1"))
    rr_t2 = _to_float(row.get("RR to T2"))
    mom_5 = _to_float(row.get("_mom_5m"))
    mom_15 = _to_float(row.get("_mom_15m"))
    earnings_days = row.get("Results Announcement (days)")
    try:
        earnings_days_num = int(earnings_days) if earnings_days is not None else None
    except Exception:
        earnings_days_num = None

    score = 38
    if rec in {"BUY", "BUY ON DIP"}:
        score += 20
    elif rec == "WATCH":
        score += 10
    else:
        score -= 18

    if confidence >= 8.0:
        score += 16
    elif confidence >= 7.0:
        score += 11
    elif confidence >= 6.5:
        score += 7
    else:
        score += 2

    if rr_t1 is not None:
        if rr_t1 >= 2.0:
            score += 12
        elif rr_t1 >= 1.6:
            score += 8
        elif rr_t1 >= 1.2:
            score += 4
    if rr_t2 is not None:
        if rr_t2 >= 2.5:
            score += 10
        elif rr_t2 >= 2.0:
            score += 7
        elif rr_t2 >= 1.5:
            score += 4

    if mom_5 is not None:
        if mom_5 >= 0.35:
            score += 6
        elif mom_5 >= 0:
            score += 3
        elif mom_5 <= -0.4:
            score -= 5
    if mom_15 is not None:
        if mom_15 >= 0.45:
            score += 8
        elif mom_15 >= 0:
            score += 4
        elif mom_15 <= -0.5:
            score -= 7

    if earnings_days_num is not None:
        if earnings_days_num < 2:
            score -= 6
        elif earnings_days_num <= 10:
            score += 2
        else:
            score += 4

    score = int(np.clip(score, 0, 100))
    if (
        score >= 75
        and rec in {"BUY", "BUY ON DIP"}
        and (rr_t1 is None or rr_t1 >= 1.6)
        and (mom_15 is None or mom_15 >= -0.15)
    ):
        return score, "Ready", "#00B386"
    if score >= 55:
        return score, "Watch", "#FFB020"
    return score, "Skip", "#EB5336"


def _get_optional_config(key: str) -> str:
    """Read optional config from env or Streamlit secrets."""
    raw = os.getenv(key)
    if raw:
        return str(raw).strip()
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        return ""
    return ""


def _learn_from_row_feedback(
    state: dict[str, Any], row: dict[str, Any], scan_date: str, current_regime: str
) -> float | None:
    """Update adaptive weights from realized movement since previous scan."""
    symbol = str(row.get("Symbol") or "").strip()
    current_price = row.get("CMP (₹)")
    if not symbol or current_price is None or pd.isna(current_price):
        return None

    weights_by_regime = state.setdefault(
        "weights_by_regime",
        {name: dict(DEFAULT_COMPONENT_WEIGHTS) for name in REGIME_BUCKETS},
    )
    history = state.setdefault("history", {})
    metrics = state.setdefault("metrics", {})
    previous = history.get(symbol, {})
    realized_return_pct: float | None = None

    prev_date = str(previous.get("date") or "")
    prev_price_raw = previous.get("price")
    prev_components = previous.get("components", {})
    prev_regime = str(previous.get("regime") or "Sideways")
    prev_regime = prev_regime if prev_regime in REGIME_BUCKETS else "Sideways"

    if prev_date and prev_date != scan_date and prev_price_raw not in (None, ""):
        prev_price = float(prev_price_raw)
        if prev_price > 0:
            realized_return_pct = ((float(current_price) / prev_price) - 1.0) * 100.0
            return_signal = float(np.clip(realized_return_pct / 3.0, -1.5, 1.5))
            regime_weights = weights_by_regime.setdefault(
                prev_regime, dict(DEFAULT_COMPONENT_WEIGHTS)
            )
            for name in DEFAULT_COMPONENT_WEIGHTS:
                base_weight = float(
                    np.clip(
                        regime_weights.get(name, DEFAULT_COMPONENT_WEIGHTS[name]),
                        MIN_COMPONENT_WEIGHT,
                        MAX_COMPONENT_WEIGHT,
                    )
                )
                component_score = float(prev_components.get(name, 0.0) or 0.0)
                scale = float(COMPONENT_SCORE_SCALE.get(name, 1.0))
                strength = float(np.clip((component_score / scale) - 0.5, -0.5, 0.5))
                adjustment = LEARNING_RATE * strength * return_signal
                regime_weights[name] = float(
                    np.clip(
                        base_weight * (1.0 + adjustment),
                        MIN_COMPONENT_WEIGHT,
                        MAX_COMPONENT_WEIGHT,
                    )
                )

            prev_ewma = float(metrics.get("avg_return_ewma", 0.0) or 0.0)
            metrics["avg_return_ewma"] = round((0.88 * prev_ewma) + (0.12 * realized_return_pct), 4)
            metrics["feedback_updates"] = int(metrics.get("feedback_updates", 0) or 0) + 1
            metrics["last_feedback_date"] = scan_date

    history[symbol] = {
        "date": scan_date,
        "price": round(float(current_price), 4),
        "components": {
            name: round(float((row.get("_component_scores", {}) or {}).get(name, 0.0) or 0.0), 4)
            for name in DEFAULT_COMPONENT_WEIGHTS
        },
        "signal": str(row.get("Recommendation", "")),
        "confidence": float(row.get("Confidence (1-10)", 0.0) or 0.0),
        "regime": current_regime if current_regime in REGIME_BUCKETS else "Sideways",
    }
    return realized_return_pct


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Return a clean Close price series for one ticker from a yfinance download."""
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                if ticker in close.columns:
                    series = close[ticker]
                else:
                    series = close.iloc[:, 0]
            else:
                series = close
        elif ticker in raw.columns.get_level_values(0):
            series = raw[ticker]["Close"]
        else:
            return None
    else:
        if "Close" not in raw.columns:
            return None
        series = raw["Close"]

    series = pd.to_numeric(series, errors="coerce").dropna()
    return series if len(series) >= MA_WINDOW else None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(
    tickers: tuple[str, ...],
    lookback_days: int = LOOKBACK_DAYS,
    chunk_size: int = 120,
) -> dict[str, pd.Series]:
    """Download recent daily closes for tickers in safe chunks."""
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=max(80, int(lookback_days)))

    result: dict[str, pd.Series] = {}
    ticker_list = list(tickers)
    for start_idx in range(0, len(ticker_list), max(20, int(chunk_size))):
        batch = ticker_list[start_idx : start_idx + max(20, int(chunk_size))]
        try:
            raw = yf.download(
                batch,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            continue

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                    close = _extract_close(raw[ticker], ticker)
                else:
                    close = _extract_close(raw, ticker)
                if close is not None:
                    result[ticker] = close
            except Exception:
                continue
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nse_equity_universe() -> dict[str, str]:
    """Fetch broad NSE listed equities universe from official CSV."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        return dict(NIFTY_50)

    if "SYMBOL" not in df.columns:
        return dict(NIFTY_50)

    series_col = " SERIES" if " SERIES" in df.columns else "SERIES"
    name_col = "NAME OF COMPANY" if "NAME OF COMPANY" in df.columns else None

    if series_col in df.columns:
        df = df[df[series_col].astype(str).str.strip().isin({"EQ", "BE"})]

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row.get("SYMBOL", "") or "").strip().upper()
        if not sym:
            continue
        name = str(row.get(name_col, sym) if name_col else sym).strip()
        out[f"{sym}.NS"] = name or sym
    return out or dict(NIFTY_50)


def _parse_custom_symbols(raw_text: str) -> list[str]:
    """Parse comma/newline separated custom symbols into Yahoo format."""
    parsed: list[str] = []
    for token in re.split(r"[\s,;|]+", str(raw_text or "")):
        clean = token.strip().upper()
        if not clean:
            continue
        if clean.endswith(".NS") or clean.endswith(".BO"):
            parsed.append(clean)
        elif clean.isalnum():
            # Default to NSE if suffix not provided.
            parsed.append(f"{clean}.NS")
    # Preserve order, remove duplicates.
    return list(dict.fromkeys(parsed))


@st.cache_data(ttl=180, show_spinner=False)
def fetch_intraday_momentum_map(
    tickers: tuple[str, ...],
    interval: str = "5m",
    chunk_size: int = 80,
) -> dict[str, float]:
    """Return latest intraday momentum percentage map."""
    out: dict[str, float] = {}
    ticker_list = list(tickers)
    for start_idx in range(0, len(ticker_list), max(20, int(chunk_size))):
        batch = ticker_list[start_idx : start_idx + max(20, int(chunk_size))]
        try:
            raw = yf.download(
                batch,
                period="1d",
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            continue

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                    close = pd.to_numeric(raw[ticker].get("Close"), errors="coerce").dropna()
                else:
                    close = pd.to_numeric(raw.get("Close"), errors="coerce").dropna()
                if close is None or len(close) < 2:
                    continue
                prev = float(close.iloc[-2])
                last = float(close.iloc[-1])
                if prev:
                    out[ticker] = round(((last / prev) - 1.0) * 100.0, 2)
            except Exception:
                continue
    return out


@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_price_snapshot(ticker: str) -> dict[str, float] | None:
    """Fetch latest close-style price and day-change for one symbol."""
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(closes) < 2:
            return None
        latest = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        day_change = latest - prev
        day_change_pct = ((latest / prev) - 1.0) * 100.0 if prev else 0.0
        return {
            "price": round(latest, 2),
            "day_change": round(day_change, 2),
            "day_change_pct": round(day_change_pct, 2),
        }
    except Exception:
        return None


def _refresh_rows_with_live_prices(
    rows: list[dict[str, Any]],
    store_key: str,
    max_rows: int = 20,
) -> None:
    """Update CMP/day-change for visible rows and refresh since-added fields."""
    for row in rows[:max_rows]:
        symbol = str(row.get("Symbol", "") or "").strip()
        if not symbol:
            continue

        snapshot = fetch_live_price_snapshot(f"{symbol}.NS")
        if snapshot:
            latest_price = float(snapshot["price"])
            row["CMP (₹)"] = latest_price
            row["Price (₹)"] = latest_price
            row["Day changed"] = float(snapshot["day_change"])
            row["Daily Change (%)"] = float(snapshot["day_change_pct"])

        row["_since_added"] = _track_since_added_change(
            store_key,
            symbol,
            row.get("CMP (₹)"),
        )
        row["_added_on"] = _get_since_added_date(store_key, symbol)


def _to_float(value: Any) -> float | None:
    """Convert unknown value to float safely."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_news_sentiment(news_rows: list[dict[str, Any]]) -> tuple[float, str]:
    """Return sentiment score in [-1, 1] and text label."""
    if not news_rows:
        return 0.0, "Neutral"

    points = 0
    hits = 0
    for row in news_rows:
        text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
        for term in POSITIVE_SENTIMENT_TERMS:
            if term in text:
                points += 1
                hits += 1
        for term in NEGATIVE_SENTIMENT_TERMS:
            if term in text:
                points -= 1
                hits += 1

    if hits == 0:
        return 0.0, "Neutral"

    score = float(np.clip(points / max(hits, 1), -1, 1))
    if score > 0.2:
        return score, "Positive"
    if score < -0.2:
        return score, "Negative"
    return score, "Neutral"


def _label_from_sentiment(score: float) -> str:
    """Convert sentiment score into a text label."""
    if score > 0.2:
        return "Positive"
    if score < -0.2:
        return "Negative"
    return "Neutral"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_external_market_news() -> list[dict[str, str]]:
    """Fetch market headlines from widely read business sources via RSS."""
    feed_map = {
        "Moneycontrol": "https://news.google.com/rss/search?q=site:moneycontrol.com+Indian+stock+market",
        "Economic Times": "https://news.google.com/rss/search?q=site:economictimes.indiatimes.com+stocks",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "Zerodha": "https://zerodha.com/z-connect/feed/",
        "Screener": "https://news.google.com/rss/search?q=site:screener.in+stock",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    items: list[dict[str, str]] = []

    for source, url in feed_map.items():
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall(".//item")[:18]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                desc = (item.findtext("description") or "").strip()
                if title:
                    items.append(
                        {
                            "source": source,
                            "title": title,
                            "summary": desc,
                            "link": link,
                        }
                    )
        except Exception:
            continue
    return items


def _score_external_news_for_stock(
    company_name: str,
    symbol: str,
    news_rows: list[dict[str, str]],
) -> tuple[float, str, int, list[str]]:
    """Score sentiment from external headlines that mention stock/company."""
    if not news_rows:
        return 0.0, "Neutral", 0, []

    company_tokens = [
        t for t in re.split(r"[^a-zA-Z0-9]+", company_name.lower()) if len(t) >= 4
    ]
    symbol_token = symbol.lower()

    filtered: list[dict[str, str]] = []
    for row in news_rows:
        text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
        if symbol_token in text or any(tok in text for tok in company_tokens):
            filtered.append(row)

    if not filtered:
        return 0.0, "Neutral", 0, []

    score, _ = _score_news_sentiment(
        [{"title": r.get("title", ""), "summary": r.get("summary", "")} for r in filtered]
    )
    sources = sorted({r.get("source", "Unknown") for r in filtered})
    return score, _label_from_sentiment(score), len(filtered), sources


def _project_ma_2_weeks(closes: pd.Series, window: int = 50) -> float | None:
    """Project moving average value after ~10 trading sessions."""
    ma = closes.rolling(window=window).mean().dropna()
    if len(ma) < 10:
        return None

    lookback = min(20, len(ma))
    y = ma.tail(lookback).values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    projected = intercept + slope * (len(y) - 1 + 10)
    return float(projected)


def _compute_rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Compute RSI indicator."""
    if closes is None or len(closes) < period + 2:
        return None
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    value = _to_float(rsi.iloc[-1])
    return value


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_alpha_vantage_market_sentiment() -> dict[str, Any]:
    """Optional: broad market sentiment from Alpha Vantage news feed."""
    api_key = _get_optional_config("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "source": "Alpha Vantage",
            "score": 0.0,
            "label": "Neutral",
            "sample_size": 0,
        }

    params = urllib.parse.urlencode(
        {
            "function": "NEWS_SENTIMENT",
            "topics": "financial_markets",
            "sort": "LATEST",
            "limit": "50",
            "apikey": api_key,
        }
    )
    url = f"https://www.alphavantage.co/query?{params}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        feed = payload.get("feed", []) if isinstance(payload, dict) else []
        scores: list[float] = []
        for item in feed:
            raw = item.get("overall_sentiment_score")
            if raw in (None, ""):
                continue
            try:
                scores.append(float(raw))
            except (TypeError, ValueError):
                continue

        if not scores:
            return {
                "enabled": True,
                "source": "Alpha Vantage",
                "score": 0.0,
                "label": "Neutral",
                "sample_size": 0,
            }

        sentiment = float(np.clip(np.mean(scores), -1.0, 1.0))
        return {
            "enabled": True,
            "source": "Alpha Vantage",
            "score": sentiment,
            "label": _label_from_sentiment(sentiment),
            "sample_size": len(scores),
        }
    except Exception:
        return {
            "enabled": True,
            "source": "Alpha Vantage",
            "score": 0.0,
            "label": "Neutral",
            "sample_size": 0,
        }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_pulse() -> dict[str, Any]:
    """Get broad market trend from NIFTY index."""
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)

    raw = yf.download(
        "^NSEI",
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    closes = _extract_close(raw, "^NSEI")
    av_sentiment = fetch_alpha_vantage_market_sentiment()
    av_score = float(av_sentiment.get("score", 0.0) or 0.0) if av_sentiment.get("enabled") else 0.0
    if closes is None or len(closes) < MA_WINDOW:
        return {
            "name": "NIFTY 50",
            "regime": "Sideways",
            "weekly_pct": 0.0,
            "score_boost": 0.0,
            "ext_sentiment_source": av_sentiment.get("source", "Alpha Vantage"),
            "ext_sentiment_label": av_sentiment.get("label", "Neutral"),
            "ext_sentiment_score": round(av_score, 3),
            "ext_sentiment_samples": int(av_sentiment.get("sample_size", 0) or 0),
        }

    price = float(closes.iloc[-1])
    ma50 = float(closes.rolling(window=MA_WINDOW).mean().iloc[-1])
    weekly_pct = float((price / closes.iloc[-6] - 1.0) * 100.0) if len(closes) >= 6 else 0.0

    if price > ma50 and weekly_pct > 0:
        regime = "Bullish"
        boost = 0.2
    elif price < ma50 and weekly_pct < 0:
        regime = "Bearish"
        boost = -0.15
    else:
        regime = "Sideways"
        boost = 0.0

    # Blend optional external sentiment API into market pulse.
    boost += float(np.clip(av_score * 0.12, -0.12, 0.12))
    if regime == "Sideways":
        if av_score >= 0.35 and weekly_pct >= -0.4 and price >= ma50 * 0.995:
            regime = "Bullish"
        elif av_score <= -0.35 and weekly_pct <= 0.4 and price <= ma50 * 1.005:
            regime = "Bearish"

    return {
        "name": "NIFTY 50",
        "regime": regime,
        "weekly_pct": round(weekly_pct, 2),
        "score_boost": round(boost, 4),
        "ext_sentiment_source": av_sentiment.get("source", "Alpha Vantage"),
        "ext_sentiment_label": av_sentiment.get("label", "Neutral"),
        "ext_sentiment_score": round(av_score, 3),
        "ext_sentiment_samples": int(av_sentiment.get("sample_size", 0) or 0),
    }


@st.cache_data(ttl=15, show_spinner=False)
def fetch_live_index_snapshot() -> dict[str, dict[str, float | str]]:
    """Fetch live-ish index values for header display."""
    index_map = {
        "BSE": "^BSESN",      # SENSEX
        "NSE": "^NSEI",       # NIFTY 50
        "NIFTY": "^NSEBANK",  # NIFTY BANK
    }
    out: dict[str, dict[str, float | str]] = {}

    for label, symbol in index_map.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns:
                out[label] = {"value": np.nan, "change_pct": np.nan, "symbol": symbol}
                continue

            closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            value = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else value
            change_points = value - prev
            change_pct = ((value / prev) - 1.0) * 100.0 if prev else 0.0
            out[label] = {
                "value": value,
                "change_pct": change_pct,
                "change_points": change_points,
                "symbol": symbol,
            }
        except Exception:
            out[label] = {
                "value": np.nan,
                "change_pct": np.nan,
                "change_points": np.nan,
                "symbol": symbol,
            }

    return out


@st.cache_data(ttl=10800, show_spinner=False)
def fetch_stock_context(ticker: str) -> dict[str, Any]:
    """Fetch fundamentals, event data, and recent news for one stock."""
    tkr = yf.Ticker(ticker)
    info: dict[str, Any] = {}
    news_rows: list[dict[str, Any]] = []

    try:
        info = tkr.get_info() or {}
    except Exception:
        info = {}

    try:
        raw_news = tkr.news or []
        for item in raw_news[:8]:
            title = item.get("title", "")
            summary = item.get("summary", "")
            if not title and not summary:
                continue
            news_rows.append(
                {
                    "title": title,
                    "summary": summary,
                    "publisher": item.get("publisher", "Unknown"),
                    "link": item.get("link", ""),
                    "published": item.get("providerPublishTime"),
                }
            )
    except Exception:
        news_rows = []

    try:
        hist = tkr.history(period="6mo", interval="1d", auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()

    volume_ratio = None
    up_down_volume_ratio = None
    avg_volume_5d = None
    current_volume = None
    if not hist.empty and "Volume" in hist.columns and "Close" in hist.columns:
        vol = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0.0)
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(vol) >= 1:
            current_volume = float(vol.iloc[-1])
        if len(vol) >= 5:
            avg_volume_5d = float(vol.tail(5).mean())
        if len(vol) >= 25:
            recent_vol = float(vol.tail(5).mean())
            base_vol = float(vol.tail(25).head(20).mean())
            if base_vol > 0:
                volume_ratio = recent_vol / base_vol
        if len(close) >= 6:
            up_mask = close.diff().fillna(0) > 0
            down_mask = close.diff().fillna(0) < 0
            up_vol = float(vol[up_mask].tail(10).sum())
            down_vol = float(vol[down_mask].tail(10).sum())
            if down_vol > 0:
                up_down_volume_ratio = up_vol / down_vol

    earnings_ts = info.get("earningsTimestamp")
    days_to_earnings = None
    if earnings_ts:
        try:
            earnings_date = dt.datetime.fromtimestamp(int(earnings_ts)).date()
            days_to_earnings = (earnings_date - dt.date.today()).days
        except Exception:
            days_to_earnings = None

    sentiment_score, sentiment_label = _score_news_sentiment(news_rows)
    promoter_stake = _to_float(info.get("heldPercentInsiders"))
    if promoter_stake is not None and promoter_stake <= 1.0:
        promoter_stake *= 100
    sector_name = _clean_text(info.get("sector"))
    industry_name = _clean_text(info.get("industry"))

    total_assets = _to_float(info.get("totalAssets"))
    total_debt = _to_float(info.get("totalDebt"))
    debt_to_assets = None
    if total_assets and total_assets > 0 and total_debt is not None:
        debt_to_assets = total_debt / total_assets

    return {
        "promoter_stake_pct": promoter_stake,
        "sector_name": sector_name,
        "industry_name": industry_name,
        "total_assets": total_assets,
        "total_debt": total_debt,
        "debt_to_assets": debt_to_assets,
        "revenue_growth": _to_float(info.get("revenueGrowth")),
        "profit_margin": _to_float(info.get("profitMargins")),
        "news_rows": news_rows,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "volume_ratio": volume_ratio,
        "up_down_volume_ratio": up_down_volume_ratio,
        "avg_volume_5d": avg_volume_5d,
        "current_volume": current_volume,
        "days_to_earnings": days_to_earnings,
    }


def analyse_stock(
    ticker: str,
    name: str,
    closes: pd.Series,
    market_pulse: dict[str, Any],
    external_news_rows: list[dict[str, str]],
    component_weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Compute technical/fundamental/news score and recommendation row."""
    if closes is None or len(closes) < MA_WINDOW:
        return None

    ma50 = float(closes.rolling(window=MA_WINDOW).mean().iloc[-1])
    ma20 = float(closes.rolling(window=20).mean().iloc[-1])
    price = float(closes.iloc[-1])
    if np.isnan(ma50) or ma50 <= 0 or np.isnan(price):
        return None
    if price > MAX_STOCK_PRICE:
        # Global budget guardrail: show only affordable stocks across the portal.
        return None

    ma200 = float(closes.rolling(window=200).mean().iloc[-1]) if len(closes) >= 200 else np.nan
    vs_ma50 = ((price - ma50) / ma50) * 100.0
    below_ma = price < ma50

    year_window = closes.tail(252) if len(closes) >= 252 else closes
    high_52 = float(year_window.max())
    low_52 = float(year_window.min())
    dist_to_high_pct = ((high_52 - price) / high_52 * 100.0) if high_52 > 0 else np.nan
    dist_to_low_pct = ((price - low_52) / low_52 * 100.0) if low_52 > 0 else np.nan

    weekly_momentum = float((price / closes.iloc[-6] - 1.0) * 100.0) if len(closes) >= 6 else 0.0
    rsi = _compute_rsi(closes)

    context = fetch_stock_context(ticker)
    promoter_stake = context.get("promoter_stake_pct")
    debt_to_assets = context.get("debt_to_assets")
    revenue_growth = context.get("revenue_growth")
    profit_margin = context.get("profit_margin")
    sentiment_score = float(context.get("sentiment_score", 0.0))
    sentiment_label = context.get("sentiment_label", "Neutral")
    volume_ratio = context.get("volume_ratio")
    up_down_volume_ratio = context.get("up_down_volume_ratio")
    avg_volume_5d = context.get("avg_volume_5d")
    days_to_earnings = context.get("days_to_earnings")
    cmp_value = round(price, 2)
    day_change_rupees = float(closes.iloc[-1] - closes.iloc[-2]) if len(closes) >= 2 else 0.0
    day_change_pct = (
        float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0) if len(closes) >= 2 else 0.0
    )

    ext_sentiment_score, ext_sentiment_label, ext_hit_count, ext_sources = (
        _score_external_news_for_stock(name, ticker.replace(".NS", ""), external_news_rows)
    )
    combined_sentiment_score = float(
        np.clip(0.65 * sentiment_score + 0.35 * ext_sentiment_score, -1.0, 1.0)
    )
    combined_sentiment_label = _label_from_sentiment(combined_sentiment_score)
    projected_ma_2w = _project_ma_2_weeks(closes, window=MA_WINDOW)

    # ------------------------
    # Scoring model (0 to 100)
    # ------------------------
    technical_score = 0.0
    if price > ma20 > ma50:
        technical_score += 12
    elif price > ma50:
        technical_score += 9
    elif price > ma20:
        technical_score += 6
    else:
        technical_score += 3

    if -7 <= vs_ma50 <= 4:
        technical_score += 8
    elif vs_ma50 < -7:
        technical_score += 6
    else:
        technical_score += 4

    if 0 <= dist_to_high_pct <= 12:
        technical_score += 8
    elif dist_to_high_pct <= 22:
        technical_score += 5
    else:
        technical_score += 2

    if weekly_momentum > 2.0:
        technical_score += 8
    elif weekly_momentum > 0:
        technical_score += 6
    elif weekly_momentum > -2:
        technical_score += 4
    else:
        technical_score += 2

    if rsi is not None:
        if 48 <= rsi <= 66:
            technical_score += 8
        elif 38 <= rsi < 48 or 66 < rsi <= 75:
            technical_score += 5
        else:
            technical_score += 3
    else:
        technical_score += 4

    fundamental_score = 0.0
    if promoter_stake is not None:
        if promoter_stake >= 55:
            fundamental_score += 9
        elif promoter_stake >= 35:
            fundamental_score += 7
        elif promoter_stake >= 20:
            fundamental_score += 5
        else:
            fundamental_score += 3
    else:
        fundamental_score += 4

    if debt_to_assets is not None:
        if debt_to_assets <= 0.35:
            fundamental_score += 9
        elif debt_to_assets <= 0.65:
            fundamental_score += 7
        elif debt_to_assets <= 1.0:
            fundamental_score += 5
        else:
            fundamental_score += 2
    else:
        fundamental_score += 4

    if revenue_growth is not None:
        if revenue_growth >= 0.15:
            fundamental_score += 8
        elif revenue_growth >= 0.05:
            fundamental_score += 6
        elif revenue_growth > 0:
            fundamental_score += 4
        else:
            fundamental_score += 2
    else:
        fundamental_score += 4

    if profit_margin is not None:
        if profit_margin >= 0.15:
            fundamental_score += 6
        elif profit_margin >= 0.08:
            fundamental_score += 5
        elif profit_margin > 0:
            fundamental_score += 3
        else:
            fundamental_score += 1
    else:
        fundamental_score += 4

    flow_score = 0.0
    flow_score += float(np.interp(combined_sentiment_score, [-1, 1], [1, 8]))

    if volume_ratio is not None:
        if volume_ratio >= 1.3:
            flow_score += 6
        elif volume_ratio >= 1.05:
            flow_score += 5
        elif volume_ratio >= 0.85:
            flow_score += 3
        else:
            flow_score += 1
    else:
        flow_score += 3

    if up_down_volume_ratio is not None:
        if up_down_volume_ratio >= 1.25:
            flow_score += 5
        elif up_down_volume_ratio >= 1.0:
            flow_score += 4
        elif up_down_volume_ratio >= 0.8:
            flow_score += 3
        else:
            flow_score += 1
    else:
        flow_score += 3

    if days_to_earnings is not None and 0 <= days_to_earnings <= 14:
        flow_score += 4
    elif days_to_earnings is not None and days_to_earnings < 0:
        flow_score += 2
    else:
        flow_score += 3

    dip_resilience_bonus = 0.0
    if market_pulse.get("regime") == "Bearish":
        if debt_to_assets is not None and debt_to_assets <= 0.6:
            dip_resilience_bonus += 4
        if promoter_stake is not None and promoter_stake >= 35:
            dip_resilience_bonus += 3
        if combined_sentiment_score >= 0:
            dip_resilience_bonus += 2
        if dist_to_low_pct <= 18:
            dip_resilience_bonus += 2

    weights = component_weights or DEFAULT_COMPONENT_WEIGHTS
    tech_w = float(weights.get("technical", DEFAULT_COMPONENT_WEIGHTS["technical"]))
    fund_w = float(weights.get("fundamental", DEFAULT_COMPONENT_WEIGHTS["fundamental"]))
    flow_w = float(weights.get("flow", DEFAULT_COMPONENT_WEIGHTS["flow"]))
    resilience_w = float(weights.get("resilience", DEFAULT_COMPONENT_WEIGHTS["resilience"]))

    raw_score = (
        technical_score * tech_w
        + fundamental_score * fund_w
        + flow_score * flow_w
        + dip_resilience_bonus * resilience_w
    )
    raw_score = raw_score + float(market_pulse.get("score_boost", 0.0)) * 10
    raw_score = float(np.clip(raw_score, 10, 100))
    confidence = round(float(np.clip(raw_score / 10.0, 1.0, 10.0)), 1)

    bullish = price > ma20 > ma50 and weekly_momentum > 0 and (rsi is None or rsi >= 48)
    bearish = price < ma20 < ma50 and weekly_momentum < 0 and (rsi is None or rsi <= 45)
    trend = "Bullish" if bullish else ("Bearish" if bearish else "Neutral")

    if confidence >= 8.2 and trend == "Bullish" and combined_sentiment_score >= -0.05:
        signal = "BUY"
    elif confidence >= 7.5 and market_pulse.get("regime") == "Bearish":
        signal = "BUY ON DIP"
    elif confidence >= 6.5:
        signal = "WATCH"
    elif confidence >= 5.0:
        signal = "HOLD"
    else:
        signal = "SELL / AVOID"

    can_shoot_week = (
        confidence >= 8.0
        and weekly_momentum > 1.0
        and (volume_ratio is None or volume_ratio >= 1.05)
        and combined_sentiment_score >= 0
    )
    if signal in {"BUY", "BUY ON DIP"}:
        exit_days = 7 if can_shoot_week else 5
        exit_preference = (dt.date.today() + dt.timedelta(days=exit_days)).isoformat()
    elif signal == "WATCH":
        exit_preference = (dt.date.today() + dt.timedelta(days=3)).isoformat()
    else:
        exit_preference = "N/A"

    if signal in {"BUY", "BUY ON DIP"}:
        met_expectation = "YES" if day_change_pct >= 0 else "NO"
        buy_window_days = "0-3"
        sell_window_days = "5-10"
    elif signal == "WATCH":
        met_expectation = "MIXED" if abs(day_change_pct) < 0.6 else ("YES" if day_change_pct > 0 else "NO")
        buy_window_days = "2-7"
        sell_window_days = "8-14"
    elif signal == "HOLD":
        met_expectation = "MIXED"
        buy_window_days = "NA"
        sell_window_days = "7-14"
    else:
        met_expectation = "YES" if day_change_pct <= 0 else "NO"
        buy_window_days = "NA"
        sell_window_days = "0-4"

    if signal in {"BUY", "BUY ON DIP"}:
        exit_multiplier = 1.06 if can_shoot_week else 1.04
        reason = (
            f"{trend} trend, confidence {confidence}/10, sentiment {combined_sentiment_label}, "
            f"volume ratio {(volume_ratio or 1.0):.2f}."
        )
    elif signal == "WATCH":
        exit_multiplier = 1.03
        reason = (
            f"Near-entry setup: confidence {confidence}/10 with {combined_sentiment_label.lower()} "
            f"sentiment; wait for stronger trigger."
        )
    elif signal == "HOLD":
        exit_multiplier = 1.02
        reason = (
            f"Mixed setup with moderate confidence {confidence}/10; monitor results and volume trend."
        )
    else:
        exit_multiplier = 0.97
        reason = (
            f"Weak setup: confidence {confidence}/10, risk-control stance due to momentum/sentiment."
        )

    proposed_exit_price = round(price * exit_multiplier, 2)

    rolling_vol = (
        float(closes.pct_change().dropna().tail(14).std() * 100.0)
        if len(closes) >= 20
        else 2.1
    )
    base_risk_pct = float(np.clip(max(1.8, rolling_vol * 1.35), 1.8, 4.8))
    swing_time_stop_days = 10 if signal in {"BUY", "BUY ON DIP"} else (7 if signal == "WATCH" else None)
    if signal in {"BUY", "BUY ON DIP"}:
        entry_low = round(price * 0.995, 2)
        entry_high = round(price * 1.01, 2)
        stop_loss = round(price * (1.0 - base_risk_pct / 100.0), 2)
        target_1 = round(price * 1.05, 2)
        target_2 = round(price * (1.08 if can_shoot_week else 1.10), 2)
        swing_grade = "A" if confidence >= 8.0 and combined_sentiment_score >= 0 else "B"
    elif signal == "WATCH":
        entry_low = round(price * 0.985, 2)
        entry_high = round(price * 1.0, 2)
        stop_loss = round(price * (1.0 - min(base_risk_pct + 0.5, 5.5) / 100.0), 2)
        target_1 = round(price * 1.04, 2)
        target_2 = round(price * 1.07, 2)
        swing_grade = "C"
    else:
        entry_low = None
        entry_high = None
        stop_loss = None
        target_1 = None
        target_2 = None
        swing_grade = "NA"
    rr_target_1 = _format_rr(price, stop_loss, target_1) if stop_loss is not None and target_1 is not None else None
    rr_target_2 = _format_rr(price, stop_loss, target_2) if stop_loss is not None and target_2 is not None else None

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Company": name,
        "CMP (₹)": cmp_value,
        "Price (₹)": round(price, 2),
        "52W High (₹)": round(high_52, 2),
        "50-Day MA (₹)": round(ma50, 2),
        "Expected MA (2W, ₹)": round(projected_ma_2w, 2) if projected_ma_2w is not None else None,
        "200-Day MA (₹)": round(ma200, 2) if not np.isnan(ma200) else None,
        "vs MA (%)": round(vs_ma50, 2),
        "Signal": signal,
        "Below MA": below_ma,
        "Trend": trend,
        "Weekly Momentum (%)": round(weekly_momentum, 2),
        "RSI(14)": round(rsi, 2) if rsi is not None else None,
        "52W High Gap (%)": round(dist_to_high_pct, 2) if not np.isnan(dist_to_high_pct) else None,
        "52W Low Gap (%)": round(dist_to_low_pct, 2) if not np.isnan(dist_to_low_pct) else None,
        "Promoter / Insider Stake (%)": round(promoter_stake, 2) if promoter_stake is not None else None,
        "Industry Sector": context.get("sector_name") or context.get("industry_name") or "NA",
        "Assets (₹ Cr)": round((context.get("total_assets") or 0.0) / 1e7, 2)
        if context.get("total_assets")
        else None,
        "Liabilities (Debt, ₹ Cr)": round((context.get("total_debt") or 0.0) / 1e7, 2)
        if context.get("total_debt")
        else None,
        "Debt/Assets": round(debt_to_assets, 2) if debt_to_assets is not None else None,
        "Sales Growth (%)": round(revenue_growth * 100, 2) if revenue_growth is not None else None,
        "Profit Margin (%)": round(profit_margin * 100, 2) if profit_margin is not None else None,
        "Volume Surge (5d/20d)": round(volume_ratio, 2) if volume_ratio is not None else None,
        "Buy/Sell Volume Ratio": round(up_down_volume_ratio, 2)
        if up_down_volume_ratio is not None
        else None,
        "News Sentiment": combined_sentiment_label,
        "Product Sentiment Score": round(combined_sentiment_score, 2),
        "Day changed": round(day_change_rupees, 2),
        "Daily Change (%)": round(day_change_pct, 2),
        "Results Announcement (days)": days_to_earnings,
        "Market Pulse": market_pulse.get("regime", "Sideways"),
        "Action": signal,
        "Recommendation": signal,
        "Entry Zone (₹)": f"{entry_low} - {entry_high}" if entry_low is not None and entry_high is not None else "NA",
        "Entry Price (₹)": entry_low,
        "Stop Loss (₹)": stop_loss,
        "Target 1 (₹)": target_1,
        "Target 2 (₹)": target_2,
        "Time Stop (days)": swing_time_stop_days if swing_time_stop_days is not None else "NA",
        "RR to T1": rr_target_1,
        "RR to T2": rr_target_2,
        "Swing Grade": swing_grade,
        "Reason for Recommendation": reason,
        "Proposed Exit Price (₹)": proposed_exit_price,
        "Met Expectation Today": met_expectation,
        "Buy Window (next 2w, days)": buy_window_days,
        "Sell Window (next 2w, days)": sell_window_days,
        "Exit Preference": exit_preference,
        "1W Potential": "High" if can_shoot_week else "Moderate",
        "Confidence (1-10)": confidence,
        "Sources Used": ", ".join(ext_sources) if ext_sources else "Yahoo Finance",
        "External News Hits": ext_hit_count,
        "Avg Volume 5D": round(avg_volume_5d, 0) if avg_volume_5d is not None else None,
        "Current Volume": round(float(context.get("current_volume")), 0)
        if context.get("current_volume") is not None
        else None,
        "_score_raw": raw_score,
        "_component_scores": {
            "technical": round(technical_score, 4),
            "fundamental": round(fundamental_score, 4),
            "flow": round(flow_score, 4),
            "resilience": round(dip_resilience_bonus, 4),
        },
        "_component_weights": {
            "technical": round(tech_w, 4),
            "fundamental": round(fund_w, 4),
            "flow": round(flow_w, 4),
            "resilience": round(resilience_w, 4),
        },
        "_news_rows": context.get("news_rows", []),
        "_closes": closes,
        "_ma": closes.rolling(window=MA_WINDOW).mean(),
    }


def build_chart(company: str, closes: pd.Series, ma: pd.Series) -> go.Figure:
    """Price vs 50-day moving average chart (dark trading theme)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=closes.index,
            y=closes.values,
            name="Close Price",
            line=dict(color="#387ED1", width=2),
            fill="tozeroy",
            fillcolor="rgba(56, 126, 209, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ma.index,
            y=ma.values,
            name="50-Day MA",
            line=dict(color="#FF9800", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{company} — Price vs 50-Day MA",
            font=dict(color="#E8EDF4", size=15),
        ),
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#111821",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(color="#9AA6B5", family="IBM Plex Sans, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor="#1E2835", zerolinecolor="#1E2835"),
        yaxis=dict(gridcolor="#1E2835", zerolinecolor="#1E2835"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_portfolio_pie(df: pd.DataFrame) -> go.Figure:
    """Build a portfolio allocation pie chart by current value."""
    top_n = 8
    pie_df = df[["Symbol", "Current (₹)"]].copy()
    pie_df = pie_df.sort_values("Current (₹)", ascending=False).reset_index(drop=True)
    if len(pie_df) > top_n:
        others_value = float(pie_df.iloc[top_n:]["Current (₹)"].sum())
        pie_df = pie_df.iloc[:top_n].copy()
        pie_df.loc[len(pie_df)] = {"Symbol": "Others", "Current (₹)": others_value}

    fig = go.Figure(
        data=[
            go.Pie(
                labels=pie_df["Symbol"],
                values=pie_df["Current (₹)"],
                hole=0.45,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(line=dict(color="#0B0F14", width=1)),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Portfolio Allocation (Current Value)", font=dict(size=14)),
        template="plotly_dark",
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#111821",
        margin=dict(l=20, r=20, t=45, b=20),
        height=420,
        font=dict(color="#E8EDF4"),
        showlegend=True,
    )
    return fig


def build_bullish_candidates_pie(df: pd.DataFrame) -> go.Figure:
    """Build favorability-share pie for selected bullish candidates."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["Symbol"],
                values=df["Bullish Score"],
                hole=0.42,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(line=dict(color="#0B0F14", width=1)),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Bullish Favorability Share (Top 6)", font=dict(size=14)),
        template="plotly_dark",
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#111821",
        margin=dict(l=20, r=20, t=45, b=20),
        height=420,
        font=dict(color="#E8EDF4"),
        showlegend=True,
    )
    return fig


def style_holdings_dataframe(df: pd.DataFrame):
    """Color P&L columns green/red for quick gain-loss visibility."""
    color_pos = "color: #00D09C; font-weight: 600;"
    color_neg = "color: #FF6B6B; font-weight: 600;"
    color_neutral = "color: #9AA6B5;"

    def pnl_style(value: Any) -> str:
        if value is None or pd.isna(value):
            return color_neutral
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("+"):
                return color_pos
            if text.startswith("-"):
                return color_neg
            return color_neutral
        numeric = float(value)
        if numeric > 0:
            return color_pos
        if numeric < 0:
            return color_neg
        return color_neutral

    style_targets = [c for c in ["Day changed", "Since added", "P&L (₹)", "P&L (%)"] if c in df.columns]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    decimal_format_map = {c: "{:,.2f}" for c in numeric_cols}
    if "P&L (%)" in decimal_format_map:
        decimal_format_map["P&L (%)"] = "{:+.2f}%"
    return (
        df.style.format(decimal_format_map)
        .map(pnl_style, subset=style_targets)
    )


def style_met_expectation_dataframe(df: pd.DataFrame):
    """Color expectation-status cells for quick actionability scan."""
    color_yes = "background-color: #123B2A; color: #A7F3D0; font-weight: 700;"
    color_mixed = "background-color: #4A2F00; color: #FCD9A5; font-weight: 700;"
    color_exit = "background-color: #4A1717; color: #FECACA; font-weight: 700;"
    color_neutral = "color: #9AA6B5;"

    def expectation_style(value: Any) -> str:
        if value is None or pd.isna(value):
            return color_neutral
        text = str(value).strip().upper()
        if text == "YES":
            return color_yes
        if text == "MIXED":
            return color_mixed
        if text in {"EXIT", "NO", "SELL / AVOID"}:
            return color_exit
        return color_neutral

    met_cols = [
        c
        for c in df.columns
        if c.strip().upper() in {"MET EXP", "MET EXPECTATION TODAY", "MET EXPECTATIONS"}
    ]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    decimal_format_map = {c: "{:,.2f}" for c in numeric_cols}

    styler = df.style.format(decimal_format_map) if decimal_format_map else df.style
    if met_cols:
        styler = styler.map(expectation_style, subset=met_cols)
    return styler


@st.cache_data(show_spinner=False)
def load_logo_data_uri() -> str:
    """Return BharatLens logo as data URI for HTML embedding."""
    logo_path = (
        CUSTOM_LOGO_PATH
        if USE_CUSTOM_LOGO and CUSTOM_LOGO_PATH.exists()
        else DEFAULT_LOGO_PATH
    )
    if not logo_path.exists():
        return ""

    # For optional raster logos, remove near-white background and crop tight for visibility.
    suffix = logo_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        img = Image.open(logo_path).convert("RGBA")
        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        # Make near-white background transparent.
        white_mask = (rgb[:, :, 0] > 242) & (rgb[:, :, 1] > 242) & (rgb[:, :, 2] > 242)
        alpha[white_mask] = 0
        arr[:, :, 3] = alpha

        # Crop to non-transparent content with a small safe padding.
        non_transparent = np.argwhere(alpha > 0)
        if non_transparent.size > 0:
            y0, x0 = non_transparent.min(axis=0)
            y1, x1 = non_transparent.max(axis=0)
            pad = 8
            y0 = max(0, y0 - pad)
            x0 = max(0, x0 - pad)
            y1 = min(arr.shape[0] - 1, y1 + pad)
            x1 = min(arr.shape[1] - 1, x1 + pad)
            arr = arr[y0 : y1 + 1, x0 : x1 + 1]

        out = Image.fromarray(arr, mode="RGBA")
        buff = io.BytesIO()
        out.save(buff, format="PNG")
        raw = buff.getvalue()
        mime = "image/png"
    else:
        raw = logo_path.read_bytes()
        mime = "image/svg+xml"

    encoded = base64.b64encode(raw).decode("utf-8")
    mime_map = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    if suffix in mime_map:
        mime = mime_map.get(suffix, mime)
    return f"data:{mime};base64,{encoded}"


def inject_trading_theme() -> None:
    """Apply a Zerodha/Upstox-style dark trading terminal look."""
    css = """
        <style>
        :root {
            --bl-bg: #0B0F14;
            --bl-panel: #151B24;
            --bl-panel-2: #1A2230;
            --bl-border: #243041;
            --bl-text: #E8EDF4;
            --bl-muted: #E8EDF4;
            --bl-blue: #387ED1;
            --bl-green: #00B386;
            --bl-red: #EB5336;
            --bl-orange: #FF9800;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--bl-bg) !important;
            color: var(--bl-text) !important;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
        }

        /* Keep Streamlit header visible so sidebar expand/collapse toggle remains available */
        [data-testid="stHeader"] {
            background: transparent !important;
            border: 0 !important;
        }
        [data-testid="collapsedControl"] {
            position: sticky !important;
            top: 0.4rem !important;
            left: 0.55rem !important;
            z-index: 1200 !important;
        }
        [data-testid="collapsedControl"] button {
            border: 1px solid #2f4764 !important;
            border-radius: 8px !important;
            background: rgba(20, 30, 44, 0.92) !important;
            color: #dff0ff !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        }
        [data-testid="collapsedControl"] button:hover {
            border-color: #3f6490 !important;
            background: rgba(28, 43, 63, 0.96) !important;
        }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        #MainMenu { visibility: hidden !important; }

        [data-testid="stSidebar"] {
            background: #0E131A !important;
            border-right: 1px solid var(--bl-border) !important;
        }
        [data-testid="stSidebar"] * { color: var(--bl-text); }
        [data-testid="stSidebar"] div[role="radiogroup"] {
            background: #10161F;
            border: 1px solid var(--bl-border);
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.35rem 0.45rem;
            border-radius: 8px;
            margin: 0.1rem 0;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #162233;
            border-color: #274260;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(56, 126, 209, 0.2);
            border-color: #387ED1;
            box-shadow: inset 0 0 0 1px rgba(56, 126, 209, 0.45);
        }

        .block-container {
            padding-top: 1.1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px;
        }

        /* Top brand bar */
        .bl-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.95rem 1.15rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(600px 80px at 10% -20%, rgba(56, 126, 209, 0.25), transparent 60%),
                linear-gradient(180deg, #182232 0%, #111a26 100%);
            border: 1px solid #2f4764;
            border-radius: 10px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        .bl-brand {
            display: flex;
            align-items: baseline;
            gap: 0.55rem;
        }
        .bl-logo-wrap {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .bl-logo-icon {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(140deg, #387ED1 0%, #244f84 100%);
            color: #ffffff;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .bl-logo-img {
            width: 30px;
            height: 30px;
            border-radius: 8px;
            box-shadow: 0 0 0 1px rgba(56, 126, 209, 0.55);
        }
        .bl-logo {
            font-size: 1.42rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
            text-shadow: 0 1px 12px rgba(56, 126, 209, 0.3);
        }
        .bl-logo span { color: var(--bl-blue); }
        .bl-tag {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #E8EDF4;
            border: 1px solid #33506f;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            background: rgba(26, 38, 54, 0.7);
        }
        .bl-status {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.82rem;
            color: #E8EDF4;
            font-family: "IBM Plex Mono", monospace;
        }
        .bl-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--bl-green);
            box-shadow: 0 0 0 3px rgba(0, 179, 134, 0.18);
        }
        .bl-hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.45rem;
            padding: 1.2rem 1.25rem;
            border-radius: 12px;
            border: 1px solid var(--bl-border);
            background:
                radial-gradient(1200px 240px at 5% -40%, rgba(56, 126, 209, 0.35), transparent 55%),
                linear-gradient(135deg, #131B27 0%, #0E141D 100%);
        }
        .bl-hero-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .bl-signature-wrap {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .bl-signature-label {
            display: inline-flex;
            width: fit-content;
            padding: 0.1rem 0.45rem;
            border-radius: 6px;
            background: rgba(56, 126, 209, 0.16);
            border: 1px solid rgba(56, 126, 209, 0.34);
            color: #BFD8F5;
            font-size: 0.62rem;
            font-family: "IBM Plex Mono", monospace;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .bl-signature {
            display: inline-block;
            width: fit-content;
            margin-top: 0.05rem;
            padding: 0.02rem 0.1rem 0.16rem 0.1rem;
            border-bottom: 2px solid rgba(0, 179, 134, 0.55);
            color: #FFD978;
            font-size: 1.08rem;
            line-height: 1.1;
            font-family: Georgia, "Times New Roman", serif;
            font-style: italic;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-shadow: 0 1px 12px rgba(255, 189, 66, 0.28);
        }
        .bl-signature-wrap.bl-signature-style-minimal .bl-signature-label {
            background: rgba(232, 237, 244, 0.08);
            border-color: rgba(232, 237, 244, 0.26);
            color: #C9D3DF;
        }
        .bl-signature-wrap.bl-signature-style-minimal .bl-signature {
            border-bottom-color: rgba(173, 185, 199, 0.55);
            color: #FFD978;
            font-style: normal;
            font-weight: 500;
            letter-spacing: 0.02em;
            text-shadow: none;
        }
        .bl-signature-wrap.bl-signature-style-neon .bl-signature-label {
            background: rgba(0, 179, 134, 0.18);
            border-color: rgba(0, 179, 134, 0.45);
            color: #C7FFE7;
        }
        .bl-signature-wrap.bl-signature-style-neon .bl-signature {
            border-bottom-color: rgba(56, 126, 209, 0.85);
            color: #FFD978;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            font-style: normal;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            text-shadow:
                0 0 8px rgba(255, 189, 66, 0.55),
                0 0 16px rgba(255, 154, 32, 0.35);
        }
        .bl-hero-logo {
            width: 188px;
            height: 92px;
            border-radius: 12px;
            background: transparent !important;
            box-shadow: none;
            object-fit: contain;
        }
        .bl-hero-logo-fallback {
            width: 84px;
            height: 84px;
            border-radius: 14px;
            background: linear-gradient(140deg,#387ED1 0%,#244f84 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 700;
            font-family: "IBM Plex Mono", monospace;
        }
        .bl-hero-title {
            color: #FFFFFF;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0;
            letter-spacing: -0.01em;
            text-shadow: 0 1px 12px rgba(56, 126, 209, 0.3);
        }
        .bl-hero-title span { color: var(--bl-blue); }
        .bl-hero-sub {
            color: var(--bl-text);
            font-size: 0.9rem;
        }
        .bl-hero-chips {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .bl-chip {
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            border: 1px solid #2B3D57;
            background: rgba(26, 38, 54, 0.75);
            color: #D6E2F1;
            font-size: 0.76rem;
            font-family: "IBM Plex Mono", monospace;
        }
        .bl-chip.green {
            border-color: rgba(0, 179, 134, 0.6);
            color: #A6F7E1;
            background: rgba(0, 179, 134, 0.14);
        }
        .bl-helptext {
            color: #CFE1F7;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.15rem 0 0.35rem 0;
        }
        .bl-helptext strong {
            color: #EAF2FF;
            font-weight: 700;
        }
        .bl-notewhite {
            color: #F3F7FF !important;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.2rem 0;
        }
        .bl-switch-caption {
            color: var(--bl-text);
            font-size: 0.9rem;
            margin: 0.1rem 0 0.05rem 0;
        }

        h1, h2, h3, h4 { color: var(--bl-text) !important; font-weight: 600 !important; }
        p, label, .stMarkdown, .stCaption { color: var(--bl-text) !important; }
        strong { color: var(--bl-text); }

        /* Metrics like trading KPI tiles */
        div[data-testid="stMetric"] {
            background: var(--bl-panel);
            border: 1px solid var(--bl-border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetric"] label {
            color: var(--bl-muted) !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-family: "IBM Plex Mono", monospace !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: var(--bl-panel);
            border: 1px solid var(--bl-border);
            border-radius: 10px;
            padding: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--bl-muted) !important;
            border-radius: 8px !important;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(56, 126, 209, 0.16) !important;
            color: #FFFFFF !important;
        }

        /* Inputs */
        .stTextInput input, .stSelectbox [data-baseweb="select"] > div {
            background: #0E131A !important;
            border: 1px solid var(--bl-border) !important;
            color: var(--bl-text) !important;
            border-radius: 8px !important;
        }

        /* Primary buttons — Kite blue */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--bl-blue) !important;
            border: 1px solid var(--bl-blue) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 18px rgba(56, 126, 209, 0.35);
        }
        .stButton > button {
            border-radius: 8px !important;
            border: 1px solid var(--bl-border) !important;
            background: var(--bl-panel-2) !important;
            color: var(--bl-text) !important;
        }
        /* Secondary buttons should still look clearly clickable */
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid*="secondary"],
        .stButton > button:not([kind="primary"]):not([data-testid*="primary"]) {
            background: linear-gradient(180deg, #6FAEF2 0%, #4F93E8 100%) !important;
            border: 1px solid #7DBAF8 !important;
            color: #F7FBFF !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(79, 147, 232, 0.35) !important;
        }
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid*="secondary"]:hover,
        .stButton > button:not([kind="primary"]):not([data-testid*="primary"]):hover {
            background: linear-gradient(180deg, #7FBCFA 0%, #5AA2EF 100%) !important;
            border-color: #9DCCFF !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(90, 162, 239, 0.45) !important;
        }
        .stButton > button span,
        .stButton > button div,
        .stButton > button p {
            color: inherit !important;
            opacity: 1 !important;
        }
        .stButton > button:disabled {
            opacity: 0.55 !important;
            cursor: not-allowed !important;
        }
        .stLinkButton > a {
            background: var(--bl-blue) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        /* Dataframes / tables */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--bl-border);
            border-radius: 10px;
            overflow: hidden;
            background: var(--bl-panel);
        }
        [data-testid="stTable"] {
            border: 1px solid var(--bl-border);
            border-radius: 10px;
            overflow: hidden;
            background: var(--bl-panel);
        }

        /* Alerts */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            border: 1px solid var(--bl-border) !important;
        }

        hr { border-color: var(--bl-border) !important; }

        /* Expander */
        [data-testid="stExpander"] {
            background: var(--bl-panel);
            border: 1px solid var(--bl-border);
            border-radius: 10px;
        }

        /* Footer strip */
        .bl-footer {
            margin-top: 1.25rem;
            padding-top: 0.85rem;
            border-top: 1px solid var(--bl-border);
            color: var(--bl-muted);
            font-size: 0.8rem;
            font-family: "IBM Plex Mono", monospace;
        }
        </style>
        """

    # Streamlit versions can render CSS text when style tags are sanitized in markdown.
    # Use st.html/components for reliable style injection.
    if hasattr(st, "html"):
        st.html(css)
    else:
        components.html(css, height=0, width=0)


def render_scanner_tab() -> None:
    # Auto-refresh paused for now.
    # st_autorefresh(interval=15_000, key="market_intel_autorefresh_15s")

    # Pull the action row closer to the hero banner.
    st.markdown("<div style='margin-top:-2.05rem;'></div>", unsafe_allow_html=True)

    run_scan = False
    force_refresh = False
    action_col_1, action_col_2, _ = st.columns([1.0, 1.0, 2.2], gap="small")
    with action_col_1:
        run_scan = st.button(
            "Run Full Market Intelligence Scan",
            type="primary",
            use_container_width=True,
        )
    with action_col_2:
        force_refresh = st.button(
            "Refresh & Re-evaluate",
            type="secondary",
            use_container_width=True,
            help="Clears cached data and reruns the full recommendation algorithm.",
        )
    st.markdown(
        """
        <div class="bl-helptext">
            <strong>Run Full Market Intelligence Scan:</strong> runs the model with current cached/live data.
            <strong>Refresh &amp; Re-evaluate:</strong> clears cache and recomputes all signals from scratch.
        </div>
        """,
        unsafe_allow_html=True,
    )
    refresh_interval = st.radio(
        "Live price refresh",
        options=["Off", "30 sec", "60 sec"],
        index=1,
        horizontal=True,
        key="scanner_live_refresh_interval",
    )
    if refresh_interval != "Off":
        interval_seconds = int(refresh_interval.split()[0])
        st_autorefresh(
            interval=interval_seconds * 1000,
            key=f"scanner_live_autorefresh_{interval_seconds}",
        )
        st.caption(
            f"Live CMP/day-change refresh is ON (every {interval_seconds} seconds)."
        )

    scan_click_dt = dt.datetime.now()
    scan_click_time = scan_click_dt.strftime("%d %b %Y · %H:%M:%S")
    if run_scan:
        st.session_state["last_scan_time"] = scan_click_time
        st.session_state["last_scan_epoch"] = scan_click_dt.timestamp()

    if force_refresh:
        st.session_state["last_full_refresh_time"] = scan_click_time
        st.session_state["last_full_refresh_epoch"] = scan_click_dt.timestamp()
        st.session_state["last_scan_time"] = scan_click_time
        st.session_state["last_scan_epoch"] = scan_click_dt.timestamp()
        fetch_market_data.clear()
        fetch_nse_equity_universe.clear()
        fetch_intraday_momentum_map.clear()
        fetch_live_price_snapshot.clear()
        fetch_market_pulse.clear()
        fetch_alpha_vantage_market_sentiment.clear()
        fetch_live_index_snapshot.clear()
        fetch_external_market_news.clear()
        fetch_stock_context.clear()
        st.session_state.pop("scan_results", None)
        st.session_state.pop("market_pulse", None)
        st.session_state.pop("external_news_rows", None)
        st.session_state.pop("scan_results_nifty50", None)
        st.session_state.pop("market_pulse_nifty50", None)
        st.session_state.pop("external_news_rows_nifty50", None)
        st.session_state.pop("scan_results_broad", None)
        st.session_state.pop("market_pulse_broad", None)
        st.session_state.pop("external_news_rows_broad", None)
        st.session_state.pop("broad_daily_movers", None)
        st.success("Data cache cleared. Re-evaluating all signals now.")
        run_scan = True

    def _freshness(time_key: str, epoch_key: str) -> tuple[str, str]:
        shown_time = st.session_state.get(time_key)
        epoch = st.session_state.get(epoch_key)
        if not shown_time or epoch in (None, ""):
            return "Never", "#E8EDF4"
        age_mins = max(0.0, (dt.datetime.now().timestamp() - float(epoch)) / 60.0)
        color = "#00B386" if age_mins <= 30 else "#FFB020"
        return f"{shown_time} ({age_mins:.0f}m ago)", color

    scan_text, scan_color = _freshness("last_scan_time", "last_scan_epoch")
    refresh_text, refresh_color = _freshness(
        "last_full_refresh_time", "last_full_refresh_epoch"
    )
    st.markdown(
        f"""
        <div class="bl-helptext" style="margin-top:0.15rem;">
            <strong>Last scan time:</strong>
            <span style="color:{scan_color};font-weight:600;">{scan_text}</span>
            &nbsp;|&nbsp;
            <strong>Last full refresh time:</strong>
            <span style="color:{refresh_color};font-weight:600;">{refresh_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    header_col, index_col = st.columns([1.75, 1.55], gap="medium")
    with header_col:
        st.subheader("Market Intelligence Dashboard")
        st.markdown(
            """
            <div class="bl-helptext">
                Multi-factor model using technicals, fundamentals, market pulse, buy/sell volume trend,
                earnings timing, and live headline sentiment from available sources.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="bl-helptext">
                <strong>Price filter active across portal:</strong> only stocks with CMP &lt;= ₹{MAX_STOCK_PRICE:,.0f}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with index_col:
        index_snapshot = fetch_live_index_snapshot()
        cards_html: list[str] = []
        for label in ("BSE", "NSE", "NIFTY"):
            info = index_snapshot.get(label, {})
            value = info.get("value", np.nan)
            chg = info.get("change_pct", np.nan)
            pts = info.get("change_points", np.nan)
            value_text = f"{value:,.2f}" if value == value else "NA"
            chg_text = f"{chg:+.2f}%" if chg == chg else "NA"
            pts_text = f"{pts:+,.2f} pts" if pts == pts else "NA"
            delta_text = f"{chg_text} · {pts_text}" if chg_text != "NA" and pts_text != "NA" else chg_text
            chg_color = "#00B386" if (chg == chg and chg >= 0) else "#EB5336"
            cards_html.append(
                f"""
                <div style="
                    flex:1;
                    min-width:0;
                    border:1px solid #2B3D57;
                    border-radius:10px;
                    background:#151B24;
                    padding:0.45rem 0.55rem;
                ">
                  <div style="color:#E8EDF4;font-size:0.72rem;line-height:1.1;">{label}</div>
                  <div style="color:#E8EDF4;font-size:1.02rem;font-weight:700;line-height:1.2;">{value_text}</div>
                  <div style="color:{chg_color};font-size:0.82rem;font-weight:600;white-space:nowrap;">{delta_text}</div>
                </div>
                """
            )
        index_html = f"""
        <div style="display:flex;gap:0.5rem;justify-content:flex-end;align-items:stretch;">
          {"".join(cards_html)}
        </div>
        """
        if hasattr(st, "html"):
            st.html(index_html)
        else:
            components.html(index_html, height=96, width=660)

    main_col = st.container()
    search_query = ""
    min_confidence = 5.5
    high_conviction_mode = st.toggle(
        "High Conviction (10% target) mode",
        value=bool(st.session_state.get("high_conviction_mode_enabled", False)),
        key="high_conviction_mode_enabled",
        help=(
            "Applies stricter BUY filtering: anti-chase day move cap, stronger momentum/volume flow, "
            "promoter >=40%, and ~10% upside viability in a 5-10 day hold window."
        ),
    )
    universe_scope = st.radio(
        "Scanner universe",
        options=["Broad market (NSE + custom)", "NIFTY 50 only"],
        horizontal=True,
        key="scanner_universe_scope",
    )
    custom_symbols_raw = ""
    top_mover_pool = 140
    if universe_scope == "Broad market (NSE + custom)":
        custom_symbols_raw = st.text_area(
            "Custom symbols (optional, comma/newline; supports .NS/.BO)",
            value=st.session_state.get("scanner_custom_symbols", ""),
            key="scanner_custom_symbols",
            placeholder="Example: OLAELEC.NS, PATELSAI.NS, TATAMOTORS, RELIANCE.BO",
        )
        top_mover_pool = int(
            st.slider(
                "Daily gainers considered for deep analysis",
                min_value=60,
                max_value=250,
                value=140,
                step=10,
                key="scanner_top_mover_pool",
            )
        )
    scope_key = "broad" if universe_scope == "Broad market (NSE + custom)" else "nifty50"
    scan_results_key = f"scan_results_{scope_key}"
    market_pulse_key = f"market_pulse_{scope_key}"
    external_news_key = f"external_news_rows_{scope_key}"
    previous_top20_key = f"previous_top20_symbols_{scope_key}"
    model_state = _load_model_state()
    current_regime = str(
        (st.session_state.get(market_pulse_key, {}) or {}).get("regime", "Sideways")
    )
    adaptive_weights = _get_regime_weights(model_state, current_regime)

    should_scan = run_scan or scan_results_key not in st.session_state
    if should_scan:
        if scope_key == "broad":
            universe_map = fetch_nse_equity_universe()
            custom_symbols = _parse_custom_symbols(custom_symbols_raw)
            for ticker in custom_symbols:
                if ticker not in universe_map:
                    universe_map[ticker] = ticker.replace(".NS", "").replace(".BO", "")
            tickers = tuple(universe_map.keys())
            scan_hint = (
                f"Scanning {len(tickers)} NSE/custom symbols for today's movers, "
                f"then deep-analysing top {top_mover_pool} + watchlist..."
            )
        else:
            universe_map = dict(NIFTY_50)
            tickers = tuple(universe_map.keys())
            custom_symbols = []
            scan_hint = "Collecting NIFTY 50 market, fundamentals, and multi-source news signals..."

        with st.spinner("Collecting market, fundamentals, and multi-source news signals… this can take 1-3 minutes"):
            try:
                st.caption(scan_hint)
                lookback_days = 140 if scope_key == "broad" else LOOKBACK_DAYS
                market = fetch_market_data(tickers, lookback_days=lookback_days)
                market_pulse = fetch_market_pulse()
                current_regime = str(market_pulse.get("regime", "Sideways"))
                adaptive_weights = _get_regime_weights(model_state, current_regime)
                external_news_rows = fetch_external_market_news()
            except Exception as exc:
                st.error("Could not download market data. Please check internet and retry.")
                st.exception(exc)
                return

        if not market:
            st.error("No price data was returned. Please try again in a few minutes.")
            return

        scan_targets: list[tuple[str, str]] = []
        if scope_key == "broad":
            mover_rows: list[dict[str, Any]] = []
            for ticker, closes in market.items():
                if closes is None or len(closes) < 2:
                    continue
                price = float(closes.iloc[-1])
                if price > MAX_STOCK_PRICE:
                    continue
                day_pct = float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)
                five_day_pct = (
                    float((closes.iloc[-1] / closes.iloc[-6] - 1.0) * 100.0)
                    if len(closes) >= 6
                    else None
                )
                mover_rows.append(
                    {
                        "Ticker": ticker,
                        "Symbol": ticker.replace(".NS", "").replace(".BO", ""),
                        "Company": universe_map.get(ticker, ticker),
                        "CMP (₹)": round(price, 2),
                        "Daily Change (%)": round(day_pct, 2),
                        "Increased % since last 5 days": round(five_day_pct, 2)
                        if five_day_pct is not None
                        else None,
                    }
                )
            mover_rows = sorted(mover_rows, key=lambda r: r["Daily Change (%)"], reverse=True)
            st.session_state["broad_daily_movers"] = mover_rows[: max(120, top_mover_pool)]
            candidate_tickers = [r["Ticker"] for r in mover_rows[:top_mover_pool]]
            for ticker in custom_symbols:
                if ticker not in candidate_tickers:
                    candidate_tickers.append(ticker)
            for ticker in NIFTY_50.keys():
                if ticker not in candidate_tickers:
                    candidate_tickers.append(ticker)
            scan_targets = [(ticker, universe_map.get(ticker, ticker)) for ticker in candidate_tickers]
        else:
            scan_targets = list(universe_map.items())

        rows: list[dict[str, Any]] = []
        feedback_returns: list[float] = []
        scan_date = dt.date.today().isoformat()
        progress = st.progress(0, text="Analysing scripts…")
        for i, (ticker, name) in enumerate(scan_targets, start=1):
            closes = market.get(ticker)
            if closes is not None:
                row = analyse_stock(
                    ticker,
                    name,
                    closes,
                    market_pulse,
                    external_news_rows,
                    component_weights=adaptive_weights,
                )
                if row is not None:
                    row["_yf_ticker"] = ticker
                    realized = _learn_from_row_feedback(
                        model_state, row, scan_date, current_regime=current_regime
                    )
                    if realized is not None:
                        feedback_returns.append(realized)
                    rows.append(row)
            progress.progress(i / max(len(scan_targets), 1), text=f"Analysing {name}…")
        progress.empty()
        _save_model_state(model_state)
        if feedback_returns:
            st.session_state["adaptive_feedback_batch"] = {
                "count": len(feedback_returns),
                "avg_return": round(float(np.mean(feedback_returns)), 3),
            }
        else:
            st.session_state["adaptive_feedback_batch"] = {"count": 0, "avg_return": 0.0}

        if not rows:
            st.error(
                f"No stocks matched the analysis under the active price cap (<= ₹{MAX_STOCK_PRICE:,.0f}). "
                "Try again later when prices or signals change."
            )
            return

        if scope_key == "broad":
            analysis_by_symbol = {str(r.get("Symbol", "")).strip(): r for r in rows}
            enriched_movers: list[dict[str, Any]] = []
            for mover in st.session_state.get("broad_daily_movers", []):
                merged = dict(mover)
                symbol_key = str(merged.get("Symbol", "")).strip()
                analysis_row = analysis_by_symbol.get(symbol_key, {})
                merged["Promoter Stake (%)"] = analysis_row.get("Promoter / Insider Stake (%)")
                merged["Industry Sector"] = analysis_row.get("Industry Sector")
                merged["Current Volume"] = analysis_row.get("Current Volume")
                merged["Entry Price (₹)"] = analysis_row.get("Entry Price (₹)")
                merged["Proposed Exit Price (₹)"] = analysis_row.get("Proposed Exit Price (₹)")
                merged["Proposed Exit (₹)"] = _format_exit_with_gain(
                    merged.get("Entry Price (₹)"),
                    merged.get("Proposed Exit Price (₹)"),
                )
                merged["Minimum Holding Period (days)"] = _extract_min_days(
                    analysis_row.get("Sell Window (next 2w, days)")
                ) or analysis_row.get("Time Stop (days)")
                enriched_movers.append(merged)
            st.session_state["broad_daily_movers"] = enriched_movers

        st.session_state[scan_results_key] = rows
        st.session_state[market_pulse_key] = market_pulse
        st.session_state[external_news_key] = external_news_rows

    results = st.session_state[scan_results_key]
    if results and (
        "Reason for Recommendation" not in results[0] or "Day changed" not in results[0]
    ):
        st.session_state.pop(scan_results_key, None)
        st.session_state.pop(market_pulse_key, None)
        st.session_state.pop(external_news_key, None)
        st.info("Refreshing scan data to include newly added analysis columns…")
        st.rerun()
    if results and any((r.get("CMP (₹)") or 0) > MAX_STOCK_PRICE for r in results):
        st.session_state.pop(scan_results_key, None)
        st.session_state.pop(market_pulse_key, None)
        st.session_state.pop(external_news_key, None)
        st.info(f"Refreshing scan data to apply price cap <= ₹{MAX_STOCK_PRICE:,.0f}…")
        st.rerun()
    market_pulse = st.session_state.get(market_pulse_key, fetch_market_pulse())
    current_regime = str(market_pulse.get("regime", "Sideways"))
    adaptive_weights = _get_regime_weights(model_state, current_regime)
    external_news_rows = st.session_state.get(external_news_key, [])
    sorted_results = sorted(results, key=lambda x: x["Confidence (1-10)"], reverse=True)

    if high_conviction_mode:
        demoted_count = 0
        high_conviction_buy_count = 0
        demoted_rows: list[dict[str, Any]] = []
        for row in sorted_results:
            passes_gate, failed_checks = _high_conviction_gate(row)
            row["_high_conviction_pass"] = passes_gate
            row["_high_conviction_failures"] = failed_checks
            rec = str(row.get("Recommendation", "")).upper().strip()
            if rec in {"BUY", "BUY ON DIP"}:
                if passes_gate:
                    high_conviction_buy_count += 1
                else:
                    proposed_entry = row.get("Entry Price (₹)") or row.get("CMP (₹)")
                    proposed_exit = row.get("Proposed Exit Price (₹)")
                    demoted_rows.append(
                        {
                            "Symbol": row.get("Symbol"),
                            "Company": row.get("Company"),
                            "Industry Sector": row.get("Industry Sector"),
                            "Confidence": row.get("Confidence (1-10)"),
                            "Promoter Stake (%)": row.get("Promoter / Insider Stake (%)"),
                            "Daily Change (%)": row.get("Daily Change (%)"),
                            "Weekly Momentum (%)": row.get("Weekly Momentum (%)"),
                            "Volume Surge (5d/20d)": row.get("Volume Surge (5d/20d)"),
                            "Buy/Sell Volume Ratio": row.get("Buy/Sell Volume Ratio"),
                            "Sell Window (days)": row.get("Sell Window (next 2w, days)"),
                            "Proposed Entry (₹)": proposed_entry,
                            "Proposed Exit (₹)": _format_exit_with_gain(proposed_entry, proposed_exit),
                            "Target 2 Upside (%)": _calc_exit_gain_pct(
                                proposed_entry,
                                row.get("Target 2 (₹)"),
                            ),
                            "Failed Checks": ", ".join(failed_checks),
                        }
                    )
                    row["Recommendation"] = "WATCH"
                    row["Reason for Recommendation"] = (
                        "High Conviction mode held entry: "
                        f"{', '.join(failed_checks[:3])}. Wait for setup improvement."
                    )
                    demoted_count += 1
        st.caption(
            "High Conviction mode active: "
            f"{high_conviction_buy_count} buy candidates passed strict gate; "
            f"{demoted_count} downgraded to WATCH."
        )
        if demoted_rows:
            st.markdown("### Why High Conviction gate rejected buys")
            st.dataframe(
                style_met_expectation_dataframe(
                    pd.DataFrame(demoted_rows).sort_values(
                        "Confidence", ascending=False, na_position="last"
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )

    for row in sorted_results:
        row["_since_added"] = _track_since_added_change(
            "scanner_since_added_baseline",
            row.get("Symbol"),
            row.get("CMP (₹)"),
        )
        row["_added_on"] = _get_since_added_date(
            "scanner_since_added_baseline",
            row.get("Symbol"),
        )

    filtered = sorted_results
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [
            r
            for r in sorted_results
            if q in r["Company"].lower() or q in r["Symbol"].lower()
        ]

    filtered = [r for r in filtered if r["Confidence (1-10)"] >= min_confidence]
    filtered = sorted(filtered, key=_confidence_value, reverse=True)
    if not filtered:
        st.warning("No scripts match your current search/filter. Lower the confidence filter and try again.")
        return

    top20 = filtered[:20]
    if len(top20) < 20:
        top20 = sorted_results[:20]
        main_col.info(
            "Fewer than 20 scripts matched your current filter/search, so the dashboard is "
            "showing the best available top 20 from the full universe."
        )
    top20 = sorted(top20, key=_confidence_value, reverse=True)
    _refresh_rows_with_live_prices(top20, "scanner_since_added_baseline", max_rows=20)
    intraday_tickers = tuple(
        dict.fromkeys(
            [
                str(
                    r.get("_yf_ticker")
                    or (
                        (str(r.get("Symbol", "")) if "." in str(r.get("Symbol", "")) else f"{r.get('Symbol', '')}.NS")
                    )
                )
                for r in top20
                if r.get("Symbol")
            ]
        )
    )
    mom_5m = fetch_intraday_momentum_map(intraday_tickers, interval="5m") if intraday_tickers else {}
    mom_15m = fetch_intraday_momentum_map(intraday_tickers, interval="15m") if intraday_tickers else {}
    for row in top20:
        ticker_key = str(
            row.get("_yf_ticker")
            or ((str(row.get("Symbol", "")) if "." in str(row.get("Symbol", "")) else f"{row.get('Symbol', '')}.NS"))
        )
        row["_mom_5m"] = mom_5m.get(ticker_key)
        row["_mom_15m"] = mom_15m.get(ticker_key)
    buy_now = sorted(
        [r for r in top20 if r["Recommendation"] in {"BUY", "BUY ON DIP"}],
        key=_confidence_value,
        reverse=True,
    )
    watch_list = sorted(
        [r for r in top20 if r["Recommendation"] == "WATCH"],
        key=_confidence_value,
        reverse=True,
    )
    high_week = sorted(
        [r for r in filtered if r["1W Potential"] == "High"],
        key=_confidence_value,
        reverse=True,
    )

    with main_col:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ranked universe", len(filtered))
        c2.metric("Top 20 Buy signals", len(buy_now))
        c3.metric("Top 20 Watch signals", len(watch_list))
        c4.metric("High 1-week potential", len(high_week))

        pulse_line = (
            f"Market pulse: {market_pulse.get('name', 'NIFTY 50')} is "
            f"{market_pulse.get('regime', 'Sideways')} "
            f"({market_pulse.get('weekly_pct', 0.0):+.2f}% weekly). "
            f"External headlines fetched: {len(external_news_rows)}. "
            f"{market_pulse.get('ext_sentiment_source', 'Alpha Vantage')}: "
            f"{market_pulse.get('ext_sentiment_label', 'Neutral')} "
            f"({float(market_pulse.get('ext_sentiment_score', 0.0) or 0.0):+.2f}, "
            f"samples {int(market_pulse.get('ext_sentiment_samples', 0) or 0)})."
        )
        st.markdown(f"<div class='bl-notewhite'>{pulse_line}</div>", unsafe_allow_html=True)
        adaptive_metrics = model_state.get("metrics", {}) if isinstance(model_state, dict) else {}
        batch_feedback = st.session_state.get("adaptive_feedback_batch", {"count": 0, "avg_return": 0.0})
        weights_line = (
            f"Self-tuning weights for current regime ({current_regime}) — "
            f"Tech {adaptive_weights['technical']:.2f}, "
            f"Fund {adaptive_weights['fundamental']:.2f}, "
            f"Flow {adaptive_weights['flow']:.2f}, "
            f"Resilience {adaptive_weights['resilience']:.2f} | "
            f"Learning updates: {int(adaptive_metrics.get('feedback_updates', 0) or 0)} "
            f"(EWMA return {float(adaptive_metrics.get('avg_return_ewma', 0.0) or 0.0):+.2f}%) | "
            f"This run feedback: {int(batch_feedback.get('count', 0) or 0)} symbols, "
            f"avg {float(batch_feedback.get('avg_return', 0.0) or 0.0):+.2f}%"
        )
        st.markdown(f"<div class='bl-notewhite'>{weights_line}</div>", unsafe_allow_html=True)

        rank_table_rows = []
        for rank, row in enumerate(top20, start=1):
            rank_table_rows.append(
                {
                    "Rank": rank,
                    "Symbol": row["Symbol"],
                    "CMP (₹)": row["CMP (₹)"],
                    "Day changed": _format_day_change(
                        row.get("Day changed"), row.get("Daily Change (%)")
                    ),
                    "Since added": row.get("_since_added", "NA"),
                    "Added on": row.get("_added_on", "NA"),
                    "Industry Sector": row.get("Industry Sector"),
                    "52W High (₹)": row["52W High (₹)"],
                    "Exp MA 2W (₹)": row["Expected MA (2W, ₹)"],
                    "Proposed Entry (₹)": row.get("Entry Price (₹)") or row.get("CMP (₹)"),
                    "Proposed Exit (₹)": _format_exit_with_gain(
                        row.get("Entry Price (₹)") or row.get("CMP (₹)"),
                        row.get("Proposed Exit Price (₹)"),
                    ),
                    "Promoter Stake (%)": row["Promoter / Insider Stake (%)"],
                    "Rec": row["Recommendation"],
                    "Met Exp": row["Met Expectation Today"],
                    "5m Mom %": row.get("_mom_5m"),
                    "15m Mom %": row.get("_mom_15m"),
                    "Buy Win": row["Buy Window (next 2w, days)"],
                    "Sell Win": row["Sell Window (next 2w, days)"],
                    "Days to Result": row["Results Announcement (days)"],
                    "Conf": row["Confidence (1-10)"],
                    "50DMA (₹)": row["50-Day MA (₹)"],
                    "Sentiment": row["News Sentiment"],
                    "Source": row["Sources Used"],
                }
            )

        def _build_met_exp_summary(values: list[Any]) -> str:
            met_exp_counts = (
                pd.Series(values, dtype="object")
                .fillna("NA")
                .astype(str)
                .str.strip()
                .str.upper()
                .replace("", "NA")
                .value_counts()
            )
            preferred_met_order = ("YES", "MIXED", "NO")
            ordered_met_counts: list[tuple[str, int]] = [
                (label, int(met_exp_counts[label]))
                for label in preferred_met_order
                if label in met_exp_counts
            ]
            ordered_met_counts.extend(
                [
                    (label, int(count))
                    for label, count in met_exp_counts.items()
                    if label not in preferred_met_order
                ]
            )
            return "  |  ".join([f"{label}: {count}" for label, count in ordered_met_counts])

        met_exp_summary = _build_met_exp_summary([r.get("Met Exp", "NA") for r in rank_table_rows])

        table_header_col, met_summary_col = st.columns([2.4, 1.2], gap="small")
        with table_header_col:
            st.markdown("### Top 20 high-performing recommendations")
        with met_summary_col:
            with st.container(border=True):
                st.caption("Expectations met overview")
                st.markdown(f"**{met_exp_summary or 'NA: 0'}**")

        st.dataframe(
            style_met_expectation_dataframe(pd.DataFrame(rank_table_rows)),
            use_container_width=True,
            hide_index=True,
        )
        previous_top20 = set(st.session_state.get(previous_top20_key, []))
        current_top20_symbols = [str(r.get("Symbol", "")) for r in top20 if r.get("Symbol")]
        if previous_top20:
            entrant_rows = [r for r in top20 if str(r.get("Symbol", "")) not in previous_top20]
            st.markdown("### New entrants today")
            if entrant_rows:
                entrants_df = pd.DataFrame(
                    [
                        {
                            "Symbol": r.get("Symbol"),
                            "Company": r.get("Company"),
                            "Recommendation": r.get("Recommendation"),
                            "Confidence": r.get("Confidence (1-10)"),
                            "CMP (₹)": r.get("CMP (₹)"),
                            "Industry Sector": r.get("Industry Sector"),
                            "Day changed": _format_day_change(
                                r.get("Day changed"), r.get("Daily Change (%)")
                            ),
                            "Promoter Stake (%)": r.get("Promoter / Insider Stake (%)"),
                            "Proposed Entry (₹)": r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                            "Proposed Exit (₹)": _format_exit_with_gain(
                                r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                                r.get("Proposed Exit Price (₹)"),
                            ),
                            "Added on": r.get("_added_on", "NA"),
                            "5m Mom %": r.get("_mom_5m"),
                            "15m Mom %": r.get("_mom_15m"),
                        }
                        for r in entrant_rows
                    ]
                )
                st.dataframe(
                    style_met_expectation_dataframe(entrants_df),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No new entrants compared with the previous scan run.")
        st.session_state[previous_top20_key] = current_top20_symbols

        st.markdown("### Swing candidate engine (3-10 days)")
        swing_candidates = sorted(
            [
                r
                for r in filtered
                if r.get("Recommendation") in {"BUY", "BUY ON DIP", "WATCH"}
                and float(r.get("Confidence (1-10)", 0.0) or 0.0) >= 6.6
            ],
            key=_confidence_value,
            reverse=True,
        )[:25]
        if swing_candidates:
            for candidate in swing_candidates:
                score, tag, color = _trade_card_score_tag(candidate)
                candidate["_card_score"] = score
                candidate["_card_tag"] = tag
                candidate["_card_color"] = color
            swing_candidates_df = pd.DataFrame(
                [
                    {
                        "Symbol": r.get("Symbol"),
                        "Added on": r.get("_added_on", "NA"),
                        "Rec": r.get("Recommendation"),
                        "Swing Grade": r.get("Swing Grade"),
                        "Card Score": r.get("_card_score"),
                        "Card Tag": r.get("_card_tag"),
                        "Confidence": r.get("Confidence (1-10)"),
                        "CMP (₹)": r.get("CMP (₹)"),
                        "Industry Sector": r.get("Industry Sector"),
                        "Proposed Entry (₹)": r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                        "Proposed Exit (₹)": _format_exit_with_gain(
                            r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                            r.get("Proposed Exit Price (₹)"),
                        ),
                        "Entry Zone (₹)": r.get("Entry Zone (₹)"),
                        "Stop Loss (₹)": r.get("Stop Loss (₹)"),
                        "Target 1 (₹)": r.get("Target 1 (₹)"),
                        "Target 2 (₹)": r.get("Target 2 (₹)"),
                        "Time Stop (days)": r.get("Time Stop (days)"),
                        "Promoter Stake (%)": r.get("Promoter / Insider Stake (%)"),
                        "RR to T1": r.get("RR to T1"),
                        "RR to T2": r.get("RR to T2"),
                        "5m Mom %": r.get("_mom_5m"),
                        "15m Mom %": r.get("_mom_15m"),
                    }
                    for r in swing_candidates
                ]
            )
            st.dataframe(
                style_met_expectation_dataframe(swing_candidates_df),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No swing candidates matched confidence and recommendation rules.")

        fav_now = sorted(
            [r for r in top20 if r["Recommendation"] in {"BUY", "BUY ON DIP"}],
            key=_confidence_value,
            reverse=True,
        )[:10]
        fav_near = sorted(
            [r for r in top20 if r["Recommendation"] == "WATCH"],
            key=_confidence_value,
            reverse=True,
        )[:10]

        fav_now_summary = _build_met_exp_summary(
            [r.get("Met Expectation Today", "NA") for r in fav_now]
        )
        fav_now_header_col, fav_now_summary_col = st.columns([2.4, 1.2], gap="small")
        with fav_now_header_col:
            st.markdown("### Favourable right now")
        with fav_now_summary_col:
            with st.container(border=True):
                st.caption("Expectations met overview")
                st.markdown(f"**{fav_now_summary or 'NA: 0'}**")
        if fav_now:
            fav_now_df = pd.DataFrame(
                [
                    {
                        "Symbol": r.get("Symbol"),
                        "CMP (₹)": r.get("CMP (₹)"),
                        "Recommendation": r.get("Recommendation"),
                        "Met Exp": r.get("Met Expectation Today"),
                        "Day changed": _format_day_change(
                            r.get("Day changed"), r.get("Daily Change (%)")
                        ),
                        "Since added": r.get("_since_added", "NA"),
                        "Added on": r.get("_added_on", "NA"),
                        "Industry Sector": r.get("Industry Sector"),
                        "Confidence": r.get("Confidence (1-10)"),
                        "Promoter Stake (%)": r.get("Promoter / Insider Stake (%)"),
                        "Proposed Entry (₹)": r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                        "Proposed Exit (₹)": _format_exit_with_gain(
                            r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                            r.get("Proposed Exit Price (₹)"),
                        ),
                        "Avg Vol 5D": r.get("Avg Volume 5D"),
                        "Buy/Sell Vol Ratio": r.get("Buy/Sell Volume Ratio"),
                        "Sell Window (days)": r.get("Sell Window (next 2w, days)"),
                        "Company": r.get("Company"),
                        "Reason": (r.get("Reason for Recommendation") or "")[:90],
                    }
                    for r in fav_now
                ]
            )
            st.dataframe(
                style_met_expectation_dataframe(fav_now_df),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No immediate entry candidates under current filter.")

        fav_near_summary = _build_met_exp_summary(
            [r.get("Met Expectation Today", "NA") for r in fav_near]
        )
        fav_near_header_col, fav_near_summary_col = st.columns([2.4, 1.2], gap="small")
        with fav_near_header_col:
            st.markdown("### Favourable in near future")
        with fav_near_summary_col:
            with st.container(border=True):
                st.caption("Expectations met overview")
                st.markdown(f"**{fav_near_summary or 'NA: 0'}**")
        if fav_near:
            fav_near_df = pd.DataFrame(
                [
                    {
                        "Symbol": r.get("Symbol"),
                        "CMP (₹)": r.get("CMP (₹)"),
                        "Recommendation": r.get("Recommendation"),
                        "Met Exp": r.get("Met Expectation Today"),
                        "Day changed": _format_day_change(
                            r.get("Day changed"), r.get("Daily Change (%)")
                        ),
                        "Since added": r.get("_since_added", "NA"),
                        "Added on": r.get("_added_on", "NA"),
                        "Industry Sector": r.get("Industry Sector"),
                        "Confidence": r.get("Confidence (1-10)"),
                        "Promoter Stake (%)": r.get("Promoter / Insider Stake (%)"),
                        "Proposed Entry (₹)": r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                        "Proposed Exit (₹)": _format_exit_with_gain(
                            r.get("Entry Price (₹)") or r.get("CMP (₹)"),
                            r.get("Proposed Exit Price (₹)"),
                        ),
                        "Avg Vol 5D": r.get("Avg Volume 5D"),
                        "Buy/Sell Vol Ratio": r.get("Buy/Sell Volume Ratio"),
                        "Buy Window (days)": r.get("Buy Window (next 2w, days)"),
                        "Company": r.get("Company"),
                        "Reason": (r.get("Reason for Recommendation") or "")[:90],
                    }
                    for r in fav_near
                ]
            )
            st.dataframe(
                style_met_expectation_dataframe(fav_near_df),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No near-future watch candidates under current filter.")

        st.markdown("### Top 6 bullish buy candidates (5-10 days)")
        strict_bullish_rows: list[dict[str, Any]] = []
        fallback_bullish_rows: list[dict[str, Any]] = []
        for row in filtered:
            recommendation = str(row.get("Recommendation", "")).upper()
            if recommendation not in {"BUY", "BUY ON DIP"}:
                continue
            promoter_stake = _to_float(row.get("Promoter / Insider Stake (%)"))
            if promoter_stake is None or promoter_stake < 40.0:
                continue
            cmp_value = _to_float(row.get("CMP (₹)"))
            target_2 = _to_float(row.get("Target 2 (₹)"))
            if cmp_value is None or target_2 is None or cmp_value <= 0:
                continue
            target_2_upside_pct = ((target_2 / cmp_value) - 1.0) * 100.0
            conf_value = float(row.get("Confidence (1-10)", 0.0) or 0.0)
            weekly_momentum = _to_float(row.get("Weekly Momentum (%)")) or 0.0
            volume_surge = _to_float(row.get("Volume Surge (5d/20d)")) or 1.0
            sentiment_label = str(row.get("News Sentiment", "Neutral")).strip().upper()
            sell_window = row.get("Sell Window (next 2w, days)")
            time_stop_days = _to_float(row.get("Time Stop (days)"))
            holding_window_ok = _window_overlaps(sell_window, 5, 10) or (
                time_stop_days is not None and 5 <= time_stop_days <= 10
            )
            if not holding_window_ok:
                continue

            sentiment_score = 1.0 if sentiment_label == "POSITIVE" else (0.7 if sentiment_label == "NEUTRAL" else 0.3)
            confidence_score = float(np.clip(conf_value / 10.0, 0.0, 1.0))
            upside_score = float(np.clip((target_2_upside_pct - 8.0) / 4.0, 0.0, 1.0))
            promoter_score = float(np.clip((promoter_stake - 40.0) / 35.0, 0.0, 1.0))
            momentum_score = float(np.clip(weekly_momentum / 4.0, 0.0, 1.0))
            volume_score = float(np.clip((volume_surge - 0.8) / 0.8, 0.0, 1.0))
            bullish_score = (
                confidence_score * 0.34
                + upside_score * 0.24
                + promoter_score * 0.16
                + momentum_score * 0.14
                + volume_score * 0.08
                + sentiment_score * 0.04
            )
            candidate_row = {
                "Symbol": row.get("Symbol"),
                "Company": row.get("Company"),
                "Industry Sector": row.get("Industry Sector"),
                "Recommendation": row.get("Recommendation"),
                "Confidence": conf_value,
                "CMP (₹)": cmp_value,
                "Proposed Entry (₹)": row.get("Entry Price (₹)") or cmp_value,
                "Proposed Exit Price (₹)": _to_float(row.get("Proposed Exit Price (₹)")),
                "Proposed Exit (₹)": _format_exit_with_gain(
                    row.get("Entry Price (₹)") or cmp_value,
                    row.get("Proposed Exit Price (₹)"),
                ),
                "Target 2 (₹)": target_2,
                "Target 2 Upside (%)": round(target_2_upside_pct, 2),
                "Sell Window (days)": sell_window,
                "Time Stop (days)": row.get("Time Stop (days)"),
                "Promoter Stake (%)": round(promoter_stake, 2),
                "Weekly Momentum (%)": round(float(weekly_momentum), 2),
                "Volume Surge (5d/20d)": round(float(volume_surge), 2),
                "Bullish Score": round(float(bullish_score), 4),
            }
            if (
                target_2_upside_pct >= 10.0
                and conf_value >= 7.0
                and weekly_momentum > 0
                and sentiment_label != "NEGATIVE"
            ):
                strict_bullish_rows.append(candidate_row)
            elif target_2_upside_pct >= 8.0 and conf_value >= 6.6:
                fallback_bullish_rows.append(candidate_row)

        strict_bullish_rows = sorted(strict_bullish_rows, key=lambda x: x.get("Bullish Score", 0.0), reverse=True)
        fallback_bullish_rows = sorted(
            fallback_bullish_rows,
            key=lambda x: x.get("Bullish Score", 0.0),
            reverse=True,
        )
        selected_bullish_rows = strict_bullish_rows[:6]
        if len(selected_bullish_rows) < 6:
            used_symbols = {str(r.get("Symbol", "")) for r in selected_bullish_rows}
            for row in fallback_bullish_rows:
                symbol = str(row.get("Symbol", ""))
                if symbol in used_symbols:
                    continue
                selected_bullish_rows.append(row)
                used_symbols.add(symbol)
                if len(selected_bullish_rows) >= 6:
                    break

        if selected_bullish_rows:
            selected_bullish_df = pd.DataFrame(selected_bullish_rows)
            score_total = float(selected_bullish_df["Bullish Score"].sum())
            if score_total > 0:
                selected_bullish_df["Bullish Share (%)"] = (
                    selected_bullish_df["Bullish Score"] / score_total * 100.0
                ).round(2)
            else:
                selected_bullish_df["Bullish Share (%)"] = 0.0
            selected_bullish_df = selected_bullish_df.sort_values(
                "Bullish Score", ascending=False, na_position="last"
            ).reset_index(drop=True)

            if len(strict_bullish_rows) < 6:
                st.caption(
                    "Filled remaining slots using near-match candidates "
                    "(>=8% Target-2 upside) to keep at least 6 stocks."
                )
            st.caption(
                "Filter used: BUY / BUY ON DIP, promoter stake >=40%, 5-10 day hold window, "
                "and strong bullish support (confidence, momentum, volume, sentiment)."
            )

            pie_col, table_col = st.columns([1.15, 1.85], gap="large")
            with pie_col:
                st.plotly_chart(
                    build_bullish_candidates_pie(selected_bullish_df),
                    use_container_width=True,
                )
            with table_col:
                st.dataframe(
                    style_met_expectation_dataframe(
                        selected_bullish_df[
                            [
                                "Symbol",
                                "Industry Sector",
                                "Recommendation",
                                "Confidence",
                                "CMP (₹)",
                                "Proposed Entry (₹)",
                                "Proposed Exit (₹)",
                                "Promoter Stake (%)",
                                "Target 2 Upside (%)",
                                "Sell Window (days)",
                                "Bullish Share (%)",
                            ]
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info(
                "No stocks currently match promoter >=40%, 5-10 day hold, and bullish buy criteria."
            )

        if scope_key == "broad":
            movers_df = pd.DataFrame(st.session_state.get("broad_daily_movers", [])[:20])
            if not movers_df.empty:
                preferred_movers_cols = [
                    "Symbol",
                    "Company",
                    "CMP (₹)",
                    "Industry Sector",
                    "Promoter Stake (%)",
                    "Current Volume",
                    "Entry Price (₹)",
                    "Proposed Exit (₹)",
                    "Minimum Holding Period (days)",
                    "Increased % since last 5 days",
                    "Daily Change (%)",
                ]
                movers_df = movers_df.drop(
                    columns=["Proposed Exit Price (₹)", "Proposed Exit Gain (%)"],
                    errors="ignore",
                )
                existing_movers_cols = [c for c in preferred_movers_cols if c in movers_df.columns]
                remaining_movers_cols = [c for c in movers_df.columns if c not in existing_movers_cols]
                movers_df = movers_df[existing_movers_cols + remaining_movers_cols]
                st.markdown("### Today's broad-market gainers")
                st.dataframe(
                    style_met_expectation_dataframe(movers_df),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Trade card popout")
        if swing_candidates:
            trade_card_options = {
                f"{r.get('Company')} ({r.get('Symbol')}) · {r.get('Recommendation')}": r
                for r in swing_candidates
            }
            selected_trade_card_label = st.selectbox(
                "Select stock for trade card",
                list(trade_card_options.keys()),
                key=f"trade_card_pick_{scope_key}",
            )
            trade_card_row = trade_card_options[selected_trade_card_label]
            with st.expander("Open trade card", expanded=True):
                tc_cap_col, tc_risk_col, tc_alloc_col = st.columns(3)
                tc_capital = float(
                    tc_cap_col.number_input(
                        "Capital (₹)",
                        min_value=10000.0,
                        value=200000.0,
                        step=10000.0,
                        key=f"trade_card_capital_{scope_key}",
                    )
                )
                tc_risk_pct = float(
                    tc_risk_col.slider(
                        "Risk/trade %",
                        min_value=0.5,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key=f"trade_card_risk_{scope_key}",
                    )
                )
                tc_alloc_pct = float(
                    tc_alloc_col.slider(
                        "Max alloc %",
                        min_value=5.0,
                        max_value=35.0,
                        value=20.0,
                        step=1.0,
                        key=f"trade_card_alloc_{scope_key}",
                    )
                )

                tc_entry = _to_float(trade_card_row.get("CMP (₹)")) or 0.0
                tc_stop = _to_float(trade_card_row.get("Stop Loss (₹)"))
                tc_t1 = _to_float(trade_card_row.get("Target 1 (₹)"))
                tc_t2 = _to_float(trade_card_row.get("Target 2 (₹)"))
                tc_rr1 = _format_rr(tc_entry, tc_stop, tc_t1) if tc_stop and tc_t1 else None
                tc_rr2 = _format_rr(tc_entry, tc_stop, tc_t2) if tc_stop and tc_t2 else None
                tc_risk_share = (tc_entry - tc_stop) if tc_stop is not None else 0.0
                tc_risk_budget = tc_capital * (tc_risk_pct / 100.0)
                tc_qty_risk = int(tc_risk_budget / tc_risk_share) if tc_risk_share > 0 else 0
                tc_qty_alloc = int((tc_capital * (tc_alloc_pct / 100.0)) / tc_entry) if tc_entry > 0 else 0
                tc_qty = max(0, min(tc_qty_risk, tc_qty_alloc))
                tc_est_invested = round(tc_qty * tc_entry, 2)

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Entry", f"₹{tc_entry:,.2f}")
                mcol2.metric("Stop", f"₹{(tc_stop or 0):,.2f}")
                mcol3.metric("Target 1", f"₹{(tc_t1 or 0):,.2f}", f"RR {tc_rr1 if tc_rr1 is not None else 'NA'}")
                mcol4.metric("Target 2", f"₹{(tc_t2 or 0):,.2f}", f"RR {tc_rr2 if tc_rr2 is not None else 'NA'}")

                tc_score, tc_tag, tc_color = _trade_card_score_tag(trade_card_row)
                st.markdown(
                    f"<div style='margin:0.25rem 0 0.35rem 0;'>"
                    f"<span style='display:inline-block;padding:0.18rem 0.6rem;border-radius:999px;"
                    f"background:rgba(14,20,29,0.65);border:1px solid {tc_color};color:{tc_color};"
                    f"font-weight:700;font-size:0.82rem;'>"
                    f"{tc_tag.upper()} · SCORE {tc_score}/100"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"Qty: {tc_qty} | Estimated invested: ₹{tc_est_invested:,.2f} | "
                    f"Risk/share: ₹{tc_risk_share:,.2f} | Time stop: {trade_card_row.get('Time Stop (days)', 'NA')} days"
                )
                tc_checklist = [
                    ("Signal is BUY / BUY ON DIP", trade_card_row.get("Recommendation") in {"BUY", "BUY ON DIP"}),
                    ("Confidence >= 7.0", float(trade_card_row.get("Confidence (1-10)", 0.0) or 0.0) >= 7.0),
                    ("RR to T1 >= 1.8", tc_rr1 is not None and tc_rr1 >= 1.8),
                    ("RR to T2 >= 2.2", tc_rr2 is not None and tc_rr2 >= 2.2),
                    (
                        "Intraday momentum not weak",
                        (trade_card_row.get("_mom_5m") is None or float(trade_card_row.get("_mom_5m") or 0.0) >= -0.2)
                        and (trade_card_row.get("_mom_15m") is None or float(trade_card_row.get("_mom_15m") or 0.0) >= -0.3),
                    ),
                    ("No near result risk (>= 2 days)", (trade_card_row.get("Results Announcement (days)") or 99) >= 2),
                ]
                st.markdown(
                    "  \n".join([f"{'PASS' if ok else 'CHECK'} - {label}" for label, ok in tc_checklist])
                )
                if st.button("Add this trade card to journal", key=f"trade_card_add_{scope_key}"):
                    if tc_qty <= 0 or tc_stop is None:
                        st.warning("Cannot add: quantity is zero or stop-loss is unavailable.")
                    else:
                        trade_id = f"{trade_card_row.get('Symbol')}-{int(dt.datetime.now().timestamp())}"
                        tc_trade = {
                            "id": trade_id,
                            "opened_on": dt.date.today().isoformat(),
                            "symbol": trade_card_row.get("Symbol"),
                            "yf_ticker": trade_card_row.get("_yf_ticker") or f"{trade_card_row.get('Symbol')}.NS",
                            "company": trade_card_row.get("Company"),
                            "industry_sector": trade_card_row.get("Industry Sector"),
                            "recommendation": trade_card_row.get("Recommendation"),
                            "qty": tc_qty,
                            "entry_price": round(tc_entry, 2),
                            "stop_loss": round(tc_stop, 2),
                            "target_1": round(tc_t1, 2) if tc_t1 is not None else None,
                            "target_2": round(tc_t2, 2) if tc_t2 is not None else None,
                            "time_stop_days": trade_card_row.get("Time Stop (days)"),
                            "status": "OPEN",
                            "exit_price": None,
                            "closed_on": None,
                            "notes": "Added via trade card",
                        }
                        if "paper_trade_journal_rows" not in st.session_state:
                            st.session_state["paper_trade_journal_rows"] = _load_trade_journal()
                        rows_now = list(st.session_state.get("paper_trade_journal_rows", []))
                        rows_now.insert(0, tc_trade)
                        _upsert_journal_in_session(rows_now)
                        st.success(f"Added {trade_card_row.get('Symbol')} from trade card.")
                        st.rerun()
        else:
            st.info("Trade card will appear after swing candidates are available.")

        st.markdown("### Position sizing and trade journal")
        if "paper_trade_journal_rows" not in st.session_state:
            st.session_state["paper_trade_journal_rows"] = _load_trade_journal()
        journal_rows: list[dict[str, Any]] = list(st.session_state.get("paper_trade_journal_rows", []))

        sizing_col, close_col = st.columns([2.1, 1.4], gap="large")
        with sizing_col:
            capital = float(st.number_input("Paper capital (₹)", min_value=10000.0, value=200000.0, step=10000.0))
            risk_per_trade_pct = float(
                st.slider("Risk per trade (% of capital)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
            )
            max_alloc_pct = float(
                st.slider("Max capital allocation per trade (%)", min_value=5.0, max_value=35.0, value=20.0, step=1.0)
            )
            if swing_candidates:
                swing_options = {
                    f"{r.get('Company')} ({r.get('Symbol')}) · {r.get('Recommendation')}": r
                    for r in swing_candidates
                }
                selected_swing_label = st.selectbox(
                    "Select candidate for sizing",
                    list(swing_options.keys()),
                    key=f"swing_sizing_pick_{scope_key}",
                )
                selected_swing = swing_options[selected_swing_label]
                entry_price = float(selected_swing.get("CMP (₹)") or 0.0)
                stop_loss = _to_float(selected_swing.get("Stop Loss (₹)"))
                target_1 = _to_float(selected_swing.get("Target 1 (₹)"))
                target_2 = _to_float(selected_swing.get("Target 2 (₹)"))
                risk_per_share = (entry_price - stop_loss) if stop_loss is not None else 0.0
                risk_budget = capital * (risk_per_trade_pct / 100.0)
                qty_by_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
                qty_by_alloc = int((capital * (max_alloc_pct / 100.0)) / entry_price) if entry_price > 0 else 0
                suggested_qty = max(0, min(qty_by_risk, qty_by_alloc))
                est_invested = round(suggested_qty * entry_price, 2)
                st.caption(
                    f"Suggested qty: {suggested_qty} | Risk/share: ₹{risk_per_share:,.2f} | "
                    f"Est. invested: ₹{est_invested:,.2f}"
                )
                add_trade = st.button("Add to paper journal", key=f"add_paper_trade_{scope_key}")
                if add_trade and suggested_qty > 0 and stop_loss is not None:
                    trade_id = f"{selected_swing.get('Symbol')}-{int(dt.datetime.now().timestamp())}"
                    new_trade = {
                        "id": trade_id,
                        "opened_on": dt.date.today().isoformat(),
                        "symbol": selected_swing.get("Symbol"),
                        "yf_ticker": selected_swing.get("_yf_ticker")
                        or f"{selected_swing.get('Symbol')}.NS",
                        "company": selected_swing.get("Company"),
                        "industry_sector": selected_swing.get("Industry Sector"),
                        "recommendation": selected_swing.get("Recommendation"),
                        "qty": suggested_qty,
                        "entry_price": round(entry_price, 2),
                        "stop_loss": round(stop_loss, 2),
                        "target_1": round(target_1, 2) if target_1 is not None else None,
                        "target_2": round(target_2, 2) if target_2 is not None else None,
                        "time_stop_days": selected_swing.get("Time Stop (days)"),
                        "status": "OPEN",
                        "exit_price": None,
                        "closed_on": None,
                        "notes": "",
                    }
                    journal_rows.insert(0, new_trade)
                    _upsert_journal_in_session(journal_rows)
                    st.success(f"Added {selected_swing.get('Symbol')} to paper journal.")
                    st.rerun()
            else:
                st.info("Run scan first to generate swing sizing candidates.")

        with close_col:
            open_trades = [t for t in journal_rows if str(t.get("status", "")).upper() == "OPEN"]
            if open_trades:
                close_options = {
                    f"{t.get('symbol')} · Qty {t.get('qty')} · Entry ₹{t.get('entry_price')}": t
                    for t in open_trades
                }
                close_label = st.selectbox(
                    "Close open trade",
                    list(close_options.keys()),
                    key=f"close_trade_pick_{scope_key}",
                )
                close_trade = close_options[close_label]
                close_price = st.number_input(
                    "Exit price (₹)",
                    min_value=0.0,
                    value=float(close_trade.get("entry_price") or 0.0),
                    step=0.5,
                    key=f"close_trade_price_{scope_key}",
                )
                close_note = st.text_input("Close note (optional)", key=f"close_trade_note_{scope_key}")
                if st.button("Mark as closed", key=f"mark_closed_{scope_key}"):
                    for row in journal_rows:
                        if row.get("id") == close_trade.get("id"):
                            row["status"] = "CLOSED"
                            row["exit_price"] = round(float(close_price), 2)
                            row["closed_on"] = dt.date.today().isoformat()
                            row["notes"] = close_note
                            break
                    _upsert_journal_in_session(journal_rows)
                    st.success(f"Closed {close_trade.get('symbol')} in paper journal.")
                    st.rerun()
            else:
                st.info("No open paper trades.")

        journal_rows = list(st.session_state.get("paper_trade_journal_rows", journal_rows))
        if journal_rows:
            journal_view_rows: list[dict[str, Any]] = []
            for row in journal_rows:
                entry = _to_float(row.get("entry_price"))
                qty = int(row.get("qty") or 0)
                exit_price = _to_float(row.get("exit_price"))
                status = str(row.get("status", "OPEN")).upper()
                current_price = None
                symbol = str(row.get("symbol", "") or "")
                yf_ticker = str(row.get("yf_ticker", "") or "").strip()
                if status == "OPEN" and symbol:
                    if not yf_ticker:
                        yf_ticker = symbol if "." in symbol else f"{symbol}.NS"
                    snapshot = fetch_live_price_snapshot(yf_ticker)
                    if snapshot:
                        current_price = float(snapshot.get("price", 0.0) or 0.0)
                used_price = exit_price if status == "CLOSED" else current_price
                pnl = ((used_price - entry) * qty) if (used_price is not None and entry is not None and qty > 0) else None
                pnl_pct = (((used_price / entry) - 1.0) * 100.0) if (used_price is not None and entry and entry > 0) else None
                journal_view_rows.append(
                    {
                        "Status": status,
                        "Opened": row.get("opened_on"),
                        "Closed": row.get("closed_on") or "-",
                        "Symbol": symbol,
                        "Industry Sector": row.get("industry_sector") or "NA",
                        "Qty": qty,
                        "Entry": entry,
                        "Live/Exit": used_price if used_price is not None else "-",
                        "Stop": row.get("stop_loss"),
                        "T1": row.get("target_1"),
                        "T2": row.get("target_2"),
                        "P/L (₹)": round(float(pnl), 2) if pnl is not None else "-",
                        "P/L (%)": round(float(pnl_pct), 2) if pnl_pct is not None else "-",
                        "Notes": row.get("notes") or "",
                    }
                )
            st.dataframe(
                style_met_expectation_dataframe(pd.DataFrame(journal_view_rows)),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Performance analytics and go-live checklist")
        closed_trades = [t for t in journal_rows if str(t.get("status", "")).upper() == "CLOSED"]
        open_trades = [t for t in journal_rows if str(t.get("status", "")).upper() == "OPEN"]
        closed_returns: list[float] = []
        closed_pnls: list[float] = []
        hold_days: list[int] = []
        for t in closed_trades:
            entry = _to_float(t.get("entry_price"))
            exit_p = _to_float(t.get("exit_price"))
            qty = int(t.get("qty") or 0)
            if entry and exit_p and qty > 0:
                ret = ((exit_p / entry) - 1.0) * 100.0
                pnl = (exit_p - entry) * qty
                closed_returns.append(ret)
                closed_pnls.append(pnl)
            try:
                d0 = dt.date.fromisoformat(str(t.get("opened_on")))
                d1 = dt.date.fromisoformat(str(t.get("closed_on")))
                hold_days.append((d1 - d0).days)
            except Exception:
                continue
        win_count = len([r for r in closed_returns if r > 0])
        loss_count = len([r for r in closed_returns if r <= 0])
        win_rate = (win_count / len(closed_returns) * 100.0) if closed_returns else 0.0
        avg_return = float(np.mean(closed_returns)) if closed_returns else 0.0
        avg_hold_days = float(np.mean(hold_days)) if hold_days else 0.0
        gross_profit = float(sum([p for p in closed_pnls if p > 0]))
        gross_loss = abs(float(sum([p for p in closed_pnls if p < 0])))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        analytics_col_1, analytics_col_2, analytics_col_3, analytics_col_4 = st.columns(4)
        analytics_col_1.metric("Closed paper trades", len(closed_trades))
        analytics_col_2.metric("Win rate", f"{win_rate:.1f}%")
        analytics_col_3.metric("Avg return / trade", f"{avg_return:+.2f}%")
        analytics_col_4.metric("Avg hold days", f"{avg_hold_days:.1f}")
        checklist = [
            ("At least 15 closed paper trades", len(closed_trades) >= 15),
            ("Win rate >= 55%", win_rate >= 55.0),
            ("Avg return >= +2.0%", avg_return >= 2.0),
            ("Profit factor >= 1.30", profit_factor >= 1.30),
            ("Avg hold <= 10 days", avg_hold_days > 0 and avg_hold_days <= 10.0),
            ("No open trade older than 10 days", not any(
                (dt.date.today() - dt.date.fromisoformat(str(t.get("opened_on")))).days > 10
                for t in open_trades
                if t.get("opened_on")
            )),
        ]
        checklist_lines = [
            f"{'PASS' if ok else 'PENDING'} - {label}" for label, ok in checklist
        ]
        st.markdown("  \n".join(checklist_lines))

        st.markdown("### Script analysis")
        options = {f"{r['Company']} ({r['Symbol']})": r for r in top20}
        labels = list(options.keys())
        selected_label = st.selectbox("Select script for detailed view", labels, index=0)
        selected = options[selected_label]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{selected['Confidence (1-10)']}/10", selected["Recommendation"])
        m2.metric("Trend", selected["Trend"], f"{selected['Weekly Momentum (%)']:+.2f}% weekly")
        m3.metric("News sentiment", selected["News Sentiment"], f"{selected['Product Sentiment Score']:+.2f}")
        m4.metric(
            "Met expectation today",
            selected["Met Expectation Today"],
            f"{selected['Daily Change (%)']:+.2f}%",
        )

        st.plotly_chart(
            build_chart(selected["Company"], selected["_closes"], selected["_ma"]),
            use_container_width=True,
        )

        with st.expander("Study material: latest market/news references", expanded=True):
            news_rows = selected.get("_news_rows", [])
            if not news_rows:
                st.caption("No news references available for this script from the source API right now.")
            else:
                for item in news_rows[:6]:
                    title = item.get("title", "Untitled update")
                    publisher = item.get("publisher", "Unknown source")
                    link = item.get("link", "")
                    if link:
                        st.markdown(f"- [{title}]({link}) — {publisher}")
                    else:
                        st.markdown(f"- {title} — {publisher}")


def _complete_zerodha_login(api_key: str, api_secret: str, request_token: str) -> bool:
    try:
        session = zc.exchange_request_token(api_key, api_secret, request_token)
        kite = zc.make_kite(api_key, session["access_token"])
        st.session_state["kite"] = kite
        st.session_state["zerodha_user"] = session.get("user_name") or session.get(
            "user_id", "Zerodha user"
        )
        st.session_state["zerodha_connected"] = True
        return True
    except Exception as exc:
        st.session_state["zerodha_connected"] = False
        st.session_state.pop("kite", None)
        st.error("Login failed. Check your API secret and request token, then try again.")
        st.caption(str(exc))
        return False


def render_holdings_tab() -> None:
    st.subheader("My Zerodha Holdings")
    st.caption(
        "Connect your Zerodha Demat account using the official Kite Connect API "
        "(Personal / free plan is enough for holdings)."
    )

    with st.expander("One-time setup (do this first)", expanded=True):
        st.markdown(
            """
            1. Open **[developers.kite.trade](https://developers.kite.trade/)** and sign up / log in with Zerodha  
            2. Choose **Personal (Free)** plan — holdings are included  
            3. Click **Create new app**  
            4. Set **Redirect URL** exactly to: `http://localhost:8501/`  
            5. Copy your **API Key** and **API Secret** into the boxes below  
            """
        )
        st.info(
            "Your API secret stays on this computer only. "
            "Never share it. Login is required once per trading day."
        )

    api_key = st.text_input("Zerodha API Key", type="default", key="api_key_input")
    api_secret = st.text_input("Zerodha API Secret", type="password", key="api_secret_input")

    # Auto-capture request_token if Zerodha redirected back to this app
    qp = st.query_params
    redirect_token = qp.get("request_token")
    if isinstance(redirect_token, list):
        redirect_token = redirect_token[0] if redirect_token else None

    if (
        redirect_token
        and api_key
        and api_secret
        and st.session_state.get("last_used_request_token") != redirect_token
    ):
        with st.spinner("Finishing Zerodha login…"):
            if _complete_zerodha_login(api_key, api_secret, redirect_token):
                st.session_state["last_used_request_token"] = redirect_token
                st.success("Zerodha connected successfully.")
                st.query_params.clear()
                st.rerun()

    # Restore earlier same-day session
    if api_key and not st.session_state.get("zerodha_connected"):
        restored = zc.try_restore_session(api_key)
        if restored is not None:
            st.session_state["kite"] = restored
            saved = zc.load_session() or {}
            st.session_state["zerodha_user"] = saved.get("user_name", "Zerodha user")
            st.session_state["zerodha_connected"] = True

    col_login, col_logout = st.columns([2, 1])
    with col_login:
        if api_key:
            st.link_button(
                "1. Open Zerodha Login",
                zc.login_url(api_key),
                use_container_width=True,
            )
        else:
            st.button("1. Open Zerodha Login", disabled=True, use_container_width=True)

    with col_logout:
        if st.button("Disconnect", use_container_width=True):
            zc.clear_session()
            st.session_state["zerodha_connected"] = False
            st.session_state.pop("kite", None)
            st.session_state.pop("holdings_df", None)
            st.rerun()

    st.markdown(
        """
        **After clicking login:**
        1. Enter your Zerodha user ID / password / TOTP  
        2. You will be sent back to this site, **or** land on a page whose address looks like  
           `http://localhost:8501/?request_token=........&action=login&status=success`  
        3. If holdings do not appear automatically, paste that full address below and click connect
        """
    )

    redirect_or_token = st.text_input(
        "2. Paste redirect URL or request_token (only if auto-connect did not work)",
        placeholder="http://localhost:8501/?request_token=......",
    )

    if st.button("3. Connect & Load Holdings", type="primary"):
        if not api_key or not api_secret:
            st.error("Please enter both API Key and API Secret.")
        else:
            token = zc.extract_request_token(redirect_or_token) or redirect_token
            if not token:
                st.error(
                    "Could not find request_token. Paste the full redirect URL from your browser."
                )
            else:
                with st.spinner("Connecting to Zerodha…"):
                    if _complete_zerodha_login(api_key, api_secret, token):
                        st.success("Connected. Loading holdings…")
                        st.rerun()

    if not st.session_state.get("zerodha_connected"):
        st.warning("Not connected to Zerodha yet.")
        return

    st.success(f"Connected as **{st.session_state.get('zerodha_user', 'Zerodha user')}**")

    st.markdown("### Live refresh controls")
    refresh_col, interval_col = st.columns([1, 2])
    with interval_col:
        refresh_interval = st.radio(
            "Auto-refresh interval",
            options=["Off", "15 sec", "30 sec", "60 sec"],
            horizontal=True,
            key="holdings_refresh_interval",
        )
    with refresh_col:
        refresh = st.button("Refresh now", use_container_width=True)

    if refresh_interval != "Off":
        interval_seconds = int(refresh_interval.split()[0])
        st_autorefresh(
            interval=interval_seconds * 1000,
            key=f"holdings_autorefresh_{interval_seconds}",
        )
        st.caption(f"Auto-refresh is ON (every {interval_seconds} seconds).")

    if refresh or "holdings_df" not in st.session_state:
        try:
            kite = st.session_state["kite"]
            holdings = zc.fetch_holdings(kite)
            st.session_state["holdings_df"] = zc.holdings_to_dataframe(holdings)
            st.session_state["holdings_raw_count"] = len(holdings)
        except Exception as exc:
            st.error("Could not fetch holdings. Try Disconnect, then login again.")
            st.caption(str(exc))
            st.session_state["zerodha_connected"] = False
            return

    df = st.session_state.get("holdings_df")
    if df is None or df.empty:
        st.info("No equity holdings found in this Zerodha account.")
        return

    # Apply global affordability cap to holdings dashboard too.
    total_holdings_count = len(df)
    df = df[pd.to_numeric(df["LTP (₹)"], errors="coerce") <= MAX_STOCK_PRICE].copy()
    if df.empty:
        st.info(
            f"None of your current holdings are priced at or below ₹{MAX_STOCK_PRICE:,.0f}. "
            "This dashboard currently shows only stocks within your chosen budget range."
        )
        return
    if len(df) < total_holdings_count:
        st.caption(
            f"Showing {len(df)} of {total_holdings_count} holdings under the global price cap "
            f"(<= ₹{MAX_STOCK_PRICE:,.0f})."
        )
    if "Day changed" not in df.columns:
        df["Day changed"] = "NA"
    if "Industry Sector" not in df.columns:
        scan_results_for_sector = (
            st.session_state.get("scan_results_broad")
            or st.session_state.get("scan_results_nifty50")
            or st.session_state.get("scan_results")
            or []
        )
        sector_map = {
            str(r.get("Symbol", "")).strip(): (r.get("Industry Sector") or "NA")
            for r in scan_results_for_sector
            if r.get("Symbol")
        }
        df["Industry Sector"] = (
            df["Symbol"]
            .astype(str)
            .str.strip()
            .map(sector_map)
            .fillna("NA")
        )
    preferred_holdings_cols = [
        "Symbol",
        "Industry Sector",
        "Exchange",
        "Qty",
        "Day changed",
        "Avg Cost (₹)",
        "LTP (₹)",
        "Invested (₹)",
        "Current (₹)",
        "P&L (₹)",
        "P&L (%)",
    ]
    existing_holdings_cols = [c for c in preferred_holdings_cols if c in df.columns]
    remaining_holdings_cols = [c for c in df.columns if c not in existing_holdings_cols]
    df = df[existing_holdings_cols + remaining_holdings_cols]

    invested = float(df["Invested (₹)"].sum())
    current = float(df["Current (₹)"].sum())
    pnl = float(df["P&L (₹)"].sum())
    pnl_pct = ((current - invested) / invested * 100.0) if invested else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Invested", f"₹{invested:,.2f}")
    m2.metric("Current value", f"₹{current:,.2f}")
    m3.metric("Overall P&L", f"₹{pnl:,.2f}", f"{pnl_pct:+.2f}%")

    sort_col, order_col, visible_col = st.columns([1.2, 1.0, 2.4], gap="small")
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=[c for c in ["Current (₹)", "P&L (₹)", "Invested (₹)", "Symbol"] if c in df.columns],
            index=0,
            key="holdings_sort_by",
        )
    with order_col:
        sort_order = st.selectbox(
            "Order",
            options=["Descending", "Ascending"],
            index=0,
            key="holdings_sort_order",
        )
    with visible_col:
        visible_columns = st.multiselect(
            "Visible columns",
            options=df.columns.tolist(),
            default=df.columns.tolist(),
            key="holdings_visible_columns",
        )

    sorted_df = df.copy()
    if sort_by in sorted_df.columns:
        ascending = sort_order == "Ascending"
        if pd.api.types.is_numeric_dtype(sorted_df[sort_by]):
            sorted_df = sorted_df.sort_values(sort_by, ascending=ascending, na_position="last")
        else:
            sorted_df["_sort_key"] = pd.to_numeric(sorted_df[sort_by], errors="coerce")
            if sorted_df["_sort_key"].notna().any():
                sorted_df = sorted_df.sort_values("_sort_key", ascending=ascending, na_position="last")
            else:
                sorted_df = sorted_df.sort_values(sort_by, ascending=ascending, na_position="last")
            sorted_df = sorted_df.drop(columns=["_sort_key"], errors="ignore")

    if not visible_columns:
        st.warning("Select at least one column to display the holdings table.")
        visible_columns = df.columns.tolist()
    display_df = sorted_df[[c for c in visible_columns if c in sorted_df.columns]]

    st.dataframe(
        style_holdings_dataframe(display_df),
        use_container_width=True,
        hide_index=True,
        height=460,
    )
    st.plotly_chart(build_portfolio_pie(df), use_container_width=True)

    # Optional: overlay dip signal for holdings that are in Nifty 50
    nifty_symbols = {sym.replace(".NS", "") for sym in NIFTY_50}
    held_nifty = [s for s in df["Symbol"].tolist() if s in nifty_symbols]
    active_scan_results = (
        st.session_state.get("scan_results_broad")
        or st.session_state.get("scan_results_nifty50")
        or st.session_state.get("scan_results")
        or []
    )
    if held_nifty and active_scan_results:
        scan_map = {r["Symbol"]: r for r in active_scan_results}
        overlap = []
        for sym in held_nifty:
            row = scan_map.get(sym)
            if row:
                overlap.append(
                    {
                        "Symbol": sym,
                        "Signal": row["Signal"],
                        "Confidence": row["Confidence (1-10)"],
                        "Price (₹)": row["Price (₹)"],
                        "Day changed": _format_day_change(
                            row.get("Day changed"), row.get("Daily Change (%)")
                        ),
                        "Since added": row.get("_since_added")
                        or _track_since_added_change(
                            "scanner_since_added_baseline",
                            sym,
                            row.get("CMP (₹)"),
                        ),
                        "Industry Sector": row.get("Industry Sector"),
                        "Promoter Stake (%)": row.get("Promoter / Insider Stake (%)"),
                        "50-Day MA (₹)": row["50-Day MA (₹)"],
                        "vs MA (%)": row["vs MA (%)"],
                    }
                )
        if overlap:
            st.markdown("### Your Nifty 50 holdings vs dip scanner")
            overlap_df = pd.DataFrame(overlap).sort_values(
                "Confidence", ascending=False, na_position="last"
            )
            st.dataframe(
                style_met_expectation_dataframe(overlap_df),
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="BharatMarketLens | Trading Terminal",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_trading_theme()
    now = dt.datetime.now().strftime("%d %b %Y · %H:%M:%S")
    focus_options = ["📈 Recommended Stocks", "💼 Kite Holdings"]
    if "left_nav_focus" not in st.session_state:
        st.session_state["left_nav_focus"] = "📈 Recommended Stocks"
    pending_focus = st.session_state.pop("left_nav_focus_requested", None)
    if pending_focus in focus_options:
        st.session_state["left_nav_focus"] = pending_focus

    with st.sidebar:
        st.markdown("### Workspace")
        st.radio(
            "Focus",
            options=focus_options,
            index=0,
            key="left_nav_focus",
        )
        st.write(
            "Switch focus from the left panel. Recommended Stocks is the default view."
        )
        st.selectbox(
            "Signature style",
            options=["Premium", "Minimal", "Neon"],
            index=0,
            key="signature_style_variant",
        )
        st.caption("Theme inspired by Kite / Upstox dark terminals.")
        st.warning(
            "Not financial advice. API keys stay on your PC. "
            "Always do your own research before trading."
        )

    view = st.session_state.get("left_nav_focus", "📈 Recommended Stocks")
    active_mode = "RECOMMENDATION ENGINE" if view == "📈 Recommended Stocks" else "KITE HOLDINGS"
    signature_variant_map = {
        "Premium": "premium",
        "Minimal": "minimal",
        "Neon": "neon",
    }
    signature_variant = signature_variant_map.get(
        st.session_state.get("signature_style_variant", "Premium"),
        "premium",
    )
    st.markdown(
        f"""
        <div class="bl-hero">
          <div class="bl-hero-left">
            <div class="bl-signature-wrap bl-signature-style-{signature_variant}">
              <div class="bl-signature">Prithvi's Zone</div>
              <div class="bl-hero-title">BharatMarket<span>Lens</span></div>
            </div>
          </div>
          <div class="bl-hero-chips">
            <span class="bl-chip green">MODE · {active_mode}</span>
            <span class="bl-chip">MARKET · NSE</span>
            <span class="bl-chip">STRATEGY · 50DMA DIP</span>
            <span class="bl-chip">LOCAL · {now}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_col_1, nav_col_2, _ = st.columns([1.1, 1.1, 2.1], gap="small")
    with nav_col_1:
        st.button(
            "📈 Market Recommendations",
            type="primary" if view == "📈 Recommended Stocks" else "secondary",
            use_container_width=True,
            key="top_focus_recommendations",
            on_click=lambda: st.session_state.update(
                {"left_nav_focus_requested": "📈 Recommended Stocks"}
            ),
        )
    with nav_col_2:
        st.button(
            "💼 Kite Holdings",
            type="primary" if view == "💼 Kite Holdings" else "secondary",
            use_container_width=True,
            key="top_focus_holdings",
            on_click=lambda: st.session_state.update(
                {"left_nav_focus_requested": "💼 Kite Holdings"}
            ),
        )

    st.markdown(
        "<div class='bl-switch-caption'>Use these buttons to switch views even when the sidebar is collapsed.</div>",
        unsafe_allow_html=True,
    )

    view = st.session_state.get("left_nav_focus", "📈 Recommended Stocks")
    if view == "💼 Kite Holdings":
        render_holdings_tab()
    else:
        render_scanner_tab()

    st.markdown(
        f"""
        <div class="bl-footer">
          BHARATMARKETLENS · YAHOO FINANCE + ZERODHA KITE CONNECT · PRITHVI'S ZONE · {now}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
