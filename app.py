"""
BharatMarketLens — Local Nifty 50 intelligence scanner + Zerodha holdings dashboard.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
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
    baseline_date = existing.get("date", "Unknown")
    change_text = _format_since_added_change(current_val, baseline_price)
    return f"{change_text} | {baseline_date}"


def _confidence_value(row: dict[str, Any]) -> float:
    """Safe confidence extractor for descending sort."""
    value = row.get("Confidence (1-10)")
    if value is None or pd.isna(value):
        return float("-inf")
    return float(value)


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
def fetch_market_data(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    """Download recent daily closes for all tickers (cached for 1 hour)."""
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)

    raw = yf.download(
        list(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    result: dict[str, pd.Series] = {}
    for ticker in tickers:
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
    if closes is None or len(closes) < MA_WINDOW:
        return {
            "name": "NIFTY 50",
            "regime": "Sideways",
            "weekly_pct": 0.0,
            "score_boost": 0.0,
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

    return {
        "name": "NIFTY 50",
        "regime": regime,
        "weekly_pct": round(weekly_pct, 2),
        "score_boost": boost,
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
    if not hist.empty and "Volume" in hist.columns and "Close" in hist.columns:
        vol = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0.0)
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
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

    total_assets = _to_float(info.get("totalAssets"))
    total_debt = _to_float(info.get("totalDebt"))
    debt_to_assets = None
    if total_assets and total_assets > 0 and total_debt is not None:
        debt_to_assets = total_debt / total_assets

    return {
        "promoter_stake_pct": promoter_stake,
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
        "days_to_earnings": days_to_earnings,
    }


def analyse_stock(
    ticker: str,
    name: str,
    closes: pd.Series,
    market_pulse: dict[str, Any],
    external_news_rows: list[dict[str, str]],
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

    raw_score = technical_score + fundamental_score + flow_score + dip_resilience_bonus
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
        "_score_raw": raw_score,
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
            --bl-muted: #8B98A8;
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
            color: #b7c8da;
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
            color: #b7c8da;
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
            color: var(--bl-muted);
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
        .bl-switch-caption {
            color: #7f8d9f;
            font-size: 0.9rem;
            margin: 0.1rem 0 0.05rem 0;
        }

        h1, h2, h3, h4 { color: var(--bl-text) !important; font-weight: 600 !important; }
        p, label, .stMarkdown, .stCaption { color: var(--bl-muted); }
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
    # Auto-refresh this view every 15 seconds.
    st_autorefresh(interval=15_000, key="market_intel_autorefresh_15s")

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

    if force_refresh:
        fetch_market_data.clear()
        fetch_market_pulse.clear()
        fetch_live_index_snapshot.clear()
        fetch_external_market_news.clear()
        fetch_stock_context.clear()
        st.session_state.pop("scan_results", None)
        st.session_state.pop("market_pulse", None)
        st.session_state.pop("external_news_rows", None)
        st.success("Data cache cleared. Re-evaluating all signals now.")
        run_scan = True

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
                  <div style="color:#8B98A8;font-size:0.72rem;line-height:1.1;">{label}</div>
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
    search_query = str(st.session_state.get("script_search_input", "")).strip()
    min_confidence = float(st.session_state.get("min_confidence_filter", 5.5))

    should_scan = run_scan or "scan_results" not in st.session_state
    if should_scan:
        tickers = tuple(NIFTY_50.keys())
        with st.spinner("Collecting market, fundamentals, and multi-source news signals… this can take 1-3 minutes"):
            try:
                market = fetch_market_data(tickers)
                market_pulse = fetch_market_pulse()
                external_news_rows = fetch_external_market_news()
            except Exception as exc:
                st.error("Could not download market data. Please check internet and retry.")
                st.exception(exc)
                return

        if not market:
            st.error("No price data was returned. Please try again in a few minutes.")
            return

        rows: list[dict[str, Any]] = []
        progress = st.progress(0, text="Analysing scripts…")
        for i, (ticker, name) in enumerate(NIFTY_50.items(), start=1):
            closes = market.get(ticker)
            if closes is not None:
                row = analyse_stock(ticker, name, closes, market_pulse, external_news_rows)
                if row is not None:
                    rows.append(row)
            progress.progress(i / len(NIFTY_50), text=f"Analysing {name}…")
        progress.empty()

        if not rows:
            st.error(
                f"No stocks matched the analysis under the active price cap (<= ₹{MAX_STOCK_PRICE:,.0f}). "
                "Try again later when prices or signals change."
            )
            return

        st.session_state["scan_results"] = rows
        st.session_state["market_pulse"] = market_pulse
        st.session_state["external_news_rows"] = external_news_rows

    results = st.session_state["scan_results"]
    if results and (
        "Reason for Recommendation" not in results[0] or "Day changed" not in results[0]
    ):
        st.session_state.pop("scan_results", None)
        st.session_state.pop("market_pulse", None)
        st.session_state.pop("external_news_rows", None)
        st.info("Refreshing scan data to include newly added analysis columns…")
        st.rerun()
    if results and any((r.get("CMP (₹)") or 0) > MAX_STOCK_PRICE for r in results):
        st.session_state.pop("scan_results", None)
        st.session_state.pop("market_pulse", None)
        st.session_state.pop("external_news_rows", None)
        st.info(f"Refreshing scan data to apply price cap <= ₹{MAX_STOCK_PRICE:,.0f}…")
        st.rerun()
    market_pulse = st.session_state.get("market_pulse", fetch_market_pulse())
    external_news_rows = st.session_state.get("external_news_rows", [])
    sorted_results = sorted(results, key=lambda x: x["Confidence (1-10)"], reverse=True)
    for row in sorted_results:
        row["_since_added"] = _track_since_added_change(
            "scanner_since_added_baseline",
            row.get("Symbol"),
            row.get("CMP (₹)"),
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

        st.caption(
            f"Market pulse: **{market_pulse.get('name', 'NIFTY 50')}** is "
            f"**{market_pulse.get('regime', 'Sideways')}** "
            f"({market_pulse.get('weekly_pct', 0.0):+.2f}% weekly). "
            f"External headlines fetched: **{len(external_news_rows)}**."
        )

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
                    "52W High (₹)": row["52W High (₹)"],
                    "Exp MA 2W (₹)": row["Expected MA (2W, ₹)"],
                    "Proposed Exit (₹)": row["Proposed Exit Price (₹)"],
                    "Promoter %": row["Promoter / Insider Stake (%)"],
                    "Rec": row["Recommendation"],
                    "Met Exp": row["Met Expectation Today"],
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

        # Removed redundant strict-actionable table to avoid duplication with
        # "Favourable right now" and keep the dashboard decision-first.

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
                        "Confidence": r.get("Confidence (1-10)"),
                        "Proposed Exit (₹)": r.get("Proposed Exit Price (₹)"),
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
                        "Confidence": r.get("Confidence (1-10)"),
                        "Proposed Exit (₹)": r.get("Proposed Exit Price (₹)"),
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

        panel_controls, panel_pulse = st.columns([1.6, 1.0], gap="large")
        with panel_controls:
            st.markdown("### Scan Controls")
            st.text_input(
                "Search script name or symbol",
                placeholder="Example: TCS, Reliance, HDFC Bank",
                key="script_search_input",
            )
            st.slider("Minimum confidence score", 1.0, 10.0, 5.5, 0.5, key="min_confidence_filter")
            with st.expander("How recommendations are generated"):
                st.markdown(
                    """
                    - **Technicals:** CMP vs moving averages, RSI, weekly momentum, and 52-week range behavior  
                    - **Fundamentals:** promoter/insider stake proxy, assets/liabilities, sales growth, margin quality  
                    - **Flow + sentiment:** buy/sell volume trend, earnings/event timing, and live headline sentiment  
                    - **Sources used (available access):** Yahoo Finance, Moneycontrol-related headlines, Economic Times-related headlines, Zerodha blog feed, and Screener-related headlines  
                    - Output is a **ranked confidence score (1 to 10)** plus **BUY / WATCH / HOLD / SELL**
                    """
                )

        with panel_pulse:
            st.markdown("### Market Pulse")
            st.metric("NIFTY Weekly", f"{market_pulse.get('weekly_pct', 0.0):+.2f}%")
            st.metric("Regime", market_pulse.get("regime", "Sideways"))
            st.metric("Headlines", len(external_news_rows))

        st.markdown("---")
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
    if "Since added" not in df.columns:
        df["Since added"] = "NA"
    df["Since added"] = df.apply(
        lambda row: _track_since_added_change(
            "holdings_since_added_baseline",
            row.get("Symbol"),
            row.get("LTP (₹)"),
        ),
        axis=1,
    )

    preferred_holdings_cols = [
        "Symbol",
        "Exchange",
        "Qty",
        "Day changed",
        "Since added",
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdings", len(df))
    m2.metric("Invested", f"₹{invested:,.2f}")
    m3.metric("Current value", f"₹{current:,.2f}")
    m4.metric("Overall P&L", f"₹{pnl:,.2f}", f"{pnl_pct:+.2f}%")

    chart_col, table_col = st.columns([1, 2])
    with chart_col:
        st.plotly_chart(build_portfolio_pie(df), use_container_width=True)
    with table_col:
        st.dataframe(
            style_holdings_dataframe(df),
            use_container_width=True,
            hide_index=True,
        )

    # Optional: overlay dip signal for holdings that are in Nifty 50
    nifty_symbols = {sym.replace(".NS", "") for sym in NIFTY_50}
    held_nifty = [s for s in df["Symbol"].tolist() if s in nifty_symbols]
    if held_nifty and "scan_results" in st.session_state:
        scan_map = {r["Symbol"]: r for r in st.session_state["scan_results"]}
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
        st.caption("Theme inspired by Kite / Upstox dark terminals.")
        st.warning(
            "Not financial advice. API keys stay on your PC. "
            "Always do your own research before trading."
        )

    view = st.session_state.get("left_nav_focus", "📈 Recommended Stocks")
    active_mode = "RECOMMENDATION ENGINE" if view == "📈 Recommended Stocks" else "KITE HOLDINGS"
    st.markdown(
        f"""
        <div class="bl-hero">
          <div class="bl-hero-left">
            <div>
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
          BHARATMARKETLENS · YAHOO FINANCE + ZERODHA KITE CONNECT · {now}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
