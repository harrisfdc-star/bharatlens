"""
Bharat Lens — Local Nifty 50 dip scanner + Zerodha holdings dashboard.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
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
LOOKBACK_DAYS = 120


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


def analyse_stock(ticker: str, name: str, closes: pd.Series) -> dict[str, Any] | None:
    """Compute price vs 50-day MA and return a recommendation row."""
    if closes is None or len(closes) < MA_WINDOW:
        return None

    ma50 = float(closes.rolling(window=MA_WINDOW).mean().iloc[-1])
    price = float(closes.iloc[-1])
    if np.isnan(ma50) or ma50 <= 0 or np.isnan(price):
        return None

    discount_pct = ((price - ma50) / ma50) * 100.0
    below_ma = price < ma50

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Company": name,
        "Price (₹)": round(price, 2),
        "50-Day MA (₹)": round(ma50, 2),
        "vs MA (%)": round(discount_pct, 2),
        "Signal": "BUY DIP" if below_ma else "HOLD / WAIT",
        "Below MA": below_ma,
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

    def pnl_style(value: float) -> str:
        if value > 0:
            return color_pos
        if value < 0:
            return color_neg
        return color_neutral

    return (
        df.style.format(
            {
                "Avg Cost (₹)": "₹{:,.2f}",
                "LTP (₹)": "₹{:,.2f}",
                "Invested (₹)": "₹{:,.2f}",
                "Current (₹)": "₹{:,.2f}",
                "P&L (₹)": "₹{:,.2f}",
                "P&L (%)": "{:+.2f}%",
            }
        )
        .applymap(pnl_style, subset=["P&L (₹)", "P&L (%)"])
    )


def inject_trading_theme() -> None:
    """Apply a Zerodha/Upstox-style dark trading terminal look."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
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

        [data-testid="stHeader"] {
            background: rgba(11, 15, 20, 0.92) !important;
            border-bottom: 1px solid var(--bl-border);
        }

        [data-testid="stToolbar"] { background: transparent !important; }

        [data-testid="stSidebar"] {
            background: #0E131A !important;
            border-right: 1px solid var(--bl-border) !important;
        }
        [data-testid="stSidebar"] * { color: var(--bl-text); }

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
            padding: 0.85rem 1.1rem;
            margin-bottom: 1rem;
            background: linear-gradient(180deg, #151B24 0%, #10161F 100%);
            border: 1px solid var(--bl-border);
            border-radius: 10px;
        }
        .bl-brand {
            display: flex;
            align-items: baseline;
            gap: 0.55rem;
        }
        .bl-logo {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
        }
        .bl-logo span { color: var(--bl-blue); }
        .bl-tag {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--bl-muted);
            border: 1px solid var(--bl-border);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
        }
        .bl-status {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.82rem;
            color: var(--bl-muted);
            font-family: "IBM Plex Mono", monospace;
        }
        .bl-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--bl-green);
            box-shadow: 0 0 0 3px rgba(0, 179, 134, 0.18);
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
        .stButton > button[data-testid="baseButton-primary"] {
            background: var(--bl-blue) !important;
            border: 1px solid var(--bl-blue) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton > button {
            border-radius: 8px !important;
            border: 1px solid var(--bl-border) !important;
            background: var(--bl-panel-2) !important;
            color: var(--bl-text) !important;
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
        """,
        unsafe_allow_html=True,
    )


def render_scanner_tab() -> None:
    st.subheader("Nifty 50 Buy-the-Dip Recommendation Dashboard")
    st.caption(
        "Simple rule: highlight stocks whose latest price is below the 50-day moving average."
    )

    run_scan = st.button("Run / Refresh Scan", type="primary")

    with st.expander("How the signal works"):
        st.markdown(
            """
            1. Download recent daily prices from Yahoo Finance  
            2. Calculate the 50-day moving average  
            3. If **Price < 50-day MA** → **BUY DIP**  
            4. Otherwise → **HOLD / WAIT**
            """
        )

    if not run_scan and "scan_results" not in st.session_state:
        st.info("Click **Run / Refresh Scan** above to start scanning Nifty 50 stocks.")
        return

    if run_scan or "scan_results" not in st.session_state:
        tickers = tuple(NIFTY_50.keys())
        with st.spinner("Downloading Nifty 50 market data… this may take 20–40 seconds"):
            try:
                market = fetch_market_data(tickers)
            except Exception as exc:
                st.error(
                    "Could not download market data. Check your internet connection and try again."
                )
                st.exception(exc)
                return

        if not market:
            st.error("No price data was returned. Please try again in a few minutes.")
            return

        rows: list[dict[str, Any]] = []
        progress = st.progress(0, text="Analysing stocks…")
        for i, (ticker, name) in enumerate(NIFTY_50.items(), start=1):
            closes = market.get(ticker)
            if closes is not None:
                row = analyse_stock(ticker, name, closes)
                if row is not None:
                    rows.append(row)
            progress.progress(i / len(NIFTY_50), text=f"Analysing {name}…")
        progress.empty()

        if not rows:
            st.error("Could not analyse any stocks. Please refresh and try again.")
            return

        st.session_state["scan_results"] = rows

    results = st.session_state["scan_results"]
    buy_dips = [r for r in results if r["Below MA"]]
    holds = [r for r in results if not r["Below MA"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", len(results))
    c2.metric("BUY DIP signals", len(buy_dips))
    c3.metric("HOLD / WAIT", len(holds))
    avg_disc = (
        round(float(np.mean([r["vs MA (%)"] for r in buy_dips])), 2) if buy_dips else 0.0
    )
    c4.metric("Avg discount vs MA", f"{avg_disc}%")

    st.markdown("---")
    st.markdown("### Recommendation Agent — Stocks Below 50-Day MA")

    if not buy_dips:
        st.success(
            "No Nifty 50 stock is currently trading below its 50-day moving average. "
            "The market may be relatively strong — wait for a clearer dip."
        )
    else:
        buy_table = pd.DataFrame(
            [
                {
                    "Symbol": r["Symbol"],
                    "Company": r["Company"],
                    "Price (₹)": r["Price (₹)"],
                    "50-Day MA (₹)": r["50-Day MA (₹)"],
                    "Discount vs MA (%)": r["vs MA (%)"],
                    "Signal": r["Signal"],
                }
                for r in sorted(buy_dips, key=lambda x: x["vs MA (%)"])
            ]
        )
        st.dataframe(buy_table, use_container_width=True, hide_index=True)
        st.caption("Sorted by deepest discount vs the 50-day MA (most negative % first).")

    with st.expander("View full Nifty 50 scan results"):
        full_table = pd.DataFrame(
            [
                {
                    "Symbol": r["Symbol"],
                    "Company": r["Company"],
                    "Price (₹)": r["Price (₹)"],
                    "50-Day MA (₹)": r["50-Day MA (₹)"],
                    "vs MA (%)": r["vs MA (%)"],
                    "Signal": r["Signal"],
                }
                for r in sorted(results, key=lambda x: x["vs MA (%)"])
            ]
        )
        st.dataframe(full_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Stock Chart Explorer")

    options = {f"{r['Company']} ({r['Symbol']})": r for r in results}
    default_key = None
    if buy_dips:
        best = sorted(buy_dips, key=lambda x: x["vs MA (%)"])[0]
        default_key = f"{best['Company']} ({best['Symbol']})"
    labels = list(options.keys())
    default_index = labels.index(default_key) if default_key in labels else 0

    selected_label = st.selectbox("Choose a stock to inspect", labels, index=default_index)
    selected = options[selected_label]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Current Price", f"₹{selected['Price (₹)']:,.2f}")
    col_b.metric("50-Day MA", f"₹{selected['50-Day MA (₹)']:,.2f}")
    col_c.metric("vs MA", f"{selected['vs MA (%)']:+.2f}%", selected["Signal"])

    st.plotly_chart(
        build_chart(selected["Company"], selected["_closes"], selected["_ma"]),
        use_container_width=True,
    )


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
                        "Price (₹)": row["Price (₹)"],
                        "50-Day MA (₹)": row["50-Day MA (₹)"],
                        "vs MA (%)": row["vs MA (%)"],
                    }
                )
        if overlap:
            st.markdown("### Your Nifty 50 holdings vs dip scanner")
            st.dataframe(pd.DataFrame(overlap), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Bharat Lens | Trading Terminal",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_trading_theme()

    now = dt.datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(
        f"""
        <div class="bl-topbar">
          <div class="bl-brand">
            <div class="bl-logo">Bharat<span>Lens</span></div>
            <div class="bl-tag">Trading Terminal</div>
          </div>
          <div class="bl-status">
            <span class="bl-dot"></span>
            LOCAL · {now}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Workspace")
        st.write(
            "Connect **Zerodha Holdings** for your Demat portfolio, "
            "or open **Nifty 50 Scanner** for dip signals."
        )
        st.caption("Theme inspired by Kite / Upstox dark terminals.")
        st.warning(
            "Not financial advice. API keys stay on your PC. "
            "Always do your own research before trading."
        )

    tab_holdings, tab_scanner = st.tabs(["Portfolio", "Scanner"])
    with tab_holdings:
        render_holdings_tab()
    with tab_scanner:
        render_scanner_tab()

    st.markdown(
        f"""
        <div class="bl-footer">
          BHARAT LENS · YAHOO FINANCE + ZERODHA KITE CONNECT · {now}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
