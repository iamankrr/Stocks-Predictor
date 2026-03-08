# Aman
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Optional Lottie (safe-guarded)
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

import plotly.graph_objects as go
import plotly.express as px
import requests

# -------------------------------
# Configuration
# -------------------------------
MODEL_FILENAME = "Stock Prediction Model.keras"
DEFAULT_TICKER = "RELIANCE.NS"
PAST_WINDOW = 100
FUTURE_DAYS = 90
DATE_FREQ = "B"  # business days

st.set_page_config(
    page_title="Stock Price Predictor — Premium",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Session defaults
# -------------------------------
if "show_table" not in st.session_state:
    st.session_state["show_table"] = True
if "live" not in st.session_state:
    st.session_state["live"] = False
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

PLOTLY_THEME = "plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white"

# -------------------------------
# Data class
# -------------------------------
@dataclass
class RunContext:
    ticker_input: str
    ticker_used: Optional[str]
    currency: str
    model: Any
    info: Dict[str, Any]
    df: pd.DataFrame

# -------------------------------
# ML Utilities
# -------------------------------
def load_model_safe(path: Path):
    try:
        model = load_model(str(path))
        return model, None
    except Exception as e:
        return None, str(e)


def try_yfinance_variants(ticker: str, start: str = "2010-01-01", end: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if "." in ticker:
        variants = [ticker]
    else:
        variants = [ticker + ".NS", ticker + ".BO", ticker + ".BSE", ticker]
    last_ex = None
    for v in variants:
        try:
            df = yf.download(v, start=start, end=end, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                continue
            if "Adj Close" in df.columns:
                df["adj_close"] = df["Adj Close"].astype(float)
            else:
                df["adj_close"] = df["Close"].astype(float)
            df["Close"] = df["Close"].astype(float)
            df["ticker_used"] = v
            df["return"] = df["adj_close"].pct_change()
            return df, v
        except Exception as e:
            last_ex = e
            continue
    raise ValueError(f"yfinance returned no usable data for ticker '{ticker}'. Last error: {last_ex}")


def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        for k in ("marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta", "dividendYield", "averageVolume"):
            if k in info and info[k] is not None:
                try:
                    info[k] = float(info[k])
                except Exception:
                    pass
        return info
    except Exception:
        return {}


def enforce_currency(ticker_input: str, ticker_used: Optional[str], info: Dict[str, Any]) -> str:
    used = (ticker_used or ticker_input).upper()
    exchange = (info.get("exchange") or "").upper()
    if used.endswith((".NS", ".BO", ".BSE")) or "NSE" in exchange or "BSE" in exchange:
        return "INR"
    if any(x in exchange for x in ("NASDAQ", "NYSE", "ARCA", "NMS")):
        return "USD"
    if "." not in ticker_input and ticker_input.isalnum():
        return "USD"
    return "INR"


def prepare_sequences(prices: pd.Series, past_window: int, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:
    arr = scaler.transform(prices.values.reshape(-1, 1))
    x, y = [], []
    for i in range(past_window, arr.shape[0]):
        x.append(arr[i - past_window:i, 0].reshape(past_window, 1))
        y.append(arr[i, 0])
    return np.array(x, dtype=float), np.array(y, dtype=float)


def safe_inverse_transform(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    a = np.array(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    try:
        inv = scaler.inverse_transform(a)
        return inv.reshape(-1)
    except Exception:
        min_v, max_v = float(scaler.data_min_[0]), float(scaler.data_max_[0])
        out = a.flatten() * (max_v - min_v) + min_v
        return out


def predict_recursive(model, last_window_scaled: np.ndarray, steps: int, scaler: MinMaxScaler) -> np.ndarray:
    future_scaled = []
    inp = last_window_scaled.copy().astype(float)
    for _ in range(steps):
        p = model.predict(inp, verbose=0)
        p_arr = np.array(p)
        if p_arr.ndim == 2:
            val = float(p_arr[0, 0])
        elif p_arr.ndim == 1:
            val = float(p_arr[0])
        else:
            val = float(p_arr)
        future_scaled.append(val)
        shift = np.array(val).reshape(1, 1, 1).astype(float)
        inp = np.concatenate([inp[:, 1:, :], shift], axis=1)
    return safe_inverse_transform(scaler, np.array(future_scaled))

# -------------------------------
# COMPLETELY REWRITTEN CSS
# -------------------------------
def inject_premium_css():
    dark_mode = st.session_state.get("dark_mode", False)

    # ── Base variables ──────────────────────────────────────────────
    if dark_mode:
        bg_app        = "#0d1117"
        bg_sidebar    = "#161b22"
        bg_card       = "#1c2128"
        bg_card2      = "#21262d"
        border_color  = "#30363d"
        text_primary  = "#e6edf3"
        text_secondary= "#8b949e"
        text_muted    = "#6e7681"
        input_bg      = "#21262d"
        input_border  = "#30363d"
        hover_shadow  = "rgba(88,166,255,0.15)"
        accent        = "#58a6ff"
    else:
        bg_app        = "#f0f2f5"
        bg_sidebar    = "#ffffff"
        bg_card       = "#ffffff"
        bg_card2      = "#f8fafc"
        border_color  = "#e2e8f0"
        text_primary  = "#0f172a"
        text_secondary= "#475569"
        text_muted    = "#94a3b8"
        input_bg      = "#ffffff"
        input_border  = "#cbd5e1"
        hover_shadow  = "rgba(102,126,234,0.12)"
        accent        = "#667eea"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Reset & Base ───────────────────────────────────── */
    * {{ font-family: 'Inter', sans-serif !important; box-sizing: border-box; }}

    /* ── HIDE ALL STREAMLIT CHROME ELEMENTS ─────────────── */
    /* Sidebar toggle / collapse button — removes "keyt" ghost text */
    [data-testid="collapsedControl"],
    button[kind="header"],
    .st-emotion-cache-zq5wmm,
    .st-emotion-cache-1egp75f,
    header[data-testid="stHeader"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* Hide default hamburger / deploy menu bar */
    #MainMenu {{ visibility: hidden !important; }}
    footer {{ visibility: hidden !important; }}

    /* ── App background ──────────────────────────────────── */
    .stApp {{ background: {bg_app} !important; }}
    .main .block-container {{
        background: transparent !important;
        padding: 1.5rem 2rem 3rem 2rem !important;
        max-width: 1400px !important;
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {bg_sidebar} !important;
        border-right: 1px solid {border_color} !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        background: {bg_sidebar} !important;
        padding-top: 1rem !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {{
        color: {text_secondary} !important;
        line-height: 1.55 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {text_primary} !important;
        margin: 0.75rem 0 0.5rem !important;
    }}

    /* ── Typography ──────────────────────────────────────── */
    p, span, div, label, li {{ color: {text_secondary} !important; line-height: 1.6 !important; }}
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-weight: 700 !important;
        margin: 1.25rem 0 0.75rem !important;
        line-height: 1.35 !important;
    }}
    strong {{ color: {text_primary} !important; font-weight: 600 !important; }}
    .stMarkdown {{ margin-bottom: 0.75rem !important; }}
    .stMarkdown p {{ margin-bottom: 0.4rem !important; }}

    /* ── Metric Cards ────────────────────────────────────── */
    .metric-card {{
        padding: 22px 20px !important;
        border-radius: 14px;
        background: {bg_card} !important;
        border: 1px solid {border_color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 12px !important;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px {hover_shadow};
    }}

    /* ── Inputs ──────────────────────────────────────────── */
    .stTextInput, .stSelectbox, .stNumberInput {{ margin-bottom: 0.85rem !important; }}
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: {input_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {input_border} !important;
        border-radius: 8px !important;
        padding: 0.55rem 0.75rem !important;
        font-size: 14px !important;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 3px {hover_shadow} !important;
        outline: none !important;
    }}
    .stSelectbox > div > div > div {{
        background: {input_bg} !important;
        color: {text_primary} !important;
        border-color: {input_border} !important;
        border-radius: 8px !important;
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton {{ margin-bottom: 0.4rem !important; }}
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.01em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(102,126,234,0.25) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.4) !important;
    }}
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
    }}

    /* ── Expander ────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {bg_card2} !important;
        color: {text_primary} !important;
        border-radius: 10px !important;
        padding: 0.9rem 1rem !important;
        font-weight: 600 !important;
        border: 1px solid {border_color} !important;
        margin-bottom: 4px !important;
    }}
    .streamlit-expanderContent {{
        background: {bg_card} !important;
        padding: 1.25rem !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid {border_color} !important;
        border-top: none !important;
    }}

    /* ── DataFrames ──────────────────────────────────────── */
    .stDataFrame {{ margin-bottom: 1.25rem !important; border-radius: 10px !important; overflow: hidden !important; }}
    .stDataFrame table {{ background: {bg_card} !important; color: {text_secondary} !important; }}
    .stDataFrame thead tr th {{
        background: {bg_card2} !important;
        color: {text_primary} !important;
        padding: 0.65rem 0.85rem !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }}
    .stDataFrame tbody tr td {{
        background: {bg_card} !important;
        color: {text_secondary} !important;
        padding: 0.5rem 0.85rem !important;
        font-size: 13px !important;
    }}
    .stDataFrame tbody tr:hover td {{ background: {bg_card2} !important; }}

    /* ── Alerts ──────────────────────────────────────────── */
    .stAlert, .stSuccess, .stInfo {{
        background: {bg_card} !important;
        color: {text_secondary} !important;
        border-color: {border_color} !important;
        border-radius: 10px !important;
        margin-bottom: 0.85rem !important;
    }}

    /* ── Checkbox ────────────────────────────────────────── */
    .stCheckbox {{ margin-bottom: 0.4rem !important; }}
    .stCheckbox label span {{ color: {text_secondary} !important; font-size: 14px !important; }}

    /* ── Charts ──────────────────────────────────────────── */
    .js-plotly-plot, .plotly {{ margin-bottom: 1.25rem !important; }}

    /* ── Spinner ─────────────────────────────────────────── */
    .stSpinner > div {{ border-top-color: #667eea !important; }}

    /* ── Caption ─────────────────────────────────────────── */
    .stCaption {{ color: {text_muted} !important; font-size: 12px !important; }}

    /* ── Column padding ──────────────────────────────────── */
    [data-testid="column"] {{ padding: 0 0.4rem !important; }}

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {border_color}; border-radius: 3px; }}
    </style>
    """, unsafe_allow_html=True)


# -------------------------------
# Lottie loader (cached)
# -------------------------------
@st.cache_data
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


LOTTIE_LOADING   = load_lottie("https://assets1.lottiefiles.com/packages/lf20_p8bfn5to.json") if LOTTIE_AVAILABLE else None
LOTTIE_PREDICTING= load_lottie("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json") if LOTTIE_AVAILABLE else None

# -------------------------------
# Charts
# -------------------------------
def plotly_actual_vs_pred(actual, predicted, currency):
    actual_list    = np.array(actual,    dtype=float).flatten().tolist()
    predicted_list = np.array(predicted, dtype=float).flatten().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actual_list, mode="lines", name="Actual",
        line=dict(width=2, color="#58a6ff"),
        hovertemplate="Index: %{x}<br>Actual: %{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=predicted_list, mode="lines", name="Predicted",
        line=dict(width=2, color="#f78166", dash="dot"),
        hovertemplate="Index: %{x}<br>Predicted: %{y:,.2f}<extra></extra>"
    ))
    fig.update_layout(
        template=PLOTLY_THEME, height=420,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Sample Index",
        yaxis_title=f"Price ({currency})",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.12)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.12)'),
    )
    st.plotly_chart(fig, use_container_width=True, key="actual_vs_pred_chart")


def plotly_forecast(history, future, currency, vol):
    hist_arr   = np.array(history, dtype=float).flatten()
    future_arr = np.array(future,  dtype=float).flatten() if future is not None and len(future) > 0 else np.array([])
    hist_display = hist_arr[-200:] if len(hist_arr) > 200 else hist_arr
    start_idx    = len(hist_arr) - len(hist_display)
    x_hist   = list(range(start_idx, start_idx + len(hist_display)))
    x_future = list(range(start_idx + len(hist_display), start_idx + len(hist_display) + len(future_arr)))
    try:
        vol_val = float(vol) if (vol is not None and not np.isnan(vol)) else 0.0
        vol_val = min(max(vol_val, 0.0), 0.5)
    except Exception:
        vol_val = 0.0
    fig = go.Figure()
    if len(hist_display) > 0:
        fig.add_trace(go.Scatter(
            x=x_hist, y=hist_display.tolist(), mode="lines",
            name="Historical Price",
            line=dict(width=2, color="#58a6ff"),
            hovertemplate="Day %{x}<br>Price: %{y:,.2f}<extra></extra>"
        ))
    if len(future_arr) > 0:
        fig.add_trace(go.Scatter(
            x=x_future, y=future_arr.tolist(), mode="lines+markers",
            name="90-Day Forecast",
            line=dict(width=2.5, color="#3fb950"),
            marker=dict(size=3, color="#3fb950"),
            hovertemplate="Day %{x}<br>Forecast: %{y:,.2f}<extra></extra>"
        ))
        if vol_val > 0.01:
            upper = future_arr * (1.0 + vol_val)
            lower = future_arr * (1.0 - vol_val)
            fig.add_trace(go.Scatter(
                x=x_future, y=upper.tolist(), mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x_future, y=lower.tolist(), mode="lines",
                name="Confidence Band", fill='tonexty',
                fillcolor="rgba(63,185,80,0.15)",
                line=dict(width=0),
                hovertemplate="Range: %{y:,.2f}<extra></extra>"
            ))
    fig.update_layout(
        template=PLOTLY_THEME, height=480,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Timeline (Days)",
        yaxis_title=f"Price ({currency})",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.12)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.12)'),
    )
    st.plotly_chart(fig, use_container_width=True, key="forecast_chart_main")


# -------------------------------
# Helpers
# -------------------------------
def real_time_price(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = getattr(tk, "fast_info", None)
        if info and hasattr(info, "last_price"):
            return info.last_price
        df = tk.history(period="1d")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        return None
    return None


def run_portfolio_backtest(initial_amount, future_prices):
    if len(future_prices) == 0:
        return np.array([])
    initial_price = float(future_prices[0])
    if initial_price == 0:
        return np.array([])
    units = initial_amount / initial_price
    return units * np.array(future_prices, dtype=float)


# -------------------------------
# Main
# -------------------------------
def main():
    inject_premium_css()

    # ── HEADER ───────────────────────────────────────────────────────
    current_time = pd.Timestamp.now().strftime("%d %b %Y, %I:%M %p")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(102,126,234,0.28);
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div>
                <div style="font-size:28px; font-weight:800; color:white; letter-spacing:-0.5px; line-height:1.2;">
                    🚀 AI Stock Predictor Pro
                </div>
                <div style="font-size:13px; color:rgba(255,255,255,0.85); margin-top:8px; font-weight:500; line-height:1.7;">
                    ⚡ Real-time Analysis &nbsp;•&nbsp; 📊 90-Day ML Forecast &nbsp;•&nbsp; 💼 Portfolio Simulator &nbsp;•&nbsp; 🎯 Smart Signals
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:rgba(255,255,255,0.7); font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">Last Updated</div>
                <div style="font-size:14px; color:white; font-weight:700; margin-top:4px;">{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            text-align: center;
        ">
            <div style="font-size:16px; font-weight:800; color:white;">⚙️ Control Panel</div>
            <div style="font-size:12px; color:rgba(255,255,255,0.8); margin-top:4px;">Configure your analysis</div>
        </div>
        """, unsafe_allow_html=True)

        model_path = Path(__file__).parent / MODEL_FILENAME
        model, err = load_model_safe(model_path)
        if model is None:
            st.error("❌ Model load failed: " + (err or "unknown"))
            st.stop()
        else:
            st.success("✅ AI Model Loaded")

        st.markdown("---")
        st.markdown("#### 🎯 Stock Selection")

        popular_stocks_dict = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS":      "Tata Consultancy Services",
            "INFY.NS":     "Infosys Limited",
            "HDFCBANK.NS": "HDFC Bank",
            "AAPL":        "Apple Inc",
            "MSFT":        "Microsoft Corporation",
            "TSLA":        "Tesla Inc",
            "GOOGL":       "Alphabet Inc",
            "AMZN":        "Amazon.com Inc",
            "META":        "Meta Platforms Inc",
        }
        display_options = [f"{k} — {v}" for k, v in popular_stocks_dict.items()]
        selected_option = st.selectbox(
            "Quick-select a stock",
            options=[""] + display_options,
            help="Select from popular stocks or enter a custom ticker below"
        )
        if selected_option:
            ticker_input = selected_option.split(" — ")[0].strip()
        else:
            ticker_input = st.text_input(
                "Custom ticker symbol",
                DEFAULT_TICKER,
                help="e.g., AAPL, TSLA, RELIANCE.NS"
            ).strip().upper()

        start_date = st.text_input("📅 History start date", "2012-01-01")

        st.markdown("---")
        st.markdown("#### 🎛️ Options")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state["live"]       = st.checkbox("📡 Live Price",  value=st.session_state.get("live", False))
        with col2:
            st.session_state["show_table"] = st.checkbox("📋 Table",       value=st.session_state.get("show_table", True))

        st.session_state["dark_mode"] = st.checkbox("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))

        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(102,126,234,0.08); border-left:3px solid #667eea; border-radius:8px; padding:14px 16px;">
            <div style="font-size:13px; font-weight:700; color:#667eea; margin-bottom:8px;">✨ Features</div>
            <div style="font-size:12px; color:#64748b; line-height:1.8;">
                AI-powered predictions<br>
                90-day forecasting<br>
                Portfolio backtesting<br>
                Interactive charts<br>
                Real-time data<br>
                Dark / Light mode
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── VALIDATE TICKER ───────────────────────────────────────────────
    if not ticker_input:
        st.info("👈 Enter a ticker in the sidebar to begin.")
        st.stop()

    # ── FETCH DATA ────────────────────────────────────────────────────
    with st.spinner("Fetching historical data…"):
        if LOTTIE_AVAILABLE and LOTTIE_LOADING:
            try:
                st_lottie(LOTTIE_LOADING, height=80, key="loading_lottie")
            except Exception:
                pass
        try:
            df, ticker_used = try_yfinance_variants(ticker_input, start=start_date)
        except Exception as e:
            st.error("Data fetch failed: " + str(e))
            st.stop()

    info     = fetch_ticker_info(ticker_used)
    currency = enforce_currency(ticker_input, ticker_used, info)
    ctx      = RunContext(ticker_input=ticker_input, ticker_used=ticker_used,
                          currency=currency, model=model, info=info, df=df)

    # ── DATA HEADER ───────────────────────────────────────────────────
    st.markdown(f"### 📄 {ctx.ticker_input} &nbsp;—&nbsp; Retrieved as `{ctx.ticker_used}`")
    st.dataframe(ctx.df.tail(5), use_container_width=True)

    # ── COMPANY METADATA ──────────────────────────────────────────────
    st.markdown("---")
    if "show_metadata" not in st.session_state:
        st.session_state["show_metadata"] = False
    btn_label = "🏢 Company Metadata  ▼" if not st.session_state["show_metadata"] else "🏢 Company Metadata  ▲"
    if st.button(btn_label, use_container_width=True, key="metadata_toggle"):
        st.session_state["show_metadata"] = not st.session_state["show_metadata"]

    if st.session_state["show_metadata"]:
        is_dark     = st.session_state.get("dark_mode", False)
        card_bg     = "#1c2128" if is_dark else "#ffffff"
        card_border = "#30363d" if is_dark else "#e2e8f0"
        lbl_color   = "#8b949e" if is_dark else "#64748b"
        val_color   = "#e6edf3" if is_dark else "#0f172a"
        chip_bg     = "rgba(88,166,255,0.08)" if is_dark else "rgba(102,126,234,0.06)"

        if ctx.info:
            name        = ctx.info.get("longName") or ctx.info.get("shortName") or "N/A"
            exchange    = ctx.info.get("exchange") or "N/A"
            sector      = ctx.info.get("sector")   or "N/A"
            industry    = ctx.info.get("industry") or "N/A"
            country     = ctx.info.get("country")  or "N/A"
            website     = ctx.info.get("website")  or "N/A"
            employees   = ctx.info.get("fullTimeEmployees")
            ceo_list    = ctx.info.get("companyOfficers")
            owner_name  = "N/A"
            if ceo_list and isinstance(ceo_list, list):
                for o in ceo_list:
                    if isinstance(o, dict):
                        t = o.get("title", "").lower()
                        if any(k in t for k in ("ceo","chief executive","chairman","managing director")):
                            owner_name = o.get("name", "N/A")
                            break

            market_cap = ctx.info.get("marketCap")
            if market_cap:
                if ctx.currency == "INR":
                    crore = market_cap / 1e7
                    market_cap_str = f"₹{crore/1e5:.2f} Lakh Cr" if crore >= 1e5 else f"₹{crore:.2f} Cr"
                else:
                    market_cap_str = f"${market_cap/1e12:.2f}T" if market_cap >= 1e12 else f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = "N/A"

            beta      = ctx.info.get("beta");        beta_str  = f"{beta:.2f}" if beta else "N/A"
            div_yield = ctx.info.get("dividendYield"); div_str  = f"{div_yield:.2%}" if div_yield else "N/A"
            pe_ratio  = ctx.info.get("trailingPE");  pe_str    = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
            w52h      = ctx.info.get("fiftyTwoWeekHigh"); w52h_str = f"{ctx.currency} {w52h:,.2f}" if w52h else "N/A"
            w52l      = ctx.info.get("fiftyTwoWeekLow");  w52l_str = f"{ctx.currency} {w52l:,.2f}" if w52l else "N/A"
            avg_vol   = ctx.info.get("averageVolume")
            avg_vol_str = f"{avg_vol/1e6:.2f}M" if avg_vol and avg_vol >= 1e6 else (f"{avg_vol/1e3:.2f}K" if avg_vol else "N/A")
            emp_str   = f"{employees:,}" if employees else "N/A"

            ipo_str = "N/A"
            ipo_ts  = ctx.info.get("firstTradeDateEpochUtc")
            if ipo_ts:
                try:
                    from datetime import datetime
                    ipo_str = datetime.fromtimestamp(ipo_ts).strftime("%d %b %Y")
                except Exception:
                    pass
            if ipo_str == "N/A" and not ctx.df.empty:
                ipo_str = ctx.df.index[0].strftime("%d %b %Y") + " (Est.)"

            def info_chip(label, value):
                return f"""
                <div style="margin-bottom:14px;">
                    <div style="font-size:11px; font-weight:600; color:{lbl_color}; text-transform:uppercase;
                                letter-spacing:0.06em; margin-bottom:5px;">{label}</div>
                    <div style="padding:10px 14px; background:{chip_bg}; border-radius:8px;
                                color:{val_color}; font-size:14px; font-weight:500;
                                border:1px solid {card_border};">{value}</div>
                </div>"""

            st.markdown(f"""
            <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:14px;
                        padding:24px; margin-top:12px;">
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(info_chip("Company", name), unsafe_allow_html=True)
                st.markdown(info_chip("CEO / MD", owner_name), unsafe_allow_html=True)
                st.markdown(info_chip("Exchange", exchange), unsafe_allow_html=True)
                st.markdown(info_chip("Sector", sector), unsafe_allow_html=True)
                st.markdown(info_chip("Industry", industry), unsafe_allow_html=True)
            with col2:
                st.markdown(info_chip(f"Market Cap ({ctx.currency})", market_cap_str), unsafe_allow_html=True)
                st.markdown(info_chip("Beta", beta_str), unsafe_allow_html=True)
                st.markdown(info_chip("Dividend Yield", div_str), unsafe_allow_html=True)
                st.markdown(info_chip("P/E Ratio", pe_str), unsafe_allow_html=True)
                st.markdown(info_chip("Avg Daily Volume", avg_vol_str), unsafe_allow_html=True)
            with col3:
                st.markdown(info_chip("Listed Since (IPO)", ipo_str), unsafe_allow_html=True)
                st.markdown(info_chip("Country", country), unsafe_allow_html=True)
                st.markdown(info_chip("Total Employees", emp_str), unsafe_allow_html=True)
                st.markdown(info_chip("52-Week High", w52h_str), unsafe_allow_html=True)
                st.markdown(info_chip("52-Week Low", w52l_str), unsafe_allow_html=True)

            if website != "N/A":
                st.markdown(f"""
                <div style="margin-top:8px; padding:12px 16px; background:{chip_bg}; border-radius:8px;
                            border:1px solid {card_border}; font-size:14px;">
                    🌐 <a href="{website}" target="_blank"
                          style="color:#667eea; text-decoration:none; font-weight:500;">
                        {website} ↗
                    </a>
                </div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No metadata available for this ticker.")

    st.markdown("---")

    # ── PREPROCESS ────────────────────────────────────────────────────
    prices    = ctx.df["Close"].astype(float).copy()
    train_len = int(len(prices) * 0.8)
    data_train = prices.iloc[:train_len]
    data_test  = prices.iloc[train_len:]

    if len(data_train) < PAST_WINDOW + 1:
        st.error("Not enough history. Use an older start date.")
        st.stop()

    past_tail    = data_train.tail(PAST_WINDOW)
    data_combined = pd.concat([past_tail, data_test], ignore_index=True)
    scaler = MinMaxScaler((0, 1))
    scaler.fit(data_combined.values.reshape(-1, 1))

    x_test, y_test_scaled = prepare_sequences(data_combined, PAST_WINDOW, scaler)
    if x_test.size == 0:
        st.error("No test sequences available.")
        st.stop()

    # ── PREDICT ───────────────────────────────────────────────────────
    with st.spinner("Running model predictions…"):
        if LOTTIE_AVAILABLE and LOTTIE_PREDICTING:
            try:
                st_lottie(LOTTIE_PREDICTING, height=80, key="predicting_lottie")
            except Exception:
                pass
        try:
            preds_scaled = ctx.model.predict(x_test, verbose=0)
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.stop()

    preds_scaled_arr = np.array(preds_scaled)
    if preds_scaled_arr.ndim == 1:
        preds_scaled_arr = preds_scaled_arr.reshape(-1, 1)
    preds       = safe_inverse_transform(scaler, preds_scaled_arr).astype(float)
    y_test_real = safe_inverse_transform(scaler, y_test_scaled).astype(float)

    latest_actual = float(y_test_real[-1])
    latest_pred   = float(preds[-1])
    pct_next      = (latest_pred - latest_actual) / latest_actual * 100 if latest_actual != 0 else 0.0

    # ── METRIC CARDS ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    <div class='metric-card' style='border-top:3px solid #58a6ff;'>
        <div style='font-size:11px; color:#8b949e; font-weight:600; text-transform:uppercase; letter-spacing:0.07em;'>Latest Actual Price</div>
        <div style='font-size:30px; font-weight:800; color:#58a6ff; margin:10px 0 6px;'>{currency} {latest_actual:,.2f}</div>
        <div style='font-size:12px; color:#8b949e;'>Current market value</div>
    </div>""", unsafe_allow_html=True)

    col2.markdown(f"""
    <div class='metric-card' style='border-top:3px solid #8b5cf6;'>
        <div style='font-size:11px; color:#8b949e; font-weight:600; text-transform:uppercase; letter-spacing:0.07em;'>AI Predicted Price</div>
        <div style='font-size:30px; font-weight:800; color:#8b5cf6; margin:10px 0 6px;'>{currency} {latest_pred:,.2f}</div>
        <div style='font-size:12px; color:#8b949e;'>Next-step forecast</div>
    </div>""", unsafe_allow_html=True)

    pct_color = "#3fb950" if pct_next >= 0 else "#f85149"
    pct_icon  = "▲" if pct_next >= 0 else "▼"
    col3.markdown(f"""
    <div class='metric-card' style='border-top:3px solid {pct_color};'>
        <div style='font-size:11px; color:#8b949e; font-weight:600; text-transform:uppercase; letter-spacing:0.07em;'>Predicted Change</div>
        <div style='font-size:30px; font-weight:800; color:{pct_color}; margin:10px 0 6px;'>{pct_icon} {pct_next:+.2f}%</div>
        <div style='font-size:12px; color:#8b949e;'>Short-term outlook</div>
    </div>""", unsafe_allow_html=True)

    # ── ACTUAL VS PREDICTED CHART ─────────────────────────────────────
    st.markdown("## 📈 Actual vs Predicted")
    try:
        plotly_actual_vs_pred(y_test_real, preds, ctx.currency)
    except Exception as e:
        st.error(f"Chart error: {e}")

    # ── 90-DAY FORECAST ───────────────────────────────────────────────
    st.markdown("## 🔮 90-Day Forecast")
    with st.spinner("Generating 90-day forecast…"):
        last_window_scaled = scaler.transform(
            prices.iloc[-PAST_WINDOW:].values.reshape(-1, 1)
        ).reshape(1, PAST_WINDOW, 1)
        future_preds = predict_recursive(ctx.model, last_window_scaled, FUTURE_DAYS, scaler).astype(float)

    future_dates = pd.date_range(
        start=ctx.df.index[-1] + pd.Timedelta(days=1),
        periods=FUTURE_DAYS, freq=DATE_FREQ
    )
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_preds})
    forecast_df["Predicted_Change_%"] = 100.0 * forecast_df["Predicted_Price"].pct_change().fillna(0)

    if st.session_state.get("show_table", True):
        try:
            st.dataframe(
                forecast_df.style.format({"Predicted_Price": "{:,.2f}", "Predicted_Change_%": "{:+.2f}%"}),
                use_container_width=True
            )
        except Exception:
            st.dataframe(forecast_df, use_container_width=True)

    buf = io.StringIO()
    forecast_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download 90-day forecast CSV",
        buf.getvalue(),
        file_name=f"{ctx.ticker_input}_90day_forecast.csv",
        mime="text/csv"
    )

    st.markdown("### 📊 Forecast Chart (with confidence band)")
    vol = float(pd.Series(future_preds).pct_change().std() if len(future_preds) > 1 else 0.0)
    try:
        plotly_forecast(prices.values, future_preds, ctx.currency, vol)
    except Exception as e:
        st.error(f"Forecast chart error: {e}")

    # ── INVESTMENT SUGGESTION ─────────────────────────────────────────
    expected_90day_change = 100.0 * (float(future_preds[-1]) / latest_actual - 1.0) if latest_actual != 0 else 0.0
    avg_forecast_vol      = float(pd.Series(future_preds).pct_change().std() * np.sqrt(252) if len(future_preds) > 1 else 0.0)

    st.markdown("## 💡 Investment Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:{'#1c2128' if st.session_state.get('dark_mode') else '#f8fafc'};
                    border:1px solid {'#30363d' if st.session_state.get('dark_mode') else '#e2e8f0'};
                    border-radius:12px; padding:20px;">
            <div style="font-size:13px; font-weight:700; color:#8b949e; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.07em;">Key Metrics</div>
            <div style="font-size:14px; line-height:2; color:{'#e6edf3' if st.session_state.get('dark_mode') else '#0f172a'};">
                Latest actual: <strong>{ctx.currency} {latest_actual:,.2f}</strong><br>
                Next-step pred: <strong>{ctx.currency} {latest_pred:,.2f}</strong><br>
                Change (next-step): <strong>{pct_next:+.2f}%</strong><br>
                Change (90 days): <strong>{expected_90day_change:+.2f}%</strong><br>
                Forecast vol (ann.): <strong>{avg_forecast_vol:.2%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if expected_90day_change >= 10:
            rec, rec_color, rec_icon, rec_bg = "STRONG BUY",  "#3fb950", "🚀", "rgba(63,185,80,0.1)"
        elif expected_90day_change >= 3:
            rec, rec_color, rec_icon, rec_bg = "BUY",         "#3fb950", "✅", "rgba(63,185,80,0.08)"
        elif expected_90day_change <= -10:
            rec, rec_color, rec_icon, rec_bg = "STRONG SELL", "#f85149", "🔴", "rgba(248,81,73,0.1)"
        elif expected_90day_change <= -3:
            rec, rec_color, rec_icon, rec_bg = "SELL",        "#f85149", "⚠️", "rgba(248,81,73,0.08)"
        else:
            rec, rec_color, rec_icon, rec_bg = "HOLD",        "#d29922", "⏸️", "rgba(210,153,34,0.1)"

        st.markdown(f"""
        <div style="background:{rec_bg}; border:1px solid {rec_color}40;
                    border-left:4px solid {rec_color}; border-radius:12px; padding:24px; height:100%;">
            <div style="font-size:11px; font-weight:700; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">AI Recommendation</div>
            <div style="font-size:40px; margin-bottom:6px;">{rec_icon}</div>
            <div style="font-size:28px; font-weight:800; color:{rec_color};">{rec}</div>
            <div style="font-size:12px; color:#8b949e; margin-top:6px;">
                Based on 90-day forecast: <strong style="color:{rec_color};">{expected_90day_change:+.2f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if ctx.info:
        notes = []
        beta      = ctx.info.get("beta")
        div_yield = ctx.info.get("dividendYield")
        sector    = ctx.info.get("sector")
        if sector:   notes.append(f"Sector: **{sector}**")
        if beta:     notes.append(f"Beta: **{beta:.2f}**")
        if div_yield: notes.append(f"Dividend yield: **{div_yield:.2%}**")
        if notes:
            st.markdown("**Context:** " + "  •  ".join(notes))

    # ── PORTFOLIO BACKTEST ────────────────────────────────────────────
    st.markdown("## 💼 Portfolio Backtest")
    invest = st.number_input(
        f"Investment amount ({currency})",
        min_value=100, max_value=10_000_000, value=10000, step=100
    )
    if st.button("▶ Run Backtest"):
        curve = run_portfolio_backtest(invest, future_preds)
        if len(curve) > 0:
            fig = px.line(
                y=curve,
                labels={"index": "Day", "value": f"Portfolio Value ({currency})"},
                title="Projected Portfolio Value Over Forecast Period"
            )
            fig.update_traces(line_color="#3fb950", line_width=2.5)
            fig.update_layout(
                template=PLOTLY_THEME, height=380,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.12)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.12)'),
            )
            st.plotly_chart(fig, use_container_width=True, key="backtest_chart")
            roi         = (curve[-1] - curve[0]) / curve[0] * 100
            final_value = curve[-1]
            profit      = final_value - invest
            c1, c2, c3 = st.columns(3)
            c1.metric("Initial Investment", f"{currency} {invest:,.2f}")
            c2.metric("Final Value",        f"{currency} {final_value:,.2f}")
            c3.metric("ROI",                f"{roi:+.2f}%", delta=f"{profit:+,.2f}")
        else:
            st.info("Backtest not available.")

    # ── LIVE PRICE ────────────────────────────────────────────────────
    if st.session_state.get("live", False):
        st.markdown("## 📡 Live Price")
        with st.spinner("Fetching live price…"):
            rp = real_time_price(ctx.ticker_used or ctx.ticker_input)
            if rp is not None:
                change = ((rp - latest_pred) / latest_pred * 100) if latest_pred != 0 else 0.0
                st.metric("Current Live Price", f"{ctx.currency} {rp:,.2f}",
                          delta=f"{change:+.2f}% vs prediction")
            else:
                st.info("Live price unavailable for this ticker.")

    st.markdown("---")
    st.caption("⚠️ This is a technical prediction tool. Not financial advice. Always verify before trading.")


if __name__ == "__main__":
    main()