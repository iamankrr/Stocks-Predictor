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

# Dynamic theme based on mode
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
# UI helpers & CSS
# -------------------------------
def inject_premium_css():
    # GLOBAL quick fixes: hide sidebar collapse icon + optional sidebar width
    st.markdown(
        """
        <style>
        /* Hide the sidebar collapse/open arrow icon */
        button[kind="header"] {
            display: none !important;
        }

        /* Optional: tighten sidebar width */
        section[data-testid="stSidebar"] {
            min-width: 270px !important;
            max-width: 270px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Get dark mode state
    dark_mode = st.session_state.get("dark_mode", False)
    
    if dark_mode:
        # Dark mode CSS - COMPLETE PAGE FIXED
        st.markdown(
            """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * { 
            font-family: 'Inter', sans-serif !important;
        }
        
        /* CRITICAL: Main app container */
        .stApp {
            background: #0f172a !important;
        }
        
        /* Main content area */
        .main .block-container {
            background: transparent !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Default text color for dark mode */
        p, span, div, label, li {
            color: #cbd5e1 !important;
        }
        
        /* Headers with proper spacing */
        h1, h2, h3, h4, h5, h6 {
            color: #f1f5f9 !important;
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
            line-height: 1.4 !important;
        }
        
        /* Preserve metric card colors */
        .metric-card h3 {
            color: #a78bfa !important;
            margin: 0 !important;
        }
        
        .metric-card div {
            margin: 4px 0 !important;
            line-height: 1.5 !important;
        }
        
        .header-bar { 
            padding: 15px 20px !important; 
            border-radius: 12px;
            background: #1e293b !important; 
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3); 
            margin-bottom: 25px !important;
            border: 1px solid #475569;
        }
        
        .metric-card { 
            padding: 20px !important; 
            border-radius: 12px;
            text-align: center; 
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important; 
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            border: 1px solid #475569;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 10px !important;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 28px rgba(102, 126, 234, 0.3);
        }
        
        /* Fix overlapping text in markdown */
        .stMarkdown {
            margin-bottom: 1rem !important;
        }
        
        .stMarkdown p {
            margin-bottom: 0.5rem !important;
            line-height: 1.6 !important;
        }
        
        .stMarkdown ul, .stMarkdown ol {
            margin-bottom: 1rem !important;
            padding-left: 1.5rem !important;
        }
        
        .stMarkdown li {
            margin-bottom: 0.3rem !important;
            line-height: 1.5 !important;
        }
        
        /* Sidebar - Complete dark theme */
        section[data-testid="stSidebar"] {
            background: #1e293b !important;
            padding-top: 2rem !important;
        }
        
        section[data-testid="stSidebar"] > div {
            background: #1e293b !important;
        }
        
        /* Sidebar text with spacing */
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #cbd5e1 !important;
            line-height: 1.5 !important;
        }
        
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f1f5f9 !important;
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Input fields - dark with spacing */
        .stTextInput, .stSelectbox, .stNumberInput {
            margin-bottom: 1rem !important;
        }
        
        .stTextInput > div > div > input {
            background-color: #334155 !important;
            color: #f1f5f9 !important;
            border-color: #475569 !important;
            padding: 0.5rem !important;
        }
        
        .stSelectbox > div > div > div {
            background-color: #334155 !important;
            color: #f1f5f9 !important;
            padding: 0.5rem !important;
        }
        
        .stNumberInput > div > div > input {
            background-color: #334155 !important;
            color: #f1f5f9 !important;
            border-color: #475569 !important;
            padding: 0.5rem !important;
        }
        
        /* Buttons - keep gradient visible with spacing */
        .stButton {
            margin-bottom: 0.5rem !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px;
            padding: 10px 20px !important;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 2px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Fix markdown in expanders - PREVENT OVERLAP */
        .streamlit-expanderContent .stMarkdown {
            margin-bottom: 0 !important;
            padding: 0 !important;
        }
        
        .streamlit-expanderContent .stMarkdown p {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.4 !important;
        }
        
        .streamlit-expanderContent .stMarkdown strong {
            display: block;
            margin-bottom: 6px !important;
        }
        
        /* Expander with spacing and better styling */
        .streamlit-expanderHeader {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border-radius: 8px;
            padding: 1rem !important;
            margin-bottom: 0.5rem !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #1e293b !important;
            padding: 1.5rem !important;
            border-radius: 8px;
            margin-top: 0.5rem !important;
        }
        
        /* DataFrame dark with spacing */
        .stDataFrame {
            margin-bottom: 1.5rem !important;
        }
        
        .stDataFrame table {
            background: #1e293b !important;
            color: #cbd5e1 !important;
        }
        
        .stDataFrame thead tr th {
            background-color: #334155 !important;
            color: #f1f5f9 !important;
            padding: 0.75rem !important;
        }
        
        .stDataFrame tbody tr td {
            background-color: #1e293b !important;
            color: #cbd5e1 !important;
            padding: 0.5rem !important;
        }
        
        /* Info/Success/Warning boxes with spacing */
        .stAlert {
            background-color: #1e293b !important;
            color: #cbd5e1 !important;
            border-color: #475569 !important;
            margin-bottom: 1rem !important;
            padding: 1rem !important;
        }
        
        /* Checkbox labels with spacing */
        .stCheckbox {
            margin-bottom: 0.5rem !important;
        }
        
        .stCheckbox label span {
            color: #cbd5e1 !important;
            line-height: 1.5 !important;
        }
        
        /* Download button */
        .stDownloadButton {
            margin-bottom: 1rem !important;
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 0.75rem 1.5rem !important;
        }
        
        /* Caption with spacing */
        .stCaption {
            color: #94a3b8 !important;
            margin-top: 0.5rem !important;
            line-height: 1.5 !important;
        }
        
        /* Success/Info messages */
        .stSuccess, .stInfo {
            background-color: rgba(30, 41, 59, 0.5) !important;
            color: #cbd5e1 !important;
            margin-bottom: 1rem !important;
            padding: 1rem !important;
        }
        
        /* Column spacing */
        [data-testid="column"] {
            padding: 0 0.5rem !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* Plotly charts with spacing */
        .js-plotly-plot, .plotly {
            background: transparent !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* Fix text overlap in custom HTML */
        [data-testid="stMarkdownContainer"] > div {
            overflow: visible !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:
        # Light mode CSS - COMPLETE PAGE with proper spacing
        st.markdown(
            """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * { 
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Main app background */
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
        }
        
        /* Main content with spacing */
        .main .block-container {
            background: transparent !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Text colors with spacing - IMPROVED */
        p, span, div, label, li {
            color: #1e293b !important;
            line-height: 1.6 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
            line-height: 1.4 !important;
            font-weight: 700 !important;
        }
        
        /* Better contrast for markdown */
        .stMarkdown strong {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        
        /* Fix markdown spacing */
        .stMarkdown {
            margin-bottom: 1rem !important;
        }
        
        .stMarkdown p {
            margin-bottom: 0.5rem !important;
        }
        
        .stMarkdown ul, .stMarkdown ol {
            margin-bottom: 1rem !important;
            padding-left: 1.5rem !important;
        }
        
        .stMarkdown li {
            margin-bottom: 0.3rem !important;
        }
        
        .header-bar { 
            padding: 15px 20px !important; 
            border-radius: 12px; 
            background: #ffffff !important; 
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); 
            margin-bottom: 25px !important;
            border: 1px solid #e2e8f0;
        }
        
        .metric-card { 
            padding: 20px !important; 
            border-radius: 12px; 
            text-align: center; 
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important; 
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 10px !important;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 28px rgba(102, 126, 234, 0.15);
        }
        
        .metric-card div {
            margin: 4px 0 !important;
            line-height: 1.5 !important;
        }
        
        /* Sidebar with spacing */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important;
            padding-top: 2rem !important;
        }
        
        /* Input fields with spacing */
        .stTextInput, .stSelectbox, .stNumberInput {
            margin-bottom: 1rem !important;
        }
        
        .stTextInput > div > div > input {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border-color: #cbd5e1 !important;
            padding: 0.5rem !important;
        }
        
        .stNumberInput > div > div > input {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border-color: #cbd5e1 !important;
            padding: 0.5rem !important;
        }
        
        /* Buttons with spacing */
        .stButton {
            margin-bottom: 0.5rem !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px;
            padding: 10px 20px !important;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 2px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Fix markdown in expanders - PREVENT OVERLAP */
        .streamlit-expanderContent .stMarkdown {
            margin-bottom: 0 !important;
            padding: 0 !important;
        }
        
        .streamlit-expanderContent .stMarkdown p {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.4 !important;
        }
        
        .streamlit-expanderContent .stMarkdown strong {
            display: block;
            margin-bottom: 6px !important;
        }
        
        /* Expander with spacing and better styling */
        .streamlit-expanderHeader {
            background-color: #f8fafc !important;
            color: #1e293b !important;
            border-radius: 8px;
            padding: 1rem !important;
            margin-bottom: 0.5rem !important;
            font-weight: 600 !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #ffffff !important;
            padding: 1.5rem !important;
            border-radius: 8px;
            margin-top: 0.5rem !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        /* DataFrame with spacing */
        .stDataFrame {
            margin-bottom: 1.5rem !important;
        }
        
        /* Metric values - IMPROVED VISIBILITY */
        .metric-card h3 {
            color: #667eea !important;
            font-weight: 800 !important;
            font-size: 32px !important;
            margin: 8px 0 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card div {
            color: #1e293b !important;
            font-weight: 600 !important;
        }
        
        /* Download button */
        .stDownloadButton {
            margin-bottom: 1rem !important;
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 0.75rem 1.5rem !important;
        }
        
        /* Column spacing */
        [data-testid="column"] {
            padding: 0 0.5rem !important;
        }
        
        /* Plotly charts with spacing */
        .js-plotly-plot, .plotly {
            margin-bottom: 1.5rem !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

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


LOTTIE_LOADING = load_lottie("https://assets1.lottiefiles.com/packages/lf20_p8bfn5to.json") if LOTTIE_AVAILABLE else None
LOTTIE_PREDICTING = load_lottie("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json") if LOTTIE_AVAILABLE else None

# -------------------------------
# FIXED: Plotly charts
# -------------------------------
def plotly_actual_vs_pred(actual, predicted, currency):
    """Fixed: Actual vs Predicted chart"""
    # Convert to lists safely
    actual_list = np.array(actual, dtype=float).flatten().tolist()
    predicted_list = np.array(predicted, dtype=float).flatten().tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=actual_list,
        mode="lines",
        name="Actual",
        line=dict(width=2, color="#1f77b4"),
        hovertemplate="Index: %{x}<br>Actual: %{y:,.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        y=predicted_list,
        mode="lines",
        name="Predicted",
        line=dict(width=2, color="#ff7f0e", dash="dot"),
        hovertemplate="Index: %{x}<br>Predicted: %{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        template=PLOTLY_THEME,
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Sample Index",
        yaxis_title=f"Price ({currency})",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key="actual_vs_pred_chart")


def plotly_forecast(history, future, currency, vol):
    """FIXED: Forecast chart with confidence bands"""
    # Convert to numpy arrays safely
    hist_arr = np.array(history, dtype=float).flatten()
    future_arr = np.array(future, dtype=float).flatten() if future is not None and len(future) > 0 else np.array([])
    
    # Take last 200 points for better visualization
    if len(hist_arr) > 200:
        hist_display = hist_arr[-200:]
        start_idx = len(hist_arr) - 200
    else:
        hist_display = hist_arr
        start_idx = 0
    
    # Create x-axis indices
    x_hist = list(range(start_idx, start_idx + len(hist_display)))
    x_future = list(range(start_idx + len(hist_display), start_idx + len(hist_display) + len(future_arr)))
    
    # Calculate volatility band safely
    try:
        vol_val = float(vol) if (vol is not None and not np.isnan(vol)) else 0.0
        vol_val = min(max(vol_val, 0.0), 0.5)  # Clamp between 0 and 0.5
    except Exception:
        vol_val = 0.0
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    if len(hist_display) > 0:
        fig.add_trace(go.Scatter(
            x=x_hist,
            y=hist_display.tolist(),
            mode="lines",
            name="Historical Price",
            line=dict(width=2, color="#1f77b4"),
            hovertemplate="Day %{x}<br>Price: %{y:,.2f}<extra></extra>"
        ))
    
    # Add forecast data
    if len(future_arr) > 0:
        fig.add_trace(go.Scatter(
            x=x_future,
            y=future_arr.tolist(),
            mode="lines+markers",
            name="90-Day Forecast",
            line=dict(width=3, color="#ff7f0e"),
            marker=dict(size=4, color="#ff7f0e"),
            hovertemplate="Day %{x}<br>Forecast: %{y:,.2f}<extra></extra>"
        ))
        
        # Add confidence band if volatility exists
        if vol_val > 0.01:  # Only show if meaningful volatility
            upper = future_arr * (1.0 + vol_val)
            lower = future_arr * (1.0 - vol_val)
            
            # Upper band (invisible line)
            fig.add_trace(go.Scatter(
                x=x_future,
                y=upper.tolist(),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower band with fill
            fig.add_trace(go.Scatter(
                x=x_future,
                y=lower.tolist(),
                mode="lines",
                name="Confidence Band",
                fill='tonexty',
                fillcolor="rgba(255,136,0,0.2)",
                line=dict(width=0),
                hovertemplate="Range: %{y:,.2f}<extra></extra>"
            ))
    
    # Update layout
    fig.update_layout(
        template=PLOTLY_THEME,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Timeline (Days from start)",
        yaxis_title=f"Price ({currency})",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    
    # Render with unique key
    st.plotly_chart(fig, use_container_width=True, key="forecast_chart_main")

# -------------------------------
# Real-time & backtest helpers
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
    portfolio_curve = units * np.array(future_prices, dtype=float)
    return portfolio_curve


# -------------------------------
# Main app
# -------------------------------
def main():
    inject_premium_css()

    # FIXED Header with no text overlap
    current_time = pd.Timestamp.now().strftime("%d %b %Y, %I:%M %p")
    
    st.markdown(
        f"""
        <div class="header-bar" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); margin-bottom: 30px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap: wrap; gap: 25px;">
                <div style="flex: 1; min-width: 320px;">
                    <div style="font-size: 32px; font-weight: 800; margin-bottom: 12px; letter-spacing: -0.5px; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        🚀 AI Stock Predictor Pro
                    </div>
                    <div style="font-size: 14px; opacity: 0.95; font-weight: 500; line-height: 1.6;">
                        ⚡ Real-time Analysis • 📊 90-Day ML Forecast<br>💼 Portfolio Simulator • 🎯 Smart Signals
                    </div>
                </div>
                <div style="text-align: right; min-width: 180px; flex-shrink: 0;">
                    <div style="font-size: 12px; opacity: 0.85; font-weight: 600; margin-bottom: 6px; white-space: nowrap;">
                        🕐 Last Updated
                    </div>
                    <div style="font-size: 13px; font-weight: 700; white-space: nowrap;">
                        {current_time}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar with modern design
    with st.sidebar:
        # COMPACT Control Panel (reduced padding)
        st.markdown("""
            <div style='text-align: center; padding: 10px 0; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; margin-bottom: 12px; color: white;'>
                <h2 style='margin: 0; font-size: 18px; font-weight: 800;'>⚙️ Control Panel</h2>
                <p style='margin: 3px 0 0 0; font-size: 12px; opacity: 0.9;'>
                    Configure your analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Dark Mode Toggle at the top (simpler controls)
        
        model_path = Path(__file__).parent / MODEL_FILENAME
        model, err = load_model_safe(model_path)
        if model is None:
            st.error("❌ Model load failed: " + (err or "unknown"))
            st.stop()
        else:
            st.success("✅ AI Model Loaded Successfully")

        st.markdown("---")
        
        st.markdown("### 🎯 Stock Selection")
        
        # Popular stocks list for suggestions
        popular_stocks_dict = {
            "RELIANCE.NS": "RELIANCE - Reliance Industries",
            "TCS.NS": "TCS - Tata Consultancy Services",
            "INFY.NS": "INFOSYS - Infosys Limited",
            "HDFCBANK.NS": "HDFC - HDFC Bank",
            "AAPL": "APPLE - Apple Inc",
            "MSFT": "MICROSOFT - Microsoft Corporation",
            "TSLA": "TESLA - Tesla Inc",
            "GOOGL": "GOOGLE - Alphabet Inc",
            "AMZN": "AMAZON - Amazon.com Inc",
            "META": "META - Meta Platforms Inc",
        }
        
        # Create list of suggestions for selectbox
        suggestions = list(popular_stocks_dict.keys())
        display_options = [f"{key} - {popular_stocks_dict[key]}" for key in suggestions]
        
        # Selectbox with search functionality
        selected_option = st.selectbox(
            "🔍 Search Stock (Type to filter or select)",
            options=[""] + display_options,
            help="Start typing to search, or select from popular stocks"
        )
        
        # Extract ticker from selection or use default
        if selected_option and selected_option != "":
            ticker_input = selected_option.split(" - ")[0].strip()
        else:
            # Fallback text input for custom ticker
            ticker_input = st.text_input(
                "Or enter custom ticker",
                DEFAULT_TICKER,
                help="e.g., AAPL, TSLA, RELIANCE.NS"
            ).strip().upper()

        start_date = st.text_input("📅 Start date (YYYY-MM-DD)", "2012-01-01", help="Historical data start date")
        
        st.markdown("---")
        st.markdown("### 🎛️ Features")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["live"] = st.checkbox("📡 Live Price", value=st.session_state.get("live", False))
        with col2:
            st.session_state["show_table"] = st.checkbox("📋 Show Table", value=st.session_state.get("show_table", True))

        st.markdown("---")
        
        # Dynamic theme for sidebar box
        box_bg = "#1e293b" if st.session_state.get("dark_mode", False) else "#f8fafc"
        box_border = "#475569" if st.session_state.get("dark_mode", False) else "#667eea"
        box_text = "#e2e8f0" if st.session_state.get("dark_mode", False) else "#475569"
        
        st.markdown(f"""
            <div style='background: {box_bg}; 
                        padding: 12px; border-radius: 10px; border-left: 4px solid {box_border};'>
                <h4 style='margin: 0 0 8px 0; color: #667eea;'>🌟 Premium Features</h4>
                <ul style='margin: 0; padding-left: 20px; font-size: 13px; color: {box_text};'>
                    <li>AI-Powered Predictions</li>
                    <li>90-Day Forecasting</li>
                    <li>Portfolio Backtesting</li>
                    <li>Interactive Charts</li>
                    <li>Real-time Analysis</li>
                    <li>Dark/Light Mode</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Validate ticker
    if not ticker_input:
        st.info("Enter a ticker in the sidebar to begin.")
        st.stop()

    # Fetch data
    with st.spinner("Fetching historical data..."):
        if LOTTIE_AVAILABLE and LOTTIE_LOADING:
            try:
                st_lottie(LOTTIE_LOADING, height=90, key="loading_lottie")
            except Exception:
                pass
        try:
            df, ticker_used = try_yfinance_variants(ticker_input, start=start_date)
        except Exception as e:
            st.error("Data fetch failed: " + str(e))
            st.stop()

    info = fetch_ticker_info(ticker_used)
    currency = enforce_currency(ticker_input, ticker_used, info)

    ctx = RunContext(ticker_input=ticker_input, ticker_used=ticker_used, currency=currency, model=model, info=info, df=df)

    st.markdown(f"### 📄 {ctx.ticker_input}  —  Data retrieved as **{ctx.ticker_used}**")
    st.dataframe(ctx.df.tail(5), use_container_width=True)

    # Company Metadata Section - Dropdown Style
    st.markdown("---")
    
    # Dark mode detection
    is_dark = st.session_state.get("dark_mode", False)
    
    if is_dark:
        dropdown_bg = "#1e293b"
        dropdown_text = "#e2e8f0"
        dropdown_border = "#475569"
        info_bg = "rgba(51, 65, 85, 0.3)"
        label_color = "#cbd5e1"
    else:
        dropdown_bg = "#f8fafc"
        dropdown_text = "#1e293b"
        dropdown_border = "#cbd5e1"
        info_bg = "rgba(102, 126, 234, 0.1)"
        label_color = "#475569"
    
    # Dropdown button
    if "show_metadata" not in st.session_state:
        st.session_state["show_metadata"] = False
    
    if st.button("🏢 Company Metadata " + ("▼" if not st.session_state["show_metadata"] else "▲"), 
                 use_container_width=True, 
                 key="metadata_toggle"):
        st.session_state["show_metadata"] = not st.session_state["show_metadata"]
    
    # Show metadata if toggled
    if st.session_state["show_metadata"]:
        st.markdown(f"""
            <div style='background: {dropdown_bg}; padding: 20px; border-radius: 8px; 
                        border: 1px solid {dropdown_border}; margin-top: 10px;'>
        """, unsafe_allow_html=True)
        
        if ctx.info:
            # Get values safely
            name = ctx.info.get("longName") or ctx.info.get("shortName") or "N/A"
            exchange = ctx.info.get("exchange") or "N/A"
            sector = ctx.info.get("sector") or "N/A"
            industry = ctx.info.get("industry") or "N/A"
            country = ctx.info.get("country") or "N/A"
            website = ctx.info.get("website") or "N/A"
            employees = ctx.info.get("fullTimeEmployees")
            ceo = ctx.info.get("companyOfficers")
            
            # CEO/Owner name from officers list
            owner_name = "N/A"
            if ceo and isinstance(ceo, list) and len(ceo) > 0:
                for officer in ceo:
                    if isinstance(officer, dict):
                        title = officer.get("title", "").lower()
                        if "ceo" in title or "chief executive" in title or "chairman" in title or "managing director" in title:
                            owner_name = officer.get("name", "N/A")
                            break
            
            # Market cap formatting based on currency
            market_cap = ctx.info.get("marketCap")
            if market_cap:
                # Check if Indian exchange
                if ctx.currency == "INR":
                    # Show in INR - Lakh Crore format
                    crore = market_cap / 1e7  # Convert to crores
                    if crore >= 1e5:  # If >= 1 lakh crore
                        lakh_crore = crore / 1e5
                        market_cap_str = f"₹{lakh_crore:.2f} Lakh Crore"
                    elif crore >= 1e3:  # If >= 1000 crore (1 Arab)
                        market_cap_str = f"₹{crore/1e3:.2f} Thousand Crore"
                    elif crore >= 1:
                        market_cap_str = f"₹{crore:.2f} Crore"
                    else:
                        market_cap_str = f"₹{market_cap:,.0f}"
                else:
                    # Show in USD - Trillion/Billion format
                    if market_cap >= 1e12:
                        market_cap_str = f"${market_cap/1e12:.2f} Trillion"
                    elif market_cap >= 1e9:
                        market_cap_str = f"${market_cap/1e9:.2f} Billion"
                    elif market_cap >= 1e6:
                        market_cap_str = f"${market_cap/1e6:.2f} Million"
                    else:
                        market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            beta = ctx.info.get("beta")
            beta_str = f"{beta:.2f}" if beta else "N/A"
            
            div_yield = ctx.info.get("dividendYield")
            div_str = f"{div_yield:.2%}" if div_yield else "N/A"
            
            # Additional financial metrics
            pe_ratio = ctx.info.get("trailingPE")
            pe_str = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
            
            week_52_high = ctx.info.get("fiftyTwoWeekHigh")
            week_52_low = ctx.info.get("fiftyTwoWeekLow")
            week_52_high_str = f"{ctx.currency} {week_52_high:,.2f}" if week_52_high else "N/A"
            week_52_low_str = f"{ctx.currency} {week_52_low:,.2f}" if week_52_low else "N/A"
            
            avg_volume = ctx.info.get("averageVolume")
            if avg_volume:
                if avg_volume >= 1e6:
                    avg_volume_str = f"{avg_volume/1e6:.2f}M"
                elif avg_volume >= 1e3:
                    avg_volume_str = f"{avg_volume/1e3:.2f}K"
                else:
                    avg_volume_str = f"{avg_volume:,.0f}"
            else:
                avg_volume_str = "N/A"
            
            employees_str = f"{employees:,}" if employees else "N/A"
            
            # IPO date - Multiple attempts
            ipo_date_str = "N/A"
            
            # Try 1: firstTradeDateEpochUtc
            ipo_date = ctx.info.get("firstTradeDateEpochUtc")
            if ipo_date:
                try:
                    from datetime import datetime
                    ipo_date_str = datetime.fromtimestamp(ipo_date).strftime("%d %b %Y")
                except Exception:
                    pass
            
            # Try 2: If still N/A, try ipoExpectedDate
            if ipo_date_str == "N/A":
                ipo_expected = ctx.info.get("ipoExpectedDate")
                if ipo_expected:
                    ipo_date_str = str(ipo_expected)
            
            # Try 3: Try to get from history data
            if ipo_date_str == "N/A":
                try:
                    # Get the first available date from our dataframe
                    if not ctx.df.empty and len(ctx.df) > 0:
                        first_date = ctx.df.index[0]
                        ipo_date_str = first_date.strftime("%d %b %Y") + " (Est.)"
                except Exception:
                    pass
            
            # Create professional 3-column layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📌 Company Name
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {name}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            👤 CEO / Chairman / Managing Director
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {owner_name}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            🏦 Exchange
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {exchange}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            🏭 Sector
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {sector}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            🏢 Industry
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {industry}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            💰 Market Cap ({ctx.currency})
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {market_cap_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📊 Beta (Volatility)
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {beta_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            💵 Dividend Yield
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {div_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📈 P/E Ratio
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {pe_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📊 Avg Daily Volume
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {avg_volume_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📅 Listed Since (IPO)
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {ipo_date_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            🌍 Country
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {country}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            👥 Total Employees
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {employees_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📈 52-Week High
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {week_52_high_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            📉 52-Week Low
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            {week_52_low_str}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Website link at bottom (full width)
            if website != "N/A":
                st.markdown(f"""
                    <div style='margin-top: 10px;'>
                        <div style='color: {label_color}; font-weight: 600; font-size: 13px; margin-bottom: 8px;'>
                            🌐 Company Website
                        </div>
                        <div style='padding: 12px 16px; background: {info_bg}; border-radius: 8px; 
                                    color: {dropdown_text}; font-size: 15px; font-weight: 500;'>
                            <a href='{website}' target='_blank' style='color: #667eea; text-decoration: none;'>
                                {website} ↗️
                            </a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No metadata available for this ticker.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Preprocess
    prices = ctx.df["Close"].astype(float).copy()
    train_len = int(len(prices) * 0.8)
    data_train = prices.iloc[:train_len]
    data_test = prices.iloc[train_len:]

    if len(data_train) < PAST_WINDOW + 1:
        st.error("Not enough history. Use older start date.")
        st.stop()

    past_tail = data_train.tail(PAST_WINDOW)
    data_combined = pd.concat([past_tail, data_test], ignore_index=True)
    scaler = MinMaxScaler((0, 1))
    scaler.fit(data_combined.values.reshape(-1, 1))

    x_test, y_test_scaled = prepare_sequences(data_combined, PAST_WINDOW, scaler)
    if x_test.size == 0:
        st.error("No test sequences available after preparing sequences.")
        st.stop()

    # Predict
    with st.spinner("Running model predictions..."):
        if LOTTIE_AVAILABLE and LOTTIE_PREDICTING:
            try:
                st_lottie(LOTTIE_PREDICTING, height=90, key="predicting_lottie")
            except Exception:
                pass
        try:
            preds_scaled = ctx.model.predict(x_test, verbose=0)
        except Exception as e:
            st.error("Model prediction error: " + str(e))
            st.stop()

    preds_scaled_arr = np.array(preds_scaled)
    if preds_scaled_arr.ndim == 1:
        preds_scaled_arr = preds_scaled_arr.reshape(-1, 1)

    preds = safe_inverse_transform(scaler, preds_scaled_arr).astype(float)
    y_test_real = safe_inverse_transform(scaler, y_test_scaled).astype(float)

    latest_actual = float(y_test_real[-1])
    latest_pred = float(preds[-1]) if preds.ndim == 1 else float(preds[-1])
    
    # Calculate percentage change
    pct_next = (latest_pred - latest_actual) / latest_actual * 100 if latest_actual != 0 else 0.0

    # Enhanced Metrics with icons and colors
    col1, col2, col3 = st.columns(3)
    
    col1.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #3b82f6;'>
            <div style='font-size: 14px; color: #64748b; font-weight: 600; margin-bottom: 8px;'>
                🔎 LATEST ACTUAL PRICE
            </div>
            <h3 style='margin: 0; color: #3b82f6 !important;'>{currency} {latest_actual:,.2f}</h3>
            <div style='font-size: 12px; color: #94a3b8; margin-top: 6px;'>Current market value</div>
        </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #8b5cf6;'>
            <div style='font-size: 14px; color: #64748b; font-weight: 600; margin-bottom: 8px;'>
                🤖 AI PREDICTED PRICE
            </div>
            <h3 style='margin: 0; color: #8b5cf6 !important;'>{currency} {latest_pred:,.2f}</h3>
            <div style='font-size: 12px; color: #94a3b8; margin-top: 6px;'>Next-step forecast</div>
        </div>
    """, unsafe_allow_html=True)
    
    pct_color = "#10b981" if pct_next >= 0 else "#ef4444"
    pct_icon = "📈" if pct_next >= 0 else "📉"
    col3.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid {pct_color};'>
            <div style='font-size: 14px; color: #64748b; font-weight: 600; margin-bottom: 8px;'>
                {pct_icon} PREDICTED CHANGE
            </div>
            <h3 style='margin: 0; color: {pct_color} !important;'>{pct_next:+.2f}%</h3>
            <div style='font-size: 12px; color: #94a3b8; margin-top: 6px;'>Short-term outlook</div>
        </div>
    """, unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown("## 📈 Actual vs Predicted (Interactive)")
    try:
        plotly_actual_vs_pred(y_test_real, preds, ctx.currency)
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        st.write("Debug - Actual shape:", y_test_real.shape, "Predicted shape:", preds.shape)

    # Forecasting
    st.markdown("## 🔮 90-Day Forecast")
    
    with st.spinner("Generating 90-day forecast..."):
        last_window_scaled = scaler.transform(prices.iloc[-PAST_WINDOW:].values.reshape(-1, 1)).reshape(1, PAST_WINDOW, 1)
        future_preds = predict_recursive(ctx.model, last_window_scaled, FUTURE_DAYS, scaler).astype(float)
    
    future_dates = pd.date_range(start=ctx.df.index[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq=DATE_FREQ)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_preds})
    forecast_df["Predicted_Change_%"] = 100.0 * forecast_df["Predicted_Price"].pct_change().fillna(0)

    # Show table if enabled
    if st.session_state.get("show_table", True):
        try:
            st.dataframe(
                forecast_df.style.format({
                    "Predicted_Price": "{:,.2f}",
                    "Predicted_Change_%": "{:+.2f}%"
                }),
                use_container_width=True
            )
        except Exception:
            st.dataframe(forecast_df, use_container_width=True)

    # Download button
    buf = io.StringIO()
    forecast_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download 90-day forecast CSV",
        buf.getvalue(),
        file_name=f"{ctx.ticker_input}_90day_forecast.csv",
        mime="text/csv"
    )

    # FIXED: Forecast chart
    st.markdown("### 📊 Interactive Forecast Chart (with confidence band)")

    
   
    
    # Calculate volatility
    vol = float(pd.Series(future_preds).pct_change().std() if len(future_preds) > 1 else 0.0)
    
    try:
        plotly_forecast(prices.values, future_preds, ctx.currency, vol)
    except Exception as e:
        st.error(f"Forecast chart error: {str(e)}")
        st.write("Attempting to display data...")
        st.write("Future predictions sample:", future_preds[:10])

    # Investment suggestions
    expected_90day_change = 100.0 * (float(future_preds[-1]) / latest_actual - 1.0) if latest_actual != 0 else 0.0
    avg_forecast_vol = float(pd.Series(future_preds).pct_change().std() * np.sqrt(252) if len(future_preds) > 1 else 0.0)

    st.markdown("## 💡 Investment Suggestions & Notes")
    st.markdown(f"- **Latest actual**: {ctx.currency} {latest_actual:,.2f}")
    st.markdown(f"- **Model next-step**: {ctx.currency} {latest_pred:,.2f}")
    st.markdown(f"- **Predicted change (next-step)**: {((latest_pred - latest_actual) / latest_actual * 100):+.2f}%")
    st.markdown(f"- **Predicted change (90 days)**: {expected_90day_change:+.2f}%")
    st.markdown(f"- **Forecast realized-vol (annualized)**: {avg_forecast_vol:.2%}")

    # Recommendation with modern card design
    if expected_90day_change >= 10:
        rec = "STRONG BUY"
        rec_color = "#10b981"
        rec_icon = "🚀"
        rec_bg = "#d1fae5"
    elif expected_90day_change >= 3:
        rec = "BUY"
        rec_color = "#22c55e"
        rec_icon = "✅"
        rec_bg = "#dcfce7"
    elif expected_90day_change <= -10:
        rec = "STRONG SELL"
        rec_color = "#ef4444"
        rec_icon = "🔴"
        rec_bg = "#fee2e2"
    elif expected_90day_change <= -3:
        rec = "SELL"
        rec_color = "#f97316"
        rec_icon = "⚠️"
        rec_bg = "#fed7aa"
    else:
        rec = "HOLD"
        rec_color = "#eab308"
        rec_icon = "⏸️"
        rec_bg = "#fef3c7"
    
    st.markdown(f"""
        <div style='background: {rec_bg}; padding: 20px; border-radius: 12px; border-left: 5px solid {rec_color}; margin: 20px 0;'>
            <div style='display: flex; align-items: center; gap: 15px;'>
                <div style='font-size: 48px;'>{rec_icon}</div>
                <div style='flex: 1;'>
                    <div style='font-size: 14px; color: #64748b; font-weight: 600; margin-bottom: 4px;'>
                        AI RECOMMENDATION
                    </div>
                    <div style='font-size: 32px; font-weight: 800; color: {rec_color}; margin-bottom: 4px;'>
                        {rec}
                    </div>
                    <div style='font-size: 13px; color: #475569;'>
                        Based on 90-day forecast: <strong>{expected_90day_change:+.2f}%</strong> expected change
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Extra metadata context
    if ctx.info:
        notes = []
        beta = ctx.info.get("beta")
        div_yield = ctx.info.get("dividendYield")
        sector = ctx.info.get("sector")
        if sector:
            notes.append(f"Sector: **{sector}** — review sector trends.")
        if beta:
            notes.append(f"Beta: **{beta:.2f}**")
        if div_yield:
            notes.append(f"Dividend yield: **{div_yield:.2%}**")
        if notes:
            st.markdown("**Extra context:**")
            for n in notes:
                st.markdown("- " + n)

    # Portfolio backtest
    st.markdown("## 💼 Portfolio Backtest (Optional)")
    invest = st.number_input(
        f"Amount to invest ({currency})",
        min_value=100,
        max_value=10_000_000,
        value=10000,
        step=100
    )
    
    if st.button("Run Backtest"):
        curve = run_portfolio_backtest(invest, future_preds)
        if len(curve) > 0:
            fig = px.line(
                y=curve,
                labels={"index": "Day", "value": f"Portfolio Value ({currency})"},
                title="Projected Portfolio Value Over Forecast Period"
            )
            fig.update_traces(line_color="#2ecc71", line_width=3)
            fig.update_layout(template=PLOTLY_THEME, height=400)
            st.plotly_chart(fig, use_container_width=True, key="backtest_chart")
            
            roi = (curve[-1] - curve[0]) / curve[0] * 100
            final_value = curve[-1]
            profit = final_value - invest
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Investment", f"{currency} {invest:,.2f}")
            col2.metric("Final Value", f"{currency} {final_value:,.2f}")
            col3.metric("ROI", f"{roi:+.2f}%", delta=f"{profit:+,.2f}")
        else:
            st.info("Backtest not available (no forecast).")

    # Live price
    if st.session_state.get("live", False):
        st.markdown("## 📡 Live Price")
        with st.spinner("Fetching live price..."):
            rp = real_time_price(ctx.ticker_used or ctx.ticker_input)
            if rp is not None:
                change_from_pred = ((rp - latest_pred) / latest_pred * 100) if latest_pred != 0 else 0.0
                st.metric(
                    "Current Live Price",
                    f"{ctx.currency} {rp:,.2f}",
                    delta=f"{change_from_pred:+.2f}% vs prediction"
                )
            else:
                st.info("Live price unavailable for this ticker.")

    st.markdown("---")
    st.caption("⚠️ This is a technical prediction tool. Not financial advice. Always verify before trading.")


if __name__ == "__main__":
    main()

    
    