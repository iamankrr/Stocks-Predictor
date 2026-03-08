# ============================================================
#  AI Stock Predictor Pro  —  Full Professional Edition
#  Author : Aman
# ============================================================
import io
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
#  PAGE CONFIG  — must be very first Streamlit call
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
#  CONSTANTS
# ────────────────────────────────────────────────────────────
MODEL_FILENAME = "Stock Prediction Model.keras"
DEFAULT_TICKER = "RELIANCE.NS"
PAST_WINDOW    = 100
DATE_FREQ      = "B"

POPULAR_STOCKS = {
    "RELIANCE.NS":   "Reliance Industries",
    "TCS.NS":        "Tata Consultancy Services",
    "INFY.NS":       "Infosys Ltd",
    "HDFCBANK.NS":   "HDFC Bank",
    "WIPRO.NS":      "Wipro Ltd",
    "ICICIBANK.NS":  "ICICI Bank",
    "SBIN.NS":       "State Bank of India",
    "TATAMOTORS.NS": "Tata Motors",
    "AAPL":          "Apple Inc",
    "MSFT":          "Microsoft",
    "TSLA":          "Tesla Inc",
    "GOOGL":         "Alphabet Inc",
    "AMZN":          "Amazon",
    "META":          "Meta Platforms",
    "NVDA":          "NVIDIA",
    "NFLX":          "Netflix",
}

# ────────────────────────────────────────────────────────────
#  SESSION STATE
# ────────────────────────────────────────────────────────────
for k, v in {
    "dark_mode": False, "show_table": True, "live": False, "show_meta": False,
    "compare_mode": False, "compare_ticker": "TCS.NS",
    "show_ohlcv": False, "show_fcast_tbl": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

DARK = st.session_state["dark_mode"]

# ────────────────────────────────────────────────────────────
#  THEME TOKENS
# ────────────────────────────────────────────────────────────
if DARK:
    BG = "#0d1117"; BG2 = "#161b22"; BG3 = "#1c2128"
    BORDER = "#30363d"; TXT = "#e6edf3"; TXT2 = "#8b949e"; TXT3 = "#6e7681"
    ACCENT = "#58a6ff"; GREEN = "#3fb950"; RED = "#f85149"
    YELLOW = "#d29922"; PURPLE = "#bc8cff"; CSHADOW = "rgba(0,0,0,0.4)"
else:
    BG = "#f0f4f8"; BG2 = "#ffffff"; BG3 = "#f8fafc"
    BORDER = "#e2e8f0"; TXT = "#0f172a"; TXT2 = "#475569"; TXT3 = "#94a3b8"
    ACCENT = "#2563eb"; GREEN = "#16a34a"; RED = "#dc2626"
    YELLOW = "#d97706"; PURPLE = "#7c3aed"; CSHADOW = "rgba(15,23,42,0.08)"

PT = "plotly_dark" if DARK else "plotly_white"

# ────────────────────────────────────────────────────────────
#  COLOR UTILITY  — Plotly needs rgba(), not 8-digit hex
# ────────────────────────────────────────────────────────────
def rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #rrggbb to rgba() for Plotly."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# Shared chart layout — NO xaxis/yaxis here to avoid duplicate-key conflicts
CHART_BASE = dict(
    template=PT,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=36, b=8),
    font=dict(family="DM Sans", size=12, color=TXT2),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

# Reusable grid axis dicts
def _xax(**kw):
    d = dict(showgrid=True, gridcolor=rgba(BORDER,0.55), zeroline=False)
    d.update(kw); return d

def _yax(**kw):
    d = dict(showgrid=True, gridcolor=rgba(BORDER,0.55), zeroline=False)
    d.update(kw); return d

# ────────────────────────────────────────────────────────────
#  CSS INJECTION
# ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&family=DM+Mono:wght@400;500&display=swap');

    /* ── KILL ALL STREAMLIT CHROME + keyboard_double_arrow icon ── */
    #MainMenu,footer,header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="collapsedControl"],
    [data-testid="baseButton-headerNoPadding"],
    button[kind="header"],
    button[kind="headerNoPadding"],
    [data-testid="baseButton-header"],
    [aria-label="Close sidebar"],
    [aria-label="Open sidebar"],
    [aria-label="Collapse sidebar"],
    [aria-label="Expand sidebar"],
    .st-emotion-cache-zq5wmm,
    .st-emotion-cache-1egp75f,
    .st-emotion-cache-15zrgzn,
    .st-emotion-cache-eczf96,
    .st-emotion-cache-1dp5vir,
    .st-emotion-cache-6tkfeg,
    .st-emotion-cache-dvne4q,
    .viewerBadge_container__r5tak,
    .viewerBadge_link__qRIco,
    .css-1rs6os,
    .css-17ziqus
    {{ display:none !important; visibility:hidden !important; width:0 !important; height:0 !important; overflow:hidden !important; }}

    /* Kill sidebar collapse arrow icon specifically */
    section[data-testid="stSidebar"] > div > div:first-child > button,
    section[data-testid="stSidebar"] button[data-testid*="collapse"],
    section[data-testid="stSidebar"] button[data-testid*="close"],
    div[data-testid="stSidebarCollapsedControl"],
    .st-emotion-cache-1cypcdb,
    .st-emotion-cache-h5rgaw
    {{ display:none !important; visibility:hidden !important; }}

    /* Nuclear: hide ANY button at very top of sidebar wrapper */
    [data-testid="stSidebarNav"],
    section[data-testid="stSidebar"]>div>button:first-of-type
    {{ display:none !important; }}

    /* ── HIDE DELTAGEN REPR / DEBUG TEXT IN SIDEBAR ── */
    section[data-testid="stSidebar"] .stText,
    section[data-testid="stSidebar"] [data-testid="stText"],
    section[data-testid="stSidebar"] pre
    {{ display:none !important; }}

    /* ── PREVENT EXPANDER LABEL OVERFLOW ── */
    .streamlit-expanderHeader p {{
        white-space:nowrap !important;
        overflow:hidden !important;
        text-overflow:ellipsis !important;
        max-width:100% !important;
    }}

    /* ── BASE ── */
    *, *::before, *::after {{ font-family:'DM Sans',sans-serif !important; box-sizing:border-box; }}
    code, pre, .stCode {{ font-family:'DM Mono',monospace !important; }}
    .stApp {{ background:{BG} !important; }}
    .main .block-container {{ background:transparent !important; padding:1.25rem 2rem 4rem !important; max-width:1440px !important; }}

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {{
        background:{BG2} !important;
        border-right:1px solid {BORDER} !important;
        min-width:290px !important; max-width:290px !important;
    }}
    section[data-testid="stSidebar"]>div:first-child {{ background:{BG2} !important; padding:0.75rem !important; }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {{ color:{TXT2} !important; font-size:13px !important; }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{ color:{TXT} !important; font-weight:600 !important; font-size:12.5px !important; margin:0.8rem 0 0.35rem !important; }}

    /* ── TEXT ── */
    p, li {{ color:{TXT2} !important; line-height:1.65 !important; }}
    h1 {{ font-size:24px !important; font-weight:800 !important; color:{TXT} !important; }}
    h2 {{ font-size:19px !important; font-weight:700 !important; color:{TXT} !important; margin:1.5rem 0 0.7rem !important; }}
    h3 {{ font-size:15px !important; font-weight:600 !important; color:{TXT} !important; margin:1.1rem 0 0.5rem !important; }}
    strong {{ color:{TXT} !important; }}
    hr {{ border-color:{BORDER} !important; margin:1rem 0 !important; }}

    /* ── CARDS ── */
    .ss-card {{
        background:{BG2}; border:1px solid {BORDER}; border-radius:14px;
        padding:18px 20px; box-shadow:0 2px 10px {CSHADOW};
        transition:transform 0.17s, box-shadow 0.17s;
        margin-bottom:12px;
    }}
    .ss-card:hover {{ transform:translateY(-2px); box-shadow:0 7px 24px {CSHADOW}; }}
    .ss-lbl {{ font-size:10.5px; font-weight:600; color:{TXT3}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:5px; }}
    .ss-val {{ font-size:27px; font-weight:700; line-height:1.1; margin-bottom:3px; }}
    .ss-sub {{ font-size:11.5px; color:{TXT3}; }}

    /* ── SECTION HEADER ── */
    .ss-sec {{ display:flex; align-items:center; gap:9px; margin:1.75rem 0 0.85rem; padding-bottom:0.55rem; border-bottom:1px solid {BORDER}; }}
    .ss-sec-t {{ font-size:16px; font-weight:700; color:{TXT}; }}

    /* ── BADGES ── */
    .ss-badge {{ display:inline-block; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600; }}

    /* ── INPUTS ── */
    .stTextInput,  .stSelectbox, .stNumberInput {{ margin-bottom:0.7rem !important; }}
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        background:{BG3} !important; color:{TXT} !important;
        border:1px solid {BORDER} !important; border-radius:9px !important;
        padding:0.5rem 0.7rem !important; font-size:13px !important;
        transition:border-color .15s, box-shadow .15s;
    }}
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
        border-color:{ACCENT} !important; box-shadow:0 0 0 3px {ACCENT}22 !important;
    }}
    .stSelectbox>div>div>div {{
        background:{BG3} !important; color:{TXT} !important;
        border:1px solid {BORDER} !important; border-radius:9px !important; font-size:13px !important;
    }}

    /* ── BUTTONS ── */
    .stButton>button {{
        background:linear-gradient(135deg,{ACCENT},{PURPLE}) !important;
        color:#fff !important; border:none !important; border-radius:9px !important;
        padding:9px 18px !important; font-weight:600 !important; font-size:13px !important;
        box-shadow:0 2px 8px {ACCENT}44 !important; transition:all .2s !important;
    }}
    .stButton>button:hover {{ transform:translateY(-2px) !important; box-shadow:0 6px 18px {ACCENT}55 !important; }}
    /* Toggle buttons (OHLCV, Forecast table) — subtle style */
    button[data-testid="baseButton-secondary"][kind="secondary"],
    div[data-testid="stButton"]:has(button[key="ohlcv_btn"]) button,
    div[data-testid="stButton"]:has(button[key="fcast_tbl_btn"]) button {{
        background:{BG3} !important;
        color:{TXT2} !important;
        border:1px solid {BORDER} !important;
        box-shadow:none !important;
        font-size:13px !important;
        font-weight:500 !important;
    }}
    .stDownloadButton>button {{
        background:{BG3} !important; color:{TXT2} !important;
        border:1px solid {BORDER} !important; border-radius:9px !important;
        font-size:13px !important; font-weight:500 !important;
    }}

    /* ── ALERTS ── */
    .stAlert {{ background:{BG3} !important; border-color:{BORDER} !important; border-radius:10px !important; }}
    div[data-testid="stSuccessMessage"] {{ border-left:3px solid {GREEN} !important; }}
    div[data-testid="stInfoMessage"]    {{ border-left:3px solid {ACCENT} !important; }}
    div[data-testid="stWarningMessage"] {{ border-left:3px solid {YELLOW} !important; }}
    div[data-testid="stErrorMessage"]   {{ border-left:3px solid {RED} !important; }}

    /* ── DATAFRAME ── */
    .stDataFrame {{ border-radius:12px !important; overflow:hidden !important; border:1px solid {BORDER} !important; }}
    .stDataFrame thead th {{ background:{BG3} !important; color:{TXT} !important; font-size:12px !important; font-weight:600 !important; padding:0.6rem 0.75rem !important; }}
    .stDataFrame tbody td {{ background:{BG2} !important; color:{TXT2} !important; font-size:12px !important; padding:0.42rem 0.75rem !important; }}
    .stDataFrame tbody tr:hover td {{ background:{BG3} !important; }}

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {{ background:{BG3} !important; border:1px solid {BORDER} !important; border-radius:10px !important; color:{TXT} !important; font-size:13px !important; font-weight:600 !important; }}
    .streamlit-expanderContent {{ background:{BG2} !important; border:1px solid {BORDER} !important; border-top:none !important; border-radius:0 0 10px 10px !important; }}

    /* ── MISC ── */
    .stCheckbox label span {{ color:{TXT2} !important; font-size:13px !important; }}
    .stRadio > label {{ color:{TXT2} !important; font-size:13px !important; }}
    .stSlider label  {{ color:{TXT2} !important; font-size:13px !important; }}
    .stCaption       {{ color:{TXT3} !important; font-size:11.5px !important; }}
    .stSpinner>div   {{ border-top-color:{ACCENT} !important; }}
    [data-testid="stMetricValue"] {{ color:{TXT} !important; font-size:22px !important; font-weight:700 !important; }}
    [data-testid="stMetricLabel"] {{ color:{TXT3} !important; font-size:12px !important; }}
    ::-webkit-scrollbar       {{ width:5px; height:5px; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}
    </style>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
#  UI HELPERS
# ────────────────────────────────────────────────────────────
def sec(icon, title):
    st.markdown(f'<div class="ss-sec"><span style="font-size:17px">{icon}</span><span class="ss-sec-t">{title}</span></div>', unsafe_allow_html=True)

def card(label, value, sub="", color=None, col=None):
    c = color or ACCENT
    html = f'<div class="ss-card"><div class="ss-lbl">{label}</div><div class="ss-val" style="color:{c}">{value}</div>{"<div class=ss-sub>"+sub+"</div>" if sub else ""}</div>'
    (col or st).markdown(html, unsafe_allow_html=True)

def badge(txt, color):
    return f'<span class="ss-badge" style="color:{color};background:{color}18;border:1px solid {color}33">{txt}</span>'


# ────────────────────────────────────────────────────────────
#  DATA & ML
# ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(path):
    try:    return load_model(path), None
    except Exception as e: return None, str(e)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, start):
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    variants = [ticker] if "." in ticker else [ticker+".NS", ticker+".BO", ticker]
    for v in variants:
        try:
            df = yf.download(v, start=start, end=end, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty: continue
            # Flatten multi-level columns that yfinance sometimes returns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            # Ensure all price columns are clean 1D float Series
            for col in ["Open","High","Low","Close","Volume","Adj Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].values.flatten(), errors="coerce")
            adj = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
            df["adj_close"]   = adj.astype(float)
            df["Close"]       = df["Close"].astype(float)
            df["return"]      = df["adj_close"].pct_change()
            df["ticker_used"] = v
            return df, v
        except Exception: continue
    raise ValueError(f"No data for '{ticker}'")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_info(ticker):
    try:    return yf.Ticker(ticker).info or {}
    except: return {}


def currency_of(ticker_input, ticker_used, info):
    used = (ticker_used or ticker_input).upper()
    exch = (info.get("exchange") or "").upper()
    if used.endswith((".NS",".BO",".BSE")) or "NSE" in exch or "BSE" in exch:
        return "₹", "INR"
    return "$", "USD"


def make_sequences(prices, window, scaler):
    arr = scaler.transform(prices.values.reshape(-1,1))
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i,0].reshape(window,1))
        y.append(arr[i,0])
    return np.array(X,float), np.array(y,float)


def inv_sc(scaler, arr):
    a = np.array(arr).reshape(-1,1)
    try:    return scaler.inverse_transform(a).reshape(-1)
    except: mn,mx=float(scaler.data_min_[0]),float(scaler.data_max_[0]); return a.flatten()*(mx-mn)+mn


def forecast_recursive(model, window_sc, steps, scaler):
    out, inp = [], window_sc.copy().astype(float)
    for _ in range(steps):
        val = float(np.array(model.predict(inp, verbose=0)).flatten()[0])
        out.append(val)
        inp = np.concatenate([inp[:,1:,:], np.array(val).reshape(1,1,1)], axis=1)
    return inv_sc(scaler, np.array(out))


# ────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ────────────────────────────────────────────────────────────
def flatten_df(df):
    """Flatten yfinance multi-level columns to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def indicators(df):
    df = flatten_df(df)
    # Extract as clean 1D Series to avoid MultiIndex issues
    close  = pd.Series(df["Close"].values.flatten(),  index=df.index, dtype=float)
    volume = pd.Series(df["Volume"].values.flatten(), index=df.index, dtype=float)
    high   = pd.Series(df["High"].values.flatten(),   index=df.index, dtype=float)
    low    = pd.Series(df["Low"].values.flatten(),    index=df.index, dtype=float)

    d = pd.DataFrame(index=df.index)
    d["Close"]  = close
    d["Volume"] = volume
    d["High"]   = high
    d["Low"]    = low

    c = close
    d["SMA20"]  = c.rolling(20).mean()
    d["SMA50"]  = c.rolling(50).mean()
    d["SMA200"] = c.rolling(200).mean()
    d["EMA12"]  = c.ewm(span=12,adjust=False).mean()
    d["EMA26"]  = c.ewm(span=26,adjust=False).mean()
    d["MACD"]      = d["EMA12"] - d["EMA26"]
    d["MACD_Sig"]  = d["MACD"].ewm(span=9,adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Sig"]

    roll20 = c.rolling(20)
    std20  = roll20.std()
    d["BB_Mid"] = roll20.mean()
    d["BB_Up"]  = d["BB_Mid"] + 2*std20
    d["BB_Lo"]  = d["BB_Mid"] - 2*std20
    d["BB_Pos"] = (c - d["BB_Lo"]) / (d["BB_Up"] - d["BB_Lo"] + 1e-9)

    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + g / (l + 1e-9))

    tr = pd.concat([
        (high - low),
        (high - c.shift()).abs(),
        (low  - c.shift()).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["OBV"] = (volume * np.sign(c.diff().fillna(0))).cumsum()
    d["Ret1"] = c.pct_change(1)
    d["Ret5"] = c.pct_change(5)
    d["Vol20"] = d["Ret1"].rolling(20).std() * np.sqrt(252) * 100
    return d


def signals_from(ind):
    r = ind.iloc[-1]; out = {}
    rsi=r.get("RSI",50)
    out["RSI"]     = ("Oversold ↑",GREEN,"BUY") if rsi<30 else (("Overbought ↓",RED,"SELL") if rsi>70 else (f"Neutral {rsi:.0f}",YELLOW,"HOLD"))
    out["MACD"]    = ("Bullish",GREEN,"BUY") if r.get("MACD_Hist",0)>0 else ("Bearish",RED,"SELL")
    bp=r.get("BB_Pos",0.5)
    out["Bollinger"]= ("Near Lower Band",GREEN,"BUY") if bp<0.2 else (("Near Upper Band",RED,"SELL") if bp>0.8 else ("Mid-Range",YELLOW,"HOLD"))
    c,s20,s50=r.get("Close",0),r.get("SMA20",0),r.get("SMA50",0)
    out["MA Trend"]= ("Above SMA20+50",GREEN,"BUY") if c>s20>s50 else (("Below SMA20+50",RED,"SELL") if c<s20<s50 else ("Mixed",YELLOW,"HOLD"))
    out["MA Cross"] = ("Golden Cross ☀️",GREEN,"BUY") if r.get("SMA50",0)>r.get("SMA200",1) else ("Death Cross 💀",RED,"SELL")
    atr=r.get("ATR",0); price=r.get("Close",1)
    out["Volatility"]= ("High — trade caution",RED,"SELL") if atr/price>0.03 else ("Low — stable",GREEN,"BUY")
    return out


def risk_metrics(prices):
    # Ensure 1D Series
    if hasattr(prices, 'values'):
        prices = pd.Series(prices.values.flatten(), index=prices.index, dtype=float)
    ret=prices.pct_change().dropna(); ann=np.sqrt(252)
    vol=float(ret.std()*ann*100)
    sharpe=float((ret.mean()/(ret.std()+1e-9))*ann)
    down=ret[ret<0]; sortino=float((ret.mean()/(down.std()+1e-9))*ann)
    rm=prices.cummax(); dd=(prices-rm)/(rm+1e-9)*100; mdd=float(dd.min())
    var95=float(np.percentile(ret,5)*100)
    cvar95=float(ret[ret<=np.percentile(ret,5)].mean()*100)
    calmar=float(abs(ret.mean()*252*100/(abs(mdd)+1e-9)))
    pos=int((ret>0).sum()); neg=int((ret<0).sum())
    wr=float(pos/(pos+neg+1e-9)*100)
    return {"Ann. Volatility %":round(vol,2),"Sharpe Ratio":round(sharpe,2),
            "Sortino Ratio":round(sortino,2),"Max Drawdown %":round(mdd,2),
            "VaR 95%":round(var95,2),"CVaR 95%":round(cvar95,2),
            "Calmar Ratio":round(calmar,2),"Win Rate %":round(wr,2)}


# ────────────────────────────────────────────────────────────
#  CHARTS
# ────────────────────────────────────────────────────────────
def chart_candle(df, ticker):
    df = flatten_df(df)
    ind = indicators(df); r = df.tail(250); ir = ind.tail(250); ix = r.index
    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                        row_heights=[0.58,0.22,0.20],vertical_spacing=0.015,
                        subplot_titles=("","Volume","RSI 14"))
    fig.add_trace(go.Candlestick(x=ix,
        open=pd.to_numeric(r["Open"].values.flatten(),errors="coerce"),
        high=pd.to_numeric(r["High"].values.flatten(),errors="coerce"),
        low=pd.to_numeric(r["Low"].values.flatten(),errors="coerce"),
        close=pd.to_numeric(r["Close"].values.flatten(),errors="coerce"),
        name="OHLC",increasing_line_color=GREEN,decreasing_line_color=RED,
        increasing_fillcolor=rgba(GREEN,0.65),decreasing_fillcolor=rgba(RED,0.65)),row=1,col=1)
    for col_n,col_c,dsh in [("SMA20",ACCENT,"solid"),("SMA50",YELLOW,"dot"),("SMA200",PURPLE,"dash")]:
        if col_n in ir.columns:
            fig.add_trace(go.Scatter(x=ix,y=ir[col_n],name=col_n,
                line=dict(color=col_c,width=1.3,dash=dsh)),row=1,col=1)
    if "BB_Up" in ir.columns:
        fig.add_trace(go.Scatter(x=ix,y=ir["BB_Up"],mode="lines",
            line=dict(color=TXT3,width=0.7,dash="dot"),showlegend=False,hoverinfo="skip"),row=1,col=1)
        fig.add_trace(go.Scatter(x=ix,y=ir["BB_Lo"],mode="lines",name="BB Bands",
            fill="tonexty",fillcolor=rgba(TXT3,0.08),line=dict(color=TXT3,width=0.7,dash="dot"),hoverinfo="skip"),row=1,col=1)
    close_v = pd.to_numeric(r["Close"].values.flatten(), errors="coerce")
    open_v  = pd.to_numeric(r["Open"].values.flatten(),  errors="coerce")
    vol_colors=[GREEN if close_v[i]>=open_v[i] else RED for i in range(len(r))]
    fig.add_trace(go.Bar(x=ix,y=pd.to_numeric(r["Volume"].values.flatten(),errors="coerce"),name="Volume",
        marker_color=vol_colors,showlegend=False),row=2,col=1)
    if "RSI" in ir.columns:
        fig.add_trace(go.Scatter(x=ix,y=ir["RSI"],name="RSI",
            line=dict(color=PURPLE,width=1.5)),row=3,col=1)
        fig.add_hline(y=70,line=dict(color=RED,dash="dot",width=0.8),row=3,col=1)
        fig.add_hline(y=30,line=dict(color=GREEN,dash="dot",width=0.8),row=3,col=1)
    fig.update_layout(**CHART_BASE,height=580,
                      title=dict(text=f"<b>{ticker}</b> · Technical Chart",font=dict(size=13,color=TXT)),
                      xaxis =_xax(), yaxis =_yax(),
                      xaxis2=_xax(), yaxis2=_yax(),
                      xaxis3=_xax(), yaxis3=_yax())
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True,key="candle")


def chart_macd(df):
    df=flatten_df(df)
    ind=indicators(df).tail(200); ix=ind.index
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ix,y=ind["MACD"],name="MACD",line=dict(color=ACCENT,width=1.5)))
    fig.add_trace(go.Scatter(x=ix,y=ind["MACD_Sig"],name="Signal",line=dict(color=YELLOW,width=1.5,dash="dot")))
    fig.add_trace(go.Bar(x=ix,y=ind["MACD_Hist"],name="Histogram",
        marker_color=[GREEN if v>=0 else RED for v in ind["MACD_Hist"]],opacity=0.7))
    fig.update_layout(**CHART_BASE,height=280,
                      xaxis=_xax(), yaxis=_yax(),
                      title=dict(text="<b>MACD</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="macd")


def chart_returns_dist(df):
    df=flatten_df(df)
    ret=pd.Series(df["Close"].values.flatten(),index=df.index,dtype=float).pct_change().dropna()*100
    fig=go.Figure(go.Histogram(x=ret,nbinsx=60,
        marker_color=rgba(ACCENT,0.73),marker_line_color=ACCENT,marker_line_width=0.4,name="Returns"))
    fig.add_vline(x=float(ret.mean()),line=dict(color=GREEN,dash="dot",width=1.5),
                  annotation_text=f" μ={ret.mean():.2f}%",annotation_font=dict(color=GREEN,size=11))
    fig.update_layout(**CHART_BASE,height=280,
                      xaxis=_xax(title="Daily Return (%)"), yaxis=_yax(),
                      title=dict(text="<b>Returns Distribution</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="ret_dist")


def chart_heatmap(df):
    df=flatten_df(df)
    ret=pd.Series(df["Close"].values.flatten(),index=df.index,dtype=float).pct_change().dropna()
    rdf=ret.to_frame("r"); rdf.index=pd.to_datetime(rdf.index)
    rdf["y"]=rdf.index.year; rdf["m"]=rdf.index.month
    mo=rdf.groupby(["y","m"])["r"].sum().unstack()*100
    mnames=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mo.columns=[mnames[int(m)-1] for m in mo.columns]
    fig=go.Figure(go.Heatmap(z=mo.values,x=mo.columns.tolist(),y=mo.index.tolist(),
        colorscale=[[0,RED],[0.5,BG3],[1,GREEN]],zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in mo.values],
        texttemplate="%{text}",textfont=dict(size=10),
        hovertemplate="Year %{y} %{x}: %{z:.2f}%<extra></extra>"))
    fig.update_layout(**CHART_BASE,height=max(240,len(mo)*30+90),
                      title=dict(text="<b>Monthly Returns Heatmap (%)</b>",font=dict(size=13,color=TXT)),
                      xaxis=_xax(side="top"), yaxis=_yax())
    st.plotly_chart(fig,use_container_width=True,key="heatmap")


def chart_pred_actual(actual, pred, sym):
    a=np.array(actual,float).flatten(); p=np.array(pred,float).flatten()
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=a,mode="lines",name="Actual",line=dict(color=ACCENT,width=1.8)))
    fig.add_trace(go.Scatter(y=p,mode="lines",name="Predicted",line=dict(color=YELLOW,width=1.8,dash="dot")))
    fig.add_trace(go.Scatter(y=a-p,mode="lines",name="Error",
        line=dict(color=rgba(RED,0.6),width=1),fill="tozeroy",fillcolor=rgba(RED,0.09)))
    fig.update_layout(**CHART_BASE,height=360,
                      xaxis=_xax(), yaxis=_yax(title=f"Price ({sym})"),
                      title=dict(text="<b>Actual vs Predicted</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="pred_actual")


def chart_forecast(history, future, sym, vol):
    h=np.array(history,float).flatten(); f=np.array(future,float).flatten()
    hd=h[-200:] if len(h)>200 else h; x0=len(h)-len(hd)
    xh=list(range(x0,x0+len(hd))); xf=list(range(x0+len(hd),x0+len(hd)+len(f)))
    vv=min(max(float(vol) if not np.isnan(float(vol)) else 0,0),0.5)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=xh,y=hd.tolist(),mode="lines",name="Historical",
        line=dict(color=ACCENT,width=1.8)))
    if len(f):
        fig.add_trace(go.Scatter(x=xf,y=f.tolist(),mode="lines+markers",name="Forecast",
            line=dict(color=GREEN,width=2.2),marker=dict(size=2.5,color=GREEN)))
        if vv>0.01:
            up=f*(1+vv); lo=f*(1-vv)
            fig.add_trace(go.Scatter(x=xf,y=up.tolist(),mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xf,y=lo.tolist(),mode="lines",name="Confidence",
                fill="tonexty",fillcolor=rgba(GREEN,0.12),line=dict(width=0)))
    fig.add_vline(x=x0+len(hd)-0.5,line=dict(color=TXT3,dash="dash",width=1.2),
                  annotation_text="  Today",annotation_font=dict(size=11,color=TXT3))
    fig.update_layout(**CHART_BASE,height=430,
                      xaxis=_xax(title="Days"), yaxis=_yax(title=f"Price ({sym})"),
                      title=dict(text="<b>AI Price Forecast</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="forecast_main")


def chart_portfolio(curve, invest, sym):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=curve,mode="lines",name="Portfolio",
        line=dict(color=GREEN,width=2.2),fill="tozeroy",fillcolor=rgba(GREEN,0.09)))
    fig.add_hline(y=invest,line=dict(color=TXT3,dash="dot",width=1.2),
                  annotation_text="  Initial",annotation_font=dict(size=11,color=TXT3))
    fig.update_layout(**CHART_BASE,height=310,
                      xaxis=_xax(title="Forecast Day"), yaxis=_yax(title=f"Value ({sym})"),
                      title=dict(text="<b>Portfolio Simulation</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="portfolio_chart")


def chart_drawdown(prices_s):
    prices_s=pd.Series(prices_s.values.flatten(),index=prices_s.index,dtype=float)
    rm=prices_s.cummax(); dd=(prices_s-rm)/(rm+1e-9)*100
    fig=go.Figure(go.Scatter(x=dd.index,y=dd.values,mode="lines",name="Drawdown",
        line=dict(color=RED,width=1.5),fill="tozeroy",fillcolor=rgba(RED,0.12)))
    fig.update_layout(**CHART_BASE,height=280,
                      xaxis=_xax(), yaxis=_yax(title="Drawdown (%)"),
                      title=dict(text="<b>Drawdown (Underwater) Chart</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="dd_chart")


def chart_rolling_vol(prices_s):
    prices_s=pd.Series(prices_s.values.flatten(),index=prices_s.index,dtype=float)
    rv=prices_s.pct_change().rolling(30).std()*np.sqrt(252)*100
    fig=go.Figure(go.Scatter(x=rv.index,y=rv.values,mode="lines",name="30d Vol",
        line=dict(color=YELLOW,width=1.5),fill="tozeroy",fillcolor=rgba(YELLOW,0.12)))
    fig.update_layout(**CHART_BASE,height=260,
                      xaxis=_xax(), yaxis=_yax(title="Ann. Volatility (%)"),
                      title=dict(text="<b>Rolling 30-Day Volatility</b>",font=dict(size=13,color=TXT)))
    st.plotly_chart(fig,use_container_width=True,key="vol_chart")


def gauge(value, label, key):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=value,
        title=dict(text=label,font=dict(size=12,color=TXT2)),
        number=dict(font=dict(size=20,color=TXT),suffix="%"),
        gauge=dict(axis=dict(range=[0,100]),
                   bar=dict(color=ACCENT,thickness=0.26),
                   bgcolor=BG3,bordercolor=BORDER,
                   steps=[dict(range=[0,33],color=rgba(GREEN,0.13)),
                          dict(range=[33,66],color=rgba(YELLOW,0.13)),
                          dict(range=[66,100],color=rgba(RED,0.13))],
                   threshold=dict(line=dict(color=RED,width=2),thickness=0.75,value=80))))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      height=190,margin=dict(l=16,r=16,t=36,b=8),
                      font=dict(family="DM Sans",color=TXT2))
    st.plotly_chart(fig,use_container_width=True,key=key)


# ────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{ACCENT},{PURPLE});
                    border-radius:13px;padding:14px 16px;margin-bottom:14px;text-align:center;">
            <div style="font-size:19px;font-weight:800;color:white;letter-spacing:-0.4px;">📈 StockSense AI</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.75);margin-top:3px;">Professional Market Intelligence</div>
        </div>""", unsafe_allow_html=True)

        model_path = Path(__file__).parent / MODEL_FILENAME
        model, err = load_model_cached(str(model_path))
        if model:
            st.success("✅ AI Model Active")
        else:
            st.error(f"❌ {err or 'Model not found'}")

        st.markdown("#### 🔍 Stock")
        disp = [f"{k}  —  {v}" for k, v in POPULAR_STOCKS.items()]
        sel  = st.selectbox("Quick-select", [""] + disp, label_visibility="collapsed")
        if sel:
            ticker = sel.split("  —  ")[0].strip()
            st.text_input("Custom ticker", DEFAULT_TICKER,
                          placeholder="AAPL, TCS.NS…",
                          label_visibility="collapsed",
                          disabled=True, key="custom_ticker_disabled")
        else:
            ticker = st.text_input("Custom ticker", DEFAULT_TICKER,
                                   placeholder="AAPL, TCS.NS…",
                                   label_visibility="collapsed").strip().upper()

        start  = st.text_input("📅 From", "2015-01-01")
        fdays  = st.slider("Forecast horizon (days)", 30, 180, 90, 10)

        st.markdown("---")
        st.markdown("#### ⚙️ Settings")
        c1,c2 = st.columns(2)
        with c1: st.session_state["dark_mode"] = st.checkbox("🌙 Dark",  value=DARK)
        with c2: st.session_state["live"]       = st.checkbox("📡 Live",  value=st.session_state["live"])
        st.session_state["show_table"] = st.checkbox("📋 Show forecast table", value=st.session_state["show_table"])

        st.markdown("---")
        st.markdown(f"""
        <div style="background:{ACCENT}0e;border-left:3px solid {ACCENT};
                    border-radius:0 8px 8px 0;padding:11px 13px;font-size:12px;color:{TXT2};line-height:1.95;">
            <strong style="color:{ACCENT}">What's inside</strong><br>
            🕯 Candlestick + Bollinger Bands<br>
            📉 MACD + RSI Signals<br>
            🔮 ML Price Forecast<br>
            📅 Monthly Returns Heatmap<br>
            ⚖️ Sharpe / Sortino / Drawdown<br>
            🎯 6 Technical Signal Indicators<br>
            💼 Portfolio What-If Simulator<br>
            🏢 Full Company Intelligence<br>
            📡 Live Price Feed
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⚖️ Compare Stocks")
        st.session_state["compare_mode"] = st.checkbox(
            "Enable comparison", value=st.session_state["compare_mode"])
        compare_ticker = ""
        if st.session_state["compare_mode"]:
            compare_disp = [f"{k}  —  {v}" for k, v in POPULAR_STOCKS.items()]
            csel = st.selectbox("Compare with", [""] + compare_disp,
                                label_visibility="collapsed", key="cmp_sel")
            if csel:
                compare_ticker = csel.split("  —  ")[0].strip()
                st.session_state["compare_ticker"] = compare_ticker
            else:
                compare_ticker = st.text_input(
                    "Or type ticker", st.session_state["compare_ticker"],
                    placeholder="e.g. TSLA, HDFCBANK.NS",
                    label_visibility="collapsed", key="cmp_input").strip().upper()
                st.session_state["compare_ticker"] = compare_ticker

        st.markdown(f"<div style='color:{TXT3};font-size:11px;text-align:center;margin-top:10px;'>v2.0 Professional Edition</div>", unsafe_allow_html=True)
        return ticker, start, fdays, model, compare_ticker


# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
def main():
    inject_css()
    ticker_input, start_date, forecast_days, model, compare_ticker = sidebar()

    if not ticker_input:
        st.info("👈 Select or enter a ticker symbol in the sidebar to begin.")
        st.stop()

    # Fetch
    with st.spinner(f"Loading {ticker_input}…"):
        try:    df, ticker_used = fetch_data(ticker_input, start_date)
        except Exception as e: st.error(str(e)); st.stop()

    info             = fetch_info(ticker_used)
    cur_sym, cur     = currency_of(ticker_input, ticker_used, info)
    company_name     = info.get("longName") or info.get("shortName") or ticker_used
    _close_flat      = pd.Series(df["Close"].values.flatten(), index=df.index, dtype=float)
    curr_price       = float(_close_flat.iloc[-1])
    prev_price       = float(_close_flat.iloc[-2]) if len(df)>1 else curr_price
    day_pct          = (curr_price-prev_price)/(prev_price+1e-9)*100
    day_color        = GREEN if day_pct>=0 else RED

    # ── HEADER ──────────────────────────────────────────────
    hc1, hc2, hc3 = st.columns([3,2,1])
    exch   = info.get("exchange",""); sector = info.get("sector","")
    with hc1:
        st.markdown(f"""
        <div style="padding-bottom:4px;">
            <span style="font-size:22px;font-weight:800;color:{TXT}">{company_name}</span>
            <span style="font-size:13px;color:{TXT3};margin-left:8px;">{ticker_used}</span>
            &nbsp;{badge(exch,ACCENT) if exch else ""}
            &nbsp;{badge(sector,PURPLE) if sector else ""}
        </div>
        <div style="font-size:12px;color:{TXT3};">{cur} · {len(df):,} trading days · since {df.index[0].strftime('%d %b %Y')}</div>
        """, unsafe_allow_html=True)
    with hc2:
        arrow = "▲" if day_pct>=0 else "▼"
        st.markdown(f"""
        <div style="text-align:right;padding-top:2px;">
            <div style="font-size:30px;font-weight:800;color:{TXT};line-height:1">{cur_sym}{curr_price:,.2f}</div>
            <div style="font-size:13px;font-weight:600;color:{day_color};margin-top:3px;">{arrow} {abs(day_pct):.2f}% today</div>
        </div>""", unsafe_allow_html=True)
    with hc3:
        if st.session_state["live"]:
            try:
                tk=yf.Ticker(ticker_used); fi=getattr(tk,"fast_info",None)
                lp=fi.last_price if fi and hasattr(fi,"last_price") else None
                if lp:
                    st.markdown(f"""<div style="text-align:right;padding-top:6px;">
                        <div style="font-size:10px;color:{TXT3};text-transform:uppercase;letter-spacing:.07em;">Live</div>
                        <div style="font-size:19px;font-weight:700;color:{GREEN}">{cur_sym}{lp:,.2f}</div>
                    </div>""", unsafe_allow_html=True)
            except: pass

    st.markdown(f'<hr style="border-color:{BORDER};margin:0.6rem 0">', unsafe_allow_html=True)

    # ── TABS ────────────────────────────────────────────────
    T = st.tabs(["📊 Overview","🕯 Charts","🔮 AI Forecast","⚖️ Risk","💼 Portfolio","🏢 Company","📊 Compare"])

    # ══ TAB 0 — OVERVIEW ══════════════════════════════════════
    with T[0]:
        sec("📈","Market Summary")
        mc=info.get("marketCap")
        if mc:
            if cur=="INR": mc_s=f"₹{mc/1e7/1e5:.2f}L Cr" if mc>=1e12 else f"₹{mc/1e7:.0f} Cr"
            else:          mc_s=f"${mc/1e12:.2f}T" if mc>=1e12 else f"${mc/1e9:.2f}B"
        else: mc_s="N/A"
        w52h=info.get("fiftyTwoWeekHigh"); w52l=info.get("fiftyTwoWeekLow"); avgvol=info.get("averageVolume")
        c1,c2,c3,c4,c5=st.columns(5)
        card("Current Price",f"{cur_sym}{curr_price:,.2f}",f"{'▲' if day_pct>=0 else '▼'} {day_pct:+.2f}%",day_color,c1)
        card("Market Cap",mc_s,"",PURPLE,c2)
        card("52W High",f"{cur_sym}{w52h:,.2f}" if w52h else "N/A","",GREEN,c3)
        card("52W Low", f"{cur_sym}{w52l:,.2f}" if w52l else "N/A","",RED,c4)
        card("Avg Volume",f"{avgvol/1e6:.1f}M" if avgvol and avgvol>=1e6 else (f"{avgvol/1e3:.0f}K" if avgvol else "N/A"),"",YELLOW,c5)

        ca,cb,cc,cd=st.columns(4)
        bv=info.get("beta"); pv=info.get("trailingPE"); dy=info.get("dividendYield"); ep=info.get("trailingEps")
        card("Beta",         f"{bv:.2f}" if bv else "N/A","Market sensitivity",TXT2,ca)
        card("P/E Ratio",    f"{pv:.1f}" if pv else "N/A","Trailing",           TXT2,cb)
        card("Dividend Yield",f"{dy:.2%}" if dy else "N/A","Annual",            TXT2,cc)
        card("EPS",          f"{cur_sym}{ep:.2f}" if ep else "N/A","Trailing",  TXT2,cd)

        sec("📅","Price Performance")
        periods={"1W":5,"1M":21,"3M":63,"6M":126,"1Y":252,"3Y":756}
        pcs=pd.Series(flatten_df(df)["Close"].values.flatten(),index=df.index,dtype=float); pcols=st.columns(6)
        for i,(lbl,d) in enumerate(periods.items()):
            if len(pcs)>d:
                p0,p1=float(pcs.iloc[-d-1]),float(pcs.iloc[-1]); pc=(p1-p0)/(p0+1e-9)*100
                clr=GREEN if pc>=0 else RED
                pcols[i].markdown(f'<div class="ss-card" style="text-align:center;padding:13px 8px;">'
                    f'<div class="ss-lbl">{lbl}</div>'
                    f'<div style="font-size:17px;font-weight:700;color:{clr}">{"▲" if pc>=0 else "▼"} {abs(pc):.1f}%</div>'
                    f'</div>',unsafe_allow_html=True)
            else:
                pcols[i].markdown(f'<div class="ss-card" style="text-align:center;padding:13px 8px;">'
                    f'<div class="ss-lbl">{lbl}</div><div style="color:{TXT3};font-size:13px">N/A</div>'
                    f'</div>',unsafe_allow_html=True)

        sec("📉","Price History")
        _cl=pd.Series(df["Close"].values.flatten(),index=df.index,dtype=float)
        fig_ph=go.Figure(go.Scatter(x=df.index,y=_cl,mode="lines",
            line=dict(color=ACCENT,width=1.8),fill="tozeroy",fillcolor=rgba(ACCENT,0.08),name="Close"))
        fig_ph.update_layout(**CHART_BASE,height=300,
                             xaxis=_xax(), yaxis=_yax(title=f"Price ({cur_sym})"))
        st.plotly_chart(fig_ph,use_container_width=True,key="price_history")

        # ── OHLCV table toggle (no expander to avoid "key_parent" ghost text)
        if "show_ohlcv" not in st.session_state:
            st.session_state["show_ohlcv"] = False
        if st.button("📋 Show / Hide Recent OHLCV Data", key="ohlcv_btn"):
            st.session_state["show_ohlcv"] = not st.session_state["show_ohlcv"]
        if st.session_state["show_ohlcv"]:
            _dff=flatten_df(df)
            rd=_dff[["Open","High","Low","Close","Volume","return"]].tail(20).copy()
            rd.index=rd.index.strftime("%d %b %Y")
            rd["return"]*=100; rd.columns=["Open","High","Low","Close","Volume","Return %"]
            st.dataframe(rd.style.format({"Open":"{:,.2f}","High":"{:,.2f}","Low":"{:,.2f}",
                                          "Close":"{:,.2f}","Volume":"{:,.0f}","Return %":"{:+.2f}%"}),
                         use_container_width=True)

    # ══ TAB 1 — CHARTS ════════════════════════════════════════
    with T[1]:
        sec("🕯","Candlestick + Bollinger + Volume + RSI")
        chart_candle(df, ticker_used)
        c1c,c2c=st.columns(2)
        with c1c: sec("📊","MACD"); chart_macd(df)
        with c2c: sec("📈","Returns Distribution"); chart_returns_dist(df)
        sec("🗓️","Monthly Returns Heatmap")
        chart_heatmap(df)

        sec("🎯","Technical Signals")
        ind_all=indicators(df); sigs=signals_from(ind_all)
        scols=st.columns(len(sigs)); buy_n=sell_n=hold_n=0
        for i,(name,(desc,clr,act)) in enumerate(sigs.items()):
            if act=="BUY": buy_n+=1
            elif act=="SELL": sell_n+=1
            else: hold_n+=1
            scols[i].markdown(f"""<div class="ss-card" style="border-top:3px solid {clr};text-align:center;padding:15px 10px;">
                <div class="ss-lbl">{name}</div>
                <div style="font-size:15px;font-weight:800;color:{clr};margin:6px 0">{act}</div>
                <div style="font-size:11px;color:{TXT3};line-height:1.4">{desc}</div>
            </div>""", unsafe_allow_html=True)

        total=buy_n+sell_n+hold_n
        if total:
            ov_lbl="BULLISH" if buy_n>sell_n else ("BEARISH" if sell_n>buy_n else "NEUTRAL")
            ov_clr=GREEN if buy_n>sell_n else (RED if sell_n>buy_n else YELLOW)
            st.markdown(f"""<div class="ss-card" style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:14px 18px;margin-top:8px;">
                <span style="font-size:13px;font-weight:600;color:{TXT}">Signal Consensus:</span>
                <span style="font-size:13px;font-weight:700;color:{GREEN}">✅ BUY: {buy_n}</span>
                <span style="font-size:13px;font-weight:700;color:{RED}">❌ SELL: {sell_n}</span>
                <span style="font-size:13px;font-weight:700;color:{YELLOW}">⏸ HOLD: {hold_n}</span>
                <span>Overall: {badge(ov_lbl,ov_clr)}</span>
            </div>""", unsafe_allow_html=True)

    # ══ TAB 2 — AI FORECAST ═══════════════════════════════════
    with T[2]:
        if model is None:
            st.error("Model file not found. Place 'Stock Prediction Model.keras' in the project folder.")
            st.stop()

        sec("🔮","AI Model — Actual vs Predicted (Test Set)")
        prices_s = pd.Series(df["Close"].values.flatten(), index=df.index, dtype=float)
        tlen = int(len(prices_s)*0.8)
        dtrain, dtest = prices_s.iloc[:tlen], prices_s.iloc[tlen:]
        if len(dtrain)<PAST_WINDOW+1: st.error("Not enough history."); st.stop()

        dcomb = pd.concat([dtrain.tail(PAST_WINDOW), dtest], ignore_index=True)
        sc    = MinMaxScaler((0,1)); sc.fit(dcomb.values.reshape(-1,1))
        xte, yte = make_sequences(dcomb, PAST_WINDOW, sc)
        if xte.size==0: st.error("No test sequences."); st.stop()

        with st.spinner("Running AI inference…"):
            try:    ps = model.predict(xte, verbose=0)
            except Exception as e: st.error(f"Prediction error: {e}"); st.stop()

        preds   = inv_sc(sc, np.array(ps).reshape(-1,1)).astype(float)
        y_real  = inv_sc(sc, yte).astype(float)
        la, lp  = float(y_real[-1]), float(preds[-1])
        pct_ns  = (lp-la)/(la+1e-9)*100

        mae  = float(np.mean(np.abs(y_real-preds)))
        rmse = float(np.sqrt(np.mean((y_real-preds)**2)))
        mape = float(np.mean(np.abs((y_real-preds)/(y_real+1e-9)))*100)
        r2   = float(1-np.sum((y_real-preds)**2)/(np.sum((y_real-np.mean(y_real))**2)+1e-9))

        c1,c2,c3,c4=st.columns(4)
        card("Actual Price",  f"{cur_sym}{la:,.2f}", "Latest test point", ACCENT, c1)
        card("Predicted",     f"{cur_sym}{lp:,.2f}", "Next-step forecast", PURPLE, c2)
        card("Pred. Change",  f"{'▲' if pct_ns>=0 else '▼'} {abs(pct_ns):.2f}%", "Short-term",
             GREEN if pct_ns>=0 else RED, c3)
        card("R² Score",      f"{r2:.4f}", "Model fit quality",
             GREEN if r2>0.9 else YELLOW, c4)

        c1b,c2b,c3b=st.columns(3)
        card("MAE",  f"{cur_sym}{mae:,.2f}", "Mean Absolute Error",  TXT2, c1b)
        card("RMSE", f"{cur_sym}{rmse:,.2f}","Root Mean Sq Error",   TXT2, c2b)
        card("MAPE", f"{mape:.2f}%",         "Mean Abs % Error",     TXT2, c3b)

        chart_pred_actual(y_real, preds, cur_sym)

        sec("🚀",f"{forecast_days}-Day AI Forecast")
        with st.spinner("Generating forecast…"):
            lw=sc.transform(prices_s.iloc[-PAST_WINDOW:].values.reshape(-1,1)).reshape(1,PAST_WINDOW,1)
            fut=forecast_recursive(model,lw,forecast_days,sc).astype(float)

        fdates=pd.date_range(start=df.index[-1]+pd.Timedelta(days=1),periods=forecast_days,freq=DATE_FREQ)
        fdf=pd.DataFrame({"Date":fdates,"Predicted_Price":fut})
        fdf["Change_%"]=100*fdf["Predicted_Price"].pct_change().fillna(0)
        fdf["Cum_Return_%"]=100*(fdf["Predicted_Price"]/fdf["Predicted_Price"].iloc[0]-1)
        exp=100*(float(fut[-1])/(la+1e-9)-1)
        volf=float(pd.Series(fut).pct_change().std())

        c1c,c2c,c3c=st.columns(3)
        card(f"Price in {forecast_days}d", f"{cur_sym}{fut[-1]:,.2f}", "End of forecast", ACCENT, c1c)
        card("Expected Return", f"{'▲' if exp>=0 else '▼'} {abs(exp):.2f}%", f"{forecast_days}-day outlook",
             GREEN if exp>=0 else RED, c2c)
        card("Forecast Volatility", f"{volf*100:.2f}%", "Daily std dev", YELLOW, c3c)

        chart_forecast(prices_s.values, fut, cur_sym, volf)

        # Recommendation
        if   exp>=12: rec,rc,ri="STRONG BUY", GREEN,"🚀"
        elif exp>=4:  rec,rc,ri="BUY",         GREEN,"✅"
        elif exp<=-12:rec,rc,ri="STRONG SELL", RED,  "🔴"
        elif exp<=-4: rec,rc,ri="SELL",         RED,  "⚠️"
        else:         rec,rc,ri="HOLD",         YELLOW,"⏸️"

        st.markdown(f"""<div class="ss-card" style="border-left:4px solid {rc};
            background:linear-gradient(135deg,{rc}0a,{rc}04);
            display:flex;align-items:center;gap:18px;padding:20px 22px;">
            <div style="font-size:40px">{ri}</div>
            <div>
                <div style="font-size:10.5px;font-weight:600;color:{TXT3};text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">AI Recommendation</div>
                <div style="font-size:28px;font-weight:800;color:{rc}">{rec}</div>
                <div style="font-size:12px;color:{TXT3};margin-top:2px">
                    {forecast_days}-day forecast: <strong style="color:{rc}">{exp:+.2f}%</strong> expected return
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state["show_table"]:
            if "show_fcast_tbl" not in st.session_state:
                st.session_state["show_fcast_tbl"] = False
            if st.button("📋 Show / Hide Forecast Table", key="fcast_tbl_btn"):
                st.session_state["show_fcast_tbl"] = not st.session_state["show_fcast_tbl"]
            if st.session_state["show_fcast_tbl"]:
                st.dataframe(fdf.style.format({"Predicted_Price":"{:,.2f}",
                    "Change_%":"{:+.2f}%","Cum_Return_%":"{:+.2f}%"}),
                    use_container_width=True)

        buf=io.StringIO(); fdf.to_csv(buf,index=False)
        st.download_button(f"⬇️ Download {forecast_days}-day CSV",buf.getvalue(),
                           file_name=f"{ticker_input}_{forecast_days}d_forecast.csv",mime="text/csv")

    # ══ TAB 3 — RISK ══════════════════════════════════════════
    with T[3]:
        sec("⚖️","Risk Metrics")
        rm=risk_metrics(pd.Series(flatten_df(df)["Close"].values.flatten(),index=df.index,dtype=float))
        c1,c2,c3,c4=st.columns(4)
        card("Ann. Volatility", f"{rm['Ann. Volatility %']:.2f}%","",
             RED if rm["Ann. Volatility %"]>30 else YELLOW,c1)
        card("Sharpe Ratio", f"{rm['Sharpe Ratio']:.2f}","Risk-adj return",
             GREEN if rm["Sharpe Ratio"]>1 else(YELLOW if rm["Sharpe Ratio"]>0 else RED),c2)
        card("Sortino Ratio",f"{rm['Sortino Ratio']:.2f}","Downside risk",
             GREEN if rm["Sortino Ratio"]>1 else YELLOW,c3)
        card("Max Drawdown", f"{rm['Max Drawdown %']:.2f}%","Peak-to-trough",
             RED if rm["Max Drawdown %"]<-30 else YELLOW,c4)

        c1b,c2b,c3b,c4b=st.columns(4)
        card("VaR 95%",    f"{rm['VaR 95%']:.2f}%",   "Daily worst-case",   RED,    c1b)
        card("CVaR 95%",   f"{rm['CVaR 95%']:.2f}%",  "Expected shortfall", RED,    c2b)
        card("Calmar",     f"{rm['Calmar Ratio']:.2f}","Return / |MaxDD|",   ACCENT, c3b)
        card("Win Rate",   f"{rm['Win Rate %']:.1f}%", "Positive days",
             GREEN if rm["Win Rate %"]>52 else YELLOW,c4b)

        c1c,c2c=st.columns(2)
        with c1c:
            sec("📉","Drawdown Chart")
            chart_drawdown(pd.Series(flatten_df(df)["Close"].values.flatten(),index=df.index,dtype=float))
        with c2c:
            sec("🌊","Rolling 30d Volatility")
            chart_rolling_vol(pd.Series(flatten_df(df)["Close"].values.flatten(),index=df.index,dtype=float))

        sec("🎯","Risk Gauges")
        g1,g2,g3=st.columns(3)
        with g1: gauge(min(rm["Ann. Volatility %"],100),"Volatility Risk","gv")
        with g2: gauge(max(0,min(100-rm["Win Rate %"],100)),"Bearish Pressure","gb")
        with g3: gauge(min(abs(rm["Max Drawdown %"]),100),"Drawdown Severity","gd")

    # ══ TAB 4 — PORTFOLIO ════════════════════════════════════
    with T[4]:
        sec("💼","Portfolio What-If Simulator")
        if model is None:
            st.warning("Requires a loaded model.")
        else:
            prices_p = pd.Series(df["Close"].values.flatten(), index=df.index, dtype=float)
            tlen_p   = int(len(prices_p)*0.8)
            dcomb_p  = pd.concat([prices_p.iloc[:tlen_p].tail(PAST_WINDOW),
                                   prices_p.iloc[tlen_p:]], ignore_index=True)
            sc_p=MinMaxScaler((0,1)); sc_p.fit(dcomb_p.values.reshape(-1,1))
            lw_p=sc_p.transform(prices_p.iloc[-PAST_WINDOW:].values.reshape(-1,1)).reshape(1,PAST_WINDOW,1)
            fut_p=forecast_recursive(model,lw_p,forecast_days,sc_p).astype(float)

            c1p,c2p=st.columns([1,2])
            with c1p:
                invest  = st.number_input(f"Investment ({cur})",1000,10_000_000,10000,500)
                tx_bps  = st.slider("Transaction cost (bps)",0,100,10)
                cost_f  = tx_bps/10000

            with c2p:
                if len(fut_p):
                    units   = invest/(float(fut_p[0])+1e-9)
                    curve   = units*fut_p*(1-cost_f)
                    roi     = (curve[-1]-curve[0])/(curve[0]+1e-9)*100
                    profit  = curve[-1]-invest
                    chart_portfolio(curve,invest,cur_sym)
                    c1m,c2m,c3m,c4m=st.columns(4)
                    c1m.metric("Initial",   f"{cur_sym}{invest:,.2f}")
                    c2m.metric("Final",     f"{cur_sym}{curve[-1]:,.2f}",delta=f"{profit:+,.2f}")
                    c3m.metric("ROI",       f"{roi:+.2f}%")
                    c4m.metric("Units",     f"{units:.3f}")

            sec("🔢","What-If Table")
            wdata=[]
            for amt in [5000,10000,25000,50000,100000,250000,500000]:
                if len(fut_p):
                    u=amt/(float(fut_p[0])+1e-9); cv=u*fut_p*(1-cost_f)
                    r=(cv[-1]-cv[0])/(cv[0]+1e-9)*100
                    wdata.append({f"Invest ({cur})":f"{cur_sym}{amt:,}",
                                  f"Final ({cur})":f"{cur_sym}{cv[-1]:,.2f}",
                                  "Profit/Loss":f"{cur_sym}{cv[-1]-amt:+,.2f}","ROI":f"{r:+.2f}%"})
            if wdata: st.dataframe(pd.DataFrame(wdata),use_container_width=True)

    # ══ TAB 5 — COMPANY ══════════════════════════════════════
    with T[5]:
        sec("🏢","Company Intelligence")
        if not info:
            st.info("No company data available."); 
        else:
            name     = info.get("longName") or info.get("shortName") or ticker_used
            website  = info.get("website","")
            summary  = info.get("longBusinessSummary","")
            officers = info.get("companyOfficers") or []
            ceo="N/A"
            for o in officers:
                t=o.get("title","").lower()
                if any(k in t for k in ("ceo","chief executive","chairman","managing director","md")):
                    ceo=o.get("name","N/A"); break
            ipo_s="N/A"
            its=info.get("firstTradeDateEpochUtc")
            if its:
                try:
                    from datetime import datetime
                    ipo_s=datetime.fromtimestamp(its).strftime("%d %b %Y")
                except: pass
            if ipo_s=="N/A" and not df.empty:
                ipo_s=df.index[0].strftime("%d %b %Y")+" (Est.)"

            def ifield(lbl,val,icon=""):
                return f"""<div style="margin-bottom:11px;">
                    <div style="font-size:10.5px;font-weight:600;color:{TXT3};text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">{icon} {lbl}</div>
                    <div style="background:{BG3};border:1px solid {BORDER};border-radius:8px;padding:9px 12px;color:{TXT};font-size:13px;font-weight:500;">{val}</div>
                </div>"""

            mc2=info.get("marketCap")
            mc2s=(f"₹{mc2/1e7/1e5:.2f}L Cr" if cur=="INR" and mc2>=1e12
                  else f"{cur_sym}{mc2/1e12:.2f}T" if mc2 and mc2>=1e12
                  else f"{cur_sym}{mc2/1e9:.2f}B" if mc2 and mc2>=1e9
                  else "N/A")
            w52h2=info.get("fiftyTwoWeekHigh"); w52l2=info.get("fiftyTwoWeekLow")
            emp=info.get("fullTimeEmployees")

            c1,c2,c3=st.columns(3)
            with c1:
                st.markdown(ifield("Company",name,"🏢"), unsafe_allow_html=True)
                st.markdown(ifield("CEO / MD",ceo,"👤"), unsafe_allow_html=True)
                st.markdown(ifield("Exchange",info.get("exchange","N/A"),"🏦"), unsafe_allow_html=True)
                st.markdown(ifield("Sector",info.get("sector","N/A"),"🏭"), unsafe_allow_html=True)
                st.markdown(ifield("Industry",info.get("industry","N/A"),"⚙️"), unsafe_allow_html=True)
            with c2:
                st.markdown(ifield("Market Cap",mc2s,"💰"), unsafe_allow_html=True)
                st.markdown(ifield("Beta",f"{info.get('beta',0):.2f}" if info.get('beta') else "N/A","📊"), unsafe_allow_html=True)
                st.markdown(ifield("P/E Ratio",f"{info.get('trailingPE',0):.2f}" if info.get('trailingPE') else "N/A","📈"), unsafe_allow_html=True)
                st.markdown(ifield("Div. Yield",f"{info.get('dividendYield',0):.2%}" if info.get('dividendYield') else "N/A","💵"), unsafe_allow_html=True)
                st.markdown(ifield("Avg Volume",f"{avgvol/1e6:.2f}M" if avgvol and avgvol>=1e6 else(f"{avgvol/1e3:.0f}K" if avgvol else "N/A"),"📊"), unsafe_allow_html=True)
            with c3:
                st.markdown(ifield("IPO Date",ipo_s,"📅"), unsafe_allow_html=True)
                st.markdown(ifield("Country",info.get("country","N/A"),"🌍"), unsafe_allow_html=True)
                st.markdown(ifield("Employees",f"{emp:,}" if emp else "N/A","👥"), unsafe_allow_html=True)
                st.markdown(ifield("52W High",f"{cur_sym}{w52h2:,.2f}" if w52h2 else "N/A","📈"), unsafe_allow_html=True)
                st.markdown(ifield("52W Low", f"{cur_sym}{w52l2:,.2f}" if w52l2 else "N/A","📉"), unsafe_allow_html=True)

            if website:
                st.markdown(f"""<div style="background:{BG3};border:1px solid {BORDER};border-radius:9px;padding:10px 13px;margin-top:2px;">
                    🌐 <a href="{website}" target="_blank" style="color:{ACCENT};text-decoration:none;font-weight:500;font-size:13px;">{website} ↗</a>
                </div>""", unsafe_allow_html=True)

            if summary:
                st.markdown("---")
                st.markdown(f"#### 📝 About {name}")
                st.markdown(f"""<div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;
                    padding:16px 20px;font-size:13.5px;color:{TXT2};line-height:1.75;">
                    {summary[:1400]}{"…" if len(summary)>1400 else ""}
                </div>""", unsafe_allow_html=True)

    # ══ TAB 6 — COMPARE ══════════════════════════════════════
    with T[6]:
        sec("📊","Stock Comparison")
        if not st.session_state["compare_mode"] or not compare_ticker:
            st.markdown(f"""
            <div class="ss-card" style="text-align:center;padding:40px 20px;">
                <div style="font-size:36px;margin-bottom:12px">⚖️</div>
                <div style="font-size:16px;font-weight:700;color:{TXT};margin-bottom:8px">Enable Stock Comparison</div>
                <div style="font-size:13px;color:{TXT3};">Enable "Compare Stocks" in the sidebar and pick a second ticker to compare.</div>
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner(f"Loading {compare_ticker}…"):
                try:
                    df2, ticker2_used = fetch_data(compare_ticker, start_date)
                except Exception as e:
                    st.error(f"Could not load {compare_ticker}: {e}")
                    df2 = None; ticker2_used = compare_ticker

            if df2 is not None:
                info2     = fetch_info(ticker2_used)
                cur2_sym, _ = currency_of(compare_ticker, ticker2_used, info2)
                name2     = info2.get("longName") or info2.get("shortName") or ticker2_used
                price2    = float(pd.Series(df2["Close"].values.flatten(), dtype=float).iloc[-1])
                prev2     = float(pd.Series(df2["Close"].values.flatten(), dtype=float).iloc[-2])
                chg2      = (price2-prev2)/(prev2+1e-9)*100
                clr2      = GREEN if chg2>=0 else RED

                # Side-by-side quick stats
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:1.5rem;">
                  <div class="ss-card" style="border-top:3px solid {ACCENT};">
                    <div class="ss-lbl">{ticker_used}</div>
                    <div style="font-size:22px;font-weight:800;color:{TXT}">{company_name}</div>
                    <div style="font-size:26px;font-weight:800;color:{ACCENT};margin-top:6px">{cur_sym}{curr_price:,.2f}</div>
                    <div style="font-size:13px;color:{'GREEN' if day_pct>=0 else RED}">{'▲' if day_pct>=0 else '▼'} {abs(day_pct):.2f}% today</div>
                  </div>
                  <div class="ss-card" style="border-top:3px solid {PURPLE};">
                    <div class="ss-lbl">{ticker2_used}</div>
                    <div style="font-size:22px;font-weight:800;color:{TXT}">{name2}</div>
                    <div style="font-size:26px;font-weight:800;color:{PURPLE};margin-top:6px">{cur2_sym}{price2:,.2f}</div>
                    <div style="font-size:13px;color:{clr2}">{'▲' if chg2>=0 else '▼'} {abs(chg2):.2f}% today</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Normalised price chart (both on same scale, base=100)
                sec("📈","Normalised Price (Base = 100)")
                p1 = pd.Series(df["Close"].values.flatten(), index=df.index, dtype=float).dropna()
                p2 = pd.Series(df2["Close"].values.flatten(), index=df2.index, dtype=float).dropna()
                # Find common start
                common_start = max(p1.index[0], p2.index[0])
                p1 = p1[p1.index >= common_start]
                p2 = p2[p2.index >= common_start]
                n1 = p1 / p1.iloc[0] * 100
                n2 = p2 / p2.iloc[0] * 100
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(x=n1.index, y=n1.values, mode="lines",
                    name=ticker_used, line=dict(color=ACCENT, width=2)))
                fig_cmp.add_trace(go.Scatter(x=n2.index, y=n2.values, mode="lines",
                    name=ticker2_used, line=dict(color=PURPLE, width=2)))
                fig_cmp.add_hline(y=100, line=dict(color=TXT3, dash="dot", width=1))
                fig_cmp.update_layout(**CHART_BASE, height=380,
                    xaxis=_xax(), yaxis=_yax(title="Indexed Price (Base=100)"),
                    title=dict(text=f"<b>{ticker_used} vs {ticker2_used}</b> — Normalised",
                               font=dict(size=13, color=TXT)))
                st.plotly_chart(fig_cmp, use_container_width=True, key="cmp_norm")

                # Returns correlation
                sec("📉","Daily Returns Comparison")
                r1 = p1.pct_change().dropna() * 100
                r2 = p2.pct_change().dropna() * 100
                common_idx = r1.index.intersection(r2.index)
                r1c, r2c = r1[common_idx], r2[common_idx]

                c1x, c2x = st.columns(2)
                with c1x:
                    fig_r1 = go.Figure(go.Histogram(x=r1c, nbinsx=50,
                        marker_color=rgba(ACCENT,0.7), name=ticker_used))
                    fig_r1.update_layout(**CHART_BASE, height=260,
                        xaxis=_xax(title="Return %"), yaxis=_yax(),
                        title=dict(text=f"<b>{ticker_used}</b> Returns",font=dict(size=12,color=TXT)))
                    st.plotly_chart(fig_r1, use_container_width=True, key="cmp_r1")
                with c2x:
                    fig_r2 = go.Figure(go.Histogram(x=r2c, nbinsx=50,
                        marker_color=rgba(PURPLE,0.7), name=ticker2_used))
                    fig_r2.update_layout(**CHART_BASE, height=260,
                        xaxis=_xax(title="Return %"), yaxis=_yax(),
                        title=dict(text=f"<b>{ticker2_used}</b> Returns",font=dict(size=12,color=TXT)))
                    st.plotly_chart(fig_r2, use_container_width=True, key="cmp_r2")

                # Scatter correlation
                sec("🔗","Return Correlation Scatter")
                corr = float(r1c.corr(r2c))
                fig_sc = go.Figure(go.Scatter(x=r1c, y=r2c, mode="markers",
                    marker=dict(color=rgba(ACCENT,0.5), size=4),
                    name="Daily Returns",
                    hovertemplate=f"{ticker_used}: %{{x:.2f}}%<br>{ticker2_used}: %{{y:.2f}}%<extra></extra>"))
                # Trend line
                z = np.polyfit(r1c, r2c, 1)
                xline = np.linspace(r1c.min(), r1c.max(), 100)
                fig_sc.add_trace(go.Scatter(x=xline, y=np.polyval(z, xline),
                    mode="lines", line=dict(color=RED, width=1.5, dash="dot"), name="Trend"))
                fig_sc.update_layout(**CHART_BASE, height=350,
                    xaxis=_xax(title=f"{ticker_used} Daily Return %"),
                    yaxis=_yax(title=f"{ticker2_used} Daily Return %"),
                    title=dict(text=f"<b>Correlation: {corr:.3f}</b>  ({'High' if abs(corr)>0.7 else ('Moderate' if abs(corr)>0.4 else 'Low')})",
                               font=dict(size=13, color=TXT)))
                st.plotly_chart(fig_sc, use_container_width=True, key="cmp_scatter")

                # Risk comparison table
                sec("⚖️","Risk Metrics Side-by-Side")
                rm1 = risk_metrics(p1)
                rm2 = risk_metrics(p2)
                rows = []
                for metric in rm1:
                    v1, v2 = rm1[metric], rm2[metric]
                    winner = ticker_used if (
                        (metric in ["Sharpe Ratio","Sortino Ratio","Win Rate %","Calmar Ratio"] and v1 > v2) or
                        (metric in ["Ann. Volatility %","Max Drawdown %","VaR 95%","CVaR 95%"] and v1 > v2 and metric == "Max Drawdown %") or
                        (metric in ["Ann. Volatility %","VaR 95%","CVaR 95%"] and v1 > v2)
                    ) else ticker2_used
                    rows.append({"Metric": metric, ticker_used: v1, ticker2_used: v2})
                cmp_df = pd.DataFrame(rows).set_index("Metric")
                st.dataframe(cmp_df.style.format("{:.2f}"), use_container_width=True)

                # Volume comparison
                sec("📊","Volume Comparison")
                v1s = pd.Series(df["Volume"].values.flatten(), index=df.index, dtype=float)
                v2s = pd.Series(df2["Volume"].values.flatten(), index=df2.index, dtype=float)
                v1r = v1s.rolling(20).mean().dropna()
                v2r = v2s.rolling(20).mean().dropna()
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=v1r.index, y=v1r.values, mode="lines",
                    name=f"{ticker_used} Vol (20d MA)", line=dict(color=ACCENT, width=1.8)))
                fig_vol.add_trace(go.Scatter(x=v2r.index, y=v2r.values, mode="lines",
                    name=f"{ticker2_used} Vol (20d MA)", line=dict(color=PURPLE, width=1.8)))
                fig_vol.update_layout(**CHART_BASE, height=280,
                    xaxis=_xax(), yaxis=_yax(title="Volume (20d MA)"),
                    title=dict(text="<b>Trading Volume</b>", font=dict(size=13,color=TXT)))
                st.plotly_chart(fig_vol, use_container_width=True, key="cmp_vol")

    # FOOTER
    st.markdown(f"""<div style="margin-top:3rem;padding:14px;text-align:center;
        border-top:1px solid {BORDER};color:{TXT3};font-size:11.5px;">
        ⚠️ StockSense AI is for informational purposes only. Not financial advice. Always verify before trading.
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()