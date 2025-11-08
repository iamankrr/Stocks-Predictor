# app.py
"""
Streamlit Stock Predictor + 90-day Forecast + Stock Info + Suggestions
Copy-paste this entire file into ~/Developer/Project/Stocks_Predict/app.py
Run with: streamlit run app.py
"""

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------------------
# Config / constants
# -------------------------------
MODEL_FILENAME = "Stock Prediction Model.keras"  # placed in project folder
DEFAULT_TICKER = "RELIANCE.NS"
FUTURE_DAYS = 90  # user requested 90 days
PAST_WINDOW = 100  # used by model
DATE_FREQ = "B"  # business day frequency for forecast dates

st.set_page_config(page_title="Stock Price Predictor — 90 day", layout="wide")
st.title("📈 Stock Price Predictor + 90-Day Forecast + Stock Info")

# -------------------------------
# Helper functions
# -------------------------------
def safe_load_model(path: Path):
    try:
        model = load_model(str(path))
        return model, None
    except Exception as e:
        return None, str(e)


def fetch_price_data(ticker: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """
    Automatically tries variations for Indian and US tickers if initial fetch fails.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers_to_try = [ticker]
    
    # If no exchange suffix, try Indian and US variants
    if "." not in ticker:
        tickers_to_try += [ticker + ".NS", ticker + ".BO"]
    
    for t in tickers_to_try:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if not df.empty:
                df["ticker_used"] = t
                # Ensure adjusted close column
                if "Adj Close" in df.columns:
                    df["adj_close"] = df["Adj Close"]
                else:
                    df["adj_close"] = df["Close"]
                df["return"] = df["adj_close"].pct_change()
                return df
        except Exception:
            continue
    raise ValueError(f"No data returned by yfinance for any variant of: {ticker}")



def get_ticker_info(ticker: str) -> dict:
    """
    Use yfinance.Ticker.info (best-effort). May return empty dict or partial info.
    """
    info = {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        info = {}
    return info


def prepare_test_sequences(series: pd.Series, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build x_test and y_test arrays like earlier script expects.
    series: pandas Series of prices (train+test combined where scaler was fit)
    scaler: fitted MinMaxScaler used on that combined series
    """
    arr = scaler.transform(series.values.reshape(-1, 1))
    x_test, y_test = [], []
    for i in range(PAST_WINDOW, arr.shape[0]):
        x_test.append(arr[i - PAST_WINDOW:i, :])
        y_test.append(arr[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test


def make_future_dates(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=DATE_FREQ)


def format_money(x: float, currency: str = "INR") -> str:
    if pd.isna(x):
        return "N/A"
    return f"{currency} {x:,.2f}"


# -------------------------------
# UI: model load & ticker input
# -------------------------------
# left column: controls
col1, col2 = st.columns([1, 2])

with col1:
    model_path = Path(__file__).parent / MODEL_FILENAME
    st.markdown("### 🔁 Model")
    model, model_err = safe_load_model(model_path)
    if model is None:
        st.error("⚠️ Model load failed: " + (model_err or "unknown"))
        st.stop()
    else:
        st.success("✅ Model loaded")

    st.markdown("### 🔎 Choose stock")
    popular = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Apple": "AAPL",
        "Microsoft": "MSFT"
    }
    sel = st.selectbox("Popular tickers", [""] + list(popular.keys()))
    if sel:
        ticker_input = popular[sel]
    else:
        ticker_input = st.text_input("Or enter ticker (e.g. RELIANCE.NS or AAPL)", DEFAULT_TICKER).strip().upper()

    start_date = st.text_input("Start date (YYYY-MM-DD)", "2012-01-01")
    # currency detection (basic)
    currency = "INR" if ticker_input.endswith((".NS", ".BO", ".BSE")) else "USD"

with col2:
    st.markdown("### ℹ️ Quick notes")
    st.markdown("- Uses your local `.keras` model file.")
    st.markdown("- 90-day forecast uses recursive multi-step prediction (feeding predictions back).")
    st.markdown("- Always verify with fundamentals & news. This is technical-only.")

# -------------------------------
# Main pipeline: data -> predict -> UI
# -------------------------------
if not ticker_input:
    st.info("Enter a ticker to start.")
    st.stop()

try:
    df = fetch_price_data(ticker_input, start=start_date)
except Exception as e:
    st.error("Data fetch failed: " + str(e))
    st.stop()

st.markdown(f"## 📄 {ticker_input} — Recent Data & Info")
st.dataframe(df.tail(5))

# ticker metadata
info = get_ticker_info(ticker_input)
with st.expander("Show company summary & key metadata"):
    if info:
        name = info.get("longName") or info.get("shortName") or ticker_input
        st.markdown(f"### {name}")
        # show some fields if present
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Sector**")
            st.write(info.get("sector", "N/A"))
            st.write("**Industry**")
            st.write(info.get("industry", "N/A"))
            st.write("**Market cap**")
            mcap = info.get("marketCap")
            st.write(f"{mcap:,}" if mcap else "N/A")
        with col_b:
            st.write("**52wk High / Low**")
            h52 = info.get("fiftyTwoWeekHigh", "N/A")
            l52 = info.get("fiftyTwoWeekLow", "N/A")
            st.write(f"{h52} / {l52}")
            st.write("**Beta**")
            st.write(info.get("beta", "N/A"))
            st.write("**Dividend Yield**")
            div = info.get("dividendYield")
            st.write(f"{div:.4f}" if div else "N/A")
        with col_c:
            st.write("**Avg Volume**")
            st.write(info.get("averageVolume", "N/A"))
            st.write("**Recommendation (mean)**")
            st.write(info.get("recommendationMean", "N/A"))
            st.write("**Exchange**")
            st.write(info.get("exchange", "N/A"))
        # business summary (shorten)
        summary = info.get("longBusinessSummary") or info.get("summary") or ""
        if summary:
            st.markdown("**Business summary**")
            st.write(summary[:1000] + ("..." if len(summary) > 1000 else ""))
    else:
        st.write("No metadata available for this ticker.")

# -------------------------------
# Build dataset for prediction using same approach as training
# -------------------------------
# We'll use closing prices for scaler like earlier code
prices = pd.DataFrame(df["Close"].copy())
prices = prices.rename(columns={"Close": "Close"})  # ensure column exists
# Split to mimic previous pipeline (use 80% train)
train_len = int(len(prices) * 0.8)
data_train = prices.iloc[:train_len]
data_test = prices.iloc[train_len:]

if len(data_train) < PAST_WINDOW + 1:
    st.error("Not enough history for prediction. Provide older start_date.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(PAST_WINDOW)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
scaler.fit(data_test_combined.values.reshape(-1, 1))
data_test_scaled = scaler.transform(data_test_combined.values.reshape(-1, 1))

# prepare x_test, y_test
x_test = []
y_test = []
for i in range(PAST_WINDOW, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i - PAST_WINDOW:i, 0].reshape(PAST_WINDOW, 1))
    y_test.append(data_test_scaled[i, 0])
x_test = np.array(x_test)
y_test = np.array(y_test)

# handle case x_test empty
if x_test.size == 0:
    st.error("Insufficient test samples after preparing sequences.")
    st.stop()

# Ensure model input shape matches (some keras models accept (batch, time, features))
try:
    preds_scaled = model.predict(x_test)
except Exception as e:
    st.error("Model prediction failed: " + str(e))
    st.stop()

# inverse transform predictions and y_test
try:
    preds = scaler.inverse_transform(preds_scaled)
except Exception:
    # if prediction shape mismatches (e.g. (n, ) ), reshape
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1))
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# Latest values
latest_actual = float(y_test_real[-1])
latest_pred = float(preds[-1][0]) if preds.ndim > 1 else float(preds[-1])

# Chart: actual vs predicted (test window)
st.subheader("📈 Actual vs Predicted (Test set)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test_real, label="Actual", color="green")
ax.plot(preds, label="Predicted", color="red")
ax.set_xlabel("Test samples")
ax.set_ylabel(f"Price ({'INR' if currency=='INR' else 'USD'})")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Multi-step (recursive) 90-day prediction
# -------------------------------
last_100_scaled = data_test_scaled[-PAST_WINDOW:].reshape(1, PAST_WINDOW, 1)
future_scaled_preds = []
future_input = last_100_scaled.copy()

for _ in range(FUTURE_DAYS):
    next_scaled = model.predict(future_input)[0]
    # ensure shape (features,)
    if next_scaled.ndim == 0:
        next_scaled = np.array([next_scaled])
    future_scaled_preds.append(next_scaled)
    # append and slide window
    next_scaled_reshaped = np.array(next_scaled).reshape(1, 1, 1)  # (1,1,1)
    future_input = np.concatenate([future_input[:, 1:, :], next_scaled_reshaped], axis=1)

# inverse transform
try:
    future_preds = scaler.inverse_transform(np.array(future_scaled_preds).reshape(-1, 1)).flatten()
except Exception:
    # fallback: if shapes odd
    future_preds = np.array(future_scaled_preds).reshape(-1)
    # attempt to map via min/max of scaler
    min_v, max_v = scaler.data_min_[0], scaler.data_max_[0]
    future_preds = future_preds * (max_v - min_v) + min_v

# build dates for forecast (business days)
last_price_date = df.index[-1]
future_dates = make_future_dates(pd.to_datetime(last_price_date), FUTURE_DAYS)
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_preds})
forecast_df["Predicted_Change_%"] = 100.0 * (forecast_df["Predicted_Price"].pct_change()).fillna(0)

# show summary metrics for forecast
expected_90day_change = 100.0 * (future_preds[-1] / latest_actual - 1.0)
avg_forecast_vol = float(pd.Series(future_preds).pct_change().std() * np.sqrt(252))

st.subheader(f"🔮 {FUTURE_DAYS}-Day Forecast (table)")
st.dataframe(forecast_df.style.format({"Predicted_Price": "{:,.2f}", "Predicted_Change_%": "{:+.2f}%"}))

# allow download CSV
buf = io.StringIO()
forecast_df.to_csv(buf, index=False)
st.download_button("⬇️ Download 90-day forecast CSV", buf.getvalue(), file_name=f"{ticker_input}_90day_forecast.csv")

# Chart: historical last section + 90-day futures
st.subheader("📉 Past (test) + 90-Day Forecast")

# Determine which column contains prices
if "Close" in df.columns:
    hist_prices = df["Close"].dropna().to_numpy()
elif "Adj Close" in df.columns:
    hist_prices = df["Adj Close"].dropna().to_numpy()
elif "adj_close" in df.columns:
    hist_prices = df["adj_close"].dropna().to_numpy()
else:
    # fallback: use first numeric column
    hist_prices = df.select_dtypes(include=np.number).iloc[:, 0].dropna().to_numpy()

# take last 200 values if available
hist_plot = hist_prices[-200:] if len(hist_prices) > 200 else hist_prices
x_hist = np.arange(len(hist_plot))

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(x_hist, hist_plot, label="Recent Close (historical)", color="black")

# overlay predicted future
x_future = np.arange(len(hist_plot), len(hist_plot) + FUTURE_DAYS)
ax2.plot(x_future, future_preds, label=f"{FUTURE_DAYS}-day forecast", color="blue", linestyle="dashed")
ax2.set_xlabel("Index")
ax2.set_ylabel(f"Price ({currency})")
ax2.legend()
st.pyplot(fig2)


# -------------------------------
# Investment suggestions & extra info
# -------------------------------
st.subheader("💡 Investment Suggestions & Notes")

# basic numeric info
st.markdown(f"- **Current (latest actual)**: {format_money(latest_actual, currency)}")
st.markdown(f"- **Model Predicted (next step)**: {format_money(latest_pred, currency)}")
st.markdown(f"- **Predicted change (next-step)**: {((latest_pred - latest_actual) / latest_actual * 100):+.2f}%")
st.markdown(f"- **Predicted change (after {FUTURE_DAYS} days)**: {expected_90day_change:+.2f}%")
st.markdown(f"- **Forecast realized-vol (annualized, proxy)**: {avg_forecast_vol:.2%}")

# Recommendation logic (simple heuristics)
rec_text = ""
if expected_90day_change >= 10:
    rec_text = "Strong BUY (long term view) — model expects >10% gain in 90 days"
elif expected_90day_change >= 3:
    rec_text = "BUY — modest upside expected"
elif expected_90day_change <= -10:
    rec_text = "Strong SELL — significant downside expected"
elif expected_90day_change <= -3:
    rec_text = "SELL"
else:
    rec_text = "HOLD — little net movement expected in 90 days"

st.markdown(f"**Model suggestion:** {rec_text}")

# add risk notes using metadata and volatility
if info:
    # use beta/dividend to augment notes
    beta = info.get("beta")
    div_yield = info.get("dividendYield")
    sector = info.get("sector")
    notes = []
    if sector:
        notes.append(f"Sector: **{sector}** — check sector outlook & macro factors.")
    if beta:
        notes.append(f"Beta: **{beta:.2f}** (higher than 1 implies higher market sensitivity).")
    if div_yield:
        notes.append(f"Dividend yield: **{div_yield:.2%}** (income characteristic).")
    if notes:
        st.markdown("**Extra context:**")
        for n in notes:
            st.markdown("- " + n)

st.markdown("**Practical suggestions:**")
st.markdown("- Use this model's predictions as one input among many: combine with earnings, sector news, and macro data.")
st.markdown("- If you trade intraday or require precise execution: model uses daily closes; latency & slippage matter.")
st.markdown("- Consider position sizing: don't risk more than X% of capital per trade (use Kelly/CVaR if you implement).")
st.markdown("- Always verify dataset and check for corporate actions (splits, dividends) that may affect short-term predictions.")

# small footer
st.caption("This tool provides technical predictions from a trained model. Not financial advice. Verify before trading.")

