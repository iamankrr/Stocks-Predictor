import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Constants for premium model
PAST_WINDOW = 100
FUTURE_DAYS = 90
PREMIUM_MODEL_FILENAME = "stock_predictor_premium.keras"

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    df["close"] = df["Close"].astype(float)
    df["volume"] = df.get("Volume", 0).fillna(0).astype(float)
    df["sma_7"] = df["close"].rolling(7).mean()
    df["sma_21"] = df["close"].rolling(21).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["std_20"] = df["close"].rolling(20).std()
    df["upper_bb"] = df["sma_21"] + 2 * df["std_20"]
    df["lower_bb"] = df["sma_21"] - 2 * df["std_20"]
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + (ma_up / (ma_down + 1e-9))))
    df["vol_7"] = df["volume"].rolling(7).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_7"] + 1e-9)
    df["ret_1"] = df["close"].pct_change(1).fillna(0)
    df["ret_7"] = df["close"].pct_change(7).fillna(0)
    df = df.dropna()
    return df

def load_premium_model_and_scalers(model_path: Path):
    try:
        model = load_model(str(model_path))
    except Exception as e:
        return None, f"Model load failed: {e}"

    try:
        scaler_X = MinMaxScaler()
        scaler_X.min_ = np.load("data/scaler_X_min.npy")
        scaler_X.scale_ = np.load("data/scaler_X_scale.npy")
        scaler_X.data_min_ = scaler_X.min_
        scaler_X.data_max_ = scaler_X.min_ + scaler_X.scale_
    except Exception:
        scaler_X = None

    try:
        scaler_y = MinMaxScaler()
        scaler_y.data_min_ = np.load("data/scaler_y_min.npy")
        scaler_y.scale_ = np.load("data/scaler_y_scale.npy")
        scaler_y.min_ = scaler_y.data_min_
        scaler_y.data_max_ = scaler_y.data_min_ + scaler_y.scale_
    except Exception:
        scaler_y = None

    return model, (scaler_X, scaler_y)

def predict_direct(model, scaler_X, scaler_y, df_recent, past_window=PAST_WINDOW, future_horizon=FUTURE_DAYS):
    df_feats = compute_indicators(df_recent)
    features = ["close", "sma_7", "sma_21", "ema_12", "macd", "rsi_14", "vol_ratio", "ret_1", "ret_7"]

    if len(df_feats) < past_window:
        raise ValueError("Not enough rows for prediction.")

    X = df_feats[features].iloc[-past_window:].values.astype(float)

    if scaler_X is not None:
        Xs = scaler_X.transform(X)
    else:
        tmp = MinMaxScaler().fit(X)
        Xs = tmp.transform(X)

    Xs = Xs.reshape(1, past_window, len(features))

    pred_scaled = model.predict(Xs, verbose=0)[0]

    if scaler_y is not None:
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    else:
        last_close = df_recent["Close"].iloc[-1]
        pred = pred_scaled * last_close

    return pred
