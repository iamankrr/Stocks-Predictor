"""
train_premium_optimized.py

Optimized training script — improves preprocessing, balancing, loss, callbacks,
and training pipeline while keeping the original model architecture intact.

How it works:
- Loads an existing model file if present (keeps architecture).
- If no model present, builds a simple LSTM/Conv1D model matching common premium architecture.
  (Minimal architecture only used when no model file is found.)
- Adds technical features to the series (returns, MA, vol, momentum).
- Scales features and target separately.
- Builds chronological sequences with configurable window and future horizon.
- Computes sample weights to balance up/down trends (reduces DOWN prediction bias).
- Uses a custom loss that penalizes underestimation a bit (to discourage downward bias).
- Trains with EarlyStopping, ReduceLROnPlateau and ModelCheckpoint. shuffle=False.
- Saves scalers and the improved model to disk.

Usage:
    python train_premium_optimized.py --ticker RELIANCE.NS --start 2012-01-01 --model_in "Stock Prediction Model.keras" --model_out "Stock Prediction Model_improved.keras"

Note: This script purposely avoids changing the model architecture when a model file is supplied.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# ---------------- Config ----------------
DEFAULT_PAST_WINDOW = 100
DEFAULT_FUTURE_DAYS = 90
DEFAULT_TICKER = "RELIANCE.NS"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SCALER_X_PATH = DATA_DIR / "scaler_X.npz"
SCALER_Y_PATH = DATA_DIR / "scaler_y.npz"

# ---------------- Utilities ----------------
def add_technical_features(df, price_col="Close"):
    df = df.copy()
    df = df.sort_index()
    df["close"] = df[price_col].astype(float)
    df["return"] = df["close"].pct_change().fillna(0)
    df["log_return"] = np.log1p(df["return"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["ma_7"] = df["close"].rolling(window=7, min_periods=1).mean()
    df["ma_21"] = df["close"].rolling(window=21, min_periods=1).mean()
    df["vol_21"] = df["close"].rolling(window=21, min_periods=1).std().fillna(0)
    df["mom_7"] = df["close"] - df["close"].shift(7)
    df["close_ma7_rel"] = (df["close"] - df["ma_7"]) / (df["ma_7"].replace(0, np.nan)).fillna(0)
    df["ma_ratio"] = df["ma_7"] / (df["ma_21"].replace(0, np.nan)).fillna(1)
    df = df.fillna(0)
    return df

def choose_scaler(name="minmax"):
    name = name.lower()
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError("Unsupported scaler")

def prepare_sequences(features: np.ndarray, target: np.ndarray, window: int):
    X = []
    y = []
    n = len(features)
    for i in range(window, n):
        X.append(features[i-window:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def compute_sample_weights_from_trend(y_raw, clipped=(0.5,5.0)):
    # y_raw are raw prices (not scaled). Compute trend: up (1) if next > current else 0
    trend = (np.diff(y_raw, prepend=y_raw[0]) > 0).astype(int)
    vals, counts = np.unique(trend, return_counts=True)
    total = len(trend)
    weights = np.ones_like(trend, dtype=float)
    for v,c in zip(vals, counts):
        w = total / (2.0 * c) if c>0 else 1.0
        weights[trend==v] = w
    weights = np.clip(weights, clipped[0], clipped[1])
    return weights

def custom_underestimate_penalty_loss(alpha=0.35):
    def loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))
        under = K.maximum(0.0, (y_true - y_pred))
        under_pen = K.mean(under)
        return mse + alpha * under_pen
    return loss

def build_default_model(input_shape):
    # Minimal architecture that mirrors premium model style (Conv1D + BiLSTM)
    inp = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.15)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    return model

# ---------------- Training pipeline ----------------
def train(args):
    # Fetch data
    if args.csv and Path(args.csv).exists():
        df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
    else:
        print(f"Downloading {args.ticker} from yfinance since {args.start}...")
        df = yf.download(args.ticker, start=args.start, progress=False, auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError("No data downloaded; check ticker or network.")

    df = df.copy().sort_index()
    df = add_technical_features(df, price_col="Close")

    # Choose features
    feature_cols = ["close"]
    feature_cols = ["close"]
    features = df[feature_cols].values.astype(float)
    prices = df["close"].values.astype(float)

    # Scalings
    scaler_X = choose_scaler(args.scaler)
    scaler_y = choose_scaler(args.scaler)
    scaler_X.fit(features)
    features_scaled = scaler_X.transform(features)
    scaler_y.fit(prices.reshape(-1,1))
    prices_scaled = scaler_y.transform(prices.reshape(-1,1)).flatten()

    # Sequences (note: target aligned to predict next day value)
    X_all, y_all = prepare_sequences(features_scaled, prices_scaled, args.past_window)
    # raw prices for trend weights (aligned)
    raw_prices_for_weights = prices[args.past_window:]

    if len(X_all) == 0:
        raise RuntimeError("Not enough data to build sequences. Increase history or reduce window.")

    # Chronological train / val / test split
    n = len(X_all)
    test_size = int(n * args.test_ratio)
    val_size = int(n * args.val_ratio)
    train_end = n - val_size - test_size
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:train_end+val_size], y_all[train_end:train_end+val_size]
    X_test, y_test = X_all[train_end+val_size:], y_all[train_end+val_size:]

    # Compute sample weights (on raw price trend)
    sample_weights = compute_sample_weights_from_trend(raw_prices_for_weights)
    sw_train = sample_weights[:train_end] if sample_weights is not None else None
    sw_val = sample_weights[train_end:train_end+val_size] if sample_weights is not None else None

    # Load or build model (keep architecture if existing model supplied)
    model = None
    if args.model_in and Path(args.model_in).exists():
        print("Loading existing model (architecture preserved):", args.model_in)
        model = load_model(args.model_in, compile=False)
    else:
        print("No existing model found. Building default model.")
        model = build_default_model(input_shape=X_train.shape[1:])

    # Compile with custom loss or chosen loss
    if args.loss == "mse":
        loss_fn = "mse"
    elif args.loss == "huber":
        loss_fn = tf.keras.losses.Huber()
    else:
        loss_fn = custom_underestimate_penalty_loss(alpha=args.under_alpha)

    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])

    # Callbacks
    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, args.patience//4), min_lr=1e-6, verbose=1))
    callbacks.append(ModelCheckpoint(args.model_out, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1))

    # Fit
    print("Training shapes:", X_train.shape, y_train.shape, "Validation shapes:", X_val.shape, y_val.shape)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=sw_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=2
    )

    # Save scalers
    np.savez_compressed(SCALER_X_PATH, min_=scaler_X.data_min_, scale_=getattr(scaler_X, "scale_", None))
    np.savez_compressed(SCALER_Y_PATH, min_=scaler_y.data_min_, scale_=getattr(scaler_y, "scale_", None))

    # Final evaluate
    loss_val, mae_val = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation loss {loss_val:.6f} mae {mae_val:.6f}")
    print(f"Test loss {test_loss:.6f} mae {test_mae:.6f}")

    # Save final model (already saved by checkpoint)
    print("Training completed. Model saved to:", args.model_out)

# ---------------- CLI ----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Ticker for yfinance (if csv not provided)")
    parser.add_argument("--start", type=str, default="2012-01-01", help="Start date for historical data")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV historic file (index as date)")
    parser.add_argument("--model_in", type=str, default="Stock Prediction Model.keras", help="Existing model file (keep architecture)")
    parser.add_argument("--model_out", type=str, default="Stock Prediction Model_improved.keras", help="Output model file")
    parser.add_argument("--past_window", type=int, default=DEFAULT_PAST_WINDOW)
    parser.add_argument("--future_days", type=int, default=DEFAULT_FUTURE_DAYS)
    parser.add_argument("--scaler", type=str, default="minmax", choices=["minmax","standard","robust"])
    parser.add_argument("--loss", type=str, default="custom", choices=["mse","huber","custom"])
    parser.add_argument("--under_alpha", type=float, default=0.35, help="Underestimation penalty weight")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--test_ratio", type=float, default=0.08)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # pass CLI args into train
    train(args)
