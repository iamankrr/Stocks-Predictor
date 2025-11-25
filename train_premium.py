import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# --- CONFIG ---
PAST_WINDOW = 100
FUTURE_DAYS = 90

# --- Mac-safe path ---
MODEL_OUT = Path("data/stock_predictor_premium.keras")

BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 1e-3


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    df["close"] = df["Close"].astype(float)
    df["volume"] = df["Volume"].fillna(0).astype(float)

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

    return df.dropna()


def build_premium_model(past_window, feat_count, future_horizon):
    inp = Input(shape=(past_window, feat_count))
    x = Conv1D(64, 3, activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Conv1D(128, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)

    out = Dense(future_horizon, activation="linear")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model


def prepare_dataset(df):
    df2 = compute_indicators(df)
    features = ["close","sma_7","sma_21","ema_12","macd",
                "rsi_14","vol_ratio","ret_1","ret_7"]

    X_raw = df2[features].values
    y_raw = df2["close"].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    Xs = scaler_X.fit_transform(X_raw)
    ys = scaler_y.fit_transform(y_raw.reshape(-1,1)).flatten()

    X, Y = [], []
    for i in range(PAST_WINDOW, len(Xs) - FUTURE_DAYS + 1):
        X.append(Xs[i - PAST_WINDOW:i])
        Y.append(ys[i:i+FUTURE_DAYS])

    return np.array(X), np.array(Y), scaler_X, scaler_y


def main():
    csv_path = Path("/tmp/sbin_history_clean.csv")
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    X, Y, scaler_X, scaler_y = prepare_dataset(df)
    print("Prepared shapes:", X.shape, Y.shape)

    model = build_premium_model(PAST_WINDOW, X.shape[2], FUTURE_DAYS)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
        ModelCheckpoint(str(MODEL_OUT), save_best_only=True, monitor="val_loss")
    ]

    model.fit(
        X, Y,
        validation_split=0.12,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    np.save("data/scaler_X_min.npy", scaler_X.data_min_)
    np.save("data/scaler_X_scale.npy", scaler_X.scale_)
    np.save("data/scaler_y_min.npy", scaler_y.data_min_)
    np.save("data/scaler_y_scale.npy", scaler_y.scale_)

    print("Training completed. Model saved to:", MODEL_OUT)


if __name__ == "__main__":
    main()
