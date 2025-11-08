"""
quant_pipeline.py — Full Quant ML Trading System
with walk-forward validation and realistic backtest
Author: Aman & ChatGPT Quant ML Edition
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lightgbm import LGBMClassifier
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class Config:
    symbol: str = "RELIANCE.NS"
    start_date: str = "2015-01-01"
    end_date: str = "2025-11-01"
    n_splits: int = 5
    gap_days: int = 5
    transaction_cost_bps: float = 10.0
    use_hmm_regime: bool = False  # set True later if needed

def download_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    print(f"📥 Downloading data for {symbol} ...")
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    df["Return"] = df["Adj Close"].pct_change()
    return df.dropna()

def feature_engineering(df: pd.DataFrame, use_hmm: bool = True) -> pd.DataFrame:
    print("🧮 Feature engineering ...")
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["SMA_ratio"] = df["SMA_10"] / df["SMA_30"]
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["Volatility_10"] = df["Return"].rolling(10).std()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Width"] = (std20 * 4) / ma20
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    if use_hmm:
        print("🔍 Detecting market regimes (HMM)...")
        hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
        X = df["Return"].dropna().to_numpy().reshape(-1, 1)
        hmm.fit(X)
        states = hmm.predict(X)
        df.loc[df["Return"].dropna().index, "Regime"] = states
    else:
        df["Regime"] = 0
    df = df.dropna()
    return df

def create_labels(df: pd.DataFrame) -> pd.Series:
    return (df["Close"].shift(-1) > df["Close"]).astype(int)

def build_model_pipeline() -> Pipeline:
    model = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def walk_forward_cv(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    X = df[["SMA_ratio", "Momentum_10", "Volatility_10", "RSI", "BB_Width", "OBV", "Regime"]]
    y = create_labels(df)
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.gap_days)
    preds, probs, actuals, folds = [], [], [], []
    print("🚶 Starting walk-forward validation ...")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe = build_model_pipeline()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        preds.extend(y_pred)
        probs.extend(y_prob)
        actuals.extend(y_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        print(f"Fold {i}: acc={acc:.3f}, auc={auc:.3f}, f1={f1:.3f}")
        folds.append({"fold": i, "acc": acc, "auc": auc, "f1": f1})
    metrics = pd.DataFrame(folds).mean().to_dict()
    results = pd.DataFrame({
        "Date": X.index[-len(preds):],
        "Pred": preds,
        "Prob": probs,
        "Actual": actuals
    })
    return results, metrics

def backtest(df: pd.DataFrame, results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("📈 Backtesting ...")
    df_bt = df.loc[results["Date"]].copy()
    df_bt["signal"] = results["Pred"]
    df_bt["ret"] = df_bt["Return"].fillna(0)
    df_bt["strategy"] = df_bt["signal"].shift(1) * df_bt["ret"]
    cost = cfg.transaction_cost_bps / 10000
    trades = df_bt["signal"].diff().abs()
    df_bt["strategy_net"] = df_bt["strategy"] - trades * cost
    df_bt["equity"] = (1 + df_bt["strategy_net"]).cumprod()
    return df_bt

def run_pipeline(cfg: Config):
    df = download_price_data(cfg.symbol, cfg.start_date, cfg.end_date)
    df = feature_engineering(df, use_hmm=cfg.use_hmm_regime)
    results, metrics = walk_forward_cv(df, cfg)
    bt = backtest(df, results, cfg)
    os.makedirs("outputs", exist_ok=True)
    sym = cfg.symbol.replace(".", "_")
    results.to_csv(f"outputs/{sym}_oos_predictions.csv", index=False)
    bt.to_csv(f"outputs/{sym}_backtest.csv")
    pd.DataFrame([metrics]).to_csv(f"outputs/{sym}_metrics.csv", index=False)
    plt.figure(figsize=(10,5))
    plt.plot(bt["equity"])
    plt.title(f"Equity Curve - {cfg.symbol}")
    plt.savefig(f"outputs/{sym}_equity.png")
    print("✅ Saved all outputs to /outputs/")

if __name__ == "__main__":
    cfg = Config()
    run_pipeline(cfg)

