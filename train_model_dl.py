"""
train_model_dl.py

Deep learning version of the stock prediction model.

- Downloads historical data with yfinance
- Engineers technical features (returns, MAs, RSI, volatility)
- Builds sliding windows of length WINDOW (sequence of past bars)
- Trains an LSTM with two heads:
    - direction (binary classification: UP vs DOWN/FLAT)
    - magnitude (regression: next_return)

Saves:
- Keras model to: models/dl_model
- Metadata (feature columns, scaler, window size) to: models/dl_meta.pkl
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models

import joblib


DL_MODEL_DIR = "models/dl_model"
DL_META_PATH = "models/dl_meta.pkl"


# ---------- Data download ----------
def download_data(
    ticker: str = "AAPL", period: str = "10y", interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker using yfinance.
    Default: daily bars for 10 years.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df


# ---------- Indicator calculation ----------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index) for a price series.
    Assumes `series` is a 1D pandas Series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def engineer_features(df: pd.DataFrame):
    """
    Create features for the DL model based on historical price data.

    This is similar to your earlier engineer_features, but used for
    sequence modeling.

    Targets:
    - y_cls: binary direction (UP vs DOWN/FLAT)
    - y_reg: numeric next_return
    """
    df = df.copy()

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close)

    # Returns (1, 2, 5 bars)
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    # Moving averages
    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_10"] = close.rolling(window=10).mean()
    df["ma_20"] = close.rolling(window=20).mean()

    # Ratios of moving averages
    df["ma_5_20_ratio"] = df["ma_5"] / (df["ma_20"] + 1e-9)
    df["ma_10_20_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # Volatility of 1-bar returns
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    # RSI
    df["rsi_14"] = compute_rsi(close, period=14)

    # Targets
    next_close = close.shift(-1)
    df["next_close"] = next_close
    df["next_return"] = (next_close - close) / close
    df["target_up"] = (df["next_return"] > 0).astype(int)

    # Drop rows with NaNs (from rolling + shift)
    df.dropna(inplace=True)

    feature_cols = [
        "return_1d",
        "return_2d",
        "return_5d",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_5_20_ratio",
        "ma_10_20_ratio",
        "vol_5",
        "vol_10",
        "rsi_14",
    ]

    X = df[feature_cols].values
    y_cls = df["target_up"].values.astype(np.float32)
    y_reg = df["next_return"].values.astype(np.float32)

    index = df.index  # keep timestamps for potential analysis

    return X, y_cls, y_reg, feature_cols, index


# ---------- Sequence builder ----------
def build_sequences(X, y_cls, y_reg, window: int = 30):
    """
    Build sliding window sequences:
    For each time t, input is X[t-window:t], target is at time t.

    Returns:
    - X_seq: shape (num_samples, window, num_features)
    - y_cls_seq: shape (num_samples,)
    - y_reg_seq: shape (num_samples,)
    """
    X_seq, y_cls_seq, y_reg_seq = [], [], []
    for i in range(window, len(X)):
        X_seq.append(X[i - window : i])
        y_cls_seq.append(y_cls[i])
        y_reg_seq.append(y_reg[i])

    return np.array(X_seq), np.array(y_cls_seq), np.array(y_reg_seq)


# ---------- Model builder ----------
def build_lstm_model(input_shape):
    """
    Build an LSTM-based model with two heads:
    - direction (binary classification, sigmoid)
    - magnitude (regression, linear)
    """
    inputs = layers.Input(shape=input_shape)  # (window, num_features)

    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)

    # Classification head
    direction_out = layers.Dense(1, activation="sigmoid", name="direction")(x)

    # Regression head
    magnitude_out = layers.Dense(1, activation="linear", name="magnitude")(x)

    model = models.Model(inputs=inputs, outputs=[direction_out, magnitude_out])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "direction": "binary_crossentropy",
            "magnitude": "mse",
        },
        loss_weights={
            "direction": 1.0,
            "magnitude": 0.5,
        },
        metrics={
            "direction": ["accuracy", tf.keras.metrics.AUC(name="auc")],
            "magnitude": ["mae"],
        },
    )

    return model


# ---------- Main training pipeline ----------
def train_dl_model(
    ticker: str = "AAPL",
    period: str = "10y",
    interval: str = "1d",
    window: int = 30,
    batch_size: int = 64,
    epochs: int = 50,
):
    print(f"Downloading data for {ticker}...")
    df = download_data(ticker=ticker, period=period, interval=interval)

    print("Engineering features...")
    X_raw, y_cls, y_reg, feature_cols, index = engineer_features(df)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"Building sequences (window={window})...")
    X_seq, y_cls_seq, y_reg_seq = build_sequences(X_scaled, y_cls, y_reg, window=window)

    print(f"Sequence shape: {X_seq.shape}, labels (cls): {y_cls_seq.shape}, (reg): {y_reg_seq.shape}")

    # Time-based train/val/test split
    n = len(X_seq)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, X_val, X_test = (
        X_seq[:train_end],
        X_seq[train_end:val_end],
        X_seq[val_end:],
    )
    y_cls_train, y_cls_val, y_cls_test = (
        y_cls_seq[:train_end],
        y_cls_seq[train_end:val_end],
        y_cls_seq[val_end:],
    )
    y_reg_train, y_reg_val, y_reg_test = (
        y_reg_seq[:train_end],
        y_reg_seq[train_end:val_end],
        y_reg_seq[val_end:],
    )

    print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape=input_shape)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    print("Training LSTM model...")
    history = model.fit(
        X_train,
        {"direction": y_cls_train, "magnitude": y_reg_train},
        validation_data=(X_val, {"direction": y_cls_val, "magnitude": y_reg_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating on test set...")
    # Predictions
    y_cls_pred_prob, y_reg_pred = model.predict(X_test)
    y_cls_pred = (y_cls_pred_prob.ravel() >= 0.5).astype(int)

    # Classification metrics
    cls_acc = accuracy_score(y_cls_test, y_cls_pred)
    print(f"\nTest classification accuracy: {cls_acc:.4f}")
    print("\nClassification report (direction):")
    print(classification_report(y_cls_test, y_cls_pred, digits=4))

    # Regression metrics
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    print(f"\nTest regression MAE: {mae:.5f} (~{mae * 100:.2f}% move)")
    print(f"Test regression R²: {r2:.4f}")

    # Save model + metadata
    os.makedirs("models", exist_ok=True)
    print(f"\nSaving DL model to {DL_MODEL_DIR} ...")
    model.save(DL_MODEL_DIR)

    meta = {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "window": window,
        "ticker": ticker,
        "period": period,
        "interval": interval,
    }
    joblib.dump(meta, DL_META_PATH)
    print(f"Saved metadata to {DL_META_PATH}")


if __name__ == "__main__":
    # You can change ticker or interval here if you like
    train_dl_model(
        ticker="AAPL",
        period="10y",
        interval="1d",   # "1d" for daily model; you could build a separate script with "60m"
        window=30,
        batch_size=64,
        epochs=50,
    )
