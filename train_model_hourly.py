import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    mean_absolute_error,
    r2_score,
)
import joblib
import os

HOURLY_MODEL_PATH = "models/saved_model_hourly.pkl"


def download_data_hourly(
    ticker: str = "AAPL", period: str = "60d", interval: str = "60m"
) -> pd.DataFrame:
    """
    Download intraday OHLCV data for a given ticker using yfinance.
    For hourly bars, yfinance typically allows up to ~60 days of data.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Same RSI logic as daily model, just used on hourly closes.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def engineer_features_hourly(df: pd.DataFrame):
    """
    Create features for the HOURLY ML model on intraday data.

    We keep the SAME column names as the daily model (return_1d, ma_5, etc.)
    so the Streamlit app can reuse the same feature names.
    Here "d" really means "1 bar" (1 hour).
    """
    df = df.copy()

    # Ensure Close is 1D Series
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close)

    # Returns over multiple hours (1, 2, 5 bars)
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    # Moving averages over hours
    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_10"] = close.rolling(window=10).mean()
    df["ma_20"] = close.rolling(window=20).mean()

    # Ratios of moving averages (trend strength)
    df["ma_5_20_ratio"] = df["ma_5"] / (df["ma_20"] + 1e-9)
    df["ma_10_20_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # Volatility of hourly returns
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    # RSI on hourly close
    df["rsi_14"] = compute_rsi(close, period=14)

    # Next-hour close and targets
    next_close = close.shift(-1)
    df["next_close"] = next_close
    df["next_return"] = (next_close - close) / close  # regression target (next hour)
    df["target_up"] = (df["next_return"] > 0).astype(int)  # classification target

    # Drop rows with NaNs from rolling/shift
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

    X = df[feature_cols]
    y_cls = df["target_up"]
    y_reg = df["next_return"]

    return X, y_cls, y_reg, feature_cols


def train_hourly_model(ticker: str = "AAPL"):
    print(f"Downloading HOURLY data for {ticker}...")
    df = download_data_hourly(ticker=ticker)

    print("Engineering HOURLY features...")
    X, y_cls, y_reg, feature_cols = engineer_features_hourly(df)

    print("Splitting train/test (hourly)...")
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_cls,
        y_reg,
        test_size=0.2,
        shuffle=False,  # keep time order
    )

    # ---- Classification model (direction UP/DOWN) ----
    print("\nTraining RandomForestClassifier (HOURLY direction)...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_cls_train)

    print("\nEvaluating HOURLY classifier...")
    y_cls_pred = clf.predict(X_test)
    cls_acc = accuracy_score(y_cls_test, y_cls_pred)
    print(f"HOURLY classification accuracy: {cls_acc:.4f}")
    print("Classification report:")
    print(classification_report(y_cls_test, y_cls_pred))

    # ---- Regression model (magnitude of move) ----
    print("\nTraining RandomForestRegressor (next-hour return)...")
    reg = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_reg_train)

    print("\nEvaluating HOURLY regressor...")
    y_reg_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    print(f"HOURLY MAE: {mae:.5f} (~{mae * 100:.2f}% move)")
    print(f"HOURLY R²: {r2:.4f}")

    # ---- Save both models + feature columns ----
    os.makedirs(os.path.dirname(HOURLY_MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "clf": clf,  # classifier
            "reg": reg,  # regressor
            "features": feature_cols,
        },
        HOURLY_MODEL_PATH,
    )
    print(f"\nHOURLY models saved to {HOURLY_MODEL_PATH}")


if __name__ == "__main__":
    # Train HOURLY model for 1 ticker (AAPL by default)
    train_hourly_model(ticker="AAPL")
