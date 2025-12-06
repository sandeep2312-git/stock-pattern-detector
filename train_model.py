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

DAILY_MODEL_PATH = "models/saved_model.pkl"


def download_data(
    ticker: str = "AAPL", period: str = "10y", interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker using yfinance.
    Daily bars by default.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df


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
    Create features for the DAILY ML model based on historical price data.

    This MUST match engineer_features_for_app() in app.py.
    """
    df = df.copy()

    # Ensure Close is 1D Series
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close)

    # Daily and multi-day returns (1, 2, 5 bars)
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    # Moving averages
    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_10"] = close.rolling(window=10).mean()
    df["ma_20"] = close.rolling(window=20).mean()

    # Ratios of moving averages (trend strength)
    df["ma_5_20_ratio"] = df["ma_5"] / (df["ma_20"] + 1e-9)
    df["ma_10_20_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # Volatility
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    # RSI
    df["rsi_14"] = compute_rsi(close, period=14)

    # Next-day close and targets
    next_close = close.shift(-1)
    df["next_close"] = next_close
    df["next_return"] = (next_close - close) / close  # regression target
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


def train_model(ticker: str = "AAPL"):
    print(f"Downloading DAILY data for {ticker}...")
    df = download_data(ticker=ticker)

    print("Engineering DAILY features...")
    X, y_cls, y_reg, feature_cols = engineer_features(df)

    print("Splitting train/test (daily)...")
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_cls,
        y_reg,
        test_size=0.2,
        shuffle=False,  # keep time order
    )

    # ---- Classification model (direction UP/DOWN) ----
    print("\nTraining RandomForestClassifier (DAILY direction)...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_cls_train)

    print("\nEvaluating DAILY classifier...")
    y_cls_pred = clf.predict(X_test)
    cls_acc = accuracy_score(y_cls_test, y_cls_pred)
    print(f"DAILY classification accuracy: {cls_acc:.4f}")
    print("Classification report:")
    print(classification_report(y_cls_test, y_cls_pred))

    # ---- Regression model (magnitude of move) ----
    print("\nTraining RandomForestRegressor (next-day return)...")
    reg = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_reg_train)

    print("\nEvaluating DAILY regressor...")
    y_reg_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    print(f"DAILY MAE: {mae:.5f} (~{mae * 100:.2f}% move)")
    print(f"DAILY R²: {r2:.4f}")

    # ---- Save both models + feature columns ----
    os.makedirs(os.path.dirname(DAILY_MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "clf": clf,  # classifier
            "reg": reg,  # regressor
            "features": feature_cols,
        },
        DAILY_MODEL_PATH,
    )
    print(f"\nDAILY models saved to {DAILY_MODEL_PATH}")


if __name__ == "__main__":
    # Train DAILY model for 1 ticker (AAPL by default)
    train_model(ticker="AAPL")
