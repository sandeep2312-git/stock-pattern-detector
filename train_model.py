import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

MODEL_PATH = "models/saved_model.pkl"


def download_data(ticker="AAPL", period="5y", interval="1d"):
    """
    Download historical OHLCV data for a given ticker.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df


def compute_rsi(series, period=14):
    """
    Compute RSI (Relative Strength Index) for a price series.
    Assumes `series` is already a pandas Series (like df["Close"]).
    """
    # Price differences
    delta = series.diff()

    # Gains (up moves) and losses (down moves)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Rolling averages
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # RS and RSI
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def engineer_features(df):
    """
    Create features for the ML model based on historical price data.
    """
    df = df.copy()

    # Daily returns
    df["return_1d"] = df["Close"].pct_change()

    # Moving averages
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # Volatility (rolling std of returns)
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    # RSI
    df["rsi_14"] = compute_rsi(df["Close"], period=14)

    # Next-day close and label (1 = up, 0 = down/flat)
    df["next_close"] = df["Close"].shift(-1)
    df["next_return"] = (df["next_close"] - df["Close"]) / df["Close"]
    df["target_up"] = (df["next_return"] > 0).astype(int)

    # Drop rows with NaN from rolling windows & shift
    df.dropna(inplace=True)

    feature_cols = [
        "return_1d",
        "ma_5", "ma_10", "ma_20",
        "vol_5", "vol_10",
        "rsi_14",
    ]

    X = df[feature_cols]
    y = df["target_up"]

    return X, y, feature_cols


def train_model(ticker="AAPL"):
    print(f"Downloading data for {ticker}...")
    df = download_data(ticker=ticker)

    print("Engineering features...")
    X, y, feature_cols = engineer_features(df)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Save model + feature columns together
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"model": clf, "features": feature_cols},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model(ticker="AAPL")
