import os
import yfinance as yf
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)

MODELS_DIR = "models"
DAILY_MODEL_PATH = os.path.join(MODELS_DIR, "saved_model.pkl")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _to_series(x) -> pd.Series:
    # yfinance sometimes returns DataFrame columns (MultiIndex-ish). Force 1D series.
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.Series(x)


def engineer_features(df: pd.DataFrame):
    """
    Must match:
      - train_model_dl.py engineer_features()
      - app.py engineer_features_for_app()
    """
    df = df.copy()

    close = _to_series(df["Close"])
    vol = _to_series(df["Volume"]) if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    # Returns
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    # MAs
    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_10"] = close.rolling(window=10).mean()
    df["ma_20"] = close.rolling(window=20).mean()

    # MA ratios
    df["ma_5_20_ratio"] = df["ma_5"] / (df["ma_20"] + 1e-9)
    df["ma_10_20_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # Volatility
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    # RSI
    df["rsi_14"] = compute_rsi(close, period=14)

    # Volume features (safe)
    if "Volume" in df.columns:
        df["volume_change_1"] = vol.pct_change()
        df["volume_ma_5"] = vol.rolling(window=5).mean()
        df["volume_ma_20"] = vol.rolling(window=20).mean()
        df["volume_relative"] = vol / (df["volume_ma_20"] + 1e-9)
    else:
        df["volume_change_1"] = np.nan
        df["volume_ma_5"] = np.nan
        df["volume_ma_20"] = np.nan
        df["volume_relative"] = np.nan

    # ATR + Bollinger
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            _to_series(df["High"]) - _to_series(df["Low"]),
            (_to_series(df["High"]) - prev_close).abs(),
            (_to_series(df["Low"]) - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    rolling_std_20 = close.rolling(window=20).std()
    df["bb_upper"] = df["ma_20"] + 2 * rolling_std_20
    df["bb_lower"] = df["ma_20"] - 2 * rolling_std_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["ma_20"] + 1e-9)
    df["bb_percent_b"] = (close - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]) + 1e-9)

    # Targets
    df["next_close"] = close.shift(-1)
    df["next_return"] = (df["next_close"] - close) / (close + 1e-9)
    df["target_up"] = (df["next_return"] > 0).astype(int)

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
        "volume_change_1",
        "volume_ma_5",
        "volume_ma_20",
        "volume_relative",
        "atr_14",
        "bb_width",
        "bb_percent_b",
    ]

    X = df[feature_cols].values
    y_class = df["target_up"].values
    y_reg = df["next_return"].values
    return X, y_class, y_reg, feature_cols


def download_data(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    return df


def train_model(ticker: str = "AAPL"):
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = download_data(ticker=ticker, period="10y", interval="1d")
    print("Engineering features...")
    X, y_class, y_reg, feature_cols = engineer_features(df)

    print("Splitting train/test...")
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.2,
        random_state=42,
        shuffle=False,
    )

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train_cls)

    print("Training RandomForestRegressor...")
    reg = RandomForestRegressor(
        n_estimators=500,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train_reg)

    print("Evaluating (classification)...")
    y_pred_cls = clf.predict(X_test)
    acc = accuracy_score(y_test_cls, y_pred_cls)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test_cls, y_pred_cls))

    print("Evaluating (regression)...")
    y_pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"Test MAE (next_return): {mae:.6f}")
    print(f"Test R² (next_return): {r2:.4f}")

    bundle = {"clf": clf, "reg": reg, "features": feature_cols}
    joblib.dump(bundle, DAILY_MODEL_PATH)
    print(f"Model saved to {DAILY_MODEL_PATH}")


if __name__ == "__main__":
    train_model(ticker="AAPL")
