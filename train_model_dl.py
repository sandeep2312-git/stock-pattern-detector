import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

MODELS_DIR = "models"
DL_MODEL_DIR = os.path.join(MODELS_DIR, "dl_model")
DL_META_PATH = os.path.join(MODELS_DIR, "dl_meta.pkl")


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
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.Series(x)


def engineer_features(df: pd.DataFrame):
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

    # Volume (safe)
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
    return df, feature_cols


def download_data(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    return df


def build_sequences(X_scaled, y_dir, y_ret, window: int):
    X_seq, y_seq_dir, y_seq_ret = [], [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i - window : i])
        y_seq_dir.append(y_dir[i])
        y_seq_ret.append(y_ret[i])
    return np.array(X_seq), np.array(y_seq_dir), np.array(y_seq_ret)


def build_lstm_model(window: int, num_features: int) -> tf.keras.Model:
    inputs = Input(shape=(window, num_features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)

    dir_out = Dense(1, activation="sigmoid", name="direction")(x)
    mag_out = Dense(1, activation="linear", name="magnitude")(x)

    model = Model(inputs=inputs, outputs=[dir_out, mag_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"direction": "binary_crossentropy", "magnitude": "mse"},
        loss_weights={"direction": 1.0, "magnitude": 0.5},
        metrics={"direction": ["accuracy"], "magnitude": ["mae"]},
    )
    return model


def train_lstm_model(ticker: str = "AAPL", window: int = 30):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DL_MODEL_DIR, exist_ok=True)

    df = download_data(ticker=ticker, period="10y", interval="1d")
    print("Engineering features...")
    df_feat, feature_cols = engineer_features(df)

    X_raw = df_feat[feature_cols].values
    y_dir = df_feat["target_up"].values
    y_ret = df_feat["next_return"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq_dir, y_seq_ret = build_sequences(X_scaled, y_dir, y_ret, window=window)
    if X_seq.shape[0] == 0:
        raise ValueError("Not enough data to create sequences. Try smaller window.")

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train_dir, y_test_dir = y_seq_dir[:split_idx], y_seq_dir[split_idx:]
    y_train_ret, y_test_ret = y_seq_ret[:split_idx], y_seq_ret[split_idx:]

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = build_lstm_model(window=window, num_features=X_seq.shape[2])

    early_stop = EarlyStopping(
        monitor="val_direction_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        {"direction": y_train_dir, "magnitude": y_train_ret},
        validation_data=(X_test, {"direction": y_test_dir, "magnitude": y_test_ret}),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate
    dir_probs_test, mag_preds_test = model.predict(X_test)
    dir_probs_test = dir_probs_test.ravel()
    mag_preds_test = mag_preds_test.ravel()
    dir_pred_test = (dir_probs_test >= 0.5).astype(int)

    acc = accuracy_score(y_test_dir, dir_pred_test)
    print(f"DL Test accuracy (direction): {acc:.4f}")
    print(classification_report(y_test_dir, dir_pred_test))

    mae = mean_absolute_error(y_test_ret, mag_preds_test)
    r2 = r2_score(y_test_ret, mag_preds_test)
    print(f"DL Test MAE (next_return): {mae:.6f}")
    print(f"DL Test R² (next_return): {r2:.4f}")

    # Save
    model.save(DL_MODEL_DIR)
    meta = {"feature_cols": feature_cols, "scaler": scaler, "window": window}
    joblib.dump(meta, DL_META_PATH)

    print(f"LSTM model saved to {DL_MODEL_DIR}")
    print(f"Metadata saved to {DL_META_PATH}")


if __name__ == "__main__":
    train_lstm_model(ticker="AAPL", window=30)
