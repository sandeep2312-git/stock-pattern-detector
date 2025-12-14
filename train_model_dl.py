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

from feature_engineering import engineer_features, add_market_context, get_feature_cols

MODELS_DIR = "models"
DL_MODEL_DIR = os.path.join(MODELS_DIR, "dl_model")
DL_META_PATH = os.path.join(MODELS_DIR, "dl_meta.pkl")


def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
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


def train_lstm_model(
    ticker: str = "AAPL",
    period: str = "10y",
    interval: str = "1d",
    window: int = 30,
    include_market: bool = True,
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DL_MODEL_DIR, exist_ok=True)

    print(f"Downloading {ticker} ({period}, {interval})...")
    df = download_data(ticker=ticker, period=period, interval=interval)

    print("Engineering features...")
    df_feat = engineer_features(df, include_targets=True)

    if include_market:
        print("Adding market context (SPY/QQQ/VIX)...")
        df_feat = add_market_context(df_feat, period=period, interval=interval)

    feature_cols = get_feature_cols(include_market=include_market)
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    df_feat = df_feat.dropna(subset=feature_cols + ["target_up", "next_return"]).copy()

    X_raw = df_feat[feature_cols].values
    y_dir = df_feat["target_up"].values
    y_ret = df_feat["next_return"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq_dir, y_seq_ret = build_sequences(X_scaled, y_dir, y_ret, window=window)
    if X_seq.shape[0] == 0:
        raise ValueError("Not enough data to create sequences. Try smaller window or larger period.")

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train_dir, y_test_dir = y_seq_dir[:split_idx], y_seq_dir[split_idx:]
    y_train_ret, y_test_ret = y_seq_ret[:split_idx], y_seq_ret[split_idx:]

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = build_lstm_model(window=window, num_features=X_seq.shape[2])

    early_stop = EarlyStopping(
        monitor="val_direction_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        {"direction": y_train_dir, "magnitude": y_train_ret},
        validation_data=(X_test, {"direction": y_test_dir, "magnitude": y_test_ret}),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

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

    model.save(DL_MODEL_DIR)
    meta = {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "window": window,
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "include_market": include_market,
    }
    joblib.dump(meta, DL_META_PATH)

    print(f"Saved DL model -> {DL_MODEL_DIR}")
    print(f"Saved DL meta  -> {DL_META_PATH}")


if __name__ == "__main__":
    train_lstm_model(ticker="AAPL", period="10y", interval="1d", window=30, include_market=True)
