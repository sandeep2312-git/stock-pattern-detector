import os
import argparse
import numpy as np
import yfinance as yf
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

from feature_engineering import engineer_features, add_market_context, get_feature_cols

MODELS_DIR = "models"


def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    return df


def train_rf(
    ticker: str,
    period: str,
    interval: str,
    out_path: str,
    include_market: bool = True,
):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Downloading {ticker} ({period}, {interval})...")
    df = download_data(ticker, period=period, interval=interval)

    print("Engineering features...")
    df_feat = engineer_features(df, include_targets=True)

    if include_market:
        print("Adding market context (SPY/QQQ/VIX)...")
        df_feat = add_market_context(df_feat, period=period, interval=interval)

    feature_cols = get_feature_cols(include_market=include_market)

    # Only keep cols that exist (VIX might be missing on some intraday intervals)
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    df_feat = df_feat.dropna(subset=feature_cols + ["target_up", "next_return"]).copy()

    X = df_feat[feature_cols].values
    y_cls = df_feat["target_up"].values
    y_reg = df_feat["next_return"].values

    # time-series split
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_cls, y_test_cls = y_cls[:split_idx], y_cls[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]

    print("Training classifier...")
    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        min_samples_split=3,
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train_cls)

    print("Training regressor...")
    reg = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        min_samples_split=3,
        min_samples_leaf=2,
    )
    reg.fit(X_train, y_train_reg)

    print("Evaluating classifier...")
    pred_cls = clf.predict(X_test)
    acc = accuracy_score(y_test_cls, pred_cls)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test_cls, pred_cls))

    print("Evaluating regressor...")
    pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, pred_reg)
    r2 = r2_score(y_test_reg, pred_reg)
    print(f"MAE(next_return): {mae:.6f}")
    print(f"R²(next_return): {r2:.4f}")

    bundle = {
        "clf": clf,
        "reg": reg,
        "features": feature_cols,
        "meta": {"ticker": ticker, "period": period, "interval": interval, "include_market": include_market},
    }
    joblib.dump(bundle, out_path)
    print(f"Saved model -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--interval", default="1d", help="Use 1d for daily, 60m for hourly")
    parser.add_argument("--out", default=os.path.join(MODELS_DIR, "saved_model.pkl"))
    parser.add_argument("--no-market", action="store_true")
    args = parser.parse_args()

    train_rf(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        out_path=args.out,
        include_market=not args.no_market,
    )
