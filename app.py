import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

MODEL_PATH = "models/saved_model.pkl"


# ---------- RSI CALCULATION (SAFE VERSION) ----------
def compute_rsi(series, period=14):
    """
    Compute RSI using 1D-safe pandas operations.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ---------- FEATURE ENGINEERING FOR APP ----------
def engineer_features_for_app(df):
    df = df.copy()

    # Ensure Close is 1D
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close)

    df["return_1d"] = close.pct_change()

    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_10"] = close.rolling(window=10).mean()
    df["ma_20"] = close.rolling(window=20).mean()

    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    df["rsi_14"] = compute_rsi(close)

    df.dropna(inplace=True)
    return df


# ---------- LOAD MODEL ----------
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run train_model.py first.")
        return None, None

    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


# ---------- PATTERN DETECTION ----------
def get_pattern_label(preds, closes):
    if len(preds) < 5:
        return "Not enough data"

    last5 = np.array(preds[-5:])
    ones_ratio = last5.mean()

    last_prices = closes[-10:]
    if len(last_prices) < 2:
        return "Not enough price history"

    x = np.arange(len(last_prices))
    slope = np.polyfit(x, last_prices, 1)[0]

    if ones_ratio > 0.6 and slope > 0:
        return "UPTREND / BULLISH"
    elif ones_ratio < 0.4 and slope < 0:
        return "DOWNTREND / BEARISH"
    else:
        return "SIDEWAYS / NEUTRAL"


# ---------- STREAMLIT APP ----------
def main():
    st.title("📈 Stock Pattern Detector (ML-Powered)")
    st.write("Predict short-term market direction using a trained ML model.")

    model, feature_cols = load_model()
    if model is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL")
    with col2:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=1)

    if st.button("🔍 Analyze"):
        with st.spinner("Fetching and analyzing data..."):
            df = yf.download(ticker, period=period, interval="1d")

            if df.empty:
                st.error("No data found for this ticker.")
                return

            df_feat = engineer_features_for_app(df)

            if df_feat.empty:
                st.error("Not enough data after feature engineering.")
                return

            # Build feature matrix
            X_latest = df_feat[feature_cols]

            preds = model.predict(X_latest)
            proba = model.predict_proba(X_latest)[:, 1]

            df_feat["pred_up"] = preds
            df_feat["prob_up"] = proba

            pattern = get_pattern_label(preds, df_feat["Close"].values)

            st.subheader(f"Detected Pattern for {ticker}:")
            st.markdown(f"### 🧭 {pattern}")

            # Latest prediction
            latest_row = df_feat.iloc[-1]
            latest_date = latest_row.name.date()

            st.subheader("Latest Prediction")
            st.write(f"- Date: `{latest_date}`")
            st.write(f"- Close Price: `{float(latest_row['Close']):.2f}`")

            # Use prob_up to decide label, so no ambiguous Series truth value
            prob_up = float(latest_row["prob_up"])
            direction_label = "UP" if prob_up >= 0.5 else "DOWN / FLAT"
            st.write(f"- Model expects: **{direction_label}**")
            st.write(f"- Probability of UP: `{prob_up:.2%}`")

            # Plot
            st.subheader("Price + Moving Averages")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_feat.index, df_feat["Close"], label="Close Price")
            ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
            ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
            ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Recent Predictions (Last 10 Days)")
            st.dataframe(
                df_feat[["Close", "rsi_14", "pred_up", "prob_up"]].tail(10)
            )

    st.caption("This app is for educational use only — not financial advice.")


if __name__ == "__main__":
    main()
