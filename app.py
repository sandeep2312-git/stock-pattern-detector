import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

MODEL_PATH = "models/saved_model.pkl"


# ---------- Helpers (must match train_model.py) ----------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)

    gain_rolling = gain_series.rolling(window=period).mean()
    loss_rolling = loss_series.rolling(window=period).mean()

    rs = gain_rolling / (loss_rolling + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features_for_app(df):
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()
    df["rsi_14"] = compute_rsi(df["Close"], period=14)
    df.dropna(inplace=True)
    return df


def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train_model.py first.")
        return None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def get_pattern_label(latest_preds, closes):
    """
    Simple heuristic:
    - If last 5 predictions mostly 1 and price slope > 0 -> Uptrend
    - If last 5 preds mostly 0 and price slope < 0 -> Downtrend
    - Else -> Sideways / Uncertain
    """
    if len(latest_preds) < 5:
        return "Not enough data for pattern"

    last5 = np.array(latest_preds[-5:])
    ones_ratio = last5.mean()

    # Price slope (last N closes)
    last_prices = closes[-10:]
    if len(last_prices) < 2:
        return "Not enough price history"

    x = np.arange(len(last_prices))
    slope = np.polyfit(x, last_prices, 1)[0]

    if ones_ratio > 0.6 and slope > 0:
        return "UPTREND / BULLISH BIAS"
    elif ones_ratio < 0.4 and slope < 0:
        return "DOWNTREND / BEARISH BIAS"
    else:
        return "SIDEWAYS / UNCERTAIN"


# ---------- Streamlit App ----------

def main():
    st.title("📈 Real-Time Stock Pattern Detector (ML + Streamlit)")
    st.write(
        "This app uses a machine learning model trained on historical prices "
        "to detect short-term bullish/bearish patterns for a given stock."
    )

    model, feature_cols = load_model()
    if model is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker symbol", value="AAPL")
    with col2:
        period = st.selectbox(
            "Historical period",
            options=["6mo", "1y", "2y", "5y"],
            index=1,
        )

    if st.button("🔍 Analyze"):
        with st.spinner("Downloading and analyzing data..."):
            df = yf.download(ticker, period=period, interval="1d")
            if df.empty:
                st.error("No data found for this ticker.")
                return

            df_feat = engineer_features_for_app(df)
            if df_feat.empty:
                st.error("Not enough data to create features.")
                return

            # Use same feature columns as training
            X_latest = df_feat[feature_cols]
            preds = model.predict(X_latest)
            proba = model.predict_proba(X_latest)[:, 1]  # prob of 'up'

            df_feat["pred_up"] = preds
            df_feat["prob_up"] = proba

            # Pattern label based on last few days
            pattern_label = get_pattern_label(
                latest_preds=preds,
                closes=df_feat["Close"].values,
            )

            st.subheader(f"Detected pattern for {ticker}:")
            st.markdown(f"### 🧠 {pattern_label}")

            # Latest prediction details
            latest_row = df_feat.iloc[-1]
            latest_date = latest_row.name.date()

            st.write("**Latest day prediction:**")
            st.write(f"- Date: `{latest_date}`")
            st.write(f"- Close Price: `{latest_row['Close']:.2f}`")
            st.write(
                f"- Model expects **{'UP' if latest_row['pred_up'] == 1 else 'DOWN/FLAT'}** next day"
            )
            st.write(
                f"- Probability of UP: `{latest_row['prob_up']:.2%}`"
            )

            # Plot price + moving averages
            st.subheader("Price & Moving Averages")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_feat.index, df_feat["Close"], label="Close")
            ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
            ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
            ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Show recent table with predictions
            st.subheader("Recent Predictions (Last 10 Days)")
            st.dataframe(
                df_feat[["Close", "rsi_14", "pred_up", "prob_up"]].tail(10)
            )

    st.markdown("---")
    st.caption(
        "This project is for educational purposes only and does not constitute "
        "financial advice."
    )


if __name__ == "__main__":
    main()
