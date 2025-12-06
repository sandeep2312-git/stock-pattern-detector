import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

DAILY_MODEL_PATH = "models/saved_model.pkl"
HOURLY_MODEL_PATH = "models/saved_model_hourly.pkl"


# ---------- RSI CALCULATION ----------
def compute_rsi(series, period=14):
    """
    Compute RSI using 1D-safe pandas operations.
    Assumes `series` is a pandas Series.
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
def engineer_features_for_app(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the SAME features as used during training in:
    - train_model.py (daily)
    - train_model_hourly.py (hourly)

    Here, "d" in return_1d, etc., just means "1 bar"
    (1 day for daily mode, 1 hour for hourly mode).
    """
    df = df.copy()

    # Ensure Close is 1D
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close)

    # Returns
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

    # Drop rows with NaNs created by rolling/shift
    df.dropna(inplace=True)

    return df


# ---------- MODEL LOADING ----------
def load_model(mode: str = "daily"):
    """
    mode: "daily" or "hourly"
    """
    model_path = DAILY_MODEL_PATH if mode == "daily" else HOURLY_MODEL_PATH

    if not os.path.exists(model_path):
        st.error(
            f"Model not found for mode={mode}. "
            f"Expected file: {model_path}. "
            "Please run the appropriate training script and commit the model."
        )
        return None, None, None

    bundle = joblib.load(model_path)

    clf = bundle.get("clf", None)
    reg = bundle.get("reg", None)
    feature_cols = bundle.get("features", None)

    if clf is None or reg is None or feature_cols is None:
        st.error(
            f"Model file {model_path} is missing required keys. "
            "Retrain with the latest train_model.py or train_model_hourly.py."
        )
        return None, None, None

    return clf, reg, feature_cols


# ---------- PATTERN DETECTION ----------
def get_pattern_label(preds, closes):
    """
    Use recent predictions + price slope to label the pattern.
    """
    if len(preds) < 5:
        return "Not enough data"

    last5 = np.array(preds[-5:])
    ones_ratio = last5.mean()  # avg of last 5 predicted up(1)/down(0)

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
    st.set_page_config(
        page_title="Stock Pattern Detector",
        page_icon="📈",
        layout="wide",
    )

    # Sidebar info / help
    st.sidebar.title("ℹ️ About this app")
    st.sidebar.markdown(
        """
This dashboard uses **two machine learning models** (per timeframe) trained on historical
prices to predict:

1. Whether the next bar is more likely to be **UP or DOWN/FLAT**  
2. By approximately **how much (%)** the price might move  

You can choose:
- **Daily mode** → predicts **next day**  
- **Hourly mode** → predicts **next 1-hour bar**  

**Key terms:**
- **RSI (14)**: momentum (0–100).  
  - >70 → overbought  
  - <30 → oversold  
- **Prob. of UP**: model’s confidence that the next bar closes higher.  
- **Predicted move**: expected % price change for the next bar.  
- **Pattern**: based on recent predictions + price slope  
  (Uptrend / Downtrend / Sideways).

⚠️ This is **educational only**, **not financial advice**.
"""
    )

    st.title("📈 Stock Pattern Detector (ML-Powered)")
    st.caption(
        "Short-term directional signal + predicted % move based on historical price patterns."
    )

    # Timeframe toggle
    timeframe = st.radio(
        "Prediction timeframe",
        ["Daily (next day)", "Hourly (next hour)"],
        index=0,
        horizontal=True,
    )
    mode = "daily" if timeframe.startswith("Daily") else "hourly"
    horizon_text = "next day" if mode == "daily" else "next hour"

    clf, reg, feature_cols = load_model(mode=mode)
    if clf is None or reg is None:
        return

    # --- Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        common_tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NFLX",
            "TSLA",
            "NVDA",
            "AMD",
            "INTC",
            "QCOM",
            "SPY",
            "QQQ",
            "V",
            "MA",
            "JPM",
            "BAC",
            "WMT",
            "TGT",
            "DIS",
            "NKE",
            "XOM",
            "CVX",
            "TSM",
            "ADBE",
            "CRM",
            "PYPL",
            "CSCO",
            "ORCL",
        ]
        selected_ticker = st.selectbox(
            "Select ticker (scrollable list):",
            options=common_tickers,
            index=0,
        )
        custom_ticker = st.text_input("Or type a custom ticker:", "")
        ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_ticker

    with col2:
        period = st.selectbox(
            "Historical period (for DAILY mode):",
            ["6mo", "1y", "2y", "5y"],
            index=1,
        )

    st.markdown("---")

    if st.button("🔍 Analyze"):
        with st.spinner(
            f"Fetching data for **{ticker}** and running predictions ({mode.upper()})..."
        ):
            # Choose interval and effective period based on mode
            if mode == "daily":
                yf_interval = "1d"
                yf_period = period  # "6mo", "1y", etc.
            else:
                yf_interval = "60m"
                # Intraday is limited; map longer choices to 30–60 days
                yf_period = "60d"  # good default for hourly

            df = yf.download(ticker, period=yf_period, interval=yf_interval)

            if df.empty:
                st.error(f"No data found for ticker `{ticker}`.")
                return

            df_feat = engineer_features_for_app(df)

            if df_feat.empty:
                st.error("Not enough data after feature engineering.")
                return

            # Try to build feature matrix that matches model
            try:
                X_latest = df_feat[feature_cols]
            except KeyError:
                missing_cols = [c for c in feature_cols if c not in df_feat.columns]
                st.error(
                    "Feature mismatch between training and app.\n\n"
                    f"Missing features in app: `{missing_cols}`.\n"
                    "Make sure engineer_features_for_app() matches both training scripts."
                )
                return

            # Classification: direction (UP/DOWN)
            preds = clf.predict(X_latest)
            proba = clf.predict_proba(X_latest)[:, 1]

            # Regression: magnitude of move (next bar return)
            pred_returns = reg.predict(X_latest)  # decimal, e.g., 0.0123 = +1.23%

            df_feat["pred_up"] = preds
            df_feat["prob_up"] = proba
            df_feat["pred_return"] = pred_returns

            closes = df_feat["Close"].values
            pattern = get_pattern_label(preds, closes)

            latest_row = df_feat.iloc[-1]
            latest_index = latest_row.name  # Timestamp (date or date+time)
            latest_close = float(latest_row["Close"])
            latest_rsi = float(latest_row["rsi_14"])
            latest_prob_up = float(latest_row["prob_up"])
            latest_pred_return = float(latest_row["pred_return"])
            direction_label = "UP" if latest_prob_up >= 0.5 else "DOWN / FLAT"

            # --- High-level pattern summary ---
            st.subheader(f"Market pattern for `{ticker}` ({mode.upper()})")
            if "UPTREND" in pattern:
                st.markdown(f"### 🟢 {pattern}")
            elif "DOWNTREND" in pattern:
                st.markdown(f"### 🔴 {pattern}")
            elif "SIDEWAYS" in pattern:
                st.markdown(f"### 🟡 {pattern}")
            else:
                st.markdown(f"### {pattern}")

            st.markdown(
                f"""
**Interpretation:**

- The model sees the recent pattern for **{ticker}** as: **{pattern}**  
- This is based on the last few bars of predicted up/down moves  
  **and** the slope of the recent price trend.
"""
            )

            # --- Key metrics row ---
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Last close", f"${latest_close:,.2f}")
            mcol2.metric(f"Prob. of UP ({horizon_text})", f"{latest_prob_up:.1%}")
            mcol3.metric("RSI (14)", f"{latest_rsi:.1f}")
            mcol4.metric(
                f"Predicted move ({horizon_text})",
                f"{latest_pred_return * 100:+.2f}%",
            )

            # --- Latest prediction details ---
            st.subheader(f"Latest prediction ({horizon_text} direction & move)")
            st.write(f"- **Bar time:** `{latest_index}`")
            st.write(f"- **Close price:** `${latest_close:,.2f}`")
            st.write(f"- **Model expectation:** **{direction_label}** for the {horizon_text}")
            st.write(f"- **Probability of UP:** `{latest_prob_up:.2%}`")
            st.write(
                f"- **Predicted price change:** `{latest_pred_return * 100:+.2f}%` ({horizon_text})"
            )

            st.markdown(
                f"""
**How to read this:**

- In **{mode.upper()} mode**, the model looks at the past bars and predicts the **next {horizon_text}**.  
- If **Prob. of UP** is closer to **1.0 / 100%**, the model is more confident
  that the next bar will close **higher** than the current one.
- The **Predicted move** is the expected percentage change in price from the current bar to the next.
"""
            )

            # --- Chart: Price & Moving Averages ---
            st.subheader("Price and moving averages")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_feat.index, df_feat["Close"], label="Close price")
            ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
            ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
            ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # --- Recent predictions table ---
            st.subheader("Recent model outputs (last 10 bars)")
            st.dataframe(
                df_feat[
                    ["Close", "rsi_14", "pred_up", "prob_up", "pred_return"]
                ]
                .tail(10)
                .rename(
                    columns={
                        "Close": "Close price",
                        "rsi_14": "RSI (14)",
                        "pred_up": "Predicted up? (1=yes)",
                        "prob_up": "Prob. of up",
                        "pred_return": "Pred. return (decimal)",
                    }
                )
            )

            # --- Explanation section ---
            with st.expander("📘 What do these numbers mean?"):
                st.markdown(
                    """
**RSI (14)**  
- Measures momentum on a 0–100 scale.  
- **> 70** → overbought (price may cool down or pull back)  
- **< 30** → oversold (price may bounce or recover)

**Predicted up? (1 / 0)**  
- `1` → model predicts next bar close **higher**  
- `0` → model predicts next bar close **lower or flat**

**Prob. of up**  
- A probability between 0 and 1 (0–100%)  
- Closer to 1 → stronger bullish bias  
- Closer to 0 → stronger bearish/flat bias

**Pred. return (decimal)**  
- Example: `0.0123` ≈ `+1.23%` expected price increase for the next bar  
- `-0.0100` ≈ `-1.00%` expected price decrease for the next bar

**Pattern label**  
- **UPTREND / BULLISH** → recent predictions mostly up, price slope rising  
- **DOWNTREND / BEARISH** → recent predictions mostly down, price slope falling  
- **SIDEWAYS / NEUTRAL** → mixed predictions, flat-ish price action
"""
                )

    st.markdown("---")
    st.caption(
        "Built for learning and portfolio demonstration. Not investment advice."
    )


if __name__ == "__main__":
    main()
