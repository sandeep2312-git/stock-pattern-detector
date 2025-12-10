import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import load_model


# --------- Paths ----------
DAILY_MODEL_PATH = "models/saved_model.pkl"
HOURLY_MODEL_PATH = "models/saved_model_hourly.pkl"

DL_MODEL_DIR = "models/dl_model"
DL_META_PATH = "models/dl_meta.pkl"


# --------- Indicators ----------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def engineer_features_for_app(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close)

    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    df["ma_5"] = close.rolling(window=5).mean()
    df["ma_10"] = close.rolling(window=10).mean()
    df["ma_20"] = close.rolling(window=20).mean()

    df["ma_5_20_ratio"] = df["ma_5"] / (df["ma_20"] + 1e-9)
    df["ma_10_20_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    df["rsi_14"] = compute_rsi(close, period=14)

    df.dropna(inplace=True)

    return df


def get_pattern_label(preds, closes) -> str:
    preds = np.array(preds)
    closes = np.array(closes)

    if len(preds) < 5 or len(closes) < 10:
        return "Not enough data"

    last5 = preds[-5:]
    ones_ratio = last5.mean()

    last_prices = closes[-10:]
    x = np.arange(len(last_prices))
    slope = np.polyfit(x, last_prices, 1)[0]

    if ones_ratio > 0.6 and slope > 0:
        return "UPTREND / BULLISH"
    elif ones_ratio < 0.4 and slope < 0:
        return "DOWNTREND / BEARISH"
    else:
        return "SIDEWAYS / NEUTRAL"


# --------- Model loading ----------
def load_rf_model(mode: str = "daily"):
    model_path = DAILY_MODEL_PATH if mode == "daily" else HOURLY_MODEL_PATH

    if not os.path.exists(model_path):
        st.error(
            f"RandomForest model not found for mode={mode}. "
            f"Expected file: {model_path}. Run the training script first."
        )
        return None, None, None

    bundle = joblib.load(model_path)
    clf = bundle.get("clf")
    reg = bundle.get("reg")
    feature_cols = bundle.get("features")

    if clf is None or reg is None or feature_cols is None:
        st.error(
            f"Model file {model_path} is missing required keys. "
            "Retrain with the latest training script."
        )
        return None, None, None

    return clf, reg, feature_cols


@st.cache_resource
def load_dl_model():
    if (not os.path.exists(DL_MODEL_DIR)) or (not os.path.exists(DL_META_PATH)):
        st.error(
            f"Deep learning model not found. "
            f"Expected: {DL_MODEL_DIR} and {DL_META_PATH}. "
            "Run train_model_dl.py first and push the files to GitHub."
        )
        return None, None

    model = load_model(DL_MODEL_DIR)
    meta = joblib.load(DL_META_PATH)

    return model, meta


def build_sequences_for_app(X_scaled: np.ndarray, window: int) -> np.ndarray:
    seqs = []
    for i in range(window, len(X_scaled)):
        seqs.append(X_scaled[i - window : i])
    if not seqs:
        return np.empty((0, window, X_scaled.shape[1]))
    return np.array(seqs)


# --------- Streamlit app ----------
def main():
    st.set_page_config(
        page_title="Stock Pattern Detector",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Stock Pattern Detector – ML + Deep Learning")
    st.caption(
        "Short-term stock direction (UP/DOWN) and expected % move using "
        "RandomForest (daily & hourly) and LSTM (daily)."
    )

    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown(
        """
**Models**

- RandomForest (classic ML) – daily & hourly  
- LSTM (deep learning) – daily only  

**Targets**

- Direction: UP (1) vs DOWN/FLAT (0)  
- Magnitude: next bar % move  

**Features**

- Returns (1, 2, 5 bars)  
- Moving averages (5 / 10 / 20)  
- MA ratios  
- Volatility  
- RSI(14)  

> Educational only. Not financial advice.
"""
    )

    col_top1, col_top2 = st.columns([1.2, 1])

    with col_top1:
        timeframe = st.radio(
            "Prediction timeframe",
            ["Daily (next day)", "Hourly (next hour)"],
            index=0,
            horizontal=True,
        )
        mode = "daily" if timeframe.startswith("Daily") else "hourly"
        horizon_text = "next day" if mode == "daily" else "next hour"

    with col_top2:
        algo = st.radio(
            "Model type",
            ["RandomForest (Classic ML)", "LSTM (Deep Learning)"],
            index=0,
            horizontal=True,
        )

    if algo.startswith("LSTM") and mode == "hourly":
        st.warning("LSTM (Deep Learning) is currently available only in DAILY mode.")
        mode = "daily"
        horizon_text = "next day"

    col_in1, col_in2 = st.columns([1.2, 1])

    with col_in1:
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
        selected_ticker = st.selectbox("Select ticker:", options=common_tickers, index=0)
        custom_ticker = st.text_input("Or type a custom ticker:", "")
        ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_ticker

    with col_in2:
        if mode == "daily":
            period = st.selectbox(
                "Historical period:",
                ["6mo", "1y", "2y", "5y", "10y"],
                index=2,
            )
        else:
            period = "60d"
            st.write("Historical period (hourly): `60d` (intraday limit)")

    st.markdown("---")

    if st.button("🔍 Run analysis", type="primary"):
        with st.spinner(
            f"Fetching data for **{ticker}** and running predictions "
            f"({mode.upper()} • {algo.split()[0]})..."
        ):
            if mode == "daily":
                yf_interval = "1d"
                yf_period = period
            else:
                yf_interval = "60m"
                yf_period = "60d"

            df = yf.download(ticker, period=yf_period, interval=yf_interval)

            if df.empty:
                st.error(f"No data found for ticker `{ticker}`.")
                return

            df_feat = engineer_features_for_app(df)
            if df_feat.empty:
                st.error("Not enough data after feature engineering.")
                return

            tab_summary, tab_charts, tab_signals, tab_model = st.tabs(
                ["📌 Summary", "📊 Charts", "📋 Recent signals", "🧠 Model details"]
            )

            # --------- RandomForest path ----------
            if algo.startswith("RandomForest"):
                clf, reg, feature_cols = load_rf_model(mode=mode)
                if clf is None or reg is None:
                    return

                try:
                    X_all = df_feat[feature_cols]
                except KeyError:
                    missing_cols = [c for c in feature_cols if c not in df_feat.columns]
                    st.error(
                        "Feature mismatch between training and app.\n\n"
                        f"Missing features in app: `{missing_cols}`.\n"
                        "Make sure engineer_features_for_app() matches training scripts."
                    )
                    return

                preds = clf.predict(X_all)
                proba = clf.predict_proba(X_all)[:, 1]
                pred_returns = reg.predict(X_all)

                df_feat["pred_up"] = preds
                df_feat["prob_up"] = proba
                df_feat["pred_return"] = pred_returns

                closes = df_feat["Close"].values
                pattern = get_pattern_label(preds, closes)

                latest_row = df_feat.iloc[-1]
                latest_index = latest_row.name
                latest_close = float(latest_row["Close"])
                latest_rsi = float(latest_row["rsi_14"])
                latest_prob_up = float(latest_row["prob_up"])
                latest_pred_return = float(latest_row["pred_return"])
                direction_label = "UP" if latest_prob_up >= 0.5 else "DOWN / FLAT"

                with tab_summary:
                    st.subheader(f"Market pattern for `{ticker}` ({mode.upper()} – RandomForest)")

                    if "UPTREND" in pattern:
                        st.markdown(f"### 🟢 {pattern}")
                    elif "DOWNTREND" in pattern:
                        st.markdown(f"### 🔴 {pattern}")
                    elif "SIDEWAYS" in pattern:
                        st.markdown(f"### 🟡 {pattern}")
                    else:
                        st.markdown(f"### {pattern}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last close", f"${latest_close:,.2f}")
                    c2.metric("Prob. of UP", f"{latest_prob_up:.1%}")
                    c3.metric("RSI (14)", f"{latest_rsi:.1f}")
                    c4.metric("Predicted move", f"{latest_pred_return * 100:+.2f}%")

                    st.markdown("### Latest bar prediction")
                    st.write(f"- **Bar time:** `{latest_index}`")
                    st.write(f"- **Model:** RandomForest ({mode.upper()})")
                    st.write(
                        f"- **Model expectation:** **{direction_label}** for the **{horizon_text}**"
                    )
                    st.write(f"- **Probability of UP:** `{latest_prob_up:.2%}`")
                    st.write(
                        f"- **Predicted price change:** `{latest_pred_return * 100:+.2f}%` ({horizon_text})"
                    )

                with tab_charts:
                    st.subheader("Price & moving averages")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Price")
                    ax.legend()
                    st.pyplot(fig)

                    st.subheader("RSI (14)")
                    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
                    ax2.plot(df_feat.index, df_feat["rsi_14"], label="RSI 14")
                    ax2.axhline(70, linestyle="--")
                    ax2.axhline(30, linestyle="--")
                    ax2.set_ylim(0, 100)
                    ax2.set_ylabel("RSI")
                    st.pyplot(fig2)

                with tab_signals:
                    st.subheader("Recent RandomForest outputs (last 15 bars)")
                    nice_df = (
                        df_feat[
                            ["Close", "rsi_14", "pred_up", "prob_up", "pred_return"]
                        ]
                        .tail(15)
                        .rename(
                            columns={
                                "Close": "Close price",
                                "rsi_14": "RSI (14)",
                                "pred_up": "Pred. up? (1=yes)",
                                "prob_up": "Prob. of up",
                                "pred_return": "Pred. return (decimal)",
                            }
                        )
                    )
                    st.dataframe(nice_df)

                with tab_model:
                    st.subheader("RandomForest model details")
                    st.markdown(
                        f"""
**Mode:** `{mode.upper()}`  
**Horizon:** `{horizon_text}`  

Models:
- RandomForestClassifier → UP/DOWN  
- RandomForestRegressor → next_return (% move)  

Features:
- Returns: return_1d, return_2d, return_5d  
- MAs: ma_5, ma_10, ma_20  
- MA ratios: ma_5_20_ratio, ma_10_20_ratio  
- Volatility: vol_5, vol_10  
- RSI(14)
"""
                    )

            # --------- Deep Learning path ----------
            else:
                model, meta = load_dl_model()
                if model is None or meta is None:
                    return

                feature_cols = meta["feature_cols"]
                scaler = meta["scaler"]
                window = meta["window"]

                try:
                    X_raw = df_feat[feature_cols].values
                except KeyError:
                    missing_cols = [c for c in feature_cols if c not in df_feat.columns]
                    st.error(
                        "Feature mismatch for DL model.\n\n"
                        f"Missing features in app: `{missing_cols}`.\n"
                        "Make sure engineer_features_for_app() matches train_model_dl.py."
                    )
                    return

                X_scaled = scaler.transform(X_raw)
                X_seq = build_sequences_for_app(X_scaled, window=window)
                if X_seq.shape[0] == 0:
                    st.error(
                        f"Not enough data to form a sequence of length {window} for LSTM."
                    )
                    return

                dir_probs, mag_preds = model.predict(X_seq)
                dir_probs = dir_probs.ravel()
                mag_preds = mag_preds.ravel()
                dir_preds = (dir_probs >= 0.5).astype(int)

                df_feat_dl = df_feat.copy()
                df_feat_dl["dl_pred_up"] = np.nan
                df_feat_dl["dl_prob_up"] = np.nan
                df_feat_dl["dl_pred_return"] = np.nan

                df_feat_dl.iloc[window:, df_feat_dl.columns.get_loc("dl_pred_up")] = dir_preds
                df_feat_dl.iloc[window:, df_feat_dl.columns.get_loc("dl_prob_up")] = dir_probs
                df_feat_dl.iloc[window:, df_feat_dl.columns.get_loc("dl_pred_return")] = mag_preds

                # ✅ No dropna(subset=...) – use mask instead to avoid KeyError
                valid_mask = df_feat_dl["dl_prob_up"].notna() & df_feat_dl["dl_pred_return"].notna()
                df_valid = df_feat_dl[valid_mask]

                if df_valid.empty:
                    st.error("No valid DL predictions found after alignment.")
                    return

                latest_row = df_valid.iloc[-1]
                latest_index = latest_row.name
                latest_close = float(latest_row["Close"])
                latest_rsi = float(latest_row["rsi_14"])
                latest_prob_up = float(latest_row["dl_prob_up"])
                latest_pred_return = float(latest_row["dl_pred_return"])
                direction_label = "UP" if latest_prob_up >= 0.5 else "DOWN / FLAT"

                closes_valid = df_valid["Close"].values
                preds_valid = df_valid["dl_pred_up"].values
                pattern = get_pattern_label(preds_valid, closes_valid)

                with tab_summary:
                    st.subheader(f"Market pattern for `{ticker}` (DAILY – LSTM)")

                    if "UPTREND" in pattern:
                        st.markdown(f"### 🟢 {pattern}")
                    elif "DOWNTREND" in pattern:
                        st.markdown(f"### 🔴 {pattern}")
                    elif "SIDEWAYS" in pattern:
                        st.markdown(f"### 🟡 {pattern}")
                    else:
                        st.markdown(f"### {pattern}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last close", f"${latest_close:,.2f}")
                    c2.metric("Prob. of UP", f"{latest_prob_up:.1%}")
                    c3.metric("RSI (14)", f"{latest_rsi:.1f}")
                    c4.metric("Predicted move", f"{latest_pred_return * 100:+.2f}%")

                    st.markdown("### Latest bar prediction (LSTM)")
                    st.write(f"- **Bar time:** `{latest_index}`")
                    st.write(f"- **Model:** LSTM (Deep Learning, DAILY)")
                    st.write(
                        f"- **Model expectation:** **{direction_label}** for the **next day**"
                    )
                    st.write(f"- **Probability of UP:** `{latest_prob_up:.2%}`")
                    st.write(
                        f"- **Predicted price change:** `{latest_pred_return * 100:+.2f}%` (next day)"
                    )

                with tab_charts:
                    st.subheader("Price & moving averages")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Price")
                    ax.legend()
                    st.pyplot(fig)

                    st.subheader("RSI (14)")
                    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
                    ax2.plot(df_feat.index, df_feat["rsi_14"], label="RSI 14")
                    ax2.axhline(70, linestyle="--")
                    ax2.axhline(30, linestyle="--")
                    ax2.set_ylim(0, 100)
                    ax2.set_ylabel("RSI")
                    st.pyplot(fig2)

                with tab_signals:
                    st.subheader("Recent LSTM outputs (last 15 bars with predictions)")
                    recent = (
                        df_valid[
                            ["Close", "rsi_14", "dl_pred_up", "dl_prob_up", "dl_pred_return"]
                        ]
                        .tail(15)
                        .rename(
                            columns={
                                "Close": "Close price",
                                "rsi_14": "RSI (14)",
                                "dl_pred_up": "Pred. up? (1=yes)",
                                "dl_prob_up": "Prob. of up",
                                "dl_pred_return": "Pred. return (decimal)",
                            }
                        )
                    )
                    st.dataframe(recent)

                with tab_model:
                    st.subheader("LSTM (Deep Learning) model details")
                    st.markdown(
                        f"""
**Mode:** `DAILY`  
**Horizon:** `next day`  

Architecture:
- 2× LSTM layers (64 → 32 units)
- Dropout (0.3)
- Two heads:
  - `direction` (sigmoid) → UP/DOWN
  - `magnitude` (linear) → next_return (decimal)

Features (same as RandomForest):
- Returns: return_1d, return_2d, return_5d  
- MAs: ma_5, ma_10, ma_20  
- MA ratios: ma_5_20_ratio, ma_10_20_ratio  
- Volatility: vol_5, vol_10  
- RSI(14)

Input to LSTM:
- Sliding window of length `{window}` bars,
  scaled using the StandardScaler fitted during training.
"""
                    )

    st.markdown("---")
    st.caption(
        "Portfolio project – compares classic ML vs deep learning on financial time series "
        "with live data and interactive visualization."
    )


if __name__ == "__main__":
    main()
