import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from tensorflow.keras.models import load_model

# ---------------- Paths ----------------
DAILY_MODEL_PATH = "models/saved_model.pkl"
HOURLY_MODEL_PATH = "models/saved_model_hourly.pkl"

DL_MODEL_DIR = "models/dl_model"
DL_META_PATH = "models/dl_meta.pkl"

TICKERS_CSV = Path("data") / "tickers.csv"


# ---------------- Helpers ----------------
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


@st.cache_data
def load_tickers_csv() -> pd.DataFrame:
    if not TICKERS_CSV.exists():
        return pd.DataFrame(columns=["symbol", "name", "exchange"])
    df = pd.read_csv(TICKERS_CSV)

    for col in ["symbol", "name", "exchange"]:
        if col not in df.columns:
            df[col] = ""

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()

    df = df.dropna(subset=["symbol"])
    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol"])
    return df


def ticker_picker_ui() -> str:
    tickers_df = load_tickers_csv()

    st.subheader("Ticker search (CSV)")
    query = st.text_input(
        "Search by symbol or company name",
        value="AAPL",
        help="This searches data/tickers.csv. Example: type 'apple' or 'AAPL'.",
    ).strip().lower()

    # If CSV missing / empty -> fallback input
    if tickers_df.empty:
        st.warning("tickers.csv not found or empty. Create data/tickers.csv to enable search.")
        return st.text_input("Enter ticker symbol (fallback)", value="AAPL").strip().upper()

    if query:
        mask = (
            tickers_df["symbol"].str.lower().str.contains(query, na=False)
            | tickers_df["name"].str.lower().str.contains(query, na=False)
            | tickers_df["exchange"].str.lower().str.contains(query, na=False)
        )
        filtered = tickers_df[mask].copy()
    else:
        filtered = tickers_df.copy()

    filtered = filtered.head(50)

    if filtered.empty:
        st.info("No matches found in tickers.csv. Add the symbol to CSV or use fallback entry.")
        return st.text_input("Enter ticker symbol (fallback)", value="AAPL").strip().upper()

    filtered["label"] = filtered.apply(
        lambda r: f"{r['symbol']} — {r['name']} ({r['exchange']})"
        if str(r["exchange"]).strip()
        else f"{r['symbol']} — {r['name']}",
        axis=1,
    )

    choice = st.selectbox("Select a ticker", filtered["label"].tolist(), index=0)
    symbol = choice.split(" — ", 1)[0].strip().upper()
    return symbol


@st.cache_data(ttl=600)
def validate_ticker(sym: str, interval: str = "1d") -> bool:
    try:
        df_test = yf.download(sym, period="10d", interval=interval)
        return df_test is not None and not df_test.empty
    except Exception:
        return False


def engineer_features_for_app(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must match the feature logic used in:
      - train_model.py
      - train_model_dl.py
    """
    df = df.copy()

    close = _to_series(df["Close"])
    vol = _to_series(df["Volume"]) if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    # Returns
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    # Moving averages
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

    df.dropna(inplace=True)
    return df


def get_pattern_label(preds, closes) -> str:
    preds = np.array(preds, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(preds) < 5 or len(closes) < 10:
        return "Not enough data"

    last5 = preds[-5:]
    ones_ratio = last5.mean()

    last_prices = closes[-10:]
    x = np.arange(len(last_prices))
    slope = np.polyfit(x, last_prices, 1)[0]

    if ones_ratio > 0.6 and slope > 0:
        return "UPTREND / BULLISH"
    if ones_ratio < 0.4 and slope < 0:
        return "DOWNTREND / BEARISH"
    return "SIDEWAYS / NEUTRAL"


def load_rf_model(mode: str = "daily"):
    model_path = DAILY_MODEL_PATH if mode == "daily" else HOURLY_MODEL_PATH
    if not os.path.exists(model_path):
        st.error(f"RandomForest model not found: {model_path}. Train it and push it to GitHub.")
        return None, None, None

    bundle = joblib.load(model_path)
    clf = bundle.get("clf")
    reg = bundle.get("reg")
    feature_cols = bundle.get("features")

    if clf is None or reg is None or feature_cols is None:
        st.error("Invalid RF model bundle. Retrain using the latest train_model.py")
        return None, None, None
    return clf, reg, feature_cols


@st.cache_resource
def load_dl_model_cached():
    if (not os.path.exists(DL_MODEL_DIR)) or (not os.path.exists(DL_META_PATH)):
        st.error(
            f"Deep learning model not found. Expected: {DL_MODEL_DIR} and {DL_META_PATH}. "
            "Run train_model_dl.py and push model files to GitHub."
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


# ---------------- Streamlit App ----------------
def main():
    st.set_page_config(page_title="Stock Pattern Detector", page_icon="📈", layout="wide")
    st.title("📈 Stock Pattern Detector — ML + Deep Learning")

    st.sidebar.header("Settings")

    timeframe = st.sidebar.radio("Prediction timeframe", ["Daily (next day)", "Hourly (next hour)"], index=0)
    mode = "daily" if timeframe.startswith("Daily") else "hourly"
    horizon_text = "next day" if mode == "daily" else "next hour"

    algo = st.sidebar.radio("Model type", ["RandomForest (Classic ML)", "LSTM (Deep Learning)"], index=0)

    if algo.startswith("LSTM") and mode == "hourly":
        st.warning("LSTM is only available for DAILY mode right now. Switching to DAILY.")
        mode = "daily"
        horizon_text = "next day"

    # -------- Ticker search (Option B) --------
    ticker = ticker_picker_ui()

    yf_interval = "1d" if mode == "daily" else "60m"
    if not validate_ticker(ticker, interval=yf_interval):
        st.error(f"Ticker '{ticker}' not found / no data available at interval {yf_interval}. Try another.")
        st.stop()

    if mode == "daily":
        period = st.sidebar.selectbox("Historical period (daily)", ["6mo", "1y", "2y", "5y", "10y"], index=2)
        yf_period = period
    else:
        st.sidebar.info("Hourly mode uses 60 days of 60m data (Yahoo limit is tighter).")
        yf_period = "60d"

    st.sidebar.caption("Tip: add more symbols to data/tickers.csv to expand search.")

    if st.button("🔍 Run analysis", type="primary"):
        with st.spinner("Downloading data + running model..."):
            df = yf.download(ticker, period=yf_period, interval=yf_interval)
            if df is None or df.empty:
                st.error("No data returned by Yahoo Finance.")
                return

            df_feat = engineer_features_for_app(df)
            if df_feat.empty:
                st.error("Not enough rows after feature engineering (needs rolling windows).")
                return

            tab_summary, tab_charts, tab_signals, tab_model = st.tabs(
                ["📌 Summary", "📊 Charts", "📋 Recent signals", "🧠 Model details"]
            )

            # ---------------- RandomForest ----------------
            if algo.startswith("RandomForest"):
                clf, reg, feature_cols = load_rf_model(mode=mode)
                if clf is None:
                    return

                missing = [c for c in feature_cols if c not in df_feat.columns]
                if missing:
                    st.error(f"Feature mismatch. Missing in app: {missing}")
                    return

                X_all = df_feat[feature_cols]
                preds = clf.predict(X_all)
                prob_up = clf.predict_proba(X_all)[:, 1]
                pred_ret = reg.predict(X_all)

                df_feat["pred_up"] = preds
                df_feat["prob_up"] = prob_up
                df_feat["pred_return"] = pred_ret

                latest = df_feat.iloc[-1]
                pattern = get_pattern_label(df_feat["pred_up"].values, df_feat["Close"].values)

                with tab_summary:
                    st.subheader(f"{ticker} ({mode.upper()} — RandomForest)")
                    st.write(f"**Pattern:** {pattern}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last close", f"${float(latest['Close']):,.2f}")
                    c2.metric("Prob. UP", f"{float(latest['prob_up']):.1%}")
                    c3.metric("RSI(14)", f"{float(latest['rsi_14']):.1f}")
                    c4.metric(f"Pred. move ({horizon_text})", f"{float(latest['pred_return']) * 100:+.2f}%")

                with tab_charts:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.legend()
                    st.pyplot(fig)

                with tab_signals:
                    st.dataframe(df_feat[["Close", "rsi_14", "pred_up", "prob_up", "pred_return"]].tail(25))

                with tab_model:
                    st.write("RandomForestClassifier + RandomForestRegressor.")
                    st.write(f"Horizon: {horizon_text}")
                    st.write(f"Features ({len(feature_cols)}): {feature_cols}")

            # ---------------- LSTM Deep Learning (Daily only) ----------------
            else:
                model, meta = load_dl_model_cached()
                if model is None:
                    return

                feature_cols = meta["feature_cols"]
                scaler = meta["scaler"]
                window = int(meta["window"])

                missing = [c for c in feature_cols if c not in df_feat.columns]
                if missing:
                    st.error(f"Feature mismatch for DL model. Missing: {missing}")
                    return

                X_raw = df_feat[feature_cols].values
                X_scaled = scaler.transform(X_raw)

                X_seq = build_sequences_for_app(X_scaled, window=window)
                if X_seq.shape[0] == 0:
                    st.error(f"Not enough data to build LSTM sequences with window={window}.")
                    return

                dir_probs, mag_preds = model.predict(X_seq)
                dir_probs = dir_probs.ravel()
                mag_preds = mag_preds.ravel()
                dir_preds = (dir_probs >= 0.5).astype(int)

                df_pred = df_feat.copy()
                df_pred["dl_pred_up"] = np.nan
                df_pred["dl_prob_up"] = np.nan
                df_pred["dl_pred_return"] = np.nan

                # align predictions from index [window:] onward
                df_pred.iloc[window:, df_pred.columns.get_loc("dl_pred_up")] = dir_preds
                df_pred.iloc[window:, df_pred.columns.get_loc("dl_prob_up")] = dir_probs
                df_pred.iloc[window:, df_pred.columns.get_loc("dl_pred_return")] = mag_preds

                valid = df_pred["dl_prob_up"].notna() & df_pred["dl_pred_return"].notna()
                df_valid = df_pred[valid]
                if df_valid.empty:
                    st.error("DL predictions produced no valid rows (unexpected).")
                    return

                latest = df_valid.iloc[-1]
                pattern = get_pattern_label(df_valid["dl_pred_up"].values, df_valid["Close"].values)

                with tab_summary:
                    st.subheader(f"{ticker} (DAILY — LSTM)")
                    st.write(f"**Pattern:** {pattern}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last close", f"${float(latest['Close']):,.2f}")
                    c2.metric("Prob. UP", f"{float(latest['dl_prob_up']):.1%}")
                    c3.metric("RSI(14)", f"{float(latest['rsi_14']):.1f}")
                    c4.metric("Pred. move (next day)", f"{float(latest['dl_pred_return']) * 100:+.2f}%")

                with tab_charts:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.legend()
                    st.pyplot(fig)

                with tab_signals:
                    st.dataframe(
                        df_valid[["Close", "rsi_14", "dl_pred_up", "dl_prob_up", "dl_pred_return"]].tail(25)
                    )

                with tab_model:
                    st.write("LSTM with two heads: direction (classification) + magnitude (regression).")
                    st.write(f"Window: {window}")
                    st.write(f"Features ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()
