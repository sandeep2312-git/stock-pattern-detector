import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from feature_engineering import engineer_features, add_market_context

# -------- Paths --------
DAILY_MODEL_PATH = "models/saved_model.pkl"
HOURLY_MODEL_PATH = "models/saved_model_hourly.pkl"
TICKERS_CSV = Path("data") / "tickers.csv"


# -------------------- helpers --------------------
def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download with strong safety: never crash app on yfinance weirdness."""
    t = str(ticker).strip().upper()
    if not t:
        return pd.DataFrame()
    try:
        df = yf.download(t, period=period, interval=interval, group_by="column")
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except ValueError as e:
        # yfinance sometimes throws this if symbol invalid / no rows
        if "No objects to concatenate" in str(e):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def prob_bucket(p: float) -> str:
    if not np.isfinite(p):
        return "Unknown"
    if p >= 0.60:
        return "Green"
    if p >= 0.40:
        return "Yellow"
    return "Red"


def bucket_emoji(bucket: str) -> str:
    return {"Green": "🟢", "Yellow": "🟡", "Red": "🔴", "Unknown": "⚪"}.get(bucket, "⚪")


def bucket_message(bucket: str) -> str:
    if bucket == "Green":
        return "More signs point UP than down (not guaranteed)."
    if bucket == "Yellow":
        return "Mixed signals. The model is not confident."
    if bucket == "Red":
        return "More signs point DOWN/flat than up (not guaranteed)."
    return "Not enough info to decide."


def pred_range_from_history(df_feat: pd.DataFrame, pred_col: str = "pred_return", z: float = 1.5):
    """Rough move-range band from prediction dispersion."""
    if pred_col not in df_feat.columns:
        return None
    s = pd.to_numeric(df_feat[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        return None
    std = float(s.tail(120).std())
    if not np.isfinite(std) or std <= 0:
        return None
    return std * z


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
        return "Uptrend"
    if ones_ratio < 0.4 and slope < 0:
        return "Downtrend"
    return "Sideways"


def load_rf_model(mode: str):
    path = DAILY_MODEL_PATH if mode == "daily" else HOURLY_MODEL_PATH
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    return joblib.load(path)


def clean_features(df_feat: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove NaN/Inf rows so sklearn won't crash."""
    if not feature_cols:
        return df_feat.iloc[0:0].copy(), df_feat.iloc[0:0].copy()
    feat = df_feat[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat = feat.replace([np.inf, -np.inf], np.nan)
    mask = feat.notna().all(axis=1)
    return df_feat.loc[mask].copy(), feat.loc[mask].copy()


@st.cache_data
def load_tickers_csv() -> pd.DataFrame:
    if not TICKERS_CSV.exists():
        return pd.DataFrame(columns=["symbol", "name", "exchange", "category"])
    df = pd.read_csv(TICKERS_CSV)
    for col in ["symbol", "name", "exchange", "category"]:
        if col not in df.columns:
            df[col] = ""
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["symbol"] != ""].drop_duplicates(subset=["symbol"])
    return df


@st.cache_data(ttl=600)
def validate_ticker(sym: str, interval: str) -> bool:
    df_test = safe_download(sym, period="10d", interval=interval)
    return df_test is not None and not df_test.empty


def ticker_picker_ui(tickers_df: pd.DataFrame) -> str:
    st.markdown("### 1) Choose a stock")
    categories = ["All"] + sorted([c for c in tickers_df["category"].dropna().unique().tolist() if str(c).strip()])
    category = st.selectbox("Filter (optional)", categories, index=0)

    df = tickers_df.copy()
    if category != "All":
        df = df[df["category"] == category]

    query = st.text_input("Search by ticker or company name", value="AAPL").strip().lower()
    if query:
        mask = (
            df["symbol"].str.lower().str.contains(query, na=False)
            | df["name"].str.lower().str.contains(query, na=False)
        )
        df = df[mask].copy()

    df = df.head(60)

    if df.empty:
        st.info("No match found. Try a different name or enter a ticker directly.")
        return st.text_input("Enter ticker (example: AAPL)", value="AAPL").strip().upper()

    df["label"] = df.apply(lambda r: f"{r['symbol']} — {r['name']}", axis=1)
    choice = st.selectbox("Pick one", df["label"].tolist(), index=0)
    return choice.split(" — ", 1)[0].strip().upper()


# -------------------- UI renderers --------------------
def render_layman_summary(df_feat: pd.DataFrame, rf_bundle: dict, timeframe_label: str):
    feature_cols = [c for c in rf_bundle["features"] if c in df_feat.columns]
    df_feat, feat = clean_features(df_feat, feature_cols)

    if df_feat.empty or feat.empty:
        st.error("Not enough clean data to make a prediction right now.")
        st.stop()

    X = feat.values
    preds = rf_bundle["clf"].predict(X)
    probs = rf_bundle["clf"].predict_proba(X)[:, 1]
    rets = rf_bundle["reg"].predict(X)

    df_feat["pred_up"] = preds
    df_feat["prob_up"] = probs
    df_feat["pred_return"] = rets

    latest = df_feat.iloc[-1]
    close = safe_float(latest.get("Close"))
    p_up = safe_float(latest.get("prob_up"))
    pred_ret = safe_float(latest.get("pred_return"))
    rsi = safe_float(latest.get("rsi_14"))

    bucket = prob_bucket(p_up)
    emoji = bucket_emoji(bucket)

    # Big summary card
    st.markdown("### 3) Result (easy view)")
    st.markdown(
        f"""
<div style="padding:18px;border-radius:16px;border:1px solid #2a2a2a;">
  <div style="font-size:28px;font-weight:700;">{emoji} {bucket} Signal</div>
  <div style="font-size:16px;opacity:0.9;margin-top:6px;">{bucket_message(bucket)}</div>
  <div style="margin-top:10px;font-size:14px;opacity:0.85;">
    Timeframe: <b>{timeframe_label}</b> — This is a short-term statistical guess, not financial advice.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Simple metrics
    st.markdown("#### Key numbers (plain English)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current price", f"${close:,.2f}")
    c2.metric("Chance of going up", f"{p_up:.0%}")
    c3.metric("Estimated move", f"{pred_ret*100:+.2f}%")

    st.caption(
        "“Chance of going up” is the model’s confidence (not a guarantee). "
        "“Estimated move” is a rough guess of the next period % change."
    )

    # Range
    band = pred_range_from_history(df_feat, pred_col="pred_return", z=1.5)
    if band is not None:
        low = (pred_ret - band) * 100
        high = (pred_ret + band) * 100
        st.info(f"Expected move range (rough): **{low:+.2f}% → {high:+.2f}%**")
    else:
        st.caption("Move range: not enough history to estimate uncertainty.")

    # Trend + RSI explanation
    st.markdown("#### Extra context (still simple)")
    trend = get_pattern_label(df_feat["pred_up"].values, df_feat["Close"].values)
    c4, c5 = st.columns(2)
    c4.metric("Recent trend", trend)
    c5.metric("RSI (momentum)", f"{rsi:.1f}")
    st.caption(
        "RSI is a momentum score. Rough rule: above ~70 can mean “overheated”, below ~30 can mean “oversold”."
    )

    # Chart
    st.markdown("#### Price chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_feat.index, df_feat["Close"], label="Close")
    if "ma_5" in df_feat.columns:
        ax.plot(df_feat.index, df_feat["ma_5"], label="Avg (5)")
    if "ma_10" in df_feat.columns:
        ax.plot(df_feat.index, df_feat["ma_10"], label="Avg (10)")
    if "ma_20" in df_feat.columns:
        ax.plot(df_feat.index, df_feat["ma_20"], label="Avg (20)")
    ax.legend()
    st.pyplot(fig)

    # Advanced (collapsed)
    with st.expander("Advanced details (optional)", expanded=False):
        st.write("Recent rows with model outputs:")
        st.dataframe(df_feat[["Close", "rsi_14", "pred_up", "prob_up", "pred_return"]].tail(25))

        st.write("Top factors the model generally uses (feature importance):")
        try:
            importances = rf_bundle["clf"].feature_importances_
            imp_df = (
                pd.DataFrame({"Feature": rf_bundle["features"], "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(12)
            )
            st.bar_chart(imp_df.set_index("Feature"))
            st.caption("These are global importances (overall), not per-day explanations.")
        except Exception as e:
            st.caption(f"Could not show importances: {str(e)[:200]}")

    with st.expander("Glossary (plain English)", expanded=False):
        st.markdown(
            """
- **Chance of going up**: how confident the model is the price will rise next period.
- **Estimated move**: the model’s guess of the next period % change.
- **RSI**: momentum score (0–100). Above ~70 can mean “overbought”, below ~30 can mean “oversold”.
- **Moving averages**: smoothed price lines to reduce noise.
"""
        )


def run_rf_for_ticker(ticker: str, period: str, interval: str, rf_bundle: dict) -> dict:
    df = safe_download(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return {"ticker": ticker, "error": "No data / invalid symbol"}

    df_feat = engineer_features(df, include_targets=False)
    df_feat = add_market_context(df_feat, period=period, interval=interval)

    feature_cols = [c for c in rf_bundle["features"] if c in df_feat.columns]
    df_feat, feat = clean_features(df_feat, feature_cols)
    if df_feat.empty or feat.empty:
        return {"ticker": ticker, "error": "Not enough clean rows"}

    X = feat.values
    probs = rf_bundle["clf"].predict_proba(X)[:, 1]
    rets = rf_bundle["reg"].predict(X)

    latest = df_feat.iloc[-1]
    p_up = float(probs[-1])
    pred_ret = float(rets[-1])
    close = safe_float(latest.get("Close"))
    rsi = safe_float(latest.get("rsi_14"))

    bucket = prob_bucket(p_up)
    return {
        "ticker": ticker,
        "price": close,
        "chance_up": p_up,
        "signal": f"{bucket_emoji(bucket)} {bucket}",
        "move_pct": pred_ret * 100,
        "rsi": rsi,
        "note": bucket_message(bucket),
        "error": "",
    }


# -------------------- main app --------------------
def main():
    st.set_page_config(page_title="Stock Trend Helper", page_icon="📈", layout="wide")

    st.title("📈 Stock Trend Helper (Easy View)")
    st.caption(
        "This app shows a short-term **signal** (Green/Yellow/Red) based on recent price patterns. "
        "It’s a learning tool — not financial advice."
    )

    tickers_df = load_tickers_csv()

    st.sidebar.header("Settings")
    timeframe = st.sidebar.radio("Timeframe", ["Daily (next day)", "Hourly (next hour)"], index=0)
    mode = "daily" if timeframe.startswith("Daily") else "hourly"
    interval = "1d" if mode == "daily" else "60m"
    period = st.sidebar.selectbox("How much history to use", ["6mo", "1y", "2y", "5y", "10y"], index=2) if mode == "daily" else "60d"

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tip:** Green ≠ buy. It only means “more signs point up”. Always manage risk.")

    tab_predict, tab_compare, tab_tickers, tab_help = st.tabs(
        ["🔮 Simple Prediction", "🧾 Compare", "📚 Available Tickers", "❓ Help"]
    )

    # ---------- Prediction ----------
    with tab_predict:
        if tickers_df.empty:
            st.warning("No ticker list found. Add tickers to data/tickers.csv (columns: symbol,name,exchange,category).")
            ticker = st.text_input("Enter ticker (example: AAPL)", value="AAPL").strip().upper()
        else:
            ticker = ticker_picker_ui(tickers_df)

        st.markdown("### 2) Run")
        run = st.button("Run Prediction", type="primary")

        if run:
            if not validate_ticker(ticker, interval=interval):
                st.error(f"Ticker '{ticker}' not found or no data available for {interval}. Try another.")
                st.stop()

            rf = load_rf_model(mode)
            if rf is None:
                st.stop()

            with st.spinner("Loading data and generating signal..."):
                df = safe_download(ticker, period=period, interval=interval)
                if df is None or df.empty:
                    st.error("No data returned for this ticker.")
                    st.stop()

                df_feat = engineer_features(df, include_targets=False)
                df_feat = add_market_context(df_feat, period=period, interval=interval)

            render_layman_summary(df_feat, rf, timeframe_label=timeframe)

    # ---------- Compare ----------
    with tab_compare:
        st.markdown("### Compare a few stocks (fast scan)")
        rf = load_rf_model(mode)
        if rf is None:
            st.info("Train and commit the model first.")
        else:
            symbols = tickers_df["symbol"].tolist() if not tickers_df.empty else ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
            default = [s for s in ["AAPL", "MSFT", "NVDA"] if s in symbols] or symbols[:3]
            picks = st.multiselect("Pick tickers", symbols, default=default)

            if st.button("Run Scan"):
                rows = []
                with st.spinner("Scanning..."):
                    for sym in picks:
                        rows.append(run_rf_for_ticker(sym, period, interval, rf))

                df_out = pd.DataFrame([{
                    "Ticker": r.get("ticker"),
                    "Price": r.get("price"),
                    "Chance Up": (None if r.get("chance_up") is None else float(r.get("chance_up"))),
                    "Signal": r.get("signal"),
                    "Est. Move %": r.get("move_pct"),
                    "RSI": r.get("rsi"),
                    "Plain-English note": r.get("note", ""),
                    "Error": r.get("error", "")
                } for r in rows])

                df_out["Chance Up"] = pd.to_numeric(df_out["Chance Up"], errors="coerce")
                df_out = df_out.sort_values(by=["Error", "Chance Up"], ascending=[True, False], na_position="last")
                st.dataframe(df_out, use_container_width=True)

    # ---------- Available Tickers ----------
    with tab_tickers:
        st.markdown("## 📚 Available Tickers")
        st.caption(
            "This list comes from your `data/tickers.csv`. "
            "You can search and filter, then copy a symbol into the Prediction tab."
        )

        if tickers_df.empty:
            st.warning("No ticker list found. Create data/tickers.csv with columns: symbol,name,exchange,category")
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox(
                "Filter by category",
                ["All"] + sorted([x for x in tickers_df["category"].dropna().unique().tolist() if str(x).strip()]),
            )
        with c2:
            exchange = st.selectbox(
                "Filter by exchange",
                ["All"] + sorted([x for x in tickers_df["exchange"].dropna().unique().tolist() if str(x).strip()]),
            )

        df = tickers_df.copy()
        if category != "All":
            df = df[df["category"] == category]
        if exchange != "All":
            df = df[df["exchange"] == exchange]

        search = st.text_input("Search by ticker or company name", value="")
        if search:
            s = search.lower()
            df = df[
                df["symbol"].str.lower().str.contains(s, na=False)
                | df["name"].str.lower().str.contains(s, na=False)
            ]

        st.markdown(f"### Showing {len(df)} tickers")
        st.dataframe(df[["symbol", "name", "exchange", "category"]], use_container_width=True)

        st.info("Tip: Copy a ticker symbol from this table and paste it into **Simple Prediction** to analyze it.")

    # ---------- Help ----------
    with tab_help:
        st.markdown("## How to read this app (beginners)")
        st.markdown(
            """
**Green / Yellow / Red Signal** is a simple summary:

- 🟢 **Green**: more signs point UP than down.
- 🟡 **Yellow**: mixed signals (model not confident).
- 🔴 **Red**: more signs point DOWN/flat than up.

**Important:** this is not a guarantee and not financial advice.

### What is “Chance Up”?
How confident the model is that the price will be higher next period.

### What is “Estimated move”?
A rough guess of the next period % change (example: +0.80%).

### Why can it be wrong?
News, earnings, macro events, and sudden volatility can move prices in ways the model can’t predict.
"""
        )

        st.markdown("## Repo setup")
        st.markdown(
            """
- Put your ticker universe in: `data/tickers.csv`
- Required columns: `symbol,name,exchange,category`
- The **Available Tickers** tab displays whatever is in that CSV.
"""
        )


if __name__ == "__main__":
    main()
