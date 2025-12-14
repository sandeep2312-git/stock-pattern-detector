import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model

from feature_engineering import engineer_features, add_market_context

# -------- Paths --------
DAILY_MODEL_PATH = "models/saved_model.pkl"
HOURLY_MODEL_PATH = "models/saved_model_hourly.pkl"

DL_MODEL_DIR = "models/dl_model"
DL_META_PATH = "models/dl_meta.pkl"

TICKERS_CSV = Path("data") / "tickers.csv"
PORTFOLIO_CSV = Path("data") / "portfolio.csv"


# -------- Data loading --------
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
    try:
        df_test = yf.download(sym, period="10d", interval=interval)
        return df_test is not None and not df_test.empty
    except Exception:
        return False


def ticker_picker_ui(tickers_df: pd.DataFrame) -> str:
    st.subheader("Ticker search (CSV)")

    categories = ["All"] + sorted([c for c in tickers_df["category"].dropna().unique().tolist() if str(c).strip()])
    category = st.selectbox("Filter category", categories, index=0)

    df = tickers_df.copy()
    if category != "All":
        df = df[df["category"] == category]

    query = st.text_input("Search symbol / company name", value="AAPL").strip().lower()

    if query:
        mask = (
            df["symbol"].str.lower().str.contains(query, na=False)
            | df["name"].str.lower().str.contains(query, na=False)
            | df["exchange"].str.lower().str.contains(query, na=False)
        )
        df = df[mask].copy()

    df = df.head(80)

    if df.empty:
        st.info("No matches. Add symbol to data/tickers.csv or use fallback input.")
        return st.text_input("Fallback ticker", value="AAPL").strip().upper()

    df["label"] = df.apply(
        lambda r: f"{r['symbol']} — {r['name']} ({r['exchange']})" if str(r["exchange"]).strip() else f"{r['symbol']} — {r['name']}",
        axis=1,
    )
    choice = st.selectbox("Select ticker", df["label"].tolist(), index=0)
    return choice.split(" — ", 1)[0].strip().upper()


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


def load_rf_model(mode: str):
    path = DAILY_MODEL_PATH if mode == "daily" else HOURLY_MODEL_PATH
    if not os.path.exists(path):
        st.error(f"RF model not found: {path}")
        return None
    return joblib.load(path)


@st.cache_resource
def load_dl_model_cached():
    if not (os.path.exists(DL_MODEL_DIR) and os.path.exists(DL_META_PATH)):
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


def run_rf_for_ticker(ticker: str, mode: str, period: str, interval: str, rf_bundle: dict) -> dict:
    df = yf.download(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return {"ticker": ticker, "error": "No data"}

    df_feat = engineer_features(df, include_targets=False)
    df_feat = add_market_context(df_feat, period=period, interval=interval)

    feature_cols = rf_bundle["features"]
    feature_cols = [c for c in feature_cols if c in df_feat.columns]  # tolerate missing VIX
    df_feat = df_feat.dropna(subset=feature_cols).copy()
    if df_feat.empty:
        return {"ticker": ticker, "error": "Not enough rows after features"}

    clf = rf_bundle["clf"]
    reg = rf_bundle["reg"]

    X = df_feat[feature_cols].values
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    rets = reg.predict(X)

    df_feat["pred_up"] = preds
    df_feat["prob_up"] = probs
    df_feat["pred_return"] = rets

    latest = df_feat.iloc[-1]
    return {
        "ticker": ticker,
        "close": float(latest["Close"]),
        "rsi": float(latest["rsi_14"]),
        "prob_up": float(latest["prob_up"]),
        "pred_return": float(latest["pred_return"]),
        "pattern": get_pattern_label(df_feat["pred_up"].values, df_feat["Close"].values),
        "df_feat": df_feat,
    }


def load_portfolio() -> pd.DataFrame:
    if not PORTFOLIO_CSV.exists():
        return pd.DataFrame(columns=["symbol", "shares", "avg_cost"])
    df = pd.read_csv(PORTFOLIO_CSV)
    for c in ["symbol", "shares", "avg_cost"]:
        if c not in df.columns:
            df[c] = np.nan
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df = df.dropna(subset=["symbol", "shares", "avg_cost"])
    df = df[df["symbol"] != ""]
    return df


# -------- Streamlit App --------
def main():
    st.set_page_config(page_title="Stock Pattern Detector", page_icon="📈", layout="wide")
    st.title("📈 Stock Pattern Detector — Ticker Search + Compare + Portfolio + Market Context")

    tickers_df = load_tickers_csv()
    if tickers_df.empty:
        st.warning("data/tickers.csv missing or empty. Run: python scripts/update_tickers.py")

    st.sidebar.header("Mode")
    timeframe = st.sidebar.radio("Timeframe", ["Daily (next day)", "Hourly (next hour)"], index=0)
    mode = "daily" if timeframe.startswith("Daily") else "hourly"
    interval = "1d" if mode == "daily" else "60m"
    period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y", "5y", "10y"], index=2) if mode == "daily" else "60d"

    st.sidebar.header("Model")
    model_type = st.sidebar.radio("Choose model", ["RandomForest (ML)", "LSTM (DL - daily only)"], index=0)
    if model_type.startswith("LSTM") and mode == "hourly":
        st.sidebar.warning("DL is daily-only. Switching to DAILY.")
        mode = "daily"
        interval = "1d"
        period = "2y"

    # ---- Tabs for options ----
    tab_predict, tab_compare, tab_portfolio, tab_admin = st.tabs(
        ["🔮 Single Ticker", "🧾 Compare Tickers", "💼 Portfolio Dashboard", "⚙️ Admin/Info"]
    )

    # ---------------- Single ticker ----------------
    with tab_predict:
        ticker = ticker_picker_ui(tickers_df) if not tickers_df.empty else st.text_input("Ticker", "AAPL").strip().upper()

        if not validate_ticker(ticker, interval=interval):
            st.error(f"Ticker '{ticker}' invalid/no data for interval {interval}.")
            st.stop()

        run = st.button("Run analysis", type="primary")

        if run:
            with st.spinner("Downloading + computing predictions..."):
                df = yf.download(ticker, period=period, interval=interval)
                if df is None or df.empty:
                    st.error("No data returned.")
                    st.stop()

                df_feat = engineer_features(df, include_targets=False)
                df_feat = add_market_context(df_feat, period=period, interval=interval)

                if model_type.startswith("RandomForest"):
                    rf = load_rf_model(mode)
                    if rf is None:
                        st.stop()

                    feature_cols = [c for c in rf["features"] if c in df_feat.columns]
                    df_feat = df_feat.dropna(subset=feature_cols).copy()
                    if df_feat.empty:
                        st.error("Not enough data after feature engineering.")
                        st.stop()

                    X = df_feat[feature_cols].values
                    preds = rf["clf"].predict(X)
                    probs = rf["clf"].predict_proba(X)[:, 1]
                    rets = rf["reg"].predict(X)

                    df_feat["pred_up"] = preds
                    df_feat["prob_up"] = probs
                    df_feat["pred_return"] = rets

                    latest = df_feat.iloc[-1]
                    pattern = get_pattern_label(df_feat["pred_up"].values, df_feat["Close"].values)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last Close", f"${float(latest['Close']):,.2f}")
                    c2.metric("Prob UP", f"{float(latest['prob_up']):.1%}")
                    c3.metric("RSI(14)", f"{float(latest['rsi_14']):.1f}")
                    c4.metric("Pred Move", f"{float(latest['pred_return'])*100:+.2f}%")

                    st.write(f"**Pattern:** {pattern}")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.legend()
                    st.pyplot(fig)

                    st.dataframe(df_feat[["Close", "rsi_14", "pred_up", "prob_up", "pred_return"]].tail(25))

                else:
                    dl_model, meta = load_dl_model_cached()
                    if dl_model is None:
                        st.error("DL model missing. Train train_model_dl.py and push models/dl_model + dl_meta.pkl")
                        st.stop()

                    feature_cols = [c for c in meta["feature_cols"] if c in df_feat.columns]
                    df_feat = df_feat.dropna(subset=feature_cols).copy()
                    if df_feat.empty:
                        st.error("Not enough data after feature engineering.")
                        st.stop()

                    scaler = meta["scaler"]
                    window = int(meta["window"])

                    X_raw = df_feat[feature_cols].values
                    X_scaled = scaler.transform(X_raw)
                    X_seq = build_sequences_for_app(X_scaled, window=window)
                    if X_seq.shape[0] == 0:
                        st.error(f"Not enough rows for DL window={window}.")
                        st.stop()

                    dir_probs, mag_preds = dl_model.predict(X_seq)
                    dir_probs = dir_probs.ravel()
                    mag_preds = mag_preds.ravel()
                    dir_preds = (dir_probs >= 0.5).astype(int)

                    df_pred = df_feat.copy()
                    df_pred["dl_pred_up"] = np.nan
                    df_pred["dl_prob_up"] = np.nan
                    df_pred["dl_pred_return"] = np.nan
                    df_pred.iloc[window:, df_pred.columns.get_loc("dl_pred_up")] = dir_preds
                    df_pred.iloc[window:, df_pred.columns.get_loc("dl_prob_up")] = dir_probs
                    df_pred.iloc[window:, df_pred.columns.get_loc("dl_pred_return")] = mag_preds

                    valid = df_pred["dl_prob_up"].notna() & df_pred["dl_pred_return"].notna()
                    df_valid = df_pred[valid]
                    latest = df_valid.iloc[-1]
                    pattern = get_pattern_label(df_valid["dl_pred_up"].values, df_valid["Close"].values)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last Close", f"${float(latest['Close']):,.2f}")
                    c2.metric("Prob UP", f"{float(latest['dl_prob_up']):.1%}")
                    c3.metric("RSI(14)", f"{float(latest['rsi_14']):.1f}")
                    c4.metric("Pred Move", f"{float(latest['dl_pred_return'])*100:+.2f}%")

                    st.write(f"**Pattern:** {pattern}")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.legend()
                    st.pyplot(fig)

                    st.dataframe(df_valid[["Close", "rsi_14", "dl_pred_up", "dl_prob_up", "dl_pred_return"]].tail(25))

    # ---------------- Compare tickers (RF only for speed) ----------------
    with tab_compare:
        st.subheader("Compare multiple tickers (scanner)")
        if model_type.startswith("LSTM"):
            st.info("Compare mode uses RandomForest for speed. Switch to RandomForest in sidebar to scan.")
        else:
            rf = load_rf_model(mode)
            if rf is None:
                st.stop()

            symbols = tickers_df["symbol"].tolist() if not tickers_df.empty else ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
            default = ["AAPL", "MSFT", "NVDA"] if set(["AAPL","MSFT","NVDA"]).issubset(set(symbols)) else symbols[:3]
            picks = st.multiselect("Pick tickers", symbols, default=default)

            if st.button("Run scan"):
                rows = []
                with st.spinner("Scanning tickers..."):
                    for sym in picks:
                        if not validate_ticker(sym, interval=interval):
                            rows.append({"ticker": sym, "error": "Invalid/no data"})
                            continue
                        res = run_rf_for_ticker(sym, mode, period, interval, rf)
                        rows.append(res)

                df_out = pd.DataFrame([{
                    "Ticker": r.get("ticker"),
                    "Close": r.get("close"),
                    "Prob_UP": r.get("prob_up"),
                    "Pred_Move_%": None if r.get("pred_return") is None else r.get("pred_return") * 100,
                    "RSI": r.get("rsi"),
                    "Pattern": r.get("pattern"),
                    "Error": r.get("error", "")
                } for r in rows])

                df_out = df_out.sort_values(by="Prob_UP", ascending=False, na_position="last")
                st.dataframe(df_out, use_container_width=True)

    # ---------------- Portfolio dashboard ----------------
    with tab_portfolio:
        st.subheader("Portfolio dashboard (from data/portfolio.csv)")
        port = load_portfolio()
        if port.empty:
            st.warning("Create data/portfolio.csv with columns: symbol, shares, avg_cost")
        else:
            rf = load_rf_model("daily")  # portfolio uses daily signals
            rows = []
            with st.spinner("Computing portfolio + signals..."):
                for _, r in port.iterrows():
                    sym = r["symbol"]
                    shares = float(r["shares"])
                    avg_cost = float(r["avg_cost"])

                    dfp = yf.download(sym, period="6mo", interval="1d")
                    if dfp is None or dfp.empty:
                        rows.append({"symbol": sym, "error": "No data"})
                        continue

                    last_close = float(dfp["Close"].iloc[-1])
                    mv = shares * last_close
                    cost = shares * avg_cost
                    pnl = mv - cost
                    pnl_pct = (pnl / cost) * 100 if cost != 0 else np.nan

                    sig = {"prob_up": np.nan, "pred_return": np.nan, "pattern": ""}

                    if rf is not None:
                        res = run_rf_for_ticker(sym, "daily", "2y", "1d", rf)
                        if "error" not in res:
                            sig["prob_up"] = res["prob_up"]
                            sig["pred_return"] = res["pred_return"]
                            sig["pattern"] = res["pattern"]

                    rows.append({
                        "Symbol": sym,
                        "Shares": shares,
                        "Avg Cost": avg_cost,
                        "Last Close": last_close,
                        "Market Value": mv,
                        "Cost Basis": cost,
                        "P&L": pnl,
                        "P&L %": pnl_pct,
                        "Prob UP": sig["prob_up"],
                        "Pred Move %": sig["pred_return"] * 100 if pd.notna(sig["pred_return"]) else np.nan,
                        "Pattern": sig["pattern"],
                    })

            df_port = pd.DataFrame(rows)
            st.dataframe(df_port, use_container_width=True)

    # ---------------- Admin/Info ----------------
    with tab_admin:
        st.subheader("Project files")
        st.write("- data/tickers.csv: ticker universe")
        st.write("- data/portfolio.csv: portfolio holdings")
        st.write("- models/saved_model.pkl: RF daily model")
        st.write("- models/saved_model_hourly.pkl: RF hourly model (optional)")
        st.write("- models/dl_model + dl_meta.pkl: LSTM daily model (optional)")
        st.write("")
        st.subheader("Quick commands")
        st.code(
            "\n".join([
                "python scripts/update_tickers.py",
                "python train_model.py --ticker AAPL --period 10y --interval 1d --out models/saved_model.pkl",
                "python train_model.py --ticker AAPL --period 60d --interval 60m --out models/saved_model_hourly.pkl",
                "python train_model_dl.py",
            ])
        )


if __name__ == "__main__":
    main()
