import numpy as np
import pandas as pd
import yfinance as yf


# ---------------- Core helpers ----------------
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns like:
      ('Close','AAPL'), ('Open','AAPL'), ...
    This converts them to single-level: 'Close', 'Open', ...
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Most common: level 0 is OHLCV, level 1 is ticker
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _to_series(x) -> pd.Series:
    # Force 1D series
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.Series(x)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ---------------- Market context features ----------------
def fetch_market_features(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_yf_columns(df)

    close = _to_series(df["Close"])
    out = pd.DataFrame(index=df.index)

    key = symbol.replace("^", "IDX_").replace("-", "_")
    out[f"{key}_close"] = close
    out[f"{key}_ret_1"] = close.pct_change()
    out[f"{key}_ma_20"] = close.rolling(20).mean()
    out[f"{key}_ma_20_ratio"] = close / (out[f"{key}_ma_20"] + 1e-9)
    out[f"{key}_vol_10"] = out[f"{key}_ret_1"].rolling(10).std()

    return out


def add_market_context(df_feat: pd.DataFrame, period: str, interval: str) -> pd.DataFrame:
    """
    Joins SPY + QQQ (+ VIX if available) features onto df_feat index.
    Also ensures df_feat columns are single-level before join.
    """
    out = df_feat.copy()
    out = _flatten_yf_columns(out)  # IMPORTANT: fix MultiIndex columns

    spy = fetch_market_features("SPY", period=period, interval=interval)
    qqq = fetch_market_features("QQQ", period=period, interval=interval)
    vix = fetch_market_features("^VIX", period=period, interval=interval)

    # Join only if not empty
    if not spy.empty:
        out = out.join(spy, how="left")
    if not qqq.empty:
        out = out.join(qqq, how="left")
    if not vix.empty:
        out = out.join(vix, how="left")

    return out


# ---------------- Stock feature engineering ----------------
def engineer_features(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = _flatten_yf_columns(df)  # IMPORTANT: fix MultiIndex from yfinance

    close = _to_series(df["Close"])
    vol = _to_series(df["Volume"]) if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    # Returns
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    # Moving averages
    df["ma_5"] = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()

    # Ratios
    df["ma_5_20_ratio"] = df["ma_5"] / (df["ma_20"] + 1e-9)
    df["ma_10_20_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # Volatility
    df["vol_5"] = df["return_1d"].rolling(5).std()
    df["vol_10"] = df["return_1d"].rolling(10).std()

    # Momentum
    df["rsi_14"] = compute_rsi(close, period=14)

    # Volume features
    if "Volume" in df.columns:
        df["volume_change_1"] = vol.pct_change()
        df["volume_ma_5"] = vol.rolling(5).mean()
        df["volume_ma_20"] = vol.rolling(20).mean()
        df["volume_relative"] = vol / (df["volume_ma_20"] + 1e-9)
    else:
        df["volume_change_1"] = np.nan
        df["volume_ma_5"] = np.nan
        df["volume_ma_20"] = np.nan
        df["volume_relative"] = np.nan

    # ATR
    prev_close = close.shift(1)
    high = _to_series(df["High"])
    low = _to_series(df["Low"])

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()

    # Bollinger position
    std_20 = close.rolling(20).std()
    bb_upper = df["ma_20"] + 2 * std_20
    bb_lower = df["ma_20"] - 2 * std_20
    df["bb_width"] = (bb_upper - bb_lower) / (df["ma_20"] + 1e-9)
    df["bb_percent_b"] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-9)

    # Targets
    if include_targets:
        next_close = close.shift(-1)
        df["next_return"] = (next_close - close) / (close + 1e-9)
        df["target_up"] = (df["next_return"] > 0).astype(int)

    return df


def get_feature_cols(include_market: bool = True) -> list[str]:
    base = [
        "return_1d", "return_2d", "return_5d",
        "ma_5", "ma_10", "ma_20",
        "ma_5_20_ratio", "ma_10_20_ratio",
        "vol_5", "vol_10",
        "rsi_14",
        "volume_change_1", "volume_ma_5", "volume_ma_20", "volume_relative",
        "atr_14",
        "bb_width", "bb_percent_b",
    ]

    if not include_market:
        return base

    market = [
        "SPY_close", "SPY_ret_1", "SPY_ma_20_ratio", "SPY_vol_10",
        "QQQ_close", "QQQ_ret_1", "QQQ_ma_20_ratio", "QQQ_vol_10",
        "IDX_VIX_close", "IDX_VIX_ret_1", "IDX_VIX_ma_20_ratio", "IDX_VIX_vol_10",
    ]
    return base + market
