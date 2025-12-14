from pathlib import Path
import pandas as pd
import urllib.request

OUT = Path("data") / "tickers.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def clean_symbol(sym: str) -> str:
    sym = str(sym).strip()
    sym = sym.replace(".", "-")  # BRK.B -> BRK-B (Yahoo style)
    return sym.upper()

def fetch_csv(url: str) -> pd.DataFrame:
    """
    Robust CSV fetch (works in Codespaces).
    Uses pandas read_csv directly from URL.
    """
    return pd.read_csv(url)

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["symbol", "name", "exchange", "category"]:
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].map(clean_symbol)
    df["name"] = df["name"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    return df[["symbol", "name", "exchange", "category"]]

def add_category(df: pd.DataFrame, category: str, exchange: str = "") -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = df["symbol"].map(clean_symbol)
    df["name"] = df["name"].astype(str).str.strip()
    df["exchange"] = exchange
    df["category"] = category
    return ensure_cols(df)

def load_sp500() -> pd.DataFrame:
    # reliable public dataset (no wiki parsing)
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = fetch_csv(url)
    df = df.rename(columns={"Symbol": "symbol", "Name": "name"})
    return add_category(df[["symbol", "name"]], "S&P 500", "NYSE/NASDAQ")

def load_nasdaq100() -> pd.DataFrame:
    # curated list
    url = "https://raw.githubusercontent.com/ranaroussi/yfinance/master/tests/data/nasdaq100.csv"
    df = fetch_csv(url)
    # this file usually has Symbol column
    if "Symbol" in df.columns:
        df = df.rename(columns={"Symbol": "symbol"})
    if "symbol" not in df.columns:
        df = df.rename(columns={df.columns[0]: "symbol"})
    if "name" not in df.columns:
        # sometimes "Name" exists
        if "Name" in df.columns:
            df = df.rename(columns={"Name": "name"})
        else:
            df["name"] = df["symbol"]
    return add_category(df[["symbol", "name"]], "NASDAQ 100", "NASDAQ")

def load_dow30() -> pd.DataFrame:
    # small curated list (stable)
    rows = [
        ("AAPL","Apple Inc"), ("AMGN","Amgen Inc"), ("AXP","American Express"),
        ("BA","Boeing"), ("CAT","Caterpillar"), ("CRM","Salesforce"),
        ("CSCO","Cisco"), ("CVX","Chevron"), ("DIS","Disney"),
        ("DOW","Dow Inc"), ("GS","Goldman Sachs"), ("HD","Home Depot"),
        ("HON","Honeywell"), ("IBM","IBM"), ("INTC","Intel"),
        ("JNJ","Johnson & Johnson"), ("JPM","JPMorgan Chase"),
        ("KO","Coca-Cola"), ("MCD","McDonald's"), ("MMM","3M"),
        ("MRK","Merck"), ("MSFT","Microsoft"), ("NKE","Nike"),
        ("PG","Procter & Gamble"), ("TRV","Travelers"),
        ("UNH","UnitedHealth"), ("V","Visa"), ("VZ","Verizon"),
        ("WBA","Walgreens Boots Alliance"), ("WMT","Walmart"),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "name"])
    return add_category(df, "Dow 30", "NYSE/NASDAQ")

def load_extras() -> pd.DataFrame:
    extras = [
        # ETFs
        ("SPY", "SPDR S&P 500 ETF Trust", "NYSEARCA", "ETF"),
        ("QQQ", "Invesco QQQ Trust", "NASDAQ", "ETF"),
        ("IWM", "iShares Russell 2000 ETF", "NYSEARCA", "ETF"),
        ("DIA", "SPDR Dow Jones Industrial Average ETF", "NYSEARCA", "ETF"),
        ("VTI", "Vanguard Total Stock Market ETF", "NYSEARCA", "ETF"),
        ("VOO", "Vanguard S&P 500 ETF", "NYSEARCA", "ETF"),
        ("TLT", "iShares 20+ Year Treasury Bond ETF", "NASDAQ", "ETF"),
        ("GLD", "SPDR Gold Shares", "NYSEARCA", "Commodity"),
        ("SLV", "iShares Silver Trust", "NYSEARCA", "Commodity"),

        # Indexes (Yahoo symbols)
        ("^GSPC", "S&P 500 Index", "Index", "Index"),
        ("^IXIC", "NASDAQ Composite Index", "Index", "Index"),
        ("^DJI", "Dow Jones Industrial Average", "Index", "Index"),
        ("^VIX", "CBOE Volatility Index", "Index", "Index"),

        # Crypto (Yahoo symbols)
        ("BTC-USD", "Bitcoin", "Crypto", "Crypto"),
        ("ETH-USD", "Ethereum", "Crypto", "Crypto"),
        ("SOL-USD", "Solana", "Crypto", "Crypto"),
        ("XRP-USD", "XRP", "Crypto", "Crypto"),
        ("DOGE-USD", "Dogecoin", "Crypto", "Crypto"),
    ]
    df = pd.DataFrame(extras, columns=["symbol", "name", "exchange", "category"])
    return ensure_cols(df)

def main():
    parts = []
    # Each loader is isolated so one failure doesn't break the whole build
    for loader in [load_sp500, load_nasdaq100, load_dow30, load_extras]:
        try:
            df = loader()
            if df is not None and not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"⚠️ Source failed: {loader.__name__}: {e}")

    if not parts:
        raise RuntimeError("No ticker sources succeeded. Check network access.")

    out = pd.concat(parts, ignore_index=True)
    out["symbol"] = out["symbol"].map(clean_symbol)
    out = out.drop_duplicates(subset=["symbol"], keep="first")
    out = out.sort_values(["category", "symbol"]).reset_index(drop=True)

    out.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(out):,} tickers to {OUT}")

if __name__ == "__main__":
    main()
