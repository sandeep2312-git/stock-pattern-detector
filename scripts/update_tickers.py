from pathlib import Path
import pandas as pd

OUT = Path("data") / "tickers.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["symbol", "name", "exchange", "category"]:
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["symbol"] != ""].drop_duplicates(subset=["symbol"])
    return df[["symbol","name","exchange","category"]]

def curated() -> pd.DataFrame:
    rows = [
        # US stocks
        ("AAPL","Apple Inc","NASDAQ","Stock"),
        ("MSFT","Microsoft Corporation","NASDAQ","Stock"),
        ("NVDA","NVIDIA Corporation","NASDAQ","Stock"),
        ("AMZN","Amazon.com Inc","NASDAQ","Stock"),
        ("GOOGL","Alphabet Inc Class A","NASDAQ","Stock"),
        ("META","Meta Platforms Inc","NASDAQ","Stock"),
        ("TSLA","Tesla Inc","NASDAQ","Stock"),
        ("AMD","Advanced Micro Devices Inc","NASDAQ","Stock"),
        ("INTC","Intel Corporation","NASDAQ","Stock"),
        ("NFLX","Netflix Inc","NASDAQ","Stock"),
        ("AVGO","Broadcom Inc","NASDAQ","Stock"),
        ("CRM","Salesforce Inc","NYSE","Stock"),
        ("ORCL","Oracle Corporation","NYSE","Stock"),
        ("IBM","IBM","NYSE","Stock"),
        ("ADBE","Adobe Inc","NASDAQ","Stock"),
        ("PYPL","PayPal Holdings Inc","NASDAQ","Stock"),
        ("UBER","Uber Technologies Inc","NYSE","Stock"),
        ("SHOP","Shopify Inc","NYSE","Stock"),
        ("SQ","Block Inc","NYSE","Stock"),

        # ETFs
        ("SPY","SPDR S&P 500 ETF Trust","NYSEARCA","ETF"),
        ("QQQ","Invesco QQQ Trust","NASDAQ","ETF"),
        ("DIA","SPDR Dow Jones Industrial Average ETF","NYSEARCA","ETF"),
        ("IWM","iShares Russell 2000 ETF","NYSEARCA","ETF"),
        ("VOO","Vanguard S&P 500 ETF","NYSEARCA","ETF"),
        ("VTI","Vanguard Total Stock Market ETF","NYSEARCA","ETF"),
        ("XLK","Technology Select Sector SPDR Fund","NYSEARCA","ETF"),
        ("XLF","Financial Select Sector SPDR Fund","NYSEARCA","ETF"),
        ("XLE","Energy Select Sector SPDR Fund","NYSEARCA","ETF"),
        ("ARKK","ARK Innovation ETF","NYSEARCA","ETF"),

        # Commodities
        ("GLD","SPDR Gold Shares","NYSEARCA","Commodity"),
        ("SLV","iShares Silver Trust","NYSEARCA","Commodity"),
        ("USO","United States Oil Fund","NYSEARCA","Commodity"),

        # Crypto (Yahoo symbols)
        ("BTC-USD","Bitcoin USD","Crypto","Crypto"),
        ("ETH-USD","Ethereum USD","Crypto","Crypto"),
        ("SOL-USD","Solana USD","Crypto","Crypto"),
        ("BNB-USD","Binance Coin USD","Crypto","Crypto"),
        ("XRP-USD","Ripple USD","Crypto","Crypto"),
        ("ADA-USD","Cardano USD","Crypto","Crypto"),
        ("DOGE-USD","Dogecoin USD","Crypto","Crypto"),

        # India (NSE suffix)
        ("RELIANCE.NS","Reliance Industries Ltd","NSE India","Stock"),
        ("TCS.NS","Tata Consultancy Services","NSE India","Stock"),
        ("INFY.NS","Infosys Ltd","NSE India","Stock"),
        ("HDFCBANK.NS","HDFC Bank Ltd","NSE India","Stock"),
        ("ICICIBANK.NS","ICICI Bank Ltd","NSE India","Stock"),
        ("SBIN.NS","State Bank of India","NSE India","Stock"),
    ]
    return pd.DataFrame(rows, columns=["symbol","name","exchange","category"])

def main():
    df = normalize(curated())
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} tickers -> {OUT}")

if __name__ == "__main__":
    main()
