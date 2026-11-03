from pathlib import Path

import yfinance as yf
import pandas as pd


# --------------------------------------------------
# Project setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_v2"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Stock universe
# --------------------------------------------------

STOCK_UNIVERSE = {
    "IT": [
        "TCS.NS",
        "INFY.NS",
        "HCLTECH.NS"
    ],

    "BANKING": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS"
    ],

    "ENERGY": [
        "RELIANCE.NS",
        "ONGC.NS",
        "NTPC.NS"
    ],

    "AUTO": [
        "MARUTI.NS",
        "M&M.NS",
        "EICHERMOT.NS"
    ],

    "PHARMA": [
        "SUNPHARMA.NS",
        "CIPLA.NS",
        "DRREDDY.NS"
    ]
}


# --------------------------------------------------
# Market / sector indices
# --------------------------------------------------

MARKET_INDICES = {
    "NIFTY_50": "^NSEI",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT"
}


# --------------------------------------------------
# Raw data directory
# --------------------------------------------------

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_v2"


# --------------------------------------------------
# Download stock data
# --------------------------------------------------

def download_stock_data(ticker, start="2021-01-01", end=None):

    print(f"\nDownloading {ticker}...")

    try:

        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print(f"WARNING: No data returned for {ticker}")
            return None

        # Handle possible MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        print(f"Rows downloaded: {len(data)}")

        return data

    except Exception as e:

        print(f"ERROR downloading {ticker}: {e}")
        return None


# --------------------------------------------------
# Save stock data
# --------------------------------------------------

def save_stock_data(data, ticker):

    filename = ticker.replace(".", "_") + ".csv"

    filepath = RAW_DATA_PATH / filename

    data.to_csv(filepath, index=False)

    print(f"Saved: {filepath}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    RAW_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("STOCKVISION-AI V2 DATA COLLECTION")
    print("=" * 60)

    for sector, stocks in STOCK_UNIVERSE.items():

        print(f"\n{'=' * 60}")
        print(f"SECTOR: {sector}")
        print(f"{'=' * 60}")

        for ticker in stocks:

            data = download_stock_data(ticker)

            if data is not None:

                save_stock_data(
                    data,
                    ticker
                )

    print("\nData collection completed.")