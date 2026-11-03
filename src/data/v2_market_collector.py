from pathlib import Path

import yfinance as yf


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_v2"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# MARKET / SECTOR INDICES
# ============================================================

MARKET_INDICES = {
    "NIFTY_50": "^NSEI",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_ENERGY": "^CNXENERGY"
}


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_index_data(index_name, ticker, start="2021-01-01", end=None):

    print(f"\nDownloading {index_name} ({ticker})...")

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
            return False

        if len(data) < 100:
            print(
                f"WARNING: Only {len(data)} rows returned for {ticker}. "
                f"Expected historical data."
            )
            return False

        # Handle Yahoo Finance MultiIndex columns
        if hasattr(data.columns, "levels"):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        print(f"Rows downloaded: {len(data)}")

        # Save using index name
        output_file = RAW_DATA_PATH / f"{index_name}.csv"

        data.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")

        return True

    except Exception as e:

        print(f"ERROR downloading {ticker}: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STOCKVISION-AI V2 MARKET / SECTOR DATA COLLECTION")
    print("=" * 60)

    successful = 0
    failed = 0

    for index_name, ticker in MARKET_INDICES.items():

        if download_index_data(index_name, ticker):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("MARKET / SECTOR DATA COLLECTION COMPLETED")
    print("=" * 60)

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")