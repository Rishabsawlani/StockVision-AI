
from pathlib import Path

import pandas as pd
from psycopg import sql

from src.data.db_connection import get_db_connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_v2"

STOCK_UNIVERSE = {
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS"],
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS"],
    "AUTO": ["MARUTI.NS", "M&M.NS", "EICHERMOT.NS"],
    "PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"],
}

COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_raw_prices():
    """Load and combine raw OHLCV data for all 15 stocks."""

    frames = []

    for sector, tickers in STOCK_UNIVERSE.items():
        for ticker in tickers:
            filename = ticker.replace(".", "_") + ".csv"
            filepath = RAW_DATA_PATH / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Missing raw CSV for {ticker}: {filepath}"
                )

            df = pd.read_csv(filepath)

            missing_columns = set(COLUMNS) - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"{ticker} is missing columns: {sorted(missing_columns)}"
                )

            df = df[COLUMNS].copy()
            df["Date"] = pd.to_datetime(
                df["Date"], errors="raise"
            ).dt.date

            df["Ticker"] = ticker
            df["Sector"] = sector

            # Convert missing numeric values to SQL NULL.
            df = df.astype(object).where(pd.notna(df), None)

            frames.append(df)

            print(f"{ticker}: {len(df)} rows loaded from CSV")

    combined = pd.concat(frames, ignore_index=True)

    if combined.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate date/ticker records found in raw CSVs.")

    return combined


def migrate_raw_prices():
    df = load_raw_prices()

    insert_query = sql.SQL("""
        INSERT INTO stock_prices
            (date, ticker, sector, open, high, low, close, volume)
        VALUES ({})
        ON CONFLICT (date, ticker) DO NOTHING
    """).format(
        sql.SQL(", ").join(sql.Placeholder() for _ in range(8))
    )

    columns = [
        "Date", "Ticker", "Sector",
        "Open", "High", "Low", "Close", "Volume"
    ]

    records = list(df[columns].itertuples(index=False, name=None))

    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(insert_query, records)

                cursor.execute("SELECT COUNT(*) FROM stock_prices;")
                total_rows = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(DISTINCT ticker) FROM stock_prices;"
                )
                total_tickers = cursor.fetchone()[0]

        print("\nRaw price migration completed.")
        print("CSV records:", len(df))
        print("Rows currently in stock_prices:", total_rows)
        print("Distinct stocks:", total_tickers)

    except Exception as error:
        print("\nRaw price migration failed.")
        print(type(error).__name__, error)
        raise


if __name__ == "__main__":
    migrate_raw_prices()
