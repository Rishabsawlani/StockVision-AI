
import yfinance as yf
import pandas as pd

from src.data.db_connection import get_db_connection


# All stocks currently tracked by StockVision-AI
STOCKS = {
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HCLTECH.NS": "IT",

    "HDFCBANK.NS": "BANKING",
    "ICICIBANK.NS": "BANKING",
    "SBIN.NS": "BANKING",

    "RELIANCE.NS": "ENERGY",
    "ONGC.NS": "ENERGY",
    "NTPC.NS": "ENERGY",

    "MARUTI.NS": "AUTO",
    "M&M.NS": "AUTO",
    "EICHERMOT.NS": "AUTO",

    "SUNPHARMA.NS": "PHARMA",
    "CIPLA.NS": "PHARMA",
    "DRREDDY.NS": "PHARMA",
}


def update_stock_data():
    """Fetch recent prices and upsert them into PostgreSQL."""

    successful = []
    failed = []

    connection = get_db_connection()

    try:
        for ticker, sector in STOCKS.items():

            try:
                print(f"\nDownloading latest data for {ticker}...")

                # Fetch a small overlapping window to capture
                # the latest completed session and correct revisions.
                data = yf.download(
                    ticker,
                    period="10d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False
                )

                if data.empty:
                    raise ValueError("Yahoo Finance returned no data.")

                # Flatten columns if yfinance returns a MultiIndex.
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                required = ["Open", "High", "Low", "Close", "Volume"]

                if not all(column in data.columns for column in required):
                    raise ValueError("Required OHLCV columns are missing.")

                data = data[required].copy()
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                # Keep only completed trading sessions before today.
                today = pd.Timestamp.now().normalize()
                data = data[data.index.normalize() < today]
                data.index.name = "date"

                # Remove rows with missing prices.
                data = data.dropna(subset=["Open", "High", "Low", "Close"])

                rows = []

                for date, row in data.iterrows():
                    rows.append((
                        date.date(),
                        ticker,
                        sector,
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row["Volume"]) if pd.notna(row["Volume"]) else None
                    ))

                if not rows:
                    raise ValueError("No valid price rows remained.")

                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO stock_prices
                            (date, ticker, sector, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, ticker)
                        DO UPDATE SET
                            sector = EXCLUDED.sector,
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                        """,
                        rows
                    )

                connection.commit()

                successful.append(ticker)
                print(f"Updated {ticker}: {len(rows)} rows processed.")

            except Exception as error:
                connection.rollback()
                failed.append(ticker)
                print(f"Failed to update {ticker}: {error}")

    finally:
        connection.close()

    print("\n========== UPDATE SUMMARY ==========")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("Failed tickers:", ", ".join(failed))

    if failed:
        raise RuntimeError(
            f"Update incomplete. Failed tickers: {', '.join(failed)}"
        )


if __name__ == "__main__":
    update_stock_data()
