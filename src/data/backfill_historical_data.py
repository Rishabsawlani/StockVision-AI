
import pandas as pd
import yfinance as yf

from src.data.db_connection import get_db_connection
from src.data.stock_universe import STOCKS


START_DATE = "2021-01-01"


def main():
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT ticker FROM stock_prices")
            existing_tickers = {
                row[0] for row in cursor.fetchall()
            }

        new_tickers = {
            ticker: sector
            for ticker, sector in STOCKS.items()
            if ticker not in existing_tickers
        }

        print(f"Existing tickers: {len(existing_tickers)}")
        print(f"New tickers to import: {len(new_tickers)}")

        if not new_tickers:
            print("No new tickers require historical data.")
            return

        today = pd.Timestamp.now().normalize()
        end_date = today.strftime("%Y-%m-%d")

        successful = 0
        failed = []

        for ticker, sector in new_tickers.items():
            try:
                print(f"\nDownloading {ticker}...")

                data = yf.download(
                    ticker,
                    start=START_DATE,
                    end=end_date,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False
                )

                if data.empty:
                    raise ValueError("No historical data returned")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                required_columns = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                data = data[required_columns].copy()

                data.index = pd.to_datetime(data.index)

                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)

                # Exclude today's potentially incomplete candle.
                data = data[data.index.normalize() < today]

                data = data.dropna(
                    subset=["Open", "High", "Low", "Close"]
                )

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
                    raise ValueError("No valid historical rows available")

                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO stock_prices (
                            date,
                            ticker,
                            sector,
                            open,
                            high,
                            low,
                            close,
                            volume
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, ticker) DO NOTHING
                        """,
                        rows
                    )

                connection.commit()

                successful += 1
                print(
                    f"[OK] {ticker}: "
                    f"{len(rows)} historical rows processed"
                )

            except Exception as error:
                connection.rollback()
                failed.append((ticker, str(error)))
                print(f"[FAIL] {ticker}: {error}")

        print("\n========== BACKFILL SUMMARY ==========")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed tickers:")
            for ticker, error in failed:
                print(f"{ticker}: {error}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
