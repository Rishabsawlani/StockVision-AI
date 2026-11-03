
from src.data.db_connection import get_db_connection
from src.data.stock_universe import STOCKS


def main():
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    ticker,
                    COUNT(*) AS row_count,
                    MIN(date) AS first_date,
                    MAX(date) AS latest_date,
                    COUNT(*) FILTER (
                        WHERE close IS NULL OR close <= 0
                    ) AS invalid_closes,
                    COUNT(*) - COUNT(DISTINCT date) AS duplicate_dates
                FROM stock_prices
                GROUP BY ticker
                ORDER BY ticker
            """)

            results = cursor.fetchall()

        print("\n========== DATABASE VALIDATION ==========")

        found_tickers = set()

        for ticker, count, first_date, latest_date, invalid, duplicates in results:
            found_tickers.add(ticker)

            if ticker in STOCKS:
                print(
                    f"{ticker:<18} rows={count:>5} | "
                    f"{first_date} to {latest_date} | "
                    f"invalid_closes={invalid} | "
                    f"duplicate_dates={duplicates}"
                )

        missing = set(STOCKS) - found_tickers

        print("\n========== SUMMARY ==========")
        print(f"Expected tickers: {len(STOCKS)}")
        print(f"Found tickers:    {len(set(STOCKS) & found_tickers)}")
        print(f"Missing tickers:  {len(missing)}")

        if missing:
            print("Missing:", ", ".join(sorted(missing)))

        print("\nCheck JIOFIN.NS specifically for its earliest date.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
