
from pathlib import Path

import pandas as pd

from src.data.db_connection import get_db_connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_v2"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connection = get_db_connection()

    try:
        query = """
            SELECT date, ticker, open, high, low, close, volume
            FROM stock_prices
            ORDER BY ticker, date
        """

        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]

        data = pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()

    data["date"] = pd.to_datetime(data["date"])

    for ticker, stock_data in data.groupby("ticker"):
        filename = ticker.replace(".", "_") + ".csv"
        output_path = OUTPUT_DIR / filename

        stock_data = stock_data.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        stock_data = stock_data[
            ["Date", "Open", "High", "Low", "Close", "Volume"]
        ]

        stock_data.to_csv(output_path, index=False)

        print(f"[OK] {ticker}: {len(stock_data)} rows exported")

    print(f"\nExport complete. Files saved to: {OUTPUT_DIR}")
    print(f"Total database rows exported: {len(data):,}")
    print(f"Total tickers exported: {data['ticker'].nunique()}")


if __name__ == "__main__":
    main()
