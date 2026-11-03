
import pandas as pd
from psycopg import sql
from src.data.db_connection import get_db_connection

# --------------------------------------------------
# 1. Load the processed V2 dataset
# --------------------------------------------------

CSV_PATH = "data/processed/v2_features.csv"
TABLE_NAME = "v2_stock_features"

df = pd.read_csv(CSV_PATH)

# Convert the Date column to actual dates.
df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.date

# Replace pandas NaN values with Python None.
# Psycopg converts None into SQL NULL.
df = df.astype(object).where(pd.notna(df), None)

# --------------------------------------------------
# 2. Define the database column order
# --------------------------------------------------

columns = [
    "Date",
    "Ticker",
    "Sector",
    "Price_vs_MA30",
    "Volatility_20",
    "ATR_Percent",
    "Volume_Ratio_20",
    "RSI_14",
    "Nifty_Return",
    "Nifty_Volatility_20",
    "Market_Breadth",
    "Sector_Return",
    "Sector_Volatility_20",
    "Sector_Breadth",
    "Stock_vs_Market",
    "Stock_vs_Sector",
]

# --------------------------------------------------
# 3. Insert records into PostgreSQL
# --------------------------------------------------

insert_query = sql.SQL("""
        INSERT INTO v2_stock_features ({fields})
        VALUES ({placeholders})
        ON CONFLICT (date, ticker) DO UPDATE SET
        price_vs_ma30 = EXCLUDED.price_vs_ma30,
        volatility_20 = EXCLUDED.volatility_20,
        atr_percent = EXCLUDED.atr_percent,
        volume_ratio_20 = EXCLUDED.volume_ratio_20,
        rsi_14 = EXCLUDED.rsi_14,
        nifty_return = EXCLUDED.nifty_return,
        nifty_volatility_20 = EXCLUDED.nifty_volatility_20,
        market_breadth = EXCLUDED.market_breadth,
        sector_return = EXCLUDED.sector_return,
        sector_volatility_20 = EXCLUDED.sector_volatility_20,
        sector_breadth = EXCLUDED.sector_breadth,
        stock_vs_market = EXCLUDED.stock_vs_market,
        stock_vs_sector = EXCLUDED.stock_vs_sector
    """).format(
    fields=sql.SQL(", ").join(
        sql.Identifier(column.lower()) for column in columns
    ),
    placeholders=sql.SQL(", ").join(
        sql.Placeholder() for _ in columns
    ),
)

try:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                insert_query,
                df[columns].itertuples(index=False, name=None),
            )

            print("Migration completed.")
            print("CSV rows:", len(df))

            cursor.execute(
                "SELECT COUNT(*) FROM v2_stock_features;"
            )
            database_rows = cursor.fetchone()[0]

            print("Rows currently in database:", database_rows)

except Exception as error:
    print("Migration failed.")
    print(type(error).__name__, error)
    raise
