
from src.data.db_connection import get_db_connection


def validate_database():
    """Validate the migrated V2 stock features in PostgreSQL."""

    queries = {
        "total_rows": """
            SELECT COUNT(*)
            FROM v2_stock_features;
        """,

        "distinct_stocks": """
            SELECT COUNT(DISTINCT ticker)
            FROM v2_stock_features;
        """,

        "date_range": """
            SELECT MIN(date), MAX(date)
            FROM v2_stock_features;
        """,

        "missing_price_feature": """
            SELECT COUNT(*)
            FROM v2_stock_features
            WHERE price_vs_ma30 IS NULL;
        """,

        "rows_per_stock": """
            SELECT ticker, COUNT(*) AS row_count
            FROM v2_stock_features
            GROUP BY ticker
            ORDER BY ticker;
        """
    }

    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:

                for check_name, query in queries.items():
                    cursor.execute(query)
                    results = cursor.fetchall()

                    print(f"\n--- {check_name.upper()} ---")

                    for row in results:
                        print(row)

        print("\nDatabase validation completed successfully.")

    except Exception as error:
        print("\nDatabase validation failed.")
        print(type(error).__name__, error)
        raise


if __name__ == "__main__":
    validate_database()
