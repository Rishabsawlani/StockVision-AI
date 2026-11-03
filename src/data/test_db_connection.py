
from src.data.db_connection import get_db_connection

try:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), current_user, version();"
            )

            result = cursor.fetchone()

            print("Database connection successful!")
            print("Database:", result[0])
            print("User:", result[1])
            print("PostgreSQL:", result[2].split(",")[0])

except Exception as error:
    print("Database connection failed.")
    print(type(error).__name__, error)
