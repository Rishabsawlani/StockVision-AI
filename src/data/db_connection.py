
import os
import psycopg
from dotenv import load_dotenv

# Load database credentials from the project's .env file
load_dotenv()

def get_db_connection():
    """Create and return a connection to the StockVision-AI database."""

    connection = psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "stockvision_ai"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD")
    )

    return connection
