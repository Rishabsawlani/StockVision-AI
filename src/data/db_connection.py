
import os
import streamlit as st
import psycopg
from dotenv import load_dotenv

# Load local environment variables for development
load_dotenv()


def get_config(key, default=None):
    """
    Read configuration from Streamlit Secrets first.
    Fall back to local environment variables.
    """
    try:
        value = st.secrets.get(key)
        if value is not None:
            return value
    except (FileNotFoundError, AttributeError):
        pass

    return os.getenv(key, default)


def get_db_connection():
    """Create and return a connection to the StockVision-AI database."""

    connection = psycopg.connect(
        host=get_config("DB_HOST", "localhost"),
        port=get_config("DB_PORT", "5432"),
        dbname=get_config("DB_NAME", "stockvision_ai"),
        user=get_config("DB_USER", "postgres"),
        password=get_config("DB_PASSWORD")
    )

    return connection
