import yfinance as yf
import pandas as pd

from pathlib import Path
import sys

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's module search path
sys.path.append(str(PROJECT_ROOT))


# Import the function from data_processor.py
from src.data.data_processor import load_stock_data


# Load existing TCS data
tcs_data = load_stock_data("TCS.NS")

# Get the date range from the TCS dataset
start_date = tcs_data.index.min().strftime("%Y-%m-%d")

# Yahoo Finance's end date is exclusive,
# so add one day to include the final TCS date.
end_date = (
    tcs_data.index.max() + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")


# Benchmark indices
benchmarks = {
    "Nifty_50": "^NSEI",
    "Nifty_IT": "^CNXIT"
}


for name, ticker in benchmarks.items():

    print(f"\nDownloading {name} data...")

    benchmark_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    # Handle MultiIndex columns if Yahoo Finance returns them
    if isinstance(benchmark_data.columns, pd.MultiIndex):
        benchmark_data.columns = (
            benchmark_data.columns.get_level_values(0)
        )

    # Keep only the closing price
    benchmark_data = benchmark_data[["Close"]].copy()

    # Rename the column
    benchmark_data.rename(
        columns={"Close": f"{name}_Close"},
        inplace=True
    )

    # Calculate daily percentage returns
    benchmark_data[f"{name}_Return"] = (
        benchmark_data[f"{name}_Close"].pct_change() * 100
    )

    print(f"\n{name} first 5 rows:")
    print(benchmark_data.head())

    print(f"\n{name} last 5 rows:")
    print(benchmark_data.tail())

    print(f"\n{name} dataset shape:")
    print(benchmark_data.shape)

    print(f"\n{name} missing values:")
    print(benchmark_data.isna().sum())