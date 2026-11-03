
import pandas as pd
from pathlib import Path

raw_folder = Path("data/raw_v2")

tickers = [
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "MARUTI.NS",
    "NTPC.NS",
]

for ticker in tickers:
    filename = ticker.replace(".", "_") + ".csv"
    file_path = raw_folder / filename

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"\n--- {ticker} ---")
    print("Total rows:", len(df))
    print("Last 10 dates:")
    print(df["Date"].tail(10).to_string(index=False))
