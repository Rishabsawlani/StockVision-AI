import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
V2_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "v2_features.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_v2"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "v2_regression_features.csv"


def build_5day_target():
    """
    Build the 5-trading-day future return target for every stock.

    Target:
        Future_Return_5D =
        ((Close[t+5] - Close[t]) / Close[t]) * 100

    The shift is performed separately for each stock so that
    t+5 means five trading observations later for that stock.
    """

    # ---------------------------------------------------------
    # Load the existing V2 feature dataset
    # ---------------------------------------------------------
    print("Loading V2 feature dataset...")

    v2_df = pd.read_csv(V2_FEATURES_PATH)

    v2_df["Date"] = pd.to_datetime(v2_df["Date"])

    print(f"V2 dataset loaded: {v2_df.shape[0]} rows")


    # ---------------------------------------------------------
    # Get the stocks present in V2
    # ---------------------------------------------------------
    tickers = v2_df["Ticker"].unique()

    print(f"Stocks found: {len(tickers)}")


    # ---------------------------------------------------------
    # Calculate 5-day future return for every stock
    # ---------------------------------------------------------
    target_frames = []

    for ticker in tickers:

        print(f"Processing {ticker}...")

        # Convert ticker name into raw filename format.
        # Example:
        # TCS.NS -> TCS_NS.csv
        filename = ticker.replace(".", "_") + ".csv"

        raw_path = RAW_DATA_PATH / filename

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found for {ticker}: {raw_path}"
            )

        # Load raw stock data
        stock_df = pd.read_csv(raw_path)

        # Convert Date into datetime
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])

        # Sort chronologically
        stock_df = stock_df.sort_values("Date").reset_index(drop=True)

        # Calculate the closing price 5 trading observations ahead.
        stock_df["Future_Close_5D"] = stock_df["Close"].shift(-5)

        # Calculate 5-day future percentage return.
        stock_df["Future_Return_5D"] = (
            (stock_df["Future_Close_5D"] - stock_df["Close"])
            / stock_df["Close"]
        ) * 100

        # Keep only what we need for merging.
        target_df = stock_df[
            ["Date", "Future_Return_5D"]
        ].copy()

        target_df["Ticker"] = ticker

        target_frames.append(target_df)


    # ---------------------------------------------------------
    # Combine all stock targets
    # ---------------------------------------------------------
    target_df = pd.concat(
        target_frames,
        ignore_index=True
    )


    # ---------------------------------------------------------
    # Merge target with V2 features
    # ---------------------------------------------------------
    regression_df = v2_df.merge(
        target_df,
        on=["Ticker", "Date"],
        how="left",
        validate="one_to_one"
    )


    # ---------------------------------------------------------
    # Remove columns that belong specifically to the
    # previous next-day classification experiment.
    # ---------------------------------------------------------
    columns_to_remove = [
        "Next_Day_Return",
        "Target"
    ]

    regression_df = regression_df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


    # ---------------------------------------------------------
    # Sort the final dataset
    # ---------------------------------------------------------
    regression_df = regression_df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)


    # ---------------------------------------------------------
    # Save dataset
    # ---------------------------------------------------------
    regression_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # ---------------------------------------------------------
    # Basic verification
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("5-DAY REGRESSION DATASET CREATED")
    print("=" * 60)

    print(f"Rows: {len(regression_df)}")
    print(f"Columns: {len(regression_df.columns)}")

    print(
        f"Missing Future_Return_5D: "
        f"{regression_df['Future_Return_5D'].isna().sum()}"
    )

    print(
        f"Future_Return_5D mean: "
        f"{regression_df['Future_Return_5D'].mean():.4f}%"
    )

    print(
        f"Future_Return_5D std: "
        f"{regression_df['Future_Return_5D'].std():.4f}%"
    )

    print(
        f"Future_Return_5D min: "
        f"{regression_df['Future_Return_5D'].min():.4f}%"
    )

    print(
        f"Future_Return_5D max: "
        f"{regression_df['Future_Return_5D'].max():.4f}%"
    )

    print("\nDataset columns:")
    print(regression_df.columns.tolist())

    print(f"\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_5day_target()