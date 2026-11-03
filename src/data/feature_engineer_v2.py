from pathlib import Path
import pandas as pd
import numpy as np
from src.data.stock_universe import STOCKS

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_v2"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# STOCK UNIVERSE
# ============================================================

STOCK_UNIVERSE = {}

for ticker, sector in STOCKS.items():
    if sector not in STOCK_UNIVERSE:
        STOCK_UNIVERSE[sector] = []

    STOCK_UNIVERSE[sector].append(ticker)


# ============================================================
# CANDIDATE MODEL FEATURES
# ============================================================
#
# These are candidate features, NOT automatically our final
# model feature set.
#
# We will later test redundancy and predictive usefulness.
#
# ============================================================

MODEL_FEATURES = [

    # Stock state
    "Price_vs_MA30",
    "Volatility_20",
    "ATR_Percent",
    "Volume_Ratio_20",
    "RSI_14",

    # Market state
    "Nifty_Return",
    "Nifty_Volatility_20",
    "Market_Breadth",

    # Sector state
    "Sector_Return",
    "Sector_Volatility_20",
    "Sector_Breadth",

    # Relative state
    "Stock_vs_Market",
    "Stock_vs_Sector"
]


# ============================================================
# HELPER: TICKER → CSV FILENAME
# ============================================================

def ticker_to_filename(ticker):
    """
    Converts Yahoo Finance ticker into our CSV filename.

    Example:
        TCS.NS → TCS_NS.csv
        M&M.NS → M&M_NS.csv
    """

    return ticker.replace(".", "_") + ".csv"


# ============================================================
# LOAD ONE STOCK
# ============================================================

def load_stock_data(ticker, sector):
    """
    Loads one stock's raw OHLCV data and adds identifiers.
    """

    file_path = RAW_DATA_PATH / ticker_to_filename(ticker)

    data = pd.read_csv(file_path)

    # Convert date
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort chronologically
    data = data.sort_values("Date").reset_index(drop=True)

    # Keep only required OHLCV columns
    data = data[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ].copy()

    # Add identifiers
    data["Ticker"] = ticker
    data["Sector"] = sector

    return data


# ============================================================
# LOAD ALL STOCKS
# ============================================================


def load_all_stocks():
    """
    Loads all stocks into one long-format dataframe.

    One row represents one stock on one trading day.
    """

    all_data = []
    failed_tickers = []

    for sector, tickers in STOCK_UNIVERSE.items():
        for ticker in tickers:
            try:
                data = load_stock_data(ticker, sector)

                if data.empty:
                    print(f"[WARNING] {ticker}: CSV contains no rows")
                    failed_tickers.append(ticker)
                    continue

                all_data.append(data)
                print(f"[OK] {ticker}: {len(data)} rows loaded")

            except Exception as error:
                print(f"[ERROR] {ticker}: {error}")
                failed_tickers.append(ticker)

    if failed_tickers:
        raise RuntimeError(
            "Could not load these tickers: "
            + ", ".join(failed_tickers)
        )

    if not all_data:
        raise ValueError("No stock data was loaded.")

    combined_data = pd.concat(
        all_data,
        ignore_index=True
    )

    print(f"\nTotal stocks loaded: {len(all_data)}")
    print(f"Total raw rows loaded: {len(combined_data):,}")

    return combined_data



# ============================================================
# STOCK-LEVEL FEATURES
# ============================================================

def create_stock_features(data):
    """
    Creates stock-level candidate features.

    All calculations use current or historical information only.
    """

    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    # --------------------------------------------------------
    # Daily Return
    # --------------------------------------------------------
    #
    # Intermediate feature.
    # Used to construct other features.
    # NOT directly used as a final model feature.
    #

    data["Daily_Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change() * 100
    )

    # --------------------------------------------------------
    # 30-Day Moving Average
    # --------------------------------------------------------

    data["MA_30"] = (
        data.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Price vs MA30
    # --------------------------------------------------------
    #
    # Measures where the stock currently sits relative
    # to its recent 30-day price level.
    #

    data["Price_vs_MA30"] = (
        (
            data["Close"] - data["MA_30"]
        )
        / data["MA_30"]
    ) * 100

    # --------------------------------------------------------
    # 20-Day Return Volatility
    # --------------------------------------------------------
    #
    # Measures variability of the stock's recent returns.
    #

    data["Volatility_20"] = (
        data.groupby("Ticker")["Daily_Return"]
        .transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).std()
        )
    )

    # --------------------------------------------------------
    # Previous Close
    # --------------------------------------------------------

    previous_close = (
        data.groupby("Ticker")["Close"]
        .shift(1)
    )

    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------
    #
    # Uses intraday High/Low and previous Close.
    #

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (
                data["High"] - previous_close
            ).abs(),
            (
                data["Low"] - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    data["True_Range"] = true_range

    # --------------------------------------------------------
    # ATR 14
    # --------------------------------------------------------

    data["ATR_14"] = (
        data.groupby("Ticker")["True_Range"]
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=14
            ).mean()
        )
    )

    # --------------------------------------------------------
    # ATR Percentage
    # --------------------------------------------------------
    #
    # Normalizes ATR by stock price so different stocks
    # can be compared.
    #

    data["ATR_Percent"] = (
        data["ATR_14"]
        / data["Close"]
    ) * 100

    # --------------------------------------------------------
    # 20-Day Average Volume
    # --------------------------------------------------------

    average_volume = (
        data.groupby("Ticker")["Volume"]
        .transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Volume Ratio
    # --------------------------------------------------------
    #
    # 1.0 = normal volume
    # >1.0 = unusually high volume
    # <1.0 = unusually low volume
    #

    data["Volume_Ratio_20"] = (
        data["Volume"]
        / average_volume
    )

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    price_change = (
        data.groupby("Ticker")["Close"]
        .diff()
    )

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = (
        gain.groupby(data["Ticker"])
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=14
            ).mean()
        )
    )

    average_loss = (
        loss.groupby(data["Ticker"])
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=14
            ).mean()
        )
    )

    rs = average_gain / average_loss

    data["RSI_14"] = (
        100
        - (
            100 / (1 + rs)
        )
    )

    return data


# ============================================================
# MARKET FEATURES
# ============================================================

def create_market_features(data):
    """
    Creates broad-market features using NIFTY 50 and
    the 15-stock universe.
    """

    # --------------------------------------------------------
    # Load NIFTY 50
    # --------------------------------------------------------

    nifty_path = (
        RAW_DATA_PATH / "NIFTY_50.csv"
    )

    nifty = pd.read_csv(nifty_path)

    nifty["Date"] = pd.to_datetime(
        nifty["Date"]
    )

    nifty = nifty.sort_values(
        "Date"
    ).copy()

    # --------------------------------------------------------
    # NIFTY Daily Return
    # --------------------------------------------------------

    nifty["Nifty_Return"] = (
        nifty["Close"].pct_change() * 100
    )

    # --------------------------------------------------------
    # NIFTY 20-Day Volatility
    # --------------------------------------------------------

    nifty["Nifty_Volatility_20"] = (
        nifty["Nifty_Return"]
        .rolling(
            window=20,
            min_periods=20
        )
        .std()
    )

    market_features = nifty[
        [
            "Date",
            "Nifty_Return",
            "Nifty_Volatility_20"
        ]
    ].copy()

    # --------------------------------------------------------
    # Market Breadth
    # --------------------------------------------------------
    #
    # Fraction of our 15 stocks with positive returns
    # on the current date.
    #

    market_breadth = (
        data.groupby("Date")["Daily_Return"]
        .apply(
            lambda x:
            (x > 0).sum() / x.notna().sum()
            if x.notna().sum() > 0
            else np.nan
        )
        .reset_index(
            name="Market_Breadth"
        )
    )

    market_features = market_features.merge(
        market_breadth,
        on="Date",
        how="left"
    )

    # --------------------------------------------------------
    # Merge market features onto stock rows
    # --------------------------------------------------------

    data = data.merge(
        market_features,
        on="Date",
        how="left"
    )

    return data


# ============================================================
# SECTOR FEATURES
# ============================================================

def create_sector_features(data):
    """
    Creates sector-level features from the three stocks
    belonging to each sector.
    """

    # --------------------------------------------------------
    # Sector Return
    # --------------------------------------------------------
    #
    # Mean return of the three stocks in the sector.
    #

    sector_features = (
        data.groupby(
            ["Date", "Sector"]
        )["Daily_Return"]
        .mean()
        .rename("Sector_Return")
        .reset_index()
    )

    # --------------------------------------------------------
    # Sector Volatility
    # --------------------------------------------------------

    sector_features = (
        sector_features
        .sort_values(
            ["Sector", "Date"]
        )
    )

    sector_features[
        "Sector_Volatility_20"
    ] = (
        sector_features
        .groupby("Sector")["Sector_Return"]
        .transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).std()
        )
    )

    # --------------------------------------------------------
    # Sector Breadth
    # --------------------------------------------------------
    #
    # Fraction of sector stocks with positive returns.
    #

    sector_breadth = (
        data.groupby(
            ["Date", "Sector"]
        )["Daily_Return"]
        .apply(
            lambda x:
            (x > 0).sum() / x.notna().sum()
            if x.notna().sum() > 0
            else np.nan
        )
        .rename("Sector_Breadth")
        .reset_index()
    )

    # --------------------------------------------------------
    # Merge breadth
    # --------------------------------------------------------

    sector_features = sector_features.merge(
        sector_breadth,
        on=["Date", "Sector"],
        how="left"
    )

    # --------------------------------------------------------
    # Merge sector features onto stock rows
    # --------------------------------------------------------

    data = data.merge(
        sector_features,
        on=["Date", "Sector"],
        how="left"
    )

    return data


# ============================================================
# RELATIVE FEATURES
# ============================================================

def create_relative_features(data):
    """
    Creates contextual features describing how a stock
    performed relative to the market and its sector.
    """

    # --------------------------------------------------------
    # Stock vs Market
    # --------------------------------------------------------
    #
    # Positive → stock outperformed NIFTY
    # Negative → stock underperformed NIFTY
    #

    data["Stock_vs_Market"] = (
        data["Daily_Return"]
        - data["Nifty_Return"]
    )

    # --------------------------------------------------------
    # Stock vs Sector
    # --------------------------------------------------------
    #
    # Positive → stock outperformed its sector
    # Negative → stock underperformed its sector
    #

    data["Stock_vs_Sector"] = (
        data["Daily_Return"]
        - data["Sector_Return"]
    )

    return data


# ============================================================
# TARGET
# ============================================================

def create_target(data):
    """
    Creates the next-day 3-class classification target.

    DOWN:
        Next-day return < -0.25%

    NEUTRAL:
        -0.25% <= next-day return <= +0.25%

    UP:
        Next-day return > +0.25%
    """

    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    # --------------------------------------------------------
    # Next day's closing price
    # --------------------------------------------------------

    next_close = (
        data.groupby("Ticker")["Close"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # Next-day return
    # --------------------------------------------------------

    data["Next_Day_Return"] = (
        (
            next_close - data["Close"]
        )
        / data["Close"]
    ) * 100


    # --------------------------------------------------------
    # Classification target
    # --------------------------------------------------------

    data["Target"] = np.select(
        [
            data["Next_Day_Return"].isna(),
            data["Next_Day_Return"] < -0.25,
            data["Next_Day_Return"] > 0.25
        ],
        [
            np.nan,  # Unknown next-day return
            0,       # DOWN
            2        # UP
        ],
        default=1  # NEUTRAL
    )

    return data


# ============================================================
# FINAL CLEANUP
# ============================================================

def prepare_final_dataset(data):
    """
    Keeps only the columns needed for the V2 analysis.

    Intermediate calculation columns are removed.
    """

    identifier_columns = [
        "Date",
        "Ticker",
        "Sector"
    ]

    target_columns = [
        "Next_Day_Return",
        "Target"
    ]

    final_columns = (
        identifier_columns
        + MODEL_FEATURES
        + target_columns
    )

    final_data = data[
        final_columns
    ].copy()

    # Sort by date, then sector, then stock
    final_data = final_data.sort_values(
        ["Date", "Sector", "Ticker"]
    ).reset_index(drop=True)

    return final_data


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_v2_dataset():

    print("=" * 60)
    print("STOCKVISION-AI V2 FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load stocks
    # --------------------------------------------------------

    data = load_all_stocks()

    print(
        f"\nRaw stock rows loaded: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # 2. Stock features
    # --------------------------------------------------------

    data = create_stock_features(data)

    # --------------------------------------------------------
    # 3. Market features
    # --------------------------------------------------------

    data = create_market_features(data)

    # --------------------------------------------------------
    # 4. Sector features
    # --------------------------------------------------------

    data = create_sector_features(data)

    # --------------------------------------------------------
    # 5. Relative features
    # --------------------------------------------------------

    data = create_relative_features(data)

    # --------------------------------------------------------
    # 6. Target
    # --------------------------------------------------------

    data = create_target(data)

    # --------------------------------------------------------
    # 7. Keep final candidate dataset
    # --------------------------------------------------------

    final_data = prepare_final_dataset(data)

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DATA_PATH
        / "v2_features.csv"
    )

    final_data.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print(
        f"\nFinal rows: "
        f"{len(final_data):,}"
    )

    print(
        f"Final columns: "
        f"{len(final_data.columns)}"
    )

    print("\nCandidate model features:")

    for feature in MODEL_FEATURES:
        print(f"  - {feature}")

    print("\nSaved:")
    print(output_path)

    return final_data


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    data = build_v2_dataset()