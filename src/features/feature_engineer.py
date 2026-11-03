from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from src.data.stock_universe import STOCKS



# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_processor import load_stock_data


def create_features(stock_data):
    """
    Create features from stock price data.

    Parameters:
        stock_data (pd.DataFrame): Processed stock price data

    Returns:
        pd.DataFrame: Data with additional features
    """

    stock_data = stock_data.copy()

    # Daily percentage return
    stock_data["Daily_Return"] = stock_data["Close"].pct_change() * 100


    # Daily price change
    stock_data["Price_Change"] = stock_data["Close"].diff()


    # 7-day moving average
    stock_data["MA_7"] = stock_data["Close"].rolling(window=7).mean()


    # 30-day moving average
    stock_data["MA_30"] = stock_data["Close"].rolling(window=30).mean()


    # 7-day volatility
    # taking std devn of last 7 daily returns 
    stock_data["Volatility_7"] = (
    stock_data["Daily_Return"]
    .rolling(window=7)
    .std()
    )


    # 7-day momentum
    # how has stock changed  over 7 days 
    stock_data["Momentum_7"] = stock_data["Close"].pct_change(periods=7) * 100


    # Daily volume percentage change
    # how much activity happend how many shares sold/bought
    stock_data["Volume_Change"] = stock_data["Volume"].pct_change() * 100

    # Percentage change is undefined when previous volume is zero
    stock_data["Volume_Change"] = stock_data["Volume_Change"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # RSI (14-day)

    delta = stock_data["Close"].diff()

    # Separate positive and negative price movements
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate 14-day average gain and loss
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Relative Strength
    rs = avg_gain / avg_loss

    # Relative Strength Index
    rsi = 100 - (100 / (1 + rs))

    stock_data["RSI_14"] = rsi


    # MACD components
    # MACD (Moving Average Convergence Divergence)
    # EMA 12 is a faster exponential moving average that gives more weight to recent prices,
    # while EMA 26 is slower and captures the longer-term price trend. We use EWM because
    # it applies exponential weighting, making the average more responsive to recent movements.
    # MACD = EMA 12 - EMA 26 measures the difference between short-term and long-term trends.
    # A 9-period EMA of MACD forms the Signal Line, while MACD - Signal Line forms the Histogram,
    # which shows the strength and changes in the relationship between the two trend measures.

    ema_12 = stock_data["Close"].ewm(span=12).mean()
    ema_26 = stock_data["Close"].ewm(span=26).mean()

    # MACD Line
    macd = ema_12 - ema_26

    # MACD Signal Line
    signal = macd.ewm(span=9).mean()

    stock_data["MACD"] = macd
    stock_data["MACD_Signal"] = signal

    # MACD Histogram
    macd_histogram = macd - signal
    stock_data["MACD_Histogram"] = macd_histogram


    # Bollinger Bands
    # Middle Band=SMA20​
    # Upper Band=Middle+2σ
    # Lower Band=Middle−2σ
    # Bollinger Bands basically tell us two things about a stock: how volatile it currently is, 
    # and where the current price sits relative to its recent price range.

    middle_band = stock_data["Close"].rolling(window=20).mean()
    std_20 = stock_data["Close"].rolling(window=20).std()
    upper_band = middle_band + (2 * std_20)

    lower_band = middle_band - (2 * std_20)
    stock_data["BB_Upper"] = upper_band
    stock_data["BB_Middle"] = middle_band
    stock_data["BB_Lower"] = lower_band

    bb_width = upper_band - lower_band
    stock_data["BB_Width"] = bb_width

    bb_percent = (
    (stock_data["Close"] - lower_band)
    / (upper_band - lower_band)
    )
    stock_data["BB_Percent"] = bb_percent



    return stock_data


def add_market_features(stock_data):

    # Work on a copy so the original data is not modified
    stock_data = stock_data.copy()

    # Get the date range from the stock dataset.
    # Start 10 days earlier so the first benchmark return
    # can be calculated using an earlier closing price.
    start_date = (
        stock_data.index.min() - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")

    # Yahoo Finance's end date is exclusive
    end_date = (
        stock_data.index.max() + pd.Timedelta(days=1)
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

        # Stop if Yahoo Finance returns no data
        if benchmark_data.empty:
            raise ValueError(
                f"No benchmark data downloaded for {ticker}"
            )

        # Extract the Close column.
        # Depending on the yfinance version, this may be
        # a Series or a DataFrame with MultiIndex columns.
        close_prices = benchmark_data["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        # Create a clean benchmark DataFrame
        benchmark_features = pd.DataFrame(
            index=benchmark_data.index
        )

        benchmark_features[f"{name}_Return"] = (
            close_prices.pct_change() * 100
        )

        # Align benchmark observations with stock dates
        stock_data = stock_data.join(
            benchmark_features,
            how="left"
        )

    # Compare TCS's daily return with the market returns
    stock_data["Relative_Strength_Nifty50"] = (
        stock_data["Daily_Return"]
        - stock_data["Nifty_50_Return"]
    )

    stock_data["Relative_Strength_NiftyIT"] = (
        stock_data["Daily_Return"]
        - stock_data["Nifty_IT_Return"]
    )

    # Average relative strength over the previous
    # five trading observations, including the current day
    stock_data["Relative_Strength_IT_5D"] = (
        stock_data["Relative_Strength_NiftyIT"]
        .rolling(window=5)
        .mean()
    )

    return stock_data


def prepare_ml_data(
        ticker,
        selected_features=None,
        target_type="regression",
        threshold=0.25,
        scale = True
    ):
    """
    Prepare stock data for machine learning.

    The function loads validated stock data, creates technical features,
    creates the next-day return target, performs a chronological train-test
    split, and standardizes features using only the training data.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled testing features
        y_train: Training target values
        y_test: Testing target values
        scaler: Fitted StandardScaler
    """

    stock_data = load_stock_data(ticker)

    featured_data = create_features(stock_data)

    # Add broader-market and IT-sector features
    featured_data = add_market_features(featured_data)

    # Check missing values before dropping rows
    print("\nMissing values after adding market features:")
    print(featured_data.isna().sum().sort_values(ascending=False))

    # Check which dates are missing benchmark returns
    missing_benchmark = featured_data[
        featured_data[
            ["Nifty_50_Return", "Nifty_IT_Return"]
        ].isna().any(axis=1)
    ]

    print("\nRows with missing benchmark returns:")
    print(missing_benchmark[
        ["Close", "Nifty_50_Return", "Nifty_IT_Return"]
    ])

    print("\nNumber of rows with missing benchmark returns:")
    print(len(missing_benchmark))

    # Remove rows with missing feature values
    clean_data = featured_data.dropna().copy()

    # Check how many rows remain after cleaning
    print("\nRows before dropna:", len(featured_data))
    print("Rows after dropna:", len(clean_data))
    print("Rows removed:", len(featured_data) - len(clean_data))

    # Identify rows removed specifically because of missing values
    removed_rows = featured_data[
        featured_data.isna().any(axis=1)
    ]

    print("\nDates removed due to missing values:")
    print(removed_rows.index)

    

    # Show which columns caused each row to be removed
    print("\nMissing-value locations:")
    print(
        featured_data.isna()
        .loc[removed_rows.index]
        .sum()
        .sort_values(ascending=False)
    )

    # Create the next-day return target.
    # shift(-1) places the next trading day's Close on the current row.
    # This allows us to calculate the percentage return from today to tomorrow.
    clean_data["Target_Return_1D"] = (
        (clean_data["Close"].shift(-1) - clean_data["Close"])
        / clean_data["Close"]
    ) * 100

    # Create binary classification target from next-day return.
    # 1 = next-day return is positive (UP)
    # 0 = next-day return is zero or negative (DOWN)
    clean_data["Target"] = (
        clean_data["Target_Return_1D"] > 0
    ).astype(int)


    if target_type == "classification":

        clean_data["Target"] = 1  # Neutral

        clean_data.loc[
            clean_data["Target_Return_1D"] < -threshold,
            "Target"
        ] = 0  # Down

        clean_data.loc[
            clean_data["Target_Return_1D"] > threshold,
            "Target"
        ] = 2  # Up


    # Remove the final row because it has no next-day target
    clean_data = clean_data.dropna(
        subset=["Target_Return_1D"]
    ).copy()


    # Separate input features (X) from the target variable (y)
    # Use the selected feature subset when provided.
    # If no subset is provided, use all available features except the target.
    if selected_features is not None:
        X = clean_data[selected_features]
    else:
        X = clean_data.drop(
            columns=["Target_Return_1D", "Target"]
        )

    if target_type == "regression":
        y = clean_data["Target_Return_1D"]

    elif target_type == "classification":
        y = clean_data["Target"]

    else:
        raise ValueError(
            "target_type must be 'regression' or 'classification'"
        )

    # Save the feature names before scaling converts X into NumPy arrays.
    feature_names = X.columns

    # Chronological train-test split.
    # Earlier observations are used for training and later observations
    # are reserved for testing to prevent future-data leakage.
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

        # Standardize features only when requested.
    # XGBoost's tree-based model can use the original values.
    scaler = None

    # Standardize features only when requested.
    # XGBoost's tree-based model can use the original values.

    scaler = None

    if scale:
        scaler = StandardScaler()

        # Learn scaling statistics from training data only
        scaler.fit(X_train)

        # Apply the same transformation to train and test data
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    else:
        # No scaling: preserve the original feature values.
        # Keep the existing variable names for compatibility.
        X_train_scaled = X_train
        X_test_scaled = X_test
    return (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
    feature_names
    )


if __name__ == "__main__":

    ticker = "TCS.NS"

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(
        ticker,
        target_type="classification")
    
    print("\nML Data Preparation Successful!")

    print("\nTraining Features:")
    print(X_train_scaled.shape)

    print("\nTesting Features:")
    print(X_test_scaled.shape)

    print("\nTraining Target:")
    print(y_train.shape)

    print("\nTesting Target:")
    print(y_test.shape)

    print("\nTraining Classification Target Distribution:")
    print(y_train.value_counts())

    print("\nTraining Classification Target Distribution (%):")
    print(y_train.value_counts(normalize=True) * 100)

    print("\nTesting Classification Target Distribution:")
    print(y_test.value_counts())

    print("\nTesting Classification Target Distribution (%):")
    print(y_test.value_counts(normalize=True) * 100)