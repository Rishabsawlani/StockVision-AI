import yfinance as yf
import pandas as pd
from pathlib import Path


def collect_stock_data(ticker, period="5y", interval="1d"):
    """
    Downloads, cleans, validates, and saves historical stock data.

    Parameters:
        ticker (str): Stock ticker symbol.
        period (str): Historical data period.
        interval (str): Data interval.

    Returns:
        pd.DataFrame: Cleaned stock data.
    """

    print(f"\nDownloading data for {ticker}...")


    # Download stock data
    stock_data = yf.download(
        ticker,
        period=period,
        interval=interval
    )


    # Check if data was downloaded
    if stock_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")


    print("Data downloaded successfully.")


    # Flatten MultiIndex columns
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
        # fetches only the single column index cause this is only for tcs or single 1 company so dont need company name

        stock_data.columns.name = None
        # removes the metadata "price" above the OHLCV columns 

    # Required OHLCV columns
    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]


    # Check for missing columns
    missing_columns = [
        column
        for column in required_columns
        if column not in stock_data.columns
    ]


    # Validate required columns
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    print("All required columns are present.")


    # Check missing values
    missing_values = stock_data.isnull().sum()

    if missing_values.sum() > 0:
        print("\nWarning: Missing values found:")
        print(missing_values)
    else:
        print("No missing values found.")


    # Define output directory
    output_dir = Path("data/raw")

    # Create directory if it does not exist
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Create filename
    filename = ticker.replace(".", "_") + ".csv"

    # Create full file path
    file_path = output_dir / filename


    # Save data
    stock_data.to_csv(file_path)
    # since by default index is true it will also store the datetime as index
    # also here index is datetime cause yfinance uses datetie as its default index 

    print(f"\nData saved successfully to: {file_path}")


    return stock_data


# this ensures that data is downloaded only when data file is run but if another file is importing collect_stock_data function whole data should not be downloaded 
if __name__ == "__main__":

    stocks = [
        "TCS.NS",
        "INFY.NS",
    #    "FAKESTOCK123.NS",         # testing fake stock error handling 
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS"
    ]

    successful_stocks = []
    failed_stocks = []


    for ticker in stocks:

        #one stock failure doesn't affects other stocks execution thus using error handling or exception handling 
        try:

            data = collect_stock_data(
                ticker=ticker
            )

            successful_stocks.append(ticker)

            print("\nCollection successful!")


        except Exception as e:

            print(f"\nFailed to collect {ticker}")
            print(f"Error: {e}")

            failed_stocks.append(ticker)


    print("\n" + "=" * 50)

    print("COLLECTION SUMMARY")

    print("=" * 50)

    print(f"\nTotal Stocks: {len(stocks)}")

    print(f"Successful: {len(successful_stocks)}")

    print(f"Failed: {len(failed_stocks)}")


    print("\nSuccessful Stocks:")

    for ticker in successful_stocks:
        print(f"✓ {ticker}")


    if failed_stocks:

        print("\nFailed Stocks:")

        for ticker in failed_stocks:
            print(f"✗ {ticker}")


    print("\n" + "=" * 50)