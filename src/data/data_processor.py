import pandas as pd
from pathlib import Path

# validates if data loaded is valid or not 
def validate_stock_data(stock_data):

    print("\nValidating stock data...")

    # Check duplicate dates
    duplicate_dates = stock_data.index.duplicated().sum()

    print(f"Duplicate dates found: {duplicate_dates}")


    # Check invalid prices
    invalid_prices = stock_data[
        (stock_data["High"] < stock_data["Low"]) |
        (stock_data["High"] < stock_data["Open"]) |
        (stock_data["High"] < stock_data["Close"]) |
        (stock_data["Low"] > stock_data["Open"]) |
        (stock_data["Low"] > stock_data["Close"])
    ]

    print(f"Invalid price records found: {len(invalid_prices)}")


    # Check invalid volume
    # volume represents how many shares traded always >= 0

    invalid_volume = stock_data[
        stock_data["Volume"] < 0
    ]

    print(f"Invalid volume records found: {len(invalid_volume)}")


    if duplicate_dates > 0:
        return False

    if len(invalid_prices) > 0:
        return False
    # cause invlaid prices returns dataframe thus checking length
    
    if len(invalid_volume) > 0:
        return False
    # cause invlaid volume returns dataframe thus checking length

    return True


def load_stock_data(ticker):

    project_root = Path(__file__).resolve().parents[2]
    # Find the root folder of our project cause our file data processor is inside data inside src inside stockvision AI.
    # need to find path even if working directory changes 

    input_path = (
        project_root
        / "data"
        / "raw"
        / f"{ticker.replace('.', '_')}.csv"
    )
# so this will load the stock data and locate its csv file in data -> raw 

    print(f"\nLoading data for {ticker}...")

    stock_data = pd.read_csv(input_path)

    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

    stock_data.set_index("Date", inplace=True)

    stock_data.sort_index(inplace=True)

    is_valid = validate_stock_data(stock_data)
    if not is_valid:
        raise ValueError("Stock data validation failed.")

    print("Data loaded successfully.")

    print("\nDataset Information:")
    print(stock_data.info())

    return stock_data


if __name__ == "__main__":

    data = load_stock_data("TCS.NS")

    print("\nFirst 5 rows:")
    print(data.head())

