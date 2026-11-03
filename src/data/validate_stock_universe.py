
import yfinance as yf

from src.data.stock_universe import STOCKS


def validate_ticker(ticker):
    try:
        data = yf.download(
            ticker,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if data.empty:
            return False, "No data returned"

        if isinstance(data.columns, __import__("pandas").MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        if not all(column in data.columns for column in required_columns):
            return False, "Missing required OHLCV columns"

        if data["Close"].dropna().empty:
            return False, "No valid closing prices"

        return True, f"{len(data)} rows returned"

    except Exception as error:
        return False, str(error)


def main():
    print(f"Validating {len(STOCKS)} stocks...\n")

    successful = []
    failed = []

    for ticker, sector in STOCKS.items():
        is_valid, message = validate_ticker(ticker)

        if is_valid:
            successful.append(ticker)
            print(f"[OK]   {ticker:<18} {sector:<20} {message}")
        else:
            failed.append((ticker, sector, message))
            print(f"[FAIL] {ticker:<18} {sector:<20} {message}")

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print("\nTickers requiring attention:")
        for ticker, sector, message in failed:
            print(f"{ticker} ({sector}): {message}")


if __name__ == "__main__":
    main()
