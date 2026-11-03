
import pandas as pd

from src.data.db_connection import get_db_connection


def load_stock_prices():
    """Load historical stock prices from PostgreSQL."""

    query = """
        SELECT date, ticker, sector, close
        FROM stock_prices
        ORDER BY ticker, date;
    """

    with get_db_connection() as connection:
        df = pd.read_sql_query(query, connection)

    df["date"] = pd.to_datetime(df["date"])

    return df


def calculate_cumulative_returns():
    """Calculate cumulative price returns for each stock."""

    df = load_stock_prices()

    # Get each stock's first available closing price.
    first_close = df.groupby("ticker")["close"].transform("first")

    # Calculate percentage return relative to that first price.
    df["Cumulative_Return_Pct"] = (
        (df["close"] / first_close) - 1
    ) * 100

    return df


def calculate_annualized_volatility():
    """Calculate annualized volatility for each stock."""

    df = load_stock_prices()

    # Calculate daily percentage returns separately for each stock.
    df["Daily_Return"] = (
        df.groupby("ticker")["close"].pct_change() * 100
    )

    # Calculate the standard deviation of daily returns.
    daily_volatility = (
        df.groupby("ticker")["Daily_Return"].std()
    )

    # Annualize using 252 assumed trading days per year.
    annualized_volatility = daily_volatility * (252 ** 0.5)

    results = (
        annualized_volatility
        .rename("Annualized_Volatility_Pct")
        .reset_index()
        .sort_values(
            "Annualized_Volatility_Pct",
            ascending=False
        )
    )

    return results



def calculate_maximum_drawdown():
    """Calculate the maximum peak-to-trough price decline per stock."""

    df = load_stock_prices()

    # Process each stock independently in chronological order.
    df = df.sort_values(["ticker", "date"]).copy()

    # Track the highest closing price reached so far.
    df["Running_Peak"] = (
        df.groupby("ticker")["close"].cummax()
    )

    # Calculate the decline from each stock's running peak.
    df["Drawdown_Pct"] = (
        (df["close"] / df["Running_Peak"]) - 1
    ) * 100

    # Find the worst drawdown observed for each stock.
    results = (
        df.groupby("ticker")["Drawdown_Pct"]
        .min()
        .reset_index(name="Maximum_Drawdown_Pct")
        .sort_values("Maximum_Drawdown_Pct")
    )

    return results


def build_stock_comparison():
    """Combine return, volatility, and drawdown into one table."""

    # Calculate each metric using our existing functions.
    returns = calculate_cumulative_returns()
    volatility = calculate_annualized_volatility()
    drawdowns = calculate_maximum_drawdown()

    # Keep the latest observation for each stock.
    latest_prices = (
        returns.sort_values("date")
        .groupby("ticker")
        .tail(1)
        [["ticker", "sector", "date", "close", "Cumulative_Return_Pct"]]
    )

    # Merge the three metrics by ticker.
    comparison = latest_prices.merge(
        volatility,
        on="ticker",
        how="inner",
    )

    comparison = comparison.merge(
        drawdowns,
        on="ticker",
        how="inner",
    )

    # Rank stocks by cumulative price return.
    comparison["Return_Rank"] = (
        comparison["Cumulative_Return_Pct"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Sort best historical return first.
    comparison = comparison.sort_values(
        "Cumulative_Return_Pct",
        ascending=False,
    ).reset_index(drop=True)

    return comparison



if __name__ == "__main__":
    comparison = build_stock_comparison()

    print("\nSTOCKVISION-AI: STOCK COMPARISON")
    print("=" * 100)

    print(comparison.to_string(
        index=False,
        formatters={
            "close": "{:.2f}".format,
            "Cumulative_Return_Pct": "{:.2f}%".format,
            "Annualized_Volatility_Pct": "{:.2f}%".format,
            "Maximum_Drawdown_Pct": "{:.2f}%".format,
        },
    ))

    # Save the results for inspection or later dashboard use.
    output_path = "data/processed/stock_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print(f"\nComparison saved to: {output_path}")
