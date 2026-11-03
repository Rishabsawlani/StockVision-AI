
import pandas as pd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock_comparison.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sector_comparison.csv"
)


def build_sector_comparison():
    """Aggregate stock performance and risk metrics by sector."""

    df = pd.read_csv(INPUT_PATH)

    sector_summary = (
        df.groupby("sector")
        .agg(
            Stock_Count=("ticker", "nunique"),
            Average_Cumulative_Return_Pct=(
                "Cumulative_Return_Pct", "mean"
            ),
            Average_Annualized_Volatility_Pct=(
                "Annualized_Volatility_Pct", "mean"
            ),
            Average_Maximum_Drawdown_Pct=(
                "Maximum_Drawdown_Pct", "mean"
            ),
        )
        .reset_index()
    )

    sector_summary = sector_summary.sort_values(
        "Average_Cumulative_Return_Pct",
        ascending=False,
    ).reset_index(drop=True)

    sector_summary.to_csv(OUTPUT_PATH, index=False)

    return sector_summary


if __name__ == "__main__":
    results = build_sector_comparison()

    print("\nSTOCKVISION-AI: SECTOR COMPARISON")
    print("=" * 90)

    print(results.to_string(
        index=False,
        formatters={
            "Average_Cumulative_Return_Pct": "{:.2f}%".format,
            "Average_Annualized_Volatility_Pct": "{:.2f}%".format,
            "Average_Maximum_Drawdown_Pct": "{:.2f}%".format,
        },
    ))

    print(f"\nSaved to: {OUTPUT_PATH}")
