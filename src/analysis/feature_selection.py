# src/analysis/feature_selection.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np


# Reduced feature set for our first feature-selection experiment.
#
# We keep representatives from different categories:
# - Close: primary price information
# - MA_7, MA_30: short- and medium-term trend
# - Daily_Return, Price_Change: daily movement
# - Volatility_7: recent volatility
# - Momentum_7, RSI_14: momentum
# - MACD, MACD_Signal, MACD_Histogram: MACD-based momentum/trend information
# - Volume, Volume_Change: trading activity
# - BB_Width, BB_Percent: Bollinger Band volatility/position information
#
# We remove highly redundant raw OHLC and Bollinger boundary features
# for this experiment rather than deleting features blindly based only on VIF.
REDUCED_FEATURES = [
    "Close",
    "MA_7",
    "MA_30",
    "Daily_Return",
    "Price_Change",
    "Volatility_7",
    "Momentum_7",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volume",
    "Volume_Change",
    "BB_Width",
    "BB_Percent"
]


# Add project root to Python path so this file can import
# the reusable ML data-preparation pipeline.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineer import prepare_ml_data


def find_highly_correlated_features(ticker, threshold=0.90):
    """
    Identify pairs of training features that contain highly similar information.

    A high correlation between two features means they may provide overlapping
    information to the model. This can contribute to multicollinearity and
    make linear-model coefficients unstable.

    Parameters:
        ticker (str): Stock ticker symbol.
        threshold (float): Correlation threshold used to identify redundancy.

    Returns:
        pd.DataFrame: Highly correlated feature pairs and their correlations.
    """

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(ticker)

    # Convert the scaled training array back into a DataFrame
    # so that every correlation can be associated with its feature name.
    X_train = pd.DataFrame(
        X_train_scaled,
        columns=feature_names
    )

    # Calculate correlations using training data only.
    correlation_matrix = X_train.corr()

    # Keep only the upper triangle so that we don't report
    # the same pair twice (A-B and B-A).
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Convert the matrix into a readable list of feature pairs.
    correlated_pairs = (
        upper_triangle
        .stack()                #converts 2d df into 1d series
        .reset_index()          #resets index or column name properly so easier to get feature 1 and 2
    )

    correlated_pairs.columns = [
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]

    # Keep only pairs whose absolute correlation exceeds
    # the chosen threshold.
    correlated_pairs = correlated_pairs[
        correlated_pairs["Correlation"].abs() >= threshold
    ]

    # Show the strongest relationships first.
    correlated_pairs = correlated_pairs.sort_values(
        by="Correlation",
        key=lambda x: x.abs(),
        ascending=False
    )

    # Identify every feature involved in at least one
    # highly correlated relationship.
    highly_correlated_features = pd.unique(
        correlated_pairs[["Feature_1", "Feature_2"]].values.ravel()
    )

    print("\nFeatures involved in high correlation:")

    for feature in highly_correlated_features:
        print(feature)

    print(
        f"\nTotal features involved: "
        f"{len(highly_correlated_features)}"
    )

    # Count how many highly correlated relationships each feature has.
    # This helps identify features that are repeatedly providing
    # information very similar to other features.

    feature_correlation_counts = pd.concat(
        [
            correlated_pairs["Feature_1"],
            correlated_pairs["Feature_2"]
        ]
    ).value_counts()

    print("\nHigh-Correlation Count Per Feature:")

    print(feature_correlation_counts)


    return correlated_pairs


if __name__ == "__main__":

    ticker = "TCS.NS"

    correlated_features = find_highly_correlated_features(
        ticker,
        threshold=0.90
    )

    print("\nHighly Correlated Feature Pairs:")

    if correlated_features.empty:
        print("No highly correlated feature pairs found.")
    else:
        print(correlated_features.to_string(index=False))


print("\nReduced Feature Set:")

for feature in REDUCED_FEATURES:
    print(feature)

print(f"\nReduced Feature Count: {len(REDUCED_FEATURES)}")