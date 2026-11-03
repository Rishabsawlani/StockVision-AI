import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.xgboost_model import prepare_ml_data

from src.data.data_processor import load_stock_data
from src.features.feature_engineer import (
    create_features,
    add_market_features
)



# --------------------------------------------------
# 1. Load the same ML dataset used by XGBoost
# --------------------------------------------------

ticker = "TCS.NS"
threshold = 0.25

(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
    feature_names
) = prepare_ml_data(
    ticker=ticker,
    threshold=threshold,
    scale=False
)


# --------------------------------------------------
# 2. Combine the target labels
# --------------------------------------------------

# Reconstruct the chronological target series.
# This assumes prepare_ml_data uses the same chronological
# 80/20 split and does not shuffle the observations.

y_all = pd.concat(
    [y_train, y_test],
    ignore_index=True
)

# Use the same cleaned feature rows and ordering as the model.
# We will make the row alignment explicit in the next step.

# --------------------------------------------------
# 3. Create analysis_data
# --------------------------------------------------

ticker = "TCS.NS"
threshold = 0.25

# Load the original stock data
stock_data = load_stock_data(ticker)

# Generate the same features used by XGBoost
featured_data = create_features(stock_data)

# Add Nifty 50 and Nifty IT market features
featured_data = add_market_features(featured_data)

# Apply the same missing-value cleaning as the ML pipeline
analysis_data = featured_data.dropna().copy()

# Calculate the next observation's return
analysis_data["Next_Day_Return"] = (
    analysis_data["Close"].shift(-1)
    / analysis_data["Close"] - 1
) * 100

# Assign the same three target classes
analysis_data["Target"] = np.select(
    [
        analysis_data["Next_Day_Return"] < -threshold,
        analysis_data["Next_Day_Return"] > threshold
    ],
    [
        0,  # DOWN
        2   # UP
    ],
    default=1  # NEUTRAL
)

# Remove the final row, which has no next observation
analysis_data = analysis_data.dropna(
    subset=["Next_Day_Return"]
).copy()

print("\nAnalysis dataset created successfully!")
print(f"Rows: {len(analysis_data)}")
print(f"Columns: {len(analysis_data.columns)}")

print("\nTarget distribution:")
print(analysis_data["Target"].value_counts().sort_index())


# --------------------------------------------------
# 4. Compare each market feature across target classes
# --------------------------------------------------

market_features = [
    "Nifty_50_Return",
    "Nifty_IT_Return",
    "Relative_Strength_Nifty50",
    "Relative_Strength_NiftyIT",
    "Relative_Strength_IT_5D"
]

class_names = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP"
}

analysis_data = analysis_data.copy()

analysis_data["Target_Name"] = (
    analysis_data["Target"].map(class_names)
)

for feature in market_features:

    print("\n" + "=" * 65)
    print(f"FEATURE: {feature}")
    print("=" * 65)

    # Summary statistics for each next-day outcome
    summary = (
        analysis_data
        .groupby("Target_Name")[feature]
        .agg(["count", "mean", "std", "median"])
        .reindex(["DOWN", "NEUTRAL", "UP"])
    )

    print("\nClass-wise statistics:")
    print(summary.round(4))

    # Quantify the range of values for each class
    quantiles = (
        analysis_data
        .groupby("Target_Name")[feature]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .reindex(["DOWN", "NEUTRAL", "UP"])
    )

    quantiles.columns = [
        "10%", "25%", "50%", "75%", "90%"
    ]

    print("\nClass-wise quantiles:")
    print(quantiles.round(4))