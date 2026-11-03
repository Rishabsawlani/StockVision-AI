from pathlib import Path
import sys

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Add project root to Python path so this file can import
# our reusable ML data-preparation pipeline.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineer import prepare_ml_data


def calculate_vif(ticker):
    """
    Calculate Variance Inflation Factor (VIF) for every feature.

    VIF measures how strongly a feature can be explained by
    the other features in the dataset.

    Higher VIF indicates stronger multicollinearity.

    We calculate VIF only on the training features so that
    information from the unseen testing period is not used.
    """

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(ticker)

    # Calculate VIF for every training feature.
    vif_data = pd.DataFrame()

    vif_data["Feature"] = feature_names

    vif_data["VIF"] = [
        variance_inflation_factor(X_train_scaled, i)
        for i in range(X_train_scaled.shape[1])
    ]
    # using only training data as to not contaminate the vif with future data ,done so as to prevent data leakage

    # Show features with the highest multicollinearity first.
    vif_data = vif_data.sort_values(
        by="VIF",
        ascending=False
    )

    return vif_data


if __name__ == "__main__":

    ticker = "TCS.NS"

    vif_results = calculate_vif(ticker)

    print("\nVIF Analysis:")
    print(vif_results.to_string(index=False))