
from pathlib import Path
import sys

import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineer import prepare_ml_data


def train_xgboost(
    ticker,
    selected_features=None,
    threshold=0.25,
    use_class_weights=False
):
    """
    Train an XGBoost classifier for three-class stock prediction.

    Classes:
        0 = DOWN
        1 = NEUTRAL
        2 = UP

    Uses a chronological train/test split.
    Features remain unscaled because scale=False.
    """

    # Prepare data using our existing function.
    # Keep the established variable names.
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(
        ticker=ticker,
        selected_features=selected_features,
        target_type="classification",
        threshold=threshold,
        scale=False
    )

    # Define the XGBoost classifier
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    if use_class_weights:

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train_scaled,
            y_train,
            sample_weight=sample_weights
        )

    else:

        model.fit(
            X_train_scaled,
            y_train
        )

    # Predict classes for the test data
    y_pred = model.predict(
        X_test_scaled
    )

    # Get probabilities for all three classes
    y_prob = model.predict_proba(
        X_test_scaled
    )

    # Calculate feature importance
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print("\nXGBoost Training Successful!")
    print(f"Ticker: {ticker}")
    print(f"Threshold: ±{threshold:.2f}%")
    print(f"Training rows: {len(X_train_scaled)}")
    print(f"Testing rows: {len(X_test_scaled)}")
    print(f"Number of features: {len(feature_names)}")

    return (
        model,
        X_test_scaled,
        y_train,
        y_test,
        y_pred,
        y_prob,
        feature_names,
        feature_importance
    )


# Run a basic training test
if __name__ == "__main__":

    results = train_xgboost(
        ticker="TCS.NS",
        threshold=0.25
    )

    (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        y_prob,
        feature_names,
        feature_importance
    ) = results

    print("\nTest data shape:")
    print(X_test_scaled.shape)

    print("\nPrediction shape:")
    print(y_pred.shape)

    print("\nProbability shape:")
    print(y_prob.shape)

    print("\nFeature Importance:")
    print(feature_importance)