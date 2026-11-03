from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineer import prepare_ml_data


def train_logistic_regression(
    ticker,
    selected_features=None
    ):
    """
    Train a Logistic Regression model for next-day
    stock-return direction classification.

    Target:
        1 = next-day return is positive
        0 = next-day return is zero or negative
    """

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(
        ticker,
        selected_features=selected_features,
        target_type="classification"
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    return (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        feature_names
    )


if __name__ == "__main__":

    ticker = "TCS.NS"

    (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        feature_names
    ) = train_logistic_regression(ticker)

    print("\nLogistic Regression Training Successful!")

    print("\nTesting Data Shape:")
    print(X_test_scaled.shape)

    print("\nFirst 20 Predictions:")
    print(y_pred[:20])

    print("\nFirst 20 Actual Values:")
    print(y_test.values[:20])