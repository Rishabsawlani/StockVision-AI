from pathlib import Path
import sys

from sklearn.linear_model import LinearRegression


# Add project root to Python path so this file can import
# the reusable ML data-preparation pipeline from feature_engineer.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineer import prepare_ml_data


def train_linear_regression(ticker, selected_features=None):
    """
    Train a Linear Regression model on prepared stock-market data.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        model: Trained Linear Regression model.
        X_test: Scaled testing features.
        y_test: Actual testing target values.
        y_pred: Model predictions for the testing period.
    """

    # Prepare the data using the same pipeline used throughout the project.
    # This includes feature engineering, target creation, chronological
    # train-test splitting, and leakage-safe feature scaling.
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(ticker, selected_features)

    # Create the Linear Regression baseline model.
    # The model will learn the relationship between our 21 features
    # and the next-day percentage return.
    model = LinearRegression()

    # Train the model using ONLY the historical training data.
    model.fit(X_train_scaled, y_train)

    # Generate predictions for the unseen testing period.
    # The model has not seen these observations during training.
    y_pred = model.predict(X_test_scaled)

    return (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        feature_names
    )


if __name__ == "__main__":

    ticker = "TCS.NS"

    model, X_test_scaled, y_test, y_pred = train_linear_regression(ticker)

    print("\nLinear Regression Training Successful!")

    print("\nTesting Data Shape:")
    print(X_test_scaled.shape)

    print("\nActual Target Shape:")
    print(y_test.shape)

    print("\nPrediction Shape:")
    print(y_pred.shape)

    print("\nFirst 10 Predictions:")
    print(y_pred[:10])

    print("\nFirst 10 Actual Values:")
    print(y_test.values[:10])