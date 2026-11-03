from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt




def evaluate_regression_model(y_test, y_pred):
    """
    Evaluate a regression model using numerical error metrics.

    Parameters:
        y_test: Actual target values from the testing period.
        y_pred: Predicted target values produced by the model.

    Returns:
        Dictionary containing MAE, RMSE, and R² scores.
    """

    # MAE measures the average absolute difference between
    # the actual and predicted next-day returns.
    mae = mean_absolute_error(y_test, y_pred)

    # MSE squares each prediction error before averaging.
    # This gives larger errors a greater penalty.
    mse = mean_squared_error(y_test, y_pred)

    # RMSE is the square root of MSE, bringing the error
    # back to the same unit as our target variable: percentage return.
    rmse = np.sqrt(mse)

    # R² measures how much of the variation in the target
    # is explained by the model relative to a simple baseline.
    r2 = r2_score(y_test, y_pred)

    # Check whether the model correctly predicts the direction
    # of the next-day return: positive (up) or negative (down).
    # We convert both actual and predicted returns into True/False
    # values and calculate the percentage of matching directions.
    actual_direction = y_test > 0
    predicted_direction = y_pred > 0

    directional_accuracy = (
        actual_direction == predicted_direction
    ).mean() * 100
    

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy
    }


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))

    from src.analysis.feature_selection import REDUCED_FEATURES

    from src.models.ridge_regression import train_ridge_regression
    ticker = "TCS.NS"

    model, X_test_scaled, y_test, y_pred, feature_names = (
        train_ridge_regression(
            ticker,
            alpha=1.0
        )
    )

    print("\nNumber of Testing Features:")
    print(X_test_scaled.shape[1])

    print("\nTesting Data Shape:")
    print(X_test_scaled.shape)

    # Extract the coefficient learned for each feature.
    # Because the features were standardized, the coefficient magnitudes
    # can be compared more fairly across different features.
    coefficients = model.coef_

    # Create a table connecting each feature to its learned coefficient.
    feature_coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Coefficient": np.abs(coefficients)
    })

    # Sort by absolute coefficient to see which features
    # have the strongest influence on the model's predictions.
    feature_coefficients = feature_coefficients.sort_values(
        by="Absolute_Coefficient",
        ascending=False
    )

    print("\nFeature Coefficients:")
    print(feature_coefficients)

    metrics = evaluate_regression_model(y_test, y_pred)


    # plt.figure(figsize=(12, 6))

    # plt.plot(
    #     y_test.index,
    #     y_test.values,
    #     label="Actual Return"
    # )

    # plt.plot(
    #     y_test.index,
    #     y_pred,
    #     label="Predicted Return"
    # )

    # plt.title("Actual vs Predicted Next-Day Returns")
    # plt.xlabel("Date")
    # plt.ylabel("Return (%)")
    # plt.legend()
    # plt.grid(True)

    # plt.show()

    
    print("\nModel Evaluation Results:")

    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")
    print(
    f"Directional Accuracy: "
    f"{metrics['Directional_Accuracy']:.2f}%"
    )