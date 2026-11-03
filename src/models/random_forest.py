import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineer import prepare_ml_data

def train_random_forest(
    ticker,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
):
    (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
    feature_names
    ) = prepare_ml_data(
        ticker,
        target_type="classification"
    )

    model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=random_state
    )

    model.fit(X_train_scaled,y_train)

    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        labels=[0, 1, 2],
        multi_class="ovr",
        average="macro"
    )

    print(f"Macro ROC-AUC: {roc_auc:.4f}")
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nTuned Random Forest Evaluation Results:")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["DOWN", "UP"],
            zero_division=0
        )
    )

    return (
    model,
    X_test_scaled,
    y_test,
    y_pred,
    y_prob,
    feature_names)


def tune_random_forest(ticker):
    """
    Find the best Random Forest hyperparameters
    using time-series cross-validation.
    """

    # Prepare the ML data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = (
        prepare_ml_data(
            ticker,
            target_type="classification"
        )
    )

    # Create the base Random Forest
    rf = RandomForestClassifier(
        random_state=42
    )

    # Hyperparameters we want to test
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    # Preserve chronological order during cross-validation
    time_split = TimeSeriesSplit(n_splits=5)

    # Grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=time_split,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )

    # Train all parameter combinations
    grid_search.fit(X_train_scaled, y_train)

    return (
        grid_search,
        X_test_scaled,
        y_test,
        feature_names
    )


def train_tuned_random_forest(ticker, threshold=0.25):
    """
    Train the Random Forest using the best hyperparameters
    found during GridSearchCV.
    """

    # Prepare classification data
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_ml_data(
        ticker,
        target_type="classification",
        threshold=threshold
    )

    # Best parameters found during hyperparameter tuning
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    )

    # Train final tuned model
    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Probability of UP
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        y_prob,
        feature_names
    )


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


if __name__ == "__main__":

    ticker = "TCS.NS"

    (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        y_prob,
        feature_names
    ) = train_random_forest(
        ticker
    )

    print("\nRandom Forest Training Successful!")

    print("\nTesting Data Shape:")
    print(X_test_scaled.shape)

    print("\nFirst 20 Predictions:")
    print(y_pred[:20])

    print("\nFirst 20 Actual Values:")
    print(y_test.values[:20])

    print("\nFirst 20 UP Probabilities:")
    print(y_prob[:20])


# if __name__ == "__main__":

#     ticker = "TCS.NS"

#     (
#         grid_search,
#         X_test_scaled,
#         y_test,
#         feature_names
#     ) = tune_random_forest(ticker)

#     print("\nRandom Forest Hyperparameter Tuning Complete!")

#     print("\nBest Parameters:")
#     print(grid_search.best_params_)

#     print("\nBest Cross-Validation Score:")
#     print(grid_search.best_score_)


# if __name__ == "__main__":

#     ticker = "TCS.NS"

#     (
#         model,
#         X_test_scaled,
#         y_test,
#         y_pred,
#         y_prob,
#         feature_names
#     ) = train_tuned_random_forest(ticker)

#     print("\nTuned Random Forest Training Successful!")

#     print("\nTesting Data Shape:")
#     print(X_test_scaled.shape)

#     print("\nActual DOWN:")
#     print((y_test == 0).sum())

#     print("Actual UP:")
#     print((y_test == 1).sum())

#     print("\nPredicted DOWN:")
#     print((y_pred == 0).sum())

#     print("Predicted UP:")
#     print((y_pred == 1).sum())

#     print("\nPredicted UP Probability:")

#     print("Minimum:", y_prob.min())
#     print("Maximum:", y_prob.max())
#     print("Mean:", y_prob.mean())
#     print("Median:", np.median(y_prob))