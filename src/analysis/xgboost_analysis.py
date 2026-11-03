import sys
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.xgboost_model import train_xgboost


def evaluate_xgboost(
    ticker,
    selected_features=None,
    threshold=0.25,
    use_class_weights=False
):

    print(
        f"\nClass weighting enabled: {use_class_weights}"
    )
    
    # Train XGBoost and retrieve its predictions
    (
        model,
        X_test_scaled,
        y_train,
        y_test,
        y_pred,
        y_prob,
        feature_names,
        feature_importance
    ) = train_xgboost(
        ticker=ticker,
        selected_features=selected_features,
        threshold=threshold,
        use_class_weights=use_class_weights
    )

    # ------------------------------------------
    # MAJORITY-CLASS BASELINE
    # ------------------------------------------

    # Find the most frequent class in training data
    baseline_class = y_train.mode().iloc[0]

    # Predict that same class for every test sample
    baseline_pred = [baseline_class] * len(y_test)

    # Evaluate the baseline
    baseline_accuracy = accuracy_score(
        y_test,
        baseline_pred
    )

    baseline_macro_f1 = f1_score(
        y_test,
        baseline_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    baseline_weighted_f1 = f1_score(
        y_test,
        baseline_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 50)
    print("MAJORITY-CLASS BASELINE")
    print("=" * 50)

    print(f"Most common training class: {baseline_class}")
    print(f"Training class distribution:")
    print(y_train.value_counts().sort_index())

    print(f"\nBaseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline Macro F1: {baseline_macro_f1:.4f}")
    print(f"Baseline Weighted F1: {baseline_weighted_f1:.4f}")

    print("\nBaseline Classification Report:")

    print(
        classification_report(
            y_test,
            baseline_pred,
            labels=[0, 1, 2],
            target_names=["DOWN", "NEUTRAL", "UP"],
            zero_division=0
        )
    )

    print("Baseline Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            baseline_pred,
            labels=[0, 1, 2]
        )
    )

    print("\nActual Test Class Distribution:")
    print(
        y_test.value_counts()
        .sort_index()
    )

    print("\nPredicted Class Distribution:")
    print(
        pd.Series(y_pred)
        .value_counts()
        .sort_index()
    )

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0
    )

    # ==========================================
    # XGBOOST CLASSIFICATION EVALUATION
    # ==========================================

    print("\n" + "=" * 50)
    print("XGBOOST CLASSIFICATION EVALUATION")
    print("=" * 50)

    print(f"Ticker: {ticker}")
    print(f"Threshold: +/- {threshold:.2f}%")
    print(f"Test samples: {len(y_test)}")

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")


    # ==========================================
    # PER-CLASS CLASSIFICATION REPORT
    # ==========================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["DOWN", "NEUTRAL", "UP"],
            zero_division=0
        )
    )


    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2]
        )
    )

    # ==========================================
    # BASELINE VS XGBOOST
    # ==========================================

    print("\n" + "=" * 50)
    print("BASELINE VS XGBOOST")
    print("=" * 50)

    print(
        f"{'Metric':<18}"
        f"{'Baseline':>12}"
        f"{'XGBoost':>12}"
    )

    print("-" * 42)

    print(
        f"{'Accuracy':<18}"
        f"{baseline_accuracy:>12.4f}"
        f"{accuracy:>12.4f}"
    )

    print(
        f"{'Macro F1':<18}"
        f"{baseline_macro_f1:>12.4f}"
        f"{macro_f1:>12.4f}"
    )

    print(
        f"{'Weighted F1':<18}"
        f"{baseline_weighted_f1:>12.4f}"
        f"{weighted_f1:>12.4f}"
    )
    


if __name__ == "__main__":

    ticker = "TCS.NS"
    threshold = 0.25

    # Use the SAME feature selection as the previous benchmark.
    # If the previous benchmark used all features, keep this as None.
    selected_features = None

    # ------------------------------------------
    # EXPERIMENT 1: UNWEIGHTED XGBOOST
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: UNWEIGHTED XGBOOST")
    print("=" * 60)

    evaluate_xgboost(
        ticker=ticker,
        selected_features=selected_features,
        threshold=threshold,
        use_class_weights=False
    )

    # ------------------------------------------
    # EXPERIMENT 2: WEIGHTED XGBOOST
    # ------------------------------------------

    # print("\n" + "=" * 60)
    # print("EXPERIMENT 2: WEIGHTED XGBOOST")
    # print("=" * 60)

    # evaluate_xgboost(
    #     ticker=ticker,
    #     selected_features=selected_features,
    #     threshold=threshold,
    #     use_class_weights=True
    # )