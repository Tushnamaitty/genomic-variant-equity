from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TEST_PATH = Path("data/processed/ml/splits/test.csv")

TARGET = "is_resolved"


def main():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    # Fisher odds ratios can be infinite.
    # Replace infinity with missing values so the imputer can handle them.
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    # gene_frequency was calculated before the train/test split,
    # so remove it for a cleaner leakage-safe baseline.
    X_train = X_train.drop(columns=["gene_frequency"])
    X_test = X_test.drop(columns=["gene_frequency"])

    categorical_features = [
        "review_status",
        "variant_type",
        "substitution",
        "comparison_category",
    ]

    numeric_features = [
        col
        for col in X_train.columns
        if col not in categorical_features
    ]

    # Numeric preprocessing:
    # learn medians only from the training set,
    # then standardize features.
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    # Categorical preprocessing:
    # fill missing categories and one-hot encode them.
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ])

    # Class balancing because only ~1.1% of variants resolve.
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("Training logistic regression...")
    pipeline.fit(X_train, y_train)

    # Predicted probability that a VUS will later resolve
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # 0.5 is only the baseline threshold.
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n=== Logistic Regression Baseline ===")

    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nPositive prevalence in test set:")
    print(f"{y_test.mean():.4f}")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()