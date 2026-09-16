from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TEST_PATH = Path("data/processed/ml/splits/test.csv")

TARGET = "is_resolved"


def main():

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    # Replace infinite Fisher odds ratios with missing values
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    # Remove pre-split gene frequency feature
    X_train = X_train.drop(columns=["gene_frequency"])
    X_test = X_test.drop(columns=["gene_frequency"])

    categorical_features = [
        "review_status",
        "variant_type",
        "substitution",
        "comparison_category",
    ]

    numeric_features = [
        c for c in X_train.columns
        if c not in categorical_features
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
    ])

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
                handle_unknown="ignore",
                sparse_output=False
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

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    scale_pos_weight = n_negative / n_positive

    model = XGBClassifier(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0,

        objective="binary:logistic",
        eval_metric="aucpr",

        scale_pos_weight=scale_pos_weight,

        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("Training tuned XGBoost...")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # 0.5 is still only a temporary baseline threshold
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

    print("\n=== Tuned XGBoost Test Results ===")

    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nPositive prevalence:")
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