from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TEST_PATH = Path("data/processed/ml/splits/test.csv")

TARGET = "is_resolved"


def build_pipeline(X_train, y_train):

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
        ("imputer", SimpleImputer(strategy="median")),
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
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    X_train = train.drop(columns=[TARGET, "gene_frequency"])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET, "gene_frequency"])
    y_test = test[TARGET]

    pipeline = build_pipeline(X_train, y_train)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("Generating out-of-fold training probabilities...")

    oof_prob = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    print("\nOOF ROC-AUC:",
          round(roc_auc_score(y_train, oof_prob), 4))

    print("OOF PR-AUC:",
          round(average_precision_score(y_train, oof_prob), 4))

    # ------------------------------------------------------
    # Find threshold with best F1 on training OOF predictions
    # ------------------------------------------------------

    thresholds = np.arange(0.05, 0.96, 0.01)

    results = []

    for threshold in thresholds:

        pred = (oof_prob >= threshold).astype(int)

        precision = precision_score(
            y_train,
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_train,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_train,
            pred,
            zero_division=0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_threshold = best_row["threshold"]

    print("\n=== Best threshold from TRAINING CV ===")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall:    {best_row['recall']:.4f}")
    print(f"F1:        {best_row['f1']:.4f}")

    print("\nTop 10 thresholds:")
    print(
        results_df
        .sort_values("f1", ascending=False)
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    # ------------------------------------------------------
    # Train final model on all training data
    # ------------------------------------------------------

    print("\nTraining final model on full training set...")

    pipeline.fit(X_train, y_train)

    test_prob = pipeline.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    print("\n=== Final Test Results ===")

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_test,
                test_prob
            ),
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y_test,
                test_prob
            ),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                test_pred,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                test_pred,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(
                y_test,
                test_pred,
                zero_division=0
            ),
            4
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            test_pred
        )
    )


if __name__ == "__main__":
    main()