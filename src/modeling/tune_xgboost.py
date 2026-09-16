from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TARGET = "is_resolved"


def main():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)

    train = train.replace([np.inf, -np.inf], np.nan)

    X = train.drop(columns=[TARGET])
    y = train[TARGET]

    # Remove leakage-prone pre-split frequency feature
    X = X.drop(columns=["gene_frequency"])

    categorical_features = [
        "review_status",
        "variant_type",
        "substitution",
        "comparison_category",
    ]

    numeric_features = [
        c for c in X.columns
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

    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    scale_pos_weight = n_negative / n_positive

    print("Training rows:", len(y))
    print("Positive:", n_positive)
    print("Negative:", n_negative)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
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

    param_distributions = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__gamma": [0, 0.1, 0.5, 1.0],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="average_precision",
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    print("\nStarting randomized search...")
    search.fit(X, y)

    print("\n=== Best CV Result ===")
    print(f"Best mean CV PR-AUC: {search.best_score_:.4f}")

    print("\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()