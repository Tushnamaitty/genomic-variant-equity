from pathlib import Path
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TEST_PATH = Path("data/processed/ml/splits/test.csv")

OUTPUT_DIR = Path("results/population_blind")

TARGET = "is_resolved"


POPULATION_FEATURES_TO_DROP = [
    "gnomad_matched",

    "AF_joint_sas",
    "AC_joint_sas",
    "AN_joint_sas",

    "AF_joint_nfe",
    "AC_joint_nfe",
    "AN_joint_nfe",

    "odds_ratio",
    "pvalue",

    "af_difference_sas_minus_nfe",
    "abs_af_difference",

    "comparison_category",

    "sas_af_missing",
    "nfe_af_missing",
    "population_test_missing",

    "gene_frequency",
]


def main():

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    y_train = train[TARGET]
    y_test = test[TARGET]

    X_train = train.drop(
        columns=[TARGET] + POPULATION_FEATURES_TO_DROP
    )

    X_test = test.drop(
        columns=[TARGET] + POPULATION_FEATURES_TO_DROP
    )

    print("=== POPULATION-BLIND FEATURES ===")
    for c in X_train.columns:
        print(c)

    categorical_features = [
        "review_status",
        "variant_type",
        "substitution",
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

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

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

    print("\nTraining population-blind XGBoost...")

    model.fit(
        X_train_processed_df,
        y_train
    )

    probabilities = model.predict_proba(
        X_test_processed_df
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print("\n=== POPULATION-BLIND TEST PERFORMANCE ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Positive prevalence: {y_test.mean():.4f}")

    print("\nCalculating SHAP...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(
        X_test_processed_df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=feature_names,
        index=X_test.index
    )

    shap_df.to_csv(
        OUTPUT_DIR / "population_blind_shap_values_test.csv"
    )

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "population_blind_feature_importance.csv",
        index=False
    )

    print("\n=== TOP 20 POPULATION-BLIND SHAP FEATURES ===")

    print(
        importance
        .head(20)
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(
        OUTPUT_DIR /
        "population_blind_shap_values_test.csv"
    )
    print(
        OUTPUT_DIR /
        "population_blind_feature_importance.csv"
    )


if __name__ == "__main__":
    main()