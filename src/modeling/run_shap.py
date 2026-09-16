from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TEST_PATH = Path("data/processed/ml/splits/test.csv")

OUTPUT_DIR = Path("results/shap")
TARGET = "is_resolved"


def main():

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    # Replace infinite Fisher odds ratios with missing values
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    X_train = train.drop(
        columns=[TARGET, "gene_frequency"]
    )
    y_train = train[TARGET]

    X_test = test.drop(
        columns=[TARGET, "gene_frequency"]
    )
    y_test = test[TARGET]

    # ---------------------------------------------------------
    # Feature groups
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Tuned XGBoost model
    # ---------------------------------------------------------

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    scale_pos_weight = (
        n_negative / n_positive
    )

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

    # ---------------------------------------------------------
    # Fit preprocessing
    # ---------------------------------------------------------

    print("Fitting preprocessing...")

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # Get real processed feature names
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Convert processed arrays to DataFrames
    # so SHAP uses the actual feature names
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

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    print("Training tuned XGBoost...")

    model.fit(
        X_train_processed_df,
        y_train
    )

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_test_processed_df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------

    mean_abs_shap = (
        np.abs(
            shap_values.values
        ).mean(axis=0)
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    importance.to_csv(
        OUTPUT_DIR /
        "shap_feature_importance.csv",
        index=False
    )

    print(
        "\n=== TOP 20 SHAP FEATURES ==="
    )

    print(
        importance
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # SHAP bar plot
    # ---------------------------------------------------------

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "shap_bar_named.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # SHAP beeswarm plot
    # ---------------------------------------------------------

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "shap_beeswarm_named.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Save SHAP values for later SAS/NFE comparison
    # ---------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=feature_names,
        index=X_test.index
    )

    shap_df.to_csv(
        OUTPUT_DIR /
        "shap_values_test.csv",
        index=True
    )

    X_test_processed_df.to_csv(
        OUTPUT_DIR /
        "processed_test_features.csv",
        index=True
    )

    print("\nSaved:")
    print(
        OUTPUT_DIR /
        "shap_feature_importance.csv"
    )

    print(
        OUTPUT_DIR /
        "shap_bar_named.png"
    )

    print(
        OUTPUT_DIR /
        "shap_beeswarm_named.png"
    )

    print(
        OUTPUT_DIR /
        "shap_values_test.csv"
    )


if __name__ == "__main__":
    main()