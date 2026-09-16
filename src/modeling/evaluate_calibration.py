from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/ml/splits/train.csv")
TEST_PATH = Path("data/processed/ml/splits/test.csv")

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


def build_model(X_train, y_train):

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


def print_calibration_table(y_true, probabilities, name):

    prob_true, prob_pred = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    table = pd.DataFrame({
        "mean_predicted_probability": prob_pred,
        "observed_resolution_rate": prob_true
    })

    print(f"\n=== {name} CALIBRATION BINS ===")
    print(table.round(4).to_string(index=False))


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

    # ---------------------------------------------------------
    # Uncalibrated model
    # ---------------------------------------------------------

    base_model = build_model(
        X_train,
        y_train
    )

    print("Training uncalibrated model...")

    base_model.fit(
        X_train,
        y_train
    )

    raw_prob = base_model.predict_proba(
        X_test
    )[:, 1]

    print("\n=== UNCALIBRATED ===")
    print(
        "ROC-AUC:",
        round(
            roc_auc_score(y_test, raw_prob),
            4
        )
    )
    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y_test,
                raw_prob
            ),
            4
        )
    )
    print(
        "Brier score:",
        round(
            brier_score_loss(
                y_test,
                raw_prob
            ),
            6
        )
    )

    print_calibration_table(
        y_test,
        raw_prob,
        "UNCALIBRATED"
    )

    # ---------------------------------------------------------
    # Platt scaling
    # ---------------------------------------------------------

    print("\nTraining Platt calibrated model...")

    platt_base = build_model(
        X_train,
        y_train
    )

    platt = CalibratedClassifierCV(
        estimator=platt_base,
        method="sigmoid",
        cv=5
    )

    platt.fit(
        X_train,
        y_train
    )

    platt_prob = platt.predict_proba(
        X_test
    )[:, 1]

    print("\n=== PLATT CALIBRATION ===")
    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_test,
                platt_prob
            ),
            4
        )
    )
    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y_test,
                platt_prob
            ),
            4
        )
    )
    print(
        "Brier score:",
        round(
            brier_score_loss(
                y_test,
                platt_prob
            ),
            6
        )
    )

    print_calibration_table(
        y_test,
        platt_prob,
        "PLATT"
    )

    # ---------------------------------------------------------
    # Isotonic calibration
    # ---------------------------------------------------------

    print("\nTraining isotonic calibrated model...")

    isotonic_base = build_model(
        X_train,
        y_train
    )

    isotonic = CalibratedClassifierCV(
        estimator=isotonic_base,
        method="isotonic",
        cv=5
    )

    isotonic.fit(
        X_train,
        y_train
    )

    isotonic_prob = isotonic.predict_proba(
        X_test
    )[:, 1]

    print("\n=== ISOTONIC CALIBRATION ===")
    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_test,
                isotonic_prob
            ),
            4
        )
    )
    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y_test,
                isotonic_prob
            ),
            4
        )
    )
    print(
        "Brier score:",
        round(
            brier_score_loss(
                y_test,
                isotonic_prob
            ),
            6
        )
    )

    print_calibration_table(
        y_test,
        isotonic_prob,
        "ISOTONIC"
    )


if __name__ == "__main__":
    main()