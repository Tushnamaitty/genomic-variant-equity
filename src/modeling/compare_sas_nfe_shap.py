from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


ENRICHED_PATH = Path(
    "data/processed/gnomad/chr22_gnomad_with_enrichment.csv"
)

TEST_PATH = Path(
    "data/processed/ml/splits/test.csv"
)

SHAP_PATH = Path(
    "results/shap/shap_values_test.csv"
)

OUTPUT_DIR = Path(
    "results/shap/subgroup"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def clean_feature_name(name):
    name = name.replace("numeric__", "")
    name = name.replace("categorical__", "")
    return name


def main():

    # ---------------------------------------------------------
    # 1. Recreate the exact raw test partition
    # ---------------------------------------------------------

    raw = pd.read_csv(
        ENRICHED_PATH,
        low_memory=False
    )

    raw = raw[
        raw["is_clean_primary_endpoint"] == True
    ].copy()

    _, raw_test = train_test_split(
        raw,
        test_size=TEST_SIZE,
        stratify=raw["is_resolved"],
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # 2. Load saved ML test set and SHAP values
    # ---------------------------------------------------------

    ml_test = pd.read_csv(
        TEST_PATH,
        low_memory=False
    )

    shap_df = pd.read_csv(
        SHAP_PATH,
        index_col=0
    )

    # ---------------------------------------------------------
    # 3. Verify alignment
    # ---------------------------------------------------------

    assert len(raw_test) == len(ml_test), (
        "Raw and ML test sizes do not match."
    )

    assert len(shap_df) == len(ml_test), (
        "SHAP rows do not match test rows."
    )

    raw_y = (
        raw_test["is_resolved"]
        .astype(int)
        .to_numpy()
    )

    ml_y = (
        ml_test["is_resolved"]
        .astype(int)
        .to_numpy()
    )

    assert np.array_equal(
        raw_y,
        ml_y
    ), "Test partition order does not match."

    groups = (
        raw_test["enrichment_group"]
        .reset_index(drop=True)
    )

    shap_df = shap_df.reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Keep SAS and NFE groups
    # ---------------------------------------------------------

    sas_mask = groups == "SAS_enriched"
    nfe_mask = groups == "NFE_enriched"

    sas_shap = shap_df.loc[sas_mask]
    nfe_shap = shap_df.loc[nfe_mask]

    print("=== TEST SUBGROUP SIZES ===")
    print("SAS-enriched:", len(sas_shap))
    print("NFE-enriched:", len(nfe_shap))

    # ---------------------------------------------------------
    # 5. Mean absolute SHAP importance by group
    # ---------------------------------------------------------

    sas_importance = (
        sas_shap
        .abs()
        .mean(axis=0)
    )

    nfe_importance = (
        nfe_shap
        .abs()
        .mean(axis=0)
    )

    comparison = pd.DataFrame({
        "feature": shap_df.columns,
        "sas_mean_abs_shap": sas_importance.values,
        "nfe_mean_abs_shap": nfe_importance.values,
    })

    comparison["difference_sas_minus_nfe"] = (
        comparison["sas_mean_abs_shap"]
        - comparison["nfe_mean_abs_shap"]
    )

    comparison["abs_difference"] = (
        comparison["difference_sas_minus_nfe"]
        .abs()
    )

    comparison["feature"] = (
        comparison["feature"]
        .apply(clean_feature_name)
    )

    comparison = comparison.sort_values(
        "abs_difference",
        ascending=False
    )

    # ---------------------------------------------------------
    # 6. Save results
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "sas_vs_nfe_shap_comparison.csv",
        index=False
    )

    print(
        "\n=== FEATURES WITH BIGGEST SAS/NFE SHAP DIFFERENCES ==="
    )

    print(
        comparison[
            [
                "feature",
                "sas_mean_abs_shap",
                "nfe_mean_abs_shap",
                "difference_sas_minus_nfe",
            ]
        ]
        .head(20)
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== TOP SAS FEATURES ==="
    )

    print(
        comparison
        .sort_values(
            "sas_mean_abs_shap",
            ascending=False
        )[
            [
                "feature",
                "sas_mean_abs_shap"
            ]
        ]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== TOP NFE FEATURES ==="
    )

    print(
        comparison
        .sort_values(
            "nfe_mean_abs_shap",
            ascending=False
        )[
            [
                "feature",
                "nfe_mean_abs_shap"
            ]
        ]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(
        OUTPUT_DIR /
        "sas_vs_nfe_shap_comparison.csv"
    )


if __name__ == "__main__":
    main()