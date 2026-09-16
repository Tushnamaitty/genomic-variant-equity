from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/processed/gnomad/chr22_gnomad_annotated.csv")
OUTPUT = Path("data/processed/ml/chr22_ml_dataset.csv")

T1_SNAPSHOT = pd.Timestamp("2024-02-29")


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    # ---------------------------------------------------------
    # 1. Keep only the clean longitudinal prediction cohort
    # ---------------------------------------------------------
    df = df[df["is_clean_primary_endpoint"] == True].copy()

    out = pd.DataFrame(index=df.index)

    # Target: whether the VUS later resolved in ClinVar
    out["is_resolved"] = df["is_resolved"].astype(int)

    # ---------------------------------------------------------
    # 2. ClinVar features available at T1
    # ---------------------------------------------------------
    out["review_status"] = df["review_status"].fillna("missing")

    evaluation_date = pd.to_datetime(
        df["date_last_evaluated"],
        errors="coerce"
    )

    out["evaluation_age_days"] = (
        T1_SNAPSHOT - evaluation_date
    ).dt.days

    out["evaluation_date_missing"] = evaluation_date.isna().astype(int)

    # ---------------------------------------------------------
    # 3. Variant structure
    # ---------------------------------------------------------
    ref = df["ref"].fillna("").astype(str)
    alt = df["alt"].fillna("").astype(str)

    out["ref_length"] = ref.str.len()
    out["alt_length"] = alt.str.len()
    out["length_change"] = out["alt_length"] - out["ref_length"]

    out["variant_type"] = np.select(
        [
            (out["ref_length"] == 1) & (out["alt_length"] == 1),
            out["ref_length"] < out["alt_length"],
            out["ref_length"] > out["alt_length"],
        ],
        [
            "SNV",
            "insertion",
            "deletion",
        ],
        default="other",
    )

    # SNV substitution type, e.g. A>G
    out["substitution"] = np.where(
        out["variant_type"] == "SNV",
        ref.str.upper() + ">" + alt.str.upper(),
        "not_snv",
    )

    # ---------------------------------------------------------
    # 4. Gene features
    # ---------------------------------------------------------
    genes = df["gene_symbols"].fillna("").astype(str)

    # Do not directly one-hot thousands of gene names.
    # Instead capture whether a variant maps to multiple genes
    # and how frequently its gene occurs in this dataset.
    out["multi_gene"] = genes.str.contains(r"[;,|]").astype(int)

    gene_counts = genes.value_counts()
    out["gene_frequency"] = genes.map(gene_counts)

    # ---------------------------------------------------------
    # 5. gnomAD availability
    # ---------------------------------------------------------
    out["gnomad_matched"] = df["gnomad_matched"].astype(int)

    # ---------------------------------------------------------
    # 6. Population-frequency features
    # ---------------------------------------------------------
    numeric_columns = [
        "AF_joint_sas",
        "AC_joint_sas",
        "AN_joint_sas",
        "AF_joint_nfe",
        "AC_joint_nfe",
        "AN_joint_nfe",
        "odds_ratio",
        "pvalue",
    ]

    for col in numeric_columns:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # Useful derived population-frequency differences
    out["af_difference_sas_minus_nfe"] = (
        out["AF_joint_sas"] - out["AF_joint_nfe"]
    )

    out["abs_af_difference"] = (
        out["AF_joint_sas"] - out["AF_joint_nfe"]
    ).abs()

    out["comparison_category"] = (
        df["comparison_category"]
        .fillna("missing")
        .astype(str)
    )

    # ---------------------------------------------------------
    # 7. Missingness indicators
    # ---------------------------------------------------------
    out["sas_af_missing"] = out["AF_joint_sas"].isna().astype(int)
    out["nfe_af_missing"] = out["AF_joint_nfe"].isna().astype(int)
    out["population_test_missing"] = out["pvalue"].isna().astype(int)

    # ---------------------------------------------------------
    # 8. Final sanity checks
    # ---------------------------------------------------------
    assert len(out) == len(df)
    assert out["is_resolved"].isin([0, 1]).all()

    # Explicit check that no T2 columns entered the feature table
    leaked = [c for c in out.columns if c.startswith("t2_")]
    assert len(leaked) == 0, f"T2 leakage detected: {leaked}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print("ML dataset created:", OUTPUT)
    print("Shape:", out.shape)

    print("\nTarget:")
    print(out["is_resolved"].value_counts())

    print("\nTarget prevalence:")
    print(out["is_resolved"].value_counts(normalize=True).round(4))

    print("\nColumns:")
    for i, col in enumerate(out.columns, 1):
        print(f"{i:2}. {col}")

    print("\nMissingness:")
    print(
        (out.isna().mean() * 100)
        .sort_values(ascending=False)
        .head(15)
        .round(2)
        .astype(str) + "%"
    )


if __name__ == "__main__":
    main()