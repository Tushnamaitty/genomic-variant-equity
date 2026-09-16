from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


INPUT = Path(
    "data/processed/gnomad/chr22_gnomad_annotated.csv"
)

OUTPUT = Path(
    "data/processed/gnomad/chr22_gnomad_with_enrichment.csv"
)


def main():

    df = pd.read_csv(INPUT, low_memory=False)

    df["pvalue"] = pd.to_numeric(
        df["pvalue"],
        errors="coerce"
    )

    df["odds_ratio"] = pd.to_numeric(
        df["odds_ratio"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 1. BH correction across ALL valid chr22 Fisher tests
    # ---------------------------------------------------------

    valid_test = (
        df["pvalue"].notna()
        & np.isfinite(df["pvalue"])
    )

    df["qvalue"] = np.nan

    if valid_test.sum() > 0:

        _, qvals, _, _ = multipletests(
            df.loc[valid_test, "pvalue"],
            alpha=0.05,
            method="fdr_bh",
        )

        df.loc[valid_test, "qvalue"] = qvals

    # ---------------------------------------------------------
    # 2. Assign population-enrichment labels
    # ---------------------------------------------------------

    df["enrichment_group"] = "no_clear_enrichment"

    # Preserve special categories first
    df.loc[
        df["comparison_category"] == "unmatched",
        "enrichment_group"
    ] = "unmatched"

    df.loc[
        df["comparison_category"] == "insufficient_data",
        "enrichment_group"
    ] = "insufficient_data"

    df.loc[
        df["comparison_category"] == "observed_zero_both",
        "enrichment_group"
    ] = "observed_zero_both"

    # SAS enriched
    sas_mask = (
        (df["comparison_category"] == "valid_comparison")
        & (df["qvalue"] < 0.05)
        & (df["odds_ratio"] >= 2.0)
    )

    df.loc[
        sas_mask,
        "enrichment_group"
    ] = "SAS_enriched"

    # NFE enriched
    nfe_mask = (
        (df["comparison_category"] == "valid_comparison")
        & (df["qvalue"] < 0.05)
        & (df["odds_ratio"] <= 0.5)
    )

    df.loc[
        nfe_mask,
        "enrichment_group"
    ] = "NFE_enriched"

    # ---------------------------------------------------------
    # 3. Save
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    # ---------------------------------------------------------
    # 4. QC
    # ---------------------------------------------------------

    print("=== ALL CHR22 VARIANTS ===")
    print(
        df["enrichment_group"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nValid Fisher tests:")
    print(valid_test.sum())

    print("\nq < 0.05:")
    print(
        (
            df["qvalue"] < 0.05
        ).sum()
    )

    # Clean longitudinal ML cohort only
    clean = df[
        df["is_clean_primary_endpoint"] == True
    ].copy()

    print("\n=== CLEAN ML COHORT ===")
    print("Rows:", len(clean))

    print(
        clean["enrichment_group"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== RESOLUTION BY ENRICHMENT GROUP ===")

    summary = (
        clean
        .groupby("enrichment_group")["is_resolved"]
        .agg(["count", "sum", "mean"])
        .sort_values("count", ascending=False)
    )

    summary["mean"] = (
        summary["mean"] * 100
    )

    summary = summary.rename(
        columns={
            "count": "n_variants",
            "sum": "n_resolved",
            "mean": "resolution_percent",
        }
    )

    print(
        summary.round(4).to_string()
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()