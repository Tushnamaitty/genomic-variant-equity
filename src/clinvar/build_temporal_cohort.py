#!/usr/bin/env python3
"""
src/clinvar/build_temporal_cohort.py

Merge the T1 VUS cohort with its T2 outcome labels to build the primary
temporal reclassification cohort for the study.

Repo-relative paths:
  input (T1 VUS):     data/interim/t1_vus.csv
  input (T2 labels):  data/interim/t2_labels_for_t1_vus.csv
  output:              data/processed/temporal_vus_cohort.csv
  summary:             results/logs/build_temporal_cohort_summary.json

Notes:
  - Merge is a left join on vcv_accession (T1 is the reference cohort).
  - is_resolved = 1 only for resolved_benign / resolved_pathogenic.
  - resolution_direction:
      resolved_benign      -> "benign"
      resolved_pathogenic  -> "pathogenic"
      still_vus            -> "unresolved"
      other                -> "conflicting_other"
      unmatched (no T2 row)-> "unmatched"
  - "rows_with_coordinates" counts rows where both chromosome and position
    are present; the current T1 file does not explicitly record assembly
    for every row, so this is not restricted to GRCh38 specifically.
  - Does NOT add gnomAD, literature, or modeling logic.
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

T1_VUS_CSV = REPO_ROOT / "data" / "interim" / "t1_vus.csv"
T2_LABELS_CSV = REPO_ROOT / "data" / "interim" / "t2_labels_for_t1_vus.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "temporal_vus_cohort.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "logs" / "build_temporal_cohort_summary.json"

T2_LABEL_COLUMNS = [
    "t2_clinical_significance",
    "t2_review_status",
    "t2_date_last_evaluated",
    "t2_outcome",
]

RESOLVED_OUTCOMES = {"resolved_benign", "resolved_pathogenic"}

OUTCOME_TO_DIRECTION = {
    "resolved_benign": "benign",
    "resolved_pathogenic": "pathogenic",
    "still_vus": "unresolved",
    "other": "conflicting_other",
}


def load_inputs():
    if not T1_VUS_CSV.exists():
        raise FileNotFoundError(f"T1 VUS CSV not found: {T1_VUS_CSV}")
    if not T2_LABELS_CSV.exists():
        raise FileNotFoundError(f"T2 labels CSV not found: {T2_LABELS_CSV}")

    t1_df = pd.read_csv(T1_VUS_CSV, dtype=str, keep_default_na=False)
    t2_df = pd.read_csv(T2_LABELS_CSV, dtype=str, keep_default_na=False)

    return t1_df, t2_df


def merge_cohort(t1_df, t2_df):
    # Keep only the columns we need from T2; avoid duplicate/overlapping
    # non-key columns by selecting explicitly.
    t2_subset = t2_df[["vcv_accession"] + T2_LABEL_COLUMNS].drop_duplicates(
        subset="vcv_accession", keep="first"
    )

    merged = t1_df.merge(t2_subset, on="vcv_accession", how="left")

    # Normalize missing T2 fields (unmatched rows) to blank strings.
    for col in T2_LABEL_COLUMNS:
        merged[col] = merged[col].fillna("")

    return merged


def compute_derived_columns(merged):
    t2_outcome = merged["t2_outcome"]

    is_resolved = t2_outcome.isin(RESOLVED_OUTCOMES).astype(int)

    def direction_for(outcome):
        if outcome == "":
            return "unmatched"
        return OUTCOME_TO_DIRECTION.get(outcome, "conflicting_other")

    resolution_direction = t2_outcome.map(direction_for)

    merged["is_resolved"] = is_resolved
    merged["resolution_direction"] = resolution_direction

    return merged


def compute_summary(merged):
    total_t1_vus = len(merged)

    matched_mask = merged["t2_outcome"] != ""
    matched_to_t2 = int(matched_mask.sum())
    unmatched = total_t1_vus - matched_to_t2

    resolved_benign = int((merged["t2_outcome"] == "resolved_benign").sum())
    resolved_pathogenic = int((merged["t2_outcome"] == "resolved_pathogenic").sum())
    still_vus = int((merged["t2_outcome"] == "still_vus").sum())
    conflicting_other = int((merged["t2_outcome"] == "other").sum())

    has_chrom = merged["chromosome"].astype(str).str.strip() != ""
    has_position = merged["position"].astype(str).str.strip() != ""
    rows_with_coordinates = int((has_chrom & has_position).sum())

    summary = {
        "t1_vus_csv": str(T1_VUS_CSV.relative_to(REPO_ROOT)),
        "t2_labels_csv": str(T2_LABELS_CSV.relative_to(REPO_ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        "total_t1_vus": total_t1_vus,
        "matched_to_t2": matched_to_t2,
        "resolved_benign": resolved_benign,
        "resolved_pathogenic": resolved_pathogenic,
        "still_vus": still_vus,
        "conflicting_other": conflicting_other,
        "unmatched": unmatched,
        "rows_with_coordinates": rows_with_coordinates,
    }

    return summary


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    t1_df, t2_df = load_inputs()
    merged = merge_cohort(t1_df, t2_df)
    merged = compute_derived_columns(merged)

    merged.to_csv(OUTPUT_CSV, index=False)

    summary = compute_summary(merged)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== build_temporal_cohort.py summary ===")
    print(f"Total T1 VUS:                 {summary['total_t1_vus']:,}")
    print(f"Matched to T2:                {summary['matched_to_t2']:,}")
    print(f"  resolved_benign:            {summary['resolved_benign']:,}")
    print(f"  resolved_pathogenic:        {summary['resolved_pathogenic']:,}")
    print(f"  still_vus:                  {summary['still_vus']:,}")
    print(f"  conflicting_other:          {summary['conflicting_other']:,}")
    print(f"Unmatched:                    {summary['unmatched']:,}")
    print(f"Rows with coordinates:        {summary['rows_with_coordinates']:,}")


if __name__ == "__main__":
    main()



# === build_temporal_cohort.py summary ===
# Total T1 VUS:                 1,168,064
# Matched to T2:                1,167,356
#   resolved_benign:            12,094
#   resolved_pathogenic:        3,439
#   still_vus:                  1,124,772
#   conflicting_other:          27,051
# Unmatched:                    708
# Rows with coordinates:        1,164,628    