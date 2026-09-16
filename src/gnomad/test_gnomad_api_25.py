#!/usr/bin/env python3
"""
src/gnomad/test_gnomad_api_25.py

Feasibility test: for the first 25 GRCh38 variants in the temporal VUS
cohort with usable coordinates, query the official gnomAD GraphQL API for
gnomAD v4.1 joint population frequency data (SAS and NFE).

Repo-relative paths:
  input:  data/processed/temporal_vus_cohort.csv
  output: data/interim/gnomad_api_test_25.csv

Notes:
  - Uses the public gnomAD GraphQL endpoint (https://gnomad.broadinstitute.org/api)
    with dataset "gnomad_r4" and the "joint" frequency field, which carries
    the joint (genome+exome) v4.1 population frequencies including SAS/NFE.
  - gnomAD variant IDs are built as "chrom-position-ref-alt" using the
    cohort's GRCh38 coordinates (e.g. "1-55516888-G-A"); the API expects
    the "chr" prefix omitted from the chromosome token.
  - The GraphQL query requests only "ac" and "an" per population; AF is
    always computed locally as AC / AN (when AN > 0), never taken from the
    API directly.
  - A minimum 1-second delay is enforced between API requests. On HTTP 429
    (rate limited), the script waits 10 seconds and retries, up to 3
    attempts total per variant, before recording the request as an error.
  - This is a feasibility/smoke test only (25 variants), not the full
    annotation pipeline.
  - Requires the `requests` package.
  - Does NOT add literature, RL, or modeling logic.
"""

import csv
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = REPO_ROOT / "data" / "processed" / "temporal_vus_cohort.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "interim" / "gnomad_api_test_25.csv"

N_VARIANTS = 25

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
GNOMAD_DATASET = "gnomad_r4"

REQUEST_TIMEOUT_SECONDS = 30
MIN_DELAY_BETWEEN_REQUESTS_SECONDS = 1.0
RATE_LIMIT_WAIT_SECONDS = 10.0
MAX_ATTEMPTS_PER_VARIANT = 3

SUPPORTED_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y"}

POPULATIONS_OF_INTEREST = {"sas", "nfe"}

OUTPUT_FIELDNAMES = [
    "vcv_accession",
    "gnomad_variant_id",
    "chromosome",
    "position",
    "ref",
    "alt",
    "gnomad_matched",
    "AC_joint_sas",
    "AN_joint_sas",
    "AF_joint_sas",
    "AC_joint_nfe",
    "AN_joint_nfe",
    "AF_joint_nfe",
    "error_message",
]

GRAPHQL_QUERY = """
query VariantJointFrequency($variantId: String!, $datasetId: DatasetId!) {
  variant(variantId: $variantId, dataset: $datasetId) {
    variant_id
    joint {
      populations {
        id
        ac
        an
      }
    }
  }
}
"""


def normalize_chromosome(raw_chrom):
    """Strip an optional 'chr' prefix and uppercase X/Y for comparison."""
    chrom = raw_chrom.strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom.upper() if chrom.upper() in {"X", "Y", "MT"} else chrom


def select_test_variants(input_csv, n):
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    base_mask = (
        (df["assembly"] == "GRCh38")
        & (df["chromosome"].str.strip() != "")
        & (df["position"].str.strip() != "")
        & (df["ref"].str.strip() != "")
        & (df["alt"].str.strip() != "")
    )
    candidates = df.loc[
        base_mask, ["vcv_accession", "chromosome", "position", "ref", "alt"]
    ].to_dict("records")

    selected = []
    for row in candidates:
        norm_chrom = normalize_chromosome(row["chromosome"])
        if norm_chrom not in SUPPORTED_CHROMOSOMES:
            continue
        selected.append(
            {
                "vcv_accession": row["vcv_accession"],
                "chromosome": row["chromosome"],
                "norm_chrom": norm_chrom,
                "position": row["position"],
                "ref": row["ref"],
                "alt": row["alt"],
            }
        )
        if len(selected) >= n:
            break

    return selected


def build_gnomad_variant_id(norm_chrom, position, ref, alt):
    return f"{norm_chrom}-{position}-{ref}-{alt}"


def compute_af(ac, an):
    """Return AF as a string, computed as AC / AN when AN > 0."""
    try:
        ac_val = int(ac)
        an_val = int(an)
    except (TypeError, ValueError):
        return ""
    if an_val <= 0:
        return ""
    return str(ac_val / an_val)


def query_gnomad_api(gnomad_variant_id):
    """
    Query the gnomAD GraphQL API for one variant's joint frequency data,
    with rate-limit handling.

    Returns (matched, values, error_message):
      matched: bool
      values: dict with AC/AN/AF for sas and nfe (empty strings if absent)
      error_message: "" on success, else a description of the failure
    """
    empty_values = {
        "AC_joint_sas": "",
        "AN_joint_sas": "",
        "AF_joint_sas": "",
        "AC_joint_nfe": "",
        "AN_joint_nfe": "",
        "AF_joint_nfe": "",
    }

    payload = {
        "query": GRAPHQL_QUERY,
        "variables": {
            "variantId": gnomad_variant_id,
            "datasetId": GNOMAD_DATASET,
        },
    }

    last_error_message = ""

    for attempt in range(1, MAX_ATTEMPTS_PER_VARIANT + 1):
        try:
            response = requests.post(
                GNOMAD_API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.RequestException as exc:
            return False, empty_values, f"request error: {exc}"

        if response.status_code == 429:
            last_error_message = "HTTP 429: rate limited"
            if attempt < MAX_ATTEMPTS_PER_VARIANT:
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            return False, empty_values, last_error_message

        if response.status_code != 200:
            return (
                False,
                empty_values,
                f"HTTP {response.status_code}: {response.text[:300]}",
            )

        try:
            payload_json = response.json()
        except ValueError as exc:
            return False, empty_values, f"invalid JSON response: {exc}"

        if "errors" in payload_json and payload_json["errors"]:
            error_text = "; ".join(
                err.get("message", str(err)) for err in payload_json["errors"]
            )
            return False, empty_values, f"GraphQL error: {error_text[:300]}"

        data = payload_json.get("data") or {}
        variant = data.get("variant")

        if not variant:
            # Variant not found in gnomAD -- not an error, just unmatched.
            return False, empty_values, ""

        joint = variant.get("joint") or {}
        populations = joint.get("populations") or []

        values = dict(empty_values)

        for pop in populations:
            pop_id = (pop.get("id") or "").lower()
            if pop_id not in POPULATIONS_OF_INTEREST:
                continue

            ac = pop.get("ac")
            an = pop.get("an")

            values[f"AC_joint_{pop_id}"] = str(ac) if ac is not None else ""
            values[f"AN_joint_{pop_id}"] = str(an) if an is not None else ""
            values[f"AF_joint_{pop_id}"] = compute_af(ac, an)

        return True, values, ""

    # Should not be reached, but fail safe rather than raise.
    return False, empty_values, last_error_message or "unknown error"


def main():
    start_time = time.time()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    variants = select_test_variants(INPUT_CSV, N_VARIANTS)

    queried = 0
    matched = 0
    unmatched = 0
    errors = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for index, variant in enumerate(variants):
            if index > 0:
                time.sleep(MIN_DELAY_BETWEEN_REQUESTS_SECONDS)

            queried += 1

            vcv_accession = variant["vcv_accession"]
            norm_chrom = variant["norm_chrom"]
            position = variant["position"]
            ref = variant["ref"]
            alt = variant["alt"]

            gnomad_variant_id = build_gnomad_variant_id(norm_chrom, position, ref, alt)

            is_matched, values, error_message = query_gnomad_api(gnomad_variant_id)

            row = {
                "vcv_accession": vcv_accession,
                "gnomad_variant_id": gnomad_variant_id,
                "chromosome": variant["chromosome"],
                "position": position,
                "ref": ref,
                "alt": alt,
                "gnomad_matched": is_matched,
                "error_message": error_message,
            }
            row.update(values)

            writer.writerow(row)

            if error_message:
                errors += 1
            elif is_matched:
                matched += 1
            else:
                unmatched += 1

    elapsed_seconds = time.time() - start_time

    print("\n=== test_gnomad_api_25.py summary ===")
    print(f"Queried:   {queried:,}")
    print(f"Matched:   {matched:,}")
    print(f"Unmatched: {unmatched:,}")
    print(f"Errors:    {errors:,}")
    print(f"Elapsed time (seconds): {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()