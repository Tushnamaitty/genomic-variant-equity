#!/usr/bin/env python3
"""
src/gnomad/test_gnomad_remote_match.py

Feasibility test: for the first 100 GRCh38 variants in the temporal VUS
cohort with usable coordinates, remotely query the public gnomAD v4.1
joint-frequency per-chromosome VCF (via bcftools, using HTTP range/tabix
streaming) and attempt an exact CHROM/POS/REF/ALT match, extracting
SAS/NFE joint allele-frequency fields.

Repo-relative paths:
  input:  data/processed/temporal_vus_cohort.csv
  output: data/interim/gnomad_test_100.csv

Notes:
  - Runs inside WSL and shells out to `bcftools` (must be installed and on
    PATH, with an htslib build that supports remote HTTPS streaming).
  - Does NOT download full gnomAD VCFs: each query uses `bcftools view -r`
    against the remote .vcf.bgz URL, which streams only the requested
    region via the accompanying remote .tbi index.
  - This is a feasibility/smoke test only (100 variants), not the full
    annotation pipeline.
  - Does NOT add literature, RL, or modeling logic.
"""

import csv
import subprocess
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = REPO_ROOT / "data" / "processed" / "temporal_vus_cohort.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "interim" / "gnomad_test_100.csv"

N_VARIANTS = 100

GNOMAD_JOINT_URL_TEMPLATE = (
    "https://gnomad-public-us-east-1.s3.amazonaws.com/"
    "release/4.1/vcf/joint/gnomad.joint.v4.1.sites.{chrom}.vcf.bgz"
)

INFO_FIELDS = [
    "AF_joint_sas",
    "AC_joint_sas",
    "AN_joint_sas",
    "AF_joint_nfe",
    "AC_joint_nfe",
    "AN_joint_nfe",
]

OUTPUT_FIELDNAMES = [
    "vcv_accession",
    "chromosome",
    "position",
    "ref",
    "alt",
    "gnomad_matched",
] + INFO_FIELDS + ["error_message"]

BCFTOOLS_TIMEOUT_SECONDS = 60


def to_gnomad_chrom(chromosome):
    """Normalize a ClinVar-style chromosome value to gnomAD's 'chrN' form."""
    chrom = chromosome.strip()
    if chrom.lower().startswith("chr"):
        return chrom
    return f"chr{chrom}"


def select_test_variants(input_csv, n):
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    mask = (
        (df["assembly"] == "GRCh38")
        & (df["chromosome"].str.strip() != "")
        & (df["position"].str.strip() != "")
        & (df["ref"].str.strip() != "")
        & (df["alt"].str.strip() != "")
    )

    filtered = df.loc[mask, ["vcv_accession", "chromosome", "position", "ref", "alt"]]
    return filtered.head(n).to_dict("records")


def parse_info_field(info_string, field_names):
    """Parse a VCF INFO string ('key=value;key=value;flag') into a dict."""
    parsed = {}
    if not info_string or info_string == ".":
        return parsed
    for entry in info_string.split(";"):
        if "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        if key in field_names:
            parsed[key] = value
    return parsed


def query_gnomad_variant(chromosome, position, ref, alt):
    """
    Query the remote gnomAD v4.1 joint VCF for an exact CHROM/POS/REF/ALT
    match. Returns (matched: bool, values: dict, error_message: str).
    """
    gnomad_chrom = to_gnomad_chrom(chromosome)
    url = GNOMAD_JOINT_URL_TEMPLATE.format(chrom=gnomad_chrom)
    region = f"{gnomad_chrom}:{position}-{position}"

    cmd = ["bcftools", "view", "-H", "-r", region, url]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BCFTOOLS_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        return False, {}, f"bcftools error: {stderr[:500]}"
    except subprocess.TimeoutExpired:
        return False, {}, "bcftools timed out"
    except FileNotFoundError:
        return False, {}, "bcftools executable not found on PATH"
    except Exception as exc:  # noqa: BLE001 - broad catch is intentional here
        return False, {}, f"unexpected error: {exc}"

    output = (result.stdout or "").strip()
    if not output:
        return False, {}, ""

    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 8:
            continue

        vcf_pos = fields[1]
        vcf_ref = fields[3]
        vcf_alt_field = fields[4]
        info_field = fields[7]

        if vcf_pos != str(position):
            continue
        if vcf_ref != ref:
            continue

        # gnomAD sites VCFs are typically already split per-allele, but
        # handle a comma-separated ALT defensively.
        alt_alleles = vcf_alt_field.split(",")
        if alt not in alt_alleles:
            continue

        info_values = parse_info_field(info_field, set(INFO_FIELDS))
        values = {field: info_values.get(field, "") for field in INFO_FIELDS}
        return True, values, ""

    # Region returned records, but none matched exactly on REF/ALT.
    return False, {}, ""


def main():
    start_time = time.time()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    variants = select_test_variants(INPUT_CSV, N_VARIANTS)

    total_queried = 0
    exact_matches = 0
    unmatched = 0
    errors = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for variant in variants:
            total_queried += 1

            vcv_accession = variant["vcv_accession"]
            chromosome = variant["chromosome"]
            position = variant["position"]
            ref = variant["ref"]
            alt = variant["alt"]

            matched, values, error_message = query_gnomad_variant(
                chromosome, position, ref, alt
            )

            row = {
                "vcv_accession": vcv_accession,
                "chromosome": chromosome,
                "position": position,
                "ref": ref,
                "alt": alt,
                "gnomad_matched": matched,
                "error_message": error_message,
            }
            for field in INFO_FIELDS:
                row[field] = values.get(field, "")

            writer.writerow(row)

            if error_message:
                errors += 1
            elif matched:
                exact_matches += 1
            else:
                unmatched += 1

    elapsed_seconds = time.time() - start_time

    print("\n=== test_gnomad_remote_match.py summary ===")
    print(f"Total queried:   {total_queried:,}")
    print(f"Exact matches:   {exact_matches:,}")
    print(f"Unmatched:       {unmatched:,}")
    print(f"Errors:          {errors:,}")
    print(f"Elapsed time (seconds): {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()