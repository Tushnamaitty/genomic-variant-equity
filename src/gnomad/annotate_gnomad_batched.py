#!/usr/bin/env python3
"""
src/gnomad/annotate_gnomad_batched.py

Annotate all eligible GRCh38 ClinVar variants in the temporal VUS cohort
with gnomAD v4.1 joint SAS/NFE allele-frequency fields, using ONE remote
bcftools query per chromosome (not one query per variant).

Repo-relative paths:
  input:   data/processed/temporal_vus_cohort.csv
  output:  data/interim/gnomad_annotations.csv
  summary: results/logs/gnomad_annotation_summary.json

Approach:
  - Filter to assembly == "GRCh38" rows with non-empty chromosome, position,
    ref, alt, and chromosome in {1..22, X, Y}. MT/Un/other chromosomes are
    skipped and counted separately.
  - Group eligible variants by chromosome.
  - Optionally restrict to a subset of chromosomes (--chromosomes) and/or
    cap the number of ClinVar rows processed per chromosome
    (--limit-per-chromosome), for cheaper test runs. With no arguments the
    full eligible cohort across all supported chromosomes is processed.
  - For each selected chromosome, write a small temporary regions file
    listing the unique positions needed, then run a single `bcftools view
    -R <regions> <remote gnomAD chromosome VCF URL>` call. bcftools streams
    only the requested regions via the remote .tbi index -- the full
    chromosome VCF is never downloaded.
  - Build a (position, ref, alt) -> INFO-field lookup from that single
    chromosome's returned records, then emit one output row per eligible
    ClinVar variant (not per gnomAD record), matching exactly on
    chromosome + position + REF + ALT.
  - Temporary regions files are created with tempfile and always removed.

This script runs inside WSL and shells out to `bcftools` (must be on PATH,
built with HTTPS-capable htslib). It does NOT add literature, RL, or
modeling logic.
"""

import argparse
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = REPO_ROOT / "data" / "processed" / "temporal_vus_cohort.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "interim" / "gnomad_annotations.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "logs" / "gnomad_annotation_summary.json"

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

SUPPORTED_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y"}
CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]

BCFTOOLS_TIMEOUT_SECONDS = 1800  # a whole-chromosome regions query can be slow


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate GRCh38 ClinVar variants with gnomAD v4.1 joint "
            "SAS/NFE frequencies via one batched bcftools query per "
            "chromosome."
        )
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=None,
        metavar="CHROM",
        help=(
            "Optional subset of chromosomes to process (e.g. "
            "--chromosomes 22 or --chromosomes 1 2 X). Accepts values with "
            "or without a 'chr' prefix. Default: all supported chromosomes "
            "(1-22, X, Y) present in the cohort."
        ),
    )
    parser.add_argument(
        "--limit-per-chromosome",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Optional cap on the number of eligible ClinVar rows processed "
            "per chromosome, for cheaper test runs. Default: no limit "
            "(process the full eligible cohort)."
        ),
    )
    return parser.parse_args()


def normalize_chromosome(raw_chrom):
    """Strip an optional 'chr' prefix and uppercase X/Y for comparison."""
    chrom = raw_chrom.strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom.upper() if chrom.upper() in {"X", "Y", "MT"} else chrom


def to_gnomad_chrom(normalized_chrom):
    """Map a normalized chromosome ('1', 'X', ...) to gnomAD's 'chrN' form."""
    return f"chr{normalized_chrom}"


def load_eligible_variants(input_csv):
    """
    Return (eligible_rows, total_rows, unsupported_chrom_rows).

    eligible_rows is a list of dicts with keys:
      vcv_accession, chromosome (raw), norm_chrom, position, ref, alt
    """
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    total_rows = len(df)

    grch38_mask = (
        (df["assembly"] == "GRCh38")
        & (df["chromosome"].str.strip() != "")
        & (df["position"].str.strip() != "")
        & (df["ref"].str.strip() != "")
        & (df["alt"].str.strip() != "")
    )
    candidates = df.loc[
        grch38_mask, ["vcv_accession", "chromosome", "position", "ref", "alt"]
    ].to_dict("records")

    eligible_rows = []
    unsupported_chrom_rows = 0

    for row in candidates:
        norm_chrom = normalize_chromosome(row["chromosome"])
        if norm_chrom not in SUPPORTED_CHROMOSOMES:
            unsupported_chrom_rows += 1
            continue
        eligible_rows.append(
            {
                "vcv_accession": row["vcv_accession"],
                "chromosome": row["chromosome"],
                "norm_chrom": norm_chrom,
                "position": row["position"],
                "ref": row["ref"],
                "alt": row["alt"],
            }
        )

    return eligible_rows, total_rows, unsupported_chrom_rows


def group_by_chromosome(eligible_rows):
    grouped = {}
    for row in eligible_rows:
        grouped.setdefault(row["norm_chrom"], []).append(row)
    return grouped


def apply_chromosome_filter(grouped, requested_chromosomes):
    """
    Restrict `grouped` to the requested chromosomes (if any), normalizing
    each requested value the same way cohort chromosomes are normalized.
    Returns the filtered dict.
    """
    if not requested_chromosomes:
        return grouped

    normalized_requested = {normalize_chromosome(c) for c in requested_chromosomes}
    return {
        chrom: rows for chrom, rows in grouped.items() if chrom in normalized_requested
    }


def apply_per_chromosome_limit(grouped, limit):
    """Cap the number of rows processed per chromosome, if a limit is given."""
    if limit is None:
        return grouped
    return {chrom: rows[:limit] for chrom, rows in grouped.items()}


def write_regions_file(positions, norm_chrom):
    """
    Write a temporary bcftools regions file (tab-separated CHROM\\tSTART\\tEND,
    1-based, inclusive) for the unique positions on this chromosome.
    Returns the Path to the temp file; caller is responsible for deletion.
    """
    gnomad_chrom = to_gnomad_chrom(norm_chrom)

    fd, path_str = tempfile.mkstemp(prefix=f"gnomad_regions_{norm_chrom}_", suffix=".tsv")
    tmp_path = Path(path_str)

    try:
        with open(fd, "w", encoding="utf-8") as f:
            for pos in sorted(positions, key=lambda p: int(p)):
                f.write(f"{gnomad_chrom}\t{pos}\t{pos}\n")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return tmp_path


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


def query_chromosome(norm_chrom, positions):
    """
    Run a single bcftools view -R <regions_file> query against the remote
    gnomAD v4.1 joint VCF for this chromosome.

    Returns (lookup, error_message):
      lookup: dict mapping (position_str, ref, alt) -> {info field: value}
      error_message: "" on success, else a description of the failure
                      (in which case lookup is empty and every variant on
                      this chromosome should be recorded as unmatched/error).
    """
    gnomad_chrom = to_gnomad_chrom(norm_chrom)
    url = GNOMAD_JOINT_URL_TEMPLATE.format(chrom=gnomad_chrom)

    lookup = {}

    regions_path = write_regions_file(positions, norm_chrom)
    try:
        cmd = ["bcftools", "view", "-H", "-R", str(regions_path), url]
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
            return {}, f"bcftools error: {stderr[:500]}"
        except subprocess.TimeoutExpired:
            return {}, "bcftools timed out"
        except FileNotFoundError:
            return {}, "bcftools executable not found on PATH"
        except Exception as exc:  # noqa: BLE001 - broad catch is intentional here
            return {}, f"unexpected error: {exc}"

        output = (result.stdout or "").strip()
        if not output:
            return {}, ""

        for line in output.splitlines():
            fields = line.split("\t")
            if len(fields) < 8:
                continue

            vcf_pos = fields[1]
            vcf_ref = fields[3]
            vcf_alt_field = fields[4]
            info_field = fields[7]

            info_values = parse_info_field(info_field, set(INFO_FIELDS))
            values = {field: info_values.get(field, "") for field in INFO_FIELDS}

            # gnomAD sites VCFs are typically already split per-allele, but
            # handle a comma-separated ALT defensively by indexing each one.
            for alt_allele in vcf_alt_field.split(","):
                lookup[(vcf_pos, vcf_ref, alt_allele)] = values

        return lookup, ""
    finally:
        regions_path.unlink(missing_ok=True)


def annotate_chromosome(norm_chrom, rows):
    """
    Annotate all eligible variants for one chromosome. Returns
    (annotated_rows, chrom_matches, chrom_unmatched, chrom_errors).
    """
    unique_positions = {row["position"] for row in rows}
    lookup, error_message = query_chromosome(norm_chrom, unique_positions)

    annotated_rows = []
    chrom_matches = 0
    chrom_unmatched = 0
    chrom_errors = 0

    for row in rows:
        key = (row["position"], row["ref"], row["alt"])

        if error_message:
            matched = False
            values = {}
            row_error = error_message
            chrom_errors += 1
        elif key in lookup:
            matched = True
            values = lookup[key]
            row_error = ""
            chrom_matches += 1
        else:
            matched = False
            values = {}
            row_error = ""
            chrom_unmatched += 1

        out_row = {
            "vcv_accession": row["vcv_accession"],
            "chromosome": row["chromosome"],
            "position": row["position"],
            "ref": row["ref"],
            "alt": row["alt"],
            "gnomad_matched": matched,
            "error_message": row_error,
        }
        for field in INFO_FIELDS:
            out_row[field] = values.get(field, "")

        annotated_rows.append(out_row)

    return annotated_rows, chrom_matches, chrom_unmatched, chrom_errors


def main():
    args = parse_args()
    start_time = time.time()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    eligible_rows, total_cohort_rows, unsupported_chrom_rows = load_eligible_variants(
        INPUT_CSV
    )
    grouped = group_by_chromosome(eligible_rows)

    # Optional filtering for cheaper/test runs. With no arguments, behavior
    # is unchanged: the full eligible cohort across all supported
    # chromosomes present in the data is processed.
    grouped = apply_chromosome_filter(grouped, args.chromosomes)
    grouped = apply_per_chromosome_limit(grouped, args.limit_per_chromosome)

    eligible_total = sum(len(rows) for rows in grouped.values())

    exact_matches = 0
    unmatched = 0
    errors = 0
    per_chromosome_counts = {}

    # Process chromosomes in a stable, human-friendly order: 1-22, X, Y.
    ordered_chroms = [c for c in CHROM_ORDER if c in grouped]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for norm_chrom in ordered_chroms:
            rows = grouped[norm_chrom]
            if not rows:
                continue

            annotated_rows, chrom_matches, chrom_unmatched, chrom_errors = (
                annotate_chromosome(norm_chrom, rows)
            )

            for out_row in annotated_rows:
                writer.writerow(out_row)

            exact_matches += chrom_matches
            unmatched += chrom_unmatched
            errors += chrom_errors

            per_chromosome_counts[norm_chrom] = {
                "eligible_variants": len(rows),
                "matched": chrom_matches,
                "unmatched": chrom_unmatched,
                "errors": chrom_errors,
            }

            print(
                f"[chr{norm_chrom}] eligible={len(rows):,} "
                f"matched={chrom_matches:,} unmatched={chrom_unmatched:,} "
                f"errors={chrom_errors:,}",
                flush=True,
            )

    elapsed_seconds = time.time() - start_time

    summary = {
        "input_csv": str(INPUT_CSV.relative_to(REPO_ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        "chromosomes_arg": args.chromosomes,
        "limit_per_chromosome_arg": args.limit_per_chromosome,
        "total_cohort_rows": total_cohort_rows,
        "eligible_grch38_rows": eligible_total,
        "unsupported_chromosome_rows": unsupported_chrom_rows,
        "exact_matches": exact_matches,
        "unmatched": unmatched,
        "errors": errors,
        "per_chromosome_counts": per_chromosome_counts,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== annotate_gnomad_batched.py summary ===")
    print(f"Total cohort rows:            {total_cohort_rows:,}")
    print(f"Eligible rows processed:      {eligible_total:,}")
    print(f"Unsupported chromosome rows:  {unsupported_chrom_rows:,}")
    print(f"Exact matches:                {exact_matches:,}")
    print(f"Unmatched:                    {unmatched:,}")
    print(f"Errors:                       {errors:,}")
    print(f"Elapsed time (seconds):       {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()