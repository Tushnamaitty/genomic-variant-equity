#!/usr/bin/env python3
"""
src/gnomad/run_chromosome_pipeline.py

Automates the validated chr22 ClinVar-VUS -> gnomAD v4.1 joint
population-frequency pipeline for an arbitrary chromosome.

Pipeline stages (mirrors the validated chr22 pilot):
  1.  Select GRCh38 T1-VUS ClinVar variants for the target chromosome.
      Report total chromosome variants, variants missing position/ref/alt,
      and the resulting usable count (chr22-pilot accounting style).
  2.  Explicitly flag REF == ALT records (excluded, not silently dropped).
  3.  Flag variants whose reference span overlaps an 'N' in the reference
      FASTA (excluded, not silently dropped).
  4.  Normalize the remaining ("safe") variants with `bcftools norm`
      against the reference FASTA. Reference-N variants are already
      excluded beforehand, so `bcftools norm` is run WITHOUT `-c w`: any
      remaining REF/reference mismatch is left at bcftools' default
      "error" behavior and fails the pipeline loudly rather than warning
      and continuing.
  5.  Query the chromosome-specific gnomAD v4.1 joint VCF using the
      validated sequential-target strategy (`bcftools view -T <targets>`,
      NOT `-R`, which the chr22 pilot showed was far too slow due to
      random-seek index access). The command's stdout is streamed
      directly to a temporary local file (never fully buffered in Python
      memory) and that file is then parsed line-by-line to extract
      AC/AN/AF_joint_{sas,nfe}.
  6.  Match ClinVar <-> gnomAD exactly on CHROM+POS+REF+ALT only.
  7.  Never impute AF=0 for unmatched variants -- those fields stay blank.
  8.  Track categories: matched (valid_comparison / insufficient_data /
      observed_zero_both), unmatched, ref_equals_alt_excluded,
      reference_N_excluded.
  9.  AN == 0 in either population -> insufficient_data.
  10. Valid AN but AC == 0 in both populations -> observed_zero_both.
  11. For all other matched variants ("valid_comparison"), run a
      two-sided Fisher exact test (SAS vs NFE AC/AN) and record the RAW
      p-value and odds ratio. This script does NOT perform BH-FDR
      correction or assign final SAS_enriched/NFE_enriched labels --
      those require pooling p-values across all chromosomes and are
      computed by a separate genome-wide merge/correction step.
  12. Merge T2 outcome labels by vcv_accession. Clean primary endpoint =
      resolved_benign, resolved_pathogenic, or still_vus. "other" and
      missing/unmatched T2 are excluded from the clean endpoint (but
      retained in the output, not dropped).
      is_resolved = True only for resolved_benign/resolved_pathogenic.
      is_clean_primary_endpoint = True for resolved_benign/
      resolved_pathogenic/still_vus.
  13. Print detailed QC counts comparable to the chr22 pilot.
  14. Save one final chromosome-level CSV under data/processed/gnomad/.

Design notes:
  - Never writes to any input path (T1 CSV, T2 CSV, gnomAD VCF, FASTA).
  - Resumable: each stage's output is cached under a per-chromosome work
    directory; a cached file is reused on re-run unless --force is passed.
  - Fails loudly (raises) on missing/unexpected required columns,
    structurally malformed input, allele-specific INFO field counts that
    don't match the number of ALT alleles, or a bcftools norm REF
    mismatch, rather than silently coercing data.
  - Multiallelic gnomAD records are handled safely: Number=A INFO fields
    (AC_joint_*, AF_joint_*) are split and indexed to their corresponding
    ALT allele; Number=1 fields (AN_joint_*) are shared across all ALT
    alleles at that site. A per-allele field whose split length does not
    match the ALT count raises immediately.
  - bcftools subprocess calls have NO timeout by default -- local
    chromosome processing may legitimately take well over 30 minutes,
    especially for large chromosomes. An optional timeout (in seconds)
    can be supplied via --bcftools-timeout if desired.
  - External dependencies: bcftools (on PATH), pandas, numpy, scipy
    (scipy.stats.fisher_exact), pysam (indexed FASTA random access).

Does NOT implement literature, RL, downstream modeling, cross-chromosome
BH-FDR, or final enrichment-label logic.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    import pysam
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "pysam is required for indexed FASTA access (pip install pysam)"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_T1_VUS_CSV = REPO_ROOT / "data" / "interim" / "t1_vus.csv"
DEFAULT_T2_LABELS_CSV = REPO_ROOT / "data" / "interim" / "t2_labels_for_t1_vus.csv"
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "interim" / "gnomad_pipeline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "gnomad"

SUPPORTED_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y"}

REQUIRED_T1_COLUMNS = {
    "vcv_accession",
    "assembly",
    "chromosome",
    "position",
    "ref",
    "alt",
}
REQUIRED_T2_COLUMNS = {
    "vcv_accession",
    "t2_clinical_significance",
    "t2_review_status",
    "t2_date_last_evaluated",
    "t2_outcome",
}

INFO_FIELDS = [
    "AF_joint_sas",
    "AC_joint_sas",
    "AN_joint_sas",
    "AF_joint_nfe",
    "AC_joint_nfe",
    "AN_joint_nfe",
]

# Number=A fields: one value per ALT allele, comma-separated in the VCF.
PER_ALLELE_INFO_FIELDS = {"AC_joint_sas", "AF_joint_sas", "AC_joint_nfe", "AF_joint_nfe"}
# Number=1 fields: a single value shared across all ALT alleles at a site.
SHARED_INFO_FIELDS = {"AN_joint_sas", "AN_joint_nfe"}

# No fixed bcftools timeout by default -- local chromosome processing may
# legitimately take well over 30 minutes. None means "wait indefinitely";
# a caller may pass an explicit number of seconds via --bcftools-timeout.
DEFAULT_BCFTOOLS_TIMEOUT_SECONDS = None

RESOLVED_OUTCOMES = {"resolved_benign", "resolved_pathogenic"}
CLEAN_PRIMARY_ENDPOINT_OUTCOMES = {"resolved_benign", "resolved_pathogenic", "still_vus"}


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated ClinVar-VUS -> gnomAD v4.1 joint "
            "population-frequency pipeline for a single chromosome. "
            "Produces raw Fisher p-values/odds ratios only; BH-FDR and "
            "final enrichment labels are computed in a separate "
            "genome-wide step after all chromosomes are merged."
        )
    )
    parser.add_argument(
        "--chromosome",
        required=True,
        help="Chromosome to process, e.g. '22', 'X', 'Y'.",
    )
    parser.add_argument(
        "--t1-vus-csv",
        type=Path,
        default=DEFAULT_T1_VUS_CSV,
        help=f"Path to T1 VUS CSV (default: {DEFAULT_T1_VUS_CSV.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--gnomad-vcf",
        type=Path,
        required=True,
        help="Path (local or remote-capable via bcftools) to the chromosome-specific gnomAD v4.1 joint VCF.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        required=True,
        help="Path to the GRCh38 chromosome FASTA (must have a .fai index or be indexable).",
    )
    parser.add_argument(
        "--t2-labels-csv",
        type=Path,
        default=DEFAULT_T2_LABELS_CSV,
        help=f"Path to T2 labels CSV (default: {DEFAULT_T2_LABELS_CSV.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Directory for cached intermediate stage outputs (default: {DEFAULT_WORK_DIR.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the final chromosome-level CSV (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached intermediate stage outputs and recompute everything.",
    )
    parser.add_argument(
        "--bcftools-timeout",
        type=float,
        default=DEFAULT_BCFTOOLS_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Optional timeout in seconds for each bcftools subprocess call. "
            "Default: no timeout (wait until the process completes), since "
            "local chromosome processing may legitimately take well over "
            "30 minutes."
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def normalize_chromosome(raw_chrom):
    chrom = str(raw_chrom).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom.upper() if chrom.upper() in {"X", "Y", "MT"} else chrom


def to_gnomad_chrom(norm_chrom):
    return f"chr{norm_chrom}"


def validate_required_columns(df, required_columns, context):
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{context}: missing required column(s) {sorted(missing)}; "
            f"found columns: {sorted(df.columns)}"
        )


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_cached(path, force):
    """Return True if a cached stage output exists and should be reused."""
    return (not force) and path.exists()


def run_bcftools(cmd, timeout=None):
    """
    Run a bcftools subprocess command with output captured, raising loudly
    on failure. `timeout` is in seconds; the default (None) means wait
    until the process completes, with no fixed limit.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"bcftools command failed ({' '.join(cmd)}):\n{exc.stderr}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"bcftools command timed out ({' '.join(cmd)})") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("bcftools executable not found on PATH") from exc
    return result


def run_bcftools_to_file(cmd, output_path, timeout=None):
    """
    Run a bcftools subprocess command, streaming stdout directly to
    `output_path` rather than buffering it in Python memory. Raises
    loudly on failure (non-zero exit, timeout, or missing executable).
    `timeout` is in seconds; the default (None) means wait until the
    process completes, with no fixed limit.
    """
    with open(output_path, "w", encoding="utf-8") as out_f:
        try:
            subprocess.run(
                cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"bcftools command failed ({' '.join(cmd)}):\n{exc.stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"bcftools command timed out ({' '.join(cmd)})") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("bcftools executable not found on PATH") from exc
    return output_path


# --------------------------------------------------------------------------
# Stage 1: load and filter T1 VUS variants for this chromosome
# --------------------------------------------------------------------------

def load_t1_variants(t1_vus_csv, norm_chrom, work_dir, force):
    """
    Select GRCh38 T1-VUS variants for this chromosome, with chr22-pilot
    style accounting:
      total chromosome variants -> missing position/ref/alt -> usable.

    Returns (usable_df, total_chrom_variants, missing_fields_count).
    """
    usable_cache = work_dir / f"01_t1_filtered_chr{norm_chrom}.csv"
    counts_cache = work_dir / f"01_counts_chr{norm_chrom}.json"

    if stage_cached(usable_cache, force) and stage_cached(counts_cache, force):
        print(f"[stage 1] using cached filtered T1 variants: {usable_cache}")
        usable = pd.read_csv(usable_cache, dtype=str, keep_default_na=False)
        with open(counts_cache, "r", encoding="utf-8") as f:
            counts = json.load(f)
        return usable, counts["total_chrom_variants"], counts["missing_fields_count"]

    if not t1_vus_csv.exists():
        raise FileNotFoundError(f"T1 VUS CSV not found: {t1_vus_csv}")

    df = pd.read_csv(t1_vus_csv, dtype=str, keep_default_na=False)
    validate_required_columns(df, REQUIRED_T1_COLUMNS, "T1 VUS CSV")

    df["_norm_chrom"] = df["chromosome"].map(normalize_chromosome)

    # Step A: all GRCh38 T1 VUS variants on this chromosome, regardless of
    # whether position/ref/alt are populated.
    chrom_mask = (df["assembly"] == "GRCh38") & (df["_norm_chrom"] == norm_chrom)
    chrom_df = df.loc[chrom_mask].drop(columns=["_norm_chrom"]).reset_index(drop=True)
    total_chrom_variants = len(chrom_df)

    # Step B: explicitly count variants missing position/ref/alt.
    missing_mask = (
        (chrom_df["position"].str.strip() == "")
        | (chrom_df["ref"].str.strip() == "")
        | (chrom_df["alt"].str.strip() == "")
    )
    missing_fields_count = int(missing_mask.sum())

    # Step C: usable variants (position/ref/alt all present).
    usable = chrom_df.loc[~missing_mask].reset_index(drop=True)

    if total_chrom_variants == 0:
        print(
            f"[stage 1] WARNING: no GRCh38 chr{norm_chrom} T1 VUS variants "
            f"found in {t1_vus_csv}"
        )

    usable.to_csv(usable_cache, index=False)
    with open(counts_cache, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_chrom_variants": total_chrom_variants,
                "missing_fields_count": missing_fields_count,
            },
            f,
            indent=2,
        )

    print(
        f"[stage 1] chr{norm_chrom}: total GRCh38 T1 VUS = {total_chrom_variants:,}; "
        f"missing position/ref/alt = {missing_fields_count:,}; "
        f"usable = {len(usable):,}"
    )
    return usable, total_chrom_variants, missing_fields_count


# --------------------------------------------------------------------------
# Stage 2: flag REF == ALT
# --------------------------------------------------------------------------

def flag_ref_equals_alt(df, work_dir, norm_chrom, force):
    safe_cache = work_dir / f"02_safe_after_refalt_chr{norm_chrom}.csv"
    excluded_cache = work_dir / f"02_ref_equals_alt_excluded_chr{norm_chrom}.csv"

    if stage_cached(safe_cache, force) and stage_cached(excluded_cache, force):
        print(f"[stage 2] using cached ref==alt split for chr{norm_chrom}")
        safe = pd.read_csv(safe_cache, dtype=str, keep_default_na=False)
        excluded = pd.read_csv(excluded_cache, dtype=str, keep_default_na=False)
        return safe, excluded

    ref_eq_alt_mask = df["ref"] == df["alt"]
    excluded = df.loc[ref_eq_alt_mask].reset_index(drop=True)
    safe = df.loc[~ref_eq_alt_mask].reset_index(drop=True)

    safe.to_csv(safe_cache, index=False)
    excluded.to_csv(excluded_cache, index=False)

    print(
        f"[stage 2] REF==ALT excluded: {len(excluded):,}; "
        f"remaining safe: {len(safe):,}"
    )
    return safe, excluded


# --------------------------------------------------------------------------
# Stage 3: flag reference-span overlap with 'N'
# --------------------------------------------------------------------------

def check_reference_n_overlap(df, reference_fasta, work_dir, norm_chrom, force):
    clean_cache = work_dir / f"03_clean_after_n_check_chr{norm_chrom}.csv"
    n_excluded_cache = work_dir / f"03_reference_n_excluded_chr{norm_chrom}.csv"

    if stage_cached(clean_cache, force) and stage_cached(n_excluded_cache, force):
        print(f"[stage 3] using cached N-overlap split for chr{norm_chrom}")
        clean = pd.read_csv(clean_cache, dtype=str, keep_default_na=False)
        n_excluded = pd.read_csv(n_excluded_cache, dtype=str, keep_default_na=False)
        return clean, n_excluded

    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    fasta = pysam.FastaFile(str(reference_fasta))
    fasta_contigs = set(fasta.references)
    gnomad_chrom = to_gnomad_chrom(norm_chrom)

    if gnomad_chrom in fasta_contigs:
        fasta_contig = gnomad_chrom
    elif norm_chrom in fasta_contigs:
        fasta_contig = norm_chrom
    else:
        fasta.close()
        raise ValueError(
            f"Reference FASTA {reference_fasta} does not contain a contig "
            f"matching '{gnomad_chrom}' or '{norm_chrom}'; found: "
            f"{sorted(fasta_contigs)[:10]}..."
        )

    has_n_flags = []
    for _, row in df.iterrows():
        try:
            position = int(row["position"])
        except ValueError as exc:
            fasta.close()
            raise ValueError(
                f"Malformed position for vcv_accession={row.get('vcv_accession')}: "
                f"{row['position']!r}"
            ) from exc

        ref_len = max(len(row["ref"]), 1)
        # 0-based half-open fetch spanning the reference allele.
        start_0based = position - 1
        end_0based = start_0based + ref_len

        try:
            span_seq = fasta.fetch(fasta_contig, start_0based, end_0based)
        except (ValueError, IndexError) as exc:
            fasta.close()
            raise ValueError(
                f"Failed to fetch reference span for vcv_accession="
                f"{row.get('vcv_accession')} at {fasta_contig}:{position}-"
                f"{position + ref_len - 1}: {exc}"
            ) from exc

        has_n_flags.append("N" in span_seq.upper())

    fasta.close()

    df = df.copy()
    df["_has_reference_n"] = has_n_flags

    n_excluded = df.loc[df["_has_reference_n"]].drop(columns=["_has_reference_n"]).reset_index(drop=True)
    clean = df.loc[~df["_has_reference_n"]].drop(columns=["_has_reference_n"]).reset_index(drop=True)

    clean.to_csv(clean_cache, index=False)
    n_excluded.to_csv(n_excluded_cache, index=False)

    print(
        f"[stage 3] reference-N excluded: {len(n_excluded):,}; "
        f"remaining clean: {len(clean):,}"
    )
    return clean, n_excluded


# --------------------------------------------------------------------------
# Stage 4: normalize safe variants with bcftools norm
# --------------------------------------------------------------------------

def build_prenorm_vcf(df, norm_chrom, vcf_path):
    gnomad_chrom = to_gnomad_chrom(norm_chrom)
    with open(vcf_path, "w", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={gnomad_chrom}>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, row in df.iterrows():
            f.write(
                f"{gnomad_chrom}\t{row['position']}\t{row['vcv_accession']}\t"
                f"{row['ref']}\t{row['alt']}\t.\t.\t.\n"
            )


def parse_normalized_vcf(vcf_path):
    """Return dict: vcv_accession -> (norm_position, norm_ref, norm_alt)."""
    normalized = {}
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            _chrom, pos, variant_id, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
            normalized[variant_id] = (pos, ref, alt)
    return normalized


def normalize_with_bcftools(df, reference_fasta, work_dir, norm_chrom, force, bcftools_timeout=None):
    normalized_cache = work_dir / f"04_normalized_chr{norm_chrom}.csv"

    if stage_cached(normalized_cache, force):
        print(f"[stage 4] using cached normalized variants for chr{norm_chrom}")
        return pd.read_csv(normalized_cache, dtype=str, keep_default_na=False)

    if df.empty:
        empty = df.copy()
        empty["norm_position"] = pd.Series(dtype=str)
        empty["norm_ref"] = pd.Series(dtype=str)
        empty["norm_alt"] = pd.Series(dtype=str)
        empty.to_csv(normalized_cache, index=False)
        return empty

    prenorm_vcf = work_dir / f"04_prenorm_chr{norm_chrom}.vcf"
    postnorm_vcf = work_dir / f"04_postnorm_chr{norm_chrom}.vcf"

    build_prenorm_vcf(df, norm_chrom, prenorm_vcf)

    # NOTE: "-c w" (warn-and-continue on REF mismatch) is intentionally
    # NOT used here. Reference-N variants are already excluded upstream
    # (stage 3), so any remaining REF/reference mismatch at this point is
    # unexpected and should fail the pipeline loudly. Without -c, bcftools
    # norm defaults to erroring out on a REF mismatch, which run_bcftools
    # converts into a RuntimeError below.
    cmd = [
        "bcftools",
        "norm",
        "-f",
        str(reference_fasta),
        "-Ov",
        "-o",
        str(postnorm_vcf),
        str(prenorm_vcf),
    ]
    run_bcftools(cmd, timeout=bcftools_timeout)

    normalized_lookup = parse_normalized_vcf(postnorm_vcf)

    norm_positions, norm_refs, norm_alts = [], [], []
    for _, row in df.iterrows():
        accession = row["vcv_accession"]
        if accession not in normalized_lookup:
            raise RuntimeError(
                f"bcftools norm did not return a record for vcv_accession="
                f"{accession}; normalization output may be malformed."
            )
        pos, ref, alt = normalized_lookup[accession]
        norm_positions.append(pos)
        norm_refs.append(ref)
        norm_alts.append(alt)

    normalized_df = df.copy()
    normalized_df["norm_position"] = norm_positions
    normalized_df["norm_ref"] = norm_refs
    normalized_df["norm_alt"] = norm_alts

    normalized_df.to_csv(normalized_cache, index=False)
    print(f"[stage 4] normalized {len(normalized_df):,} variants -> {normalized_cache}")

    prenorm_vcf.unlink(missing_ok=True)
    postnorm_vcf.unlink(missing_ok=True)

    return normalized_df


# --------------------------------------------------------------------------
# Stage 5: batched gnomAD query for this chromosome (sequential -T targets)
# --------------------------------------------------------------------------

def write_targets_file(positions, norm_chrom, work_dir):
    """
    Write a temporary bcftools TARGETS file (tab-separated CHROM\\tSTART\\tEND,
    1-based, inclusive), sorted ascending by position. Used with
    `bcftools view -T`, which streams the VCF sequentially and matches it
    against the sorted targets list -- this is the validated strategy from
    the chr22 pilot. `-R` (which builds a random-access index seek per
    region) was found to be far too slow and must NOT be used here.
    """
    gnomad_chrom = to_gnomad_chrom(norm_chrom)
    targets_path = work_dir / f"05_targets_chr{norm_chrom}.tsv"
    with open(targets_path, "w", encoding="utf-8") as f:
        for pos in sorted({int(p) for p in positions}):
            f.write(f"{gnomad_chrom}\t{pos}\t{pos}\n")
    return targets_path


def parse_info_field(info_string, field_names):
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


def expand_multiallelic_record(vcf_pos, vcf_ref, vcf_alt_field, info_field):
    """
    Safely expand a (possibly multiallelic) gnomAD VCF record into one row
    per ALT allele.

    Number=A INFO fields (AC_joint_*, AF_joint_*) are comma-separated with
    one value per ALT allele and are indexed to their corresponding ALT.
    Number=1 fields (AN_joint_*) carry a single value shared across every
    ALT allele at the site.

    Raises RuntimeError if a per-allele field's split length does not
    match the number of ALT alleles -- this must fail loudly rather than
    silently mis-assign values across alleles.
    """
    alt_alleles = vcf_alt_field.split(",")
    n_alts = len(alt_alleles)

    info_values = parse_info_field(info_field, set(INFO_FIELDS))

    per_allele_parsed = {}
    for field in PER_ALLELE_INFO_FIELDS:
        raw = info_values.get(field, "")
        if raw == "":
            per_allele_parsed[field] = [""] * n_alts
            continue
        parts = raw.split(",")
        if len(parts) != n_alts:
            raise RuntimeError(
                f"Allele-specific field '{field}' at {vcf_pos} has "
                f"{len(parts)} value(s) but record has {n_alts} ALT "
                f"allele(s) ({vcf_alt_field}); refusing to guess an "
                f"assignment."
            )
        per_allele_parsed[field] = parts

    shared_parsed = {
        field: info_values.get(field, "") for field in SHARED_INFO_FIELDS
    }

    rows = []
    for i, alt_allele in enumerate(alt_alleles):
        row = {"position": vcf_pos, "ref": vcf_ref, "alt": alt_allele}
        for field in PER_ALLELE_INFO_FIELDS:
            row[field] = per_allele_parsed[field][i]
        for field in SHARED_INFO_FIELDS:
            row[field] = shared_parsed[field]
        rows.append(row)

    return rows


def query_gnomad_chromosome(gnomad_vcf, positions, norm_chrom, work_dir, force, bcftools_timeout=None):
    lookup_cache = work_dir / f"05_gnomad_lookup_chr{norm_chrom}.csv"

    if stage_cached(lookup_cache, force):
        print(f"[stage 5] using cached gnomAD lookup for chr{norm_chrom}")
        lookup_df = pd.read_csv(lookup_cache, dtype=str, keep_default_na=False)
    else:
        if not positions:
            lookup_df = pd.DataFrame(columns=["position", "ref", "alt"] + INFO_FIELDS)
            lookup_df.to_csv(lookup_cache, index=False)
            return lookup_df

        targets_path = write_targets_file(positions, norm_chrom, work_dir)
        raw_output_path = work_dir / f"05_raw_gnomad_output_chr{norm_chrom}.tsv"
        try:
            # Validated sequential-target strategy: -T streams the VCF once
            # and matches against the sorted targets list. Do NOT use -R
            # (random-access region seeks), which the chr22 pilot showed
            # to be far too slow. Stdout is streamed straight to a
            # temporary file rather than captured into Python memory.
            cmd = ["bcftools", "view", "-H", "-T", str(targets_path), str(gnomad_vcf)]
            run_bcftools_to_file(cmd, raw_output_path, timeout=bcftools_timeout)

            rows = []
            with open(raw_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 8:
                        continue
                    vcf_pos, vcf_ref, vcf_alt_field, info_field = (
                        fields[1],
                        fields[3],
                        fields[4],
                        fields[7],
                    )
                    rows.extend(
                        expand_multiallelic_record(vcf_pos, vcf_ref, vcf_alt_field, info_field)
                    )
        finally:
            targets_path.unlink(missing_ok=True)
            raw_output_path.unlink(missing_ok=True)

        lookup_df = pd.DataFrame(rows, columns=["position", "ref", "alt"] + INFO_FIELDS)
        lookup_df.to_csv(lookup_cache, index=False)

    print(f"[stage 5] gnomAD chr{norm_chrom} allele records retrieved: {len(lookup_df):,}")
    return lookup_df


# --------------------------------------------------------------------------
# Stage 6: exact match on CHROM+POS+REF+ALT
# --------------------------------------------------------------------------

def match_against_gnomad(normalized_df, gnomad_lookup_df):
    lookup = {
        (str(row["position"]), row["ref"], row["alt"]): {
            field: row[field] for field in INFO_FIELDS
        }
        for _, row in gnomad_lookup_df.iterrows()
    }

    matched_flags = []
    info_columns = {field: [] for field in INFO_FIELDS}

    for _, row in normalized_df.iterrows():
        key = (str(row["norm_position"]), row["norm_ref"], row["norm_alt"])
        if key in lookup:
            matched_flags.append(True)
            for field in INFO_FIELDS:
                info_columns[field].append(lookup[key][field])
        else:
            matched_flags.append(False)
            for field in INFO_FIELDS:
                # Never impute AF=0 (or AC/AN) for unmatched variants.
                info_columns[field].append("")

    result = normalized_df.copy()
    result["gnomad_matched"] = matched_flags
    for field in INFO_FIELDS:
        result[field] = info_columns[field]

    return result


# --------------------------------------------------------------------------
# Stage 7: comparison categories + raw Fisher exact test (no FDR/labels here)
# --------------------------------------------------------------------------

def to_optional_int(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def classify_comparison_category(row):
    if not row["gnomad_matched"]:
        return "unmatched"

    an_sas = to_optional_int(row["AN_joint_sas"])
    an_nfe = to_optional_int(row["AN_joint_nfe"])

    if an_sas is None or an_nfe is None or an_sas == 0 or an_nfe == 0:
        return "insufficient_data"

    ac_sas = to_optional_int(row["AC_joint_sas"]) or 0
    ac_nfe = to_optional_int(row["AC_joint_nfe"]) or 0

    if ac_sas == 0 and ac_nfe == 0:
        return "observed_zero_both"

    return "valid_comparison"


def run_fisher_tests(df):
    """
    Run two-sided Fisher exact tests for rows with comparison_category ==
    'valid_comparison'. Returns RAW pvalue and odds_ratio Series aligned
    to df (NaN for non-eligible rows). No multiple-testing correction is
    applied here -- BH-FDR is deferred to a genome-wide step performed
    after merging all chromosomes' outputs.
    """
    pvalues = pd.Series(np.nan, index=df.index, dtype=float)
    odds_ratios = pd.Series(np.nan, index=df.index, dtype=float)

    eligible_mask = df["comparison_category"] == "valid_comparison"

    for idx in df.index[eligible_mask]:
        ac_sas = to_optional_int(df.at[idx, "AC_joint_sas"]) or 0
        an_sas = to_optional_int(df.at[idx, "AN_joint_sas"])
        ac_nfe = to_optional_int(df.at[idx, "AC_joint_nfe"]) or 0
        an_nfe = to_optional_int(df.at[idx, "AN_joint_nfe"])

        table = [
            [ac_sas, an_sas - ac_sas],
            [ac_nfe, an_nfe - ac_nfe],
        ]

        odds_ratio, pvalue = fisher_exact(table, alternative="two-sided")

        pvalues.at[idx] = pvalue
        odds_ratios.at[idx] = odds_ratio

    return pvalues, odds_ratios


def compute_statistics(df):
    """
    Assign comparison_category and, for valid_comparison rows only, a raw
    two-sided Fisher p-value and odds ratio. Does NOT compute qvalue or
    assign SAS_enriched/NFE_enriched/no_clear_enrichment labels -- those
    require genome-wide BH-FDR correction across all chromosomes and are
    computed by a separate downstream step.
    """
    df = df.copy()
    df["comparison_category"] = df.apply(classify_comparison_category, axis=1)

    pvalues, odds_ratios = run_fisher_tests(df)
    df["pvalue"] = pvalues
    df["odds_ratio"] = odds_ratios

    return df


# --------------------------------------------------------------------------
# Stage 8: merge T2 outcome labels
# --------------------------------------------------------------------------

def merge_t2_labels(df, t2_labels_csv):
    """
    Merge T2 outcome labels by vcv_accession.

    Clean primary endpoint = resolved_benign, resolved_pathogenic, or
    still_vus. "other" and missing/unmatched T2 rows are excluded from the
    clean endpoint (but retained in the output, not dropped).

    is_resolved       = True only for resolved_benign / resolved_pathogenic.
    is_clean_primary_endpoint = True for resolved_benign / resolved_pathogenic
                                  / still_vus.
    """
    if not t2_labels_csv.exists():
        raise FileNotFoundError(f"T2 labels CSV not found: {t2_labels_csv}")

    t2_df = pd.read_csv(t2_labels_csv, dtype=str, keep_default_na=False)
    validate_required_columns(t2_df, REQUIRED_T2_COLUMNS, "T2 labels CSV")

    t2_subset = t2_df[list(REQUIRED_T2_COLUMNS)].drop_duplicates(
        subset="vcv_accession", keep="first"
    )

    merged = df.merge(t2_subset, on="vcv_accession", how="left")
    for col in REQUIRED_T2_COLUMNS - {"vcv_accession"}:
        merged[col] = merged[col].fillna("")

    merged["is_resolved"] = merged["t2_outcome"].isin(RESOLVED_OUTCOMES)
    merged["is_clean_primary_endpoint"] = merged["t2_outcome"].isin(
        CLEAN_PRIMARY_ENDPOINT_OUTCOMES
    )

    return merged


# --------------------------------------------------------------------------
# QC reporting
# --------------------------------------------------------------------------

def print_qc_summary(
    norm_chrom,
    total_chrom_variants,
    missing_fields_count,
    usable_count,
    ref_eq_alt_excluded,
    n_excluded,
    qc_df,
):
    matched = int((qc_df["gnomad_matched"] == True).sum())  # noqa: E712
    unmatched = int((qc_df["gnomad_matched"] == False).sum())  # noqa: E712

    category_counts = qc_df["comparison_category"].value_counts().to_dict()
    valid_comparison_count = int((qc_df["comparison_category"] == "valid_comparison").sum())

    resolved_count = int(qc_df["is_resolved"].sum())
    clean_endpoint_count = int(qc_df["is_clean_primary_endpoint"].sum())
    still_vus_count = int((qc_df["t2_outcome"] == "still_vus").sum())
    other_t2_count = int((qc_df["t2_outcome"] == "other").sum())
    unmatched_t2_count = int((qc_df["t2_outcome"] == "").sum())

    print(f"\n=== run_chromosome_pipeline.py QC summary (chr{norm_chrom}) ===")
    print(f"Total GRCh38 T1 VUS on chromosome:      {total_chrom_variants:,}")
    print(f"  missing position/ref/alt:             {missing_fields_count:,}")
    print(f"  usable (has position/ref/alt):        {usable_count:,}")
    print(f"REF==ALT excluded:                      {ref_eq_alt_excluded:,}")
    print(f"Reference-N excluded:                   {n_excluded:,}")
    print(f"Normalized & attempted gnomAD match:     {len(qc_df):,}")
    print(f"  gnomAD matched:                        {matched:,}")
    print(f"  gnomAD unmatched:                       {unmatched:,}")
    print("Comparison categories:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count:,}")
    print(f"Raw Fisher tests computed (valid_comparison): {valid_comparison_count:,}")
    print("  (qvalue / SAS_enriched / NFE_enriched are assigned in the")
    print("   separate genome-wide merge + BH-FDR step, not here.)")
    print("T2 outcome linkage:")
    print(f"  resolved (benign/pathogenic):                 {resolved_count:,}")
    print(f"  still_vus:                                     {still_vus_count:,}")
    print(f"  clean primary endpoint (resolved + still_vus): {clean_endpoint_count:,}")
    print(f"  other T2 outcome (excluded from clean endpoint): {other_t2_count:,}")
    print(f"  no T2 match:                                    {unmatched_t2_count:,}")


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------

def main():
    start_time = time.time()
    args = parse_args()

    norm_chrom = normalize_chromosome(args.chromosome)
    if norm_chrom not in SUPPORTED_CHROMOSOMES:
        raise ValueError(
            f"Unsupported chromosome '{args.chromosome}' "
            f"(normalized: '{norm_chrom}'); supported: 1-22, X, Y."
        )

    if not args.gnomad_vcf.exists():
        raise FileNotFoundError(f"gnomAD VCF not found: {args.gnomad_vcf}")
    if not args.reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {args.reference_fasta}")

    work_dir = ensure_dir(args.work_dir / f"chr{norm_chrom}")
    output_dir = ensure_dir(args.output_dir)

    bcftools_timeout = args.bcftools_timeout  # None by default: no timeout

    # Stage 1: filter T1 VUS to this chromosome / GRCh38, with explicit
    # total -> missing -> usable accounting.
    t1_usable, total_chrom_variants, missing_fields_count = load_t1_variants(
        args.t1_vus_csv, norm_chrom, work_dir, args.force
    )
    usable_count = len(t1_usable)

    if usable_count == 0:
        print(f"[pipeline] no usable variants for chr{norm_chrom}; nothing to do.")
        return

    # Stage 2: flag REF==ALT.
    safe_after_refalt, ref_eq_alt_excluded_df = flag_ref_equals_alt(
        t1_usable, work_dir, norm_chrom, args.force
    )

    # Stage 3: flag reference-N overlap.
    clean, n_excluded_df = check_reference_n_overlap(
        safe_after_refalt, args.reference_fasta, work_dir, norm_chrom, args.force
    )

    # Stage 4: normalize the clean/safe variants (no -c w: fail loudly on
    # any remaining REF mismatch).
    normalized = normalize_with_bcftools(
        clean, args.reference_fasta, work_dir, norm_chrom, args.force,
        bcftools_timeout=bcftools_timeout,
    )

    # Stage 5: batched gnomAD query (sequential -T targets, streamed to a
    # temp file) for the unique normalized positions.
    unique_positions = normalized["norm_position"].tolist() if not normalized.empty else []
    gnomad_lookup = query_gnomad_chromosome(
        args.gnomad_vcf, unique_positions, norm_chrom, work_dir, args.force,
        bcftools_timeout=bcftools_timeout,
    )

    # Stage 6: exact CHROM+POS+REF+ALT matching.
    matched = match_against_gnomad(normalized, gnomad_lookup)

    # Stage 7: comparison categories + raw Fisher p-value/odds ratio only
    # (no BH-FDR, no enrichment labels -- those are genome-wide steps).
    with_stats = compute_statistics(matched)

    # Combine excluded rows back in with explicit comparison_category so
    # nothing is silently dropped from the final CSV.
    ref_eq_alt_excluded_df = ref_eq_alt_excluded_df.copy()
    ref_eq_alt_excluded_df["comparison_category"] = "ref_equals_alt_excluded"

    n_excluded_df = n_excluded_df.copy()
    n_excluded_df["comparison_category"] = "reference_N_excluded"

    for extra_df in (ref_eq_alt_excluded_df, n_excluded_df):
        extra_df["gnomad_matched"] = False
        for field in INFO_FIELDS:
            extra_df[field] = ""
        for col in ["norm_position", "norm_ref", "norm_alt", "pvalue", "odds_ratio"]:
            extra_df[col] = ""

    combined = pd.concat(
        [with_stats, ref_eq_alt_excluded_df, n_excluded_df],
        ignore_index=True,
        sort=False,
    )

    # Stage 8: merge T2 outcome labels.
    final_df = merge_t2_labels(combined, args.t2_labels_csv)

    # QC summary uses the normalized/matched subset (excluding the
    # REF==ALT / reference-N exclusions, whose counts are reported
    # separately) but WITH the merged T2 columns, so is_resolved /
    # is_clean_primary_endpoint are available.
    qc_subset = final_df[
        final_df["comparison_category"].isin(
            ["unmatched", "insufficient_data", "observed_zero_both", "valid_comparison"]
        )
    ]
    print_qc_summary(
        norm_chrom,
        total_chrom_variants,
        missing_fields_count,
        usable_count,
        len(ref_eq_alt_excluded_df),
        len(n_excluded_df),
        qc_subset,
    )

    output_path = output_dir / f"chr{norm_chrom}_gnomad_annotated.csv"
    final_df.to_csv(output_path, index=False)

    elapsed_seconds = time.time() - start_time
    print(f"\n[pipeline] wrote {len(final_df):,} rows to {output_path}")
    print(
        "[pipeline] NOTE: 'pvalue'/'odds_ratio' are raw, per-chromosome "
        "values. Run the genome-wide merge + BH-FDR step across all "
        "chromosome outputs to obtain 'qvalue' and final "
        "SAS_enriched/NFE_enriched labels."
    )
    print(f"[pipeline] elapsed time (seconds): {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()