#!/usr/bin/env python3
"""
src/clinvar/parse_t2_labels.py

Load the T1 VUS accession set produced by parse_t1_vus.py, then stream a
later ClinVar VCV XML release (T2) and extract the later germline
classification for every variant that was VUS at T1. Classifies each
matched variant's T2 outcome as resolved_benign / resolved_pathogenic /
still_vus / other.

Repo-relative paths:
  input (T1 VUS list): data/interim/t1_vus.csv
  input (T2 release):  data/raw/clinvar/t2_2025-07/ClinVarVCVRelease_2025-07.xml.gz
  output:               data/interim/t2_labels_for_t1_vus.csv
  summary:              results/logs/parse_t2_labels_summary.json

Notes:
  - Uses lxml.etree.iterparse with huge_tree=True and the same aggressive
    element/sibling-clearing pattern as parse_t1_vus.py to keep memory
    bounded on multi-GB releases.
  - A requested T1 accession found in T2 is always counted as matched, even
    if it lacks a ClassifiedRecord or germline classification block; in
    that case classification fields are left blank and t2_outcome="other".
  - Does NOT process gnomAD, literature, or RL logic.
"""

import csv
import gzip
import json
import time
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[2]

T1_VUS_CSV = REPO_ROOT / "data" / "interim" / "t1_vus.csv"
T2_INPUT_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "clinvar"
    / "t2_2025-07"
    / "ClinVarVCVRelease_2025-07.xml.gz"
)
OUTPUT_CSV = REPO_ROOT / "data" / "interim" / "t2_labels_for_t1_vus.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "logs" / "parse_t2_labels_summary.json"

FIELDNAMES = [
    "vcv_accession",
    "t2_clinical_significance",
    "t2_review_status",
    "t2_date_last_evaluated",
    "t2_outcome",
]

BENIGN_STRINGS = {"benign", "likely benign", "benign/likely benign"}
PATHOGENIC_STRINGS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
VUS_STRINGS = {"uncertain significance"}

PROGRESS_EVERY = 50_000


def local_tag(tag):
    """Strip an XML namespace (if present) from a tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_first(elem, tag_name):
    """Depth-first search for the first descendant with this local tag name."""
    for child in elem.iter():
        if local_tag(child.tag) == tag_name:
            return child
    return None


def get_text(elem):
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def load_t1_vus_accessions(path):
    accessions = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            accession = (row.get("vcv_accession") or "").strip()
            if accession:
                accessions.add(accession)
    return accessions


def extract_germline_classification(classified_record):
    """Return (description, review_status, date_last_evaluated)."""
    classifications = find_first(classified_record, "Classifications")
    if classifications is None:
        return "", "", ""

    germline = None
    for child in classifications:
        if local_tag(child.tag) == "GermlineClassification":
            germline = child
            break
    if germline is None:
        return "", "", ""

    description = ""
    review_status = ""
    date_last_evaluated = germline.attrib.get("DateLastEvaluated", "")

    for child in germline:
        tag = local_tag(child.tag)
        if tag == "Description":
            description = get_text(child)
        elif tag == "ReviewStatus":
            review_status = get_text(child)
        elif tag == "DateLastEvaluated" and not date_last_evaluated:
            date_last_evaluated = get_text(child)

    return description, review_status, date_last_evaluated


def classify_outcome(description):
    normalized = description.strip().lower()
    if normalized in BENIGN_STRINGS:
        return "resolved_benign"
    if normalized in PATHOGENIC_STRINGS:
        return "resolved_pathogenic"
    if normalized in VUS_STRINGS:
        return "still_vus"
    return "other"


def parse_variation_archive(archive, t1_accessions):
    """
    Return a row dict if this archive's accession is in the T1 VUS set, else
    None. A matching accession always yields a row (matched), even when a
    ClassifiedRecord or germline classification block is absent; in that
    case classification fields are blank and t2_outcome is "other".
    """
    accession = archive.attrib.get("Accession", "")
    if not accession or accession not in t1_accessions:
        return None

    classified_record = None
    for child in archive:
        if local_tag(child.tag) == "ClassifiedRecord":
            classified_record = child
            break

    description, review_status, date_last_evaluated = "", "", ""
    if classified_record is not None:
        description, review_status, date_last_evaluated = extract_germline_classification(
            classified_record
        )

    outcome = classify_outcome(description)

    return {
        "vcv_accession": accession,
        "t2_clinical_significance": description,
        "t2_review_status": review_status,
        "t2_date_last_evaluated": date_last_evaluated,
        "t2_outcome": outcome,
    }


def free_element(elem):
    """Aggressively free a processed XML element and its preceding siblings."""
    elem.clear(keep_tail=False)

    while elem.getprevious() is not None:
        del elem.getparent()[0]


def main():
    start_time = time.time()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    if not T1_VUS_CSV.exists():
        raise FileNotFoundError(f"T1 VUS CSV not found: {T1_VUS_CSV}")
    if not T2_INPUT_PATH.exists():
        raise FileNotFoundError(f"T2 input file not found: {T2_INPUT_PATH}")

    t1_accessions = load_t1_vus_accessions(T1_VUS_CSV)
    t1_vus_requested = len(t1_accessions)
    matched_accessions = set()

    total_t2_records = 0
    matched = 0
    outcome_counts = {
        "resolved_benign": 0,
        "resolved_pathogenic": 0,
        "still_vus": 0,
        "other": 0,
    }

    with gzip.open(T2_INPUT_PATH, "rb") as xml_file, open(
        OUTPUT_CSV, "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        context = etree.iterparse(
            xml_file,
            events=("end",),
            tag="VariationArchive",
            huge_tree=True,
            remove_blank_text=False,
        )

        for _event, elem in context:
            total_t2_records += 1

            row = parse_variation_archive(elem, t1_accessions)

            if row is not None:
                matched += 1
                matched_accessions.add(row["vcv_accession"])
                outcome_counts[row["t2_outcome"]] += 1
                writer.writerow(row)

            # Aggressively free memory: clear this element's subtree and
            # drop its already-processed preceding siblings from the parent.
            free_element(elem)

            if total_t2_records % PROGRESS_EVERY == 0:
                print(f"...scanned {total_t2_records:,} T2 VariationArchive records", flush=True)

        # Release the root/context tree itself once parsing is complete.
        del context

    unmatched = t1_vus_requested - len(matched_accessions)
    elapsed_seconds = time.time() - start_time

    summary = {
        "t1_vus_csv": str(T1_VUS_CSV.relative_to(REPO_ROOT)),
        "t2_input_path": str(T2_INPUT_PATH.relative_to(REPO_ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        "total_t2_records_scanned": total_t2_records,
        "t1_vus_requested": t1_vus_requested,
        "matched": matched,
        "outcome_counts": outcome_counts,
        "unmatched": unmatched,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== parse_t2_labels.py summary ===")
    print(f"Total T2 records scanned:      {total_t2_records:,}")
    print(f"T1 VUS accessions requested:   {t1_vus_requested:,}")
    print(f"Matched in T2:                 {matched:,}")
    print(f"  resolved_benign:             {outcome_counts['resolved_benign']:,}")
    print(f"  resolved_pathogenic:         {outcome_counts['resolved_pathogenic']:,}")
    print(f"  still_vus:                   {outcome_counts['still_vus']:,}")
    print(f"  other:                       {outcome_counts['other']:,}")
    print(f"Unmatched (T1 VUS not in T2):  {unmatched:,}")
    print(f"Elapsed time (seconds):        {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()


# === parse_t2_labels.py summary ===
# Total T2 records scanned:      3,595,245
# T1 VUS accessions requested:   1,168,064
# Matched in T2:                 1,167,356
#   resolved_benign:             12,094
#   resolved_pathogenic:         3,439
#   still_vus:                   1,124,772
#   other:                       27,051
# Unmatched (T1 VUS not in T2):  708
# Elapsed time (seconds):        810.27    