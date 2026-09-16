#!/usr/bin/env python3
"""
src/clinvar/parse_t1_vus.py

Stream-parse a ClinVar VCV XML release (.xml.gz) and extract records whose
aggregate germline classification is "Uncertain significance" (VUS) into a
flat CSV for downstream T1 cohort construction.

Repo-relative paths:
  input:   data/raw/clinvar/t1_2024-02/ClinVarVCVRelease_2024-02.xml.gz
  output:  data/interim/t1_vus.csv
  summary: results/logs/parse_t1_vus_summary.json

Notes:
  - Uses lxml.etree.iterparse (huge_tree=True) instead of the stdlib
    xml.etree.ElementTree, which was running out of memory on this release:
    lxml's C-backed parser combined with aggressive element/sibling clearing
    keeps memory bounded on multi-GB files.
  - Reads the .xml.gz directly via gzip.open(); the parser never sees more
    than the current record's subtree at once.
  - extract_location() only inspects the allele-level SequenceLocation
    elements under ClassifiedRecord/SimpleAllele/Location -- it does NOT
    search all descendants, which previously risked matching an unrelated
    gene-level SequenceLocation nested under GeneList. It prefers a GRCh38
    SequenceLocation but will fall back to another allele-level assembly if
    GRCh38 is absent; the assembly actually used is recorded explicitly in
    the "assembly" output column. VCF-specific position/REF/ALT fields are
    preferred, with a cautious fallback to start/referenceAllele/
    alternateAllele only when the VCF-specific fields are absent.
  - Does NOT process T2, gnomAD, literature, or RL logic.
"""

import csv
import gzip
import json
import time
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "raw" / "clinvar" / "t1_2024-02" / "ClinVarVCVRelease_2024-02.xml.gz"
OUTPUT_CSV = REPO_ROOT / "data" / "interim" / "t1_vus.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "logs" / "parse_t1_vus_summary.json"

FIELDNAMES = [
    "vcv_accession",
    "variation_id",
    "preferred_name",
    "clinical_significance",
    "review_status",
    "date_last_evaluated",
    "gene_symbols",
    "chromosome",
    "position",
    "ref",
    "alt",
    "assembly",
    "hgvs_genomic",
    "hgvs_coding",
    "hgvs_protein",
]

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


def find_direct_child(elem, tag_name):
    """Return the first DIRECT child of elem with this local tag name, or None."""
    if elem is None:
        return None
    for child in elem:
        if local_tag(child.tag) == tag_name:
            return child
    return None


def get_text(elem):
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


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


def extract_genes(classified_record):
    gene_list = find_first(classified_record, "GeneList")
    if gene_list is None:
        return ""
    symbols = []
    for gene in gene_list:
        if local_tag(gene.tag) == "Gene":
            symbol = gene.attrib.get("Symbol", "")
            if symbol:
                symbols.append(symbol)
    return ";".join(symbols)


def extract_location(classified_record):
    """
    Extract allele-level genomic coordinates.

    Only the primary SimpleAllele directly under ClassifiedRecord is
    considered, and only the SequenceLocation children of that allele's own
    direct Location element are inspected. This deliberately avoids the
    broader descendant search used previously, which could incorrectly pick
    up an unrelated SequenceLocation nested under GeneList.

    Prefers a GRCh38 SequenceLocation; falls back to another allele-level
    assembly if GRCh38 is absent, recording that assembly explicitly.

    VCF-specific fields (positionVCF, referenceAlleleVCF,
    alternateAlleleVCF) are preferred; if a given field is absent, we fall
    back cautiously to the corresponding non-VCF field (start,
    referenceAllele, alternateAllele) only for that missing field.

    Returns (chromosome, position, ref, alt, assembly).
    """
    simple_allele = find_direct_child(classified_record, "SimpleAllele")
    if simple_allele is None:
        return "", "", "", "", ""

    location = find_direct_child(simple_allele, "Location")
    if location is None:
        return "", "", "", "", ""

    seq_locations = [
        child for child in location if local_tag(child.tag) == "SequenceLocation"
    ]
    if not seq_locations:
        return "", "", "", "", ""

    chosen = None
    for loc in seq_locations:
        if loc.attrib.get("Assembly") == "GRCh38":
            chosen = loc
            break
    if chosen is None:
        chosen = seq_locations[0]

    chrom = chosen.attrib.get("Chr", "")
    assembly = chosen.attrib.get("Assembly", "")

    position = chosen.attrib.get("positionVCF", "")
    if not position:
        position = chosen.attrib.get("start", "")

    ref = chosen.attrib.get("referenceAlleleVCF", "")
    if not ref:
        ref = chosen.attrib.get("referenceAllele", "")

    alt = chosen.attrib.get("alternateAlleleVCF", "")
    if not alt:
        alt = chosen.attrib.get("alternateAllele", "")

    return chrom, position, ref, alt, assembly


def extract_hgvs(classified_record):
    """Return (genomic, coding, protein) HGVS expressions."""
    hgvs_list = find_first(classified_record, "HGVSlist")
    if hgvs_list is None:
        return "", "", ""

    genomic = ""
    coding = ""
    protein = ""

    for hgvs in hgvs_list:
        if local_tag(hgvs.tag) != "HGVS":
            continue
        hgvs_type = hgvs.attrib.get("Type", "")

        nuc_expr = ""
        prot_expr = ""
        for child in hgvs:
            child_tag = local_tag(child.tag)
            if child_tag == "NucleotideExpression":
                nuc_expr = get_text(find_first(child, "Expression"))
            elif child_tag == "ProteinExpression":
                prot_expr = get_text(find_first(child, "Expression"))

        if hgvs_type == "genomic" and not genomic and nuc_expr:
            genomic = nuc_expr
        elif hgvs_type == "coding" and not coding and nuc_expr:
            coding = nuc_expr
            if prot_expr and not protein:
                protein = prot_expr
        elif not protein and prot_expr:
            protein = prot_expr

    return genomic, coding, protein


def parse_variation_archive(archive):
    """Return a row dict for a VUS VariationArchive, or None if not VUS / unusable."""
    accession = archive.attrib.get("Accession", "")
    variation_id = archive.attrib.get("VariationID", "")
    preferred_name = archive.attrib.get("VariationName", "")

    classified_record = None
    for child in archive:
        if local_tag(child.tag) == "ClassifiedRecord":
            classified_record = child
            break
    if classified_record is None:
        return None

    description, review_status, date_last_evaluated = extract_germline_classification(
        classified_record
    )
    if description.strip().lower() not in VUS_STRINGS:
        return None

    gene_symbols = extract_genes(classified_record)
    chrom, position, ref, alt, assembly = extract_location(classified_record)
    hgvs_genomic, hgvs_coding, hgvs_protein = extract_hgvs(classified_record)

    return {
        "vcv_accession": accession,
        "variation_id": variation_id,
        "preferred_name": preferred_name,
        "clinical_significance": description,
        "review_status": review_status,
        "date_last_evaluated": date_last_evaluated,
        "gene_symbols": gene_symbols,
        "chromosome": chrom,
        "position": position,
        "ref": ref,
        "alt": alt,
        "assembly": assembly,
        "hgvs_genomic": hgvs_genomic,
        "hgvs_coding": hgvs_coding,
        "hgvs_protein": hgvs_protein,
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

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    total_records = 0
    vus_records = 0
    rows_written = 0
    rows_with_coords = 0
    rows_with_genes = 0
    rows_with_position = 0
    rows_with_ref = 0
    rows_with_alt = 0
    rows_with_complete_pos_ref_alt = 0
    grch38_count = 0
    grch37_count = 0
    other_missing_assembly_count = 0
    seen_accessions = set()

    with gzip.open(INPUT_PATH, "rb") as xml_file, open(
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
            total_records += 1

            row = parse_variation_archive(elem)

            if row is not None:
                vus_records += 1
                accession = row["vcv_accession"]

                if accession and accession not in seen_accessions:
                    seen_accessions.add(accession)
                    writer.writerow(row)
                    rows_written += 1

                    has_chrom = bool(row["chromosome"])
                    has_position = bool(row["position"])
                    has_ref = bool(row["ref"])
                    has_alt = bool(row["alt"])

                    if has_chrom and has_position:
                        rows_with_coords += 1
                    if row["gene_symbols"]:
                        rows_with_genes += 1
                    if has_position:
                        rows_with_position += 1
                    if has_ref:
                        rows_with_ref += 1
                    if has_alt:
                        rows_with_alt += 1
                    if has_position and has_ref and has_alt:
                        rows_with_complete_pos_ref_alt += 1

                    assembly = row["assembly"]
                    if assembly == "GRCh38":
                        grch38_count += 1
                    elif assembly == "GRCh37":
                        grch37_count += 1
                    else:
                        other_missing_assembly_count += 1

            # Aggressively free memory: clear this element's subtree and
            # drop its already-processed preceding siblings from the parent.
            free_element(elem)

            if total_records % PROGRESS_EVERY == 0:
                print(f"...processed {total_records:,} VariationArchive records", flush=True)

        # Release the root/context tree itself once parsing is complete.
        del context

    elapsed_seconds = time.time() - start_time

    summary = {
        "input_path": str(INPUT_PATH.relative_to(REPO_ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        "total_records": total_records,
        "vus_records": vus_records,
        "rows_written": rows_written,
        "rows_with_coordinates": rows_with_coords,
        "rows_with_genes": rows_with_genes,
        "rows_with_position": rows_with_position,
        "rows_with_ref": rows_with_ref,
        "rows_with_alt": rows_with_alt,
        "rows_with_complete_pos_ref_alt": rows_with_complete_pos_ref_alt,
        "grch38_count": grch38_count,
        "grch37_count": grch37_count,
        "other_missing_assembly_count": other_missing_assembly_count,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== parse_t1_vus.py summary ===")
    print(f"Total VariationArchive records seen: {total_records:,}")
    print(f"Records classified as VUS:           {vus_records:,}")
    print(f"Rows written (deduplicated):          {rows_written:,}")
    print(f"Rows with coordinates:                {rows_with_coords:,}")
    print(f"Rows with gene symbol(s):             {rows_with_genes:,}")
    print(f"Rows with position:                   {rows_with_position:,}")
    print(f"Rows with REF:                         {rows_with_ref:,}")
    print(f"Rows with ALT:                         {rows_with_alt:,}")
    print(f"Rows with complete position+REF+ALT:  {rows_with_complete_pos_ref_alt:,}")
    print(f"GRCh38 assembly:                      {grch38_count:,}")
    print(f"GRCh37 assembly:                      {grch37_count:,}")
    print(f"Other/missing assembly:               {other_missing_assembly_count:,}")
    print(f"Elapsed time (seconds):               {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()