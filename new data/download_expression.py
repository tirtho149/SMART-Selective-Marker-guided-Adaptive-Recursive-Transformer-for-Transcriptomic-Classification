#!/usr/bin/env python3
"""
Download TCGA gene-level data (expression, CNV, mutation) from UCSC Xena.

Outputs (per dataset directory under data_tcga/):
    expression_data.csv  — log2(RSEM+1) RNA-seq (HiSeqV2)
    cnv_data.csv         — Gistic2 continuous gene-level copy number
    mutation_data.csv    — binary gene-level mutation calls

Format: patients x genes, aligned to data/<name>/mutation_data.csv (the reference).

Usage:
    cd /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga
    python download_expression.py                         # all datasets, all types
    python download_expression.py --datasets brca         # one dataset, all types
    python download_expression.py --types expression cnv  # all datasets, subset of types
    python download_expression.py --datasets brca --types mutation --force
"""

import os
import gzip
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

# UCSC Xena hubs
XENA_HUB         = "https://tcga.xenahubs.net/download"          # TCGA per-cancer (legacy)
PANCANATLAS_HUB  = "https://pancanatlas.xenahubs.net/download"   # TCGA PanCanAtlas (MC3 mutations, etc.)

# Reference data directory (read-only): provides mutation_data.csv for patient/gene alignment
DATA_DIR = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data"

# Output root: where expression_data.csv is written
OUT_DIR = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga"

# Cache for raw Xena downloads (reused across runs)
CACHE_DIR = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/xena_cache"

# Per-type Xena dataset IDs and output filenames.
# {code} is replaced with the TCGA cancer code (e.g. BRCA).
# xena_ids is a list of candidates tried in order — first one that downloads wins
# (different mutation callers are available for different cancer cohorts).
DATA_TYPE_CONFIG = {
    "expression": {
        "hub":      XENA_HUB,
        "xena_ids": ["TCGA.{code}.sampleMap/HiSeqV2"],
        "out_file": "expression_data.csv",
        "fill":     0.0,
        "as_int":   False,
    },
    "cnv": {
        "hub":      XENA_HUB,
        "xena_ids": ["TCGA.{code}.sampleMap/Gistic2_CopyNumber_Gistic2_all_data_by_genes"],
        "out_file": "cnv_data.csv",
        "fill":     0.0,
        "as_int":   False,
    },
    # MC3 is a pan-TCGA gene-level binary mutation matrix (single file, all cancers).
    # No per-cancer files on the legacy TCGA hub are reachable anymore (S3 403s).
    "mutation": {
        "hub":      PANCANATLAS_HUB,
        "xena_ids": ["mc3.v0.2.8.PUBLIC.nonsilentGene.xena"],
        "out_file": "mutation_data.csv",
        "fill":     0,
        "as_int":   True,
    },
    # Phenotype = full clinical matrix (samples × clinical fields, not genes).
    # Raw file is samples × fields, so we skip the .T transpose and gene reindex.
    "phenotype": {
        "hub":         XENA_HUB,
        "xena_ids":    ["TCGA.{code}.sampleMap/{code}_clinicalMatrix"],
        "out_file":    "phenotype.csv",
        "samples_as_rows": True,   # raw matrix already has samples on rows
        "align_genes": False,      # keep all original columns, no gene reindex
    },
    # Curated survival from Liu et al. 2018, pan-TCGA single file.
    "survival": {
        "hub":         PANCANATLAS_HUB,
        "xena_ids":    ["Survival_SupplementalTable_S1_20171025_xena_sp"],
        "out_file":    "survival.csv",
        "samples_as_rows": True,
        "align_genes": False,
    },
}

# Mapping: local data directory → TCGA cancer type code(s)
# For pancancer directories, multiple codes are listed and combined.
CANCER_MAP = {
    "kirc_pan":          ["KIRC"],
    "brca":              ["BRCA"],
    "brca_tcga":         ["BRCA"],
    "blca":              ["BLCA"],
    "gbm":               ["GBM"],
    "stad":              ["STAD"],
    "ucec":              ["UCEC"],
    "lung_laud":         ["LUAD"],
    "lung_lusc":         ["LUSC"],
    "LGG":               ["LGG"],
    "prostate":          ["PRAD"],
    "thca":              ["THCA"],
    "pan":               ["BRCA", "KIRC", "BLCA", "GBM", "STAD", "UCEC",
                          "LUAD", "LUSC", "LGG", "PRAD", "THCA"],
    "pancancer":         ["BRCA", "KIRC", "BLCA", "GBM", "STAD", "UCEC",
                          "LUAD", "LUSC", "LGG", "PRAD", "THCA"],
    "brca_pan_stage":    ["BRCA"],
    "pancancer_stage":   ["BRCA", "KIRC", "BLCA", "GBM", "STAD", "UCEC",
                          "LUAD", "LUSC", "LGG", "PRAD", "THCA"],
}

# ─── Download ─────────────────────────────────────────────────────────────────

def download_xena(cancer_code: str, data_type: str, cache_dir: str) -> pd.DataFrame:
    """
    Download a gene-level Xena matrix (expression, cnv, or mutation) for one cancer.
    Tries each candidate dataset_id in DATA_TYPE_CONFIG[data_type]['xena_ids'] in order,
    returning the first that succeeds. Uses cached file if available.

    Returns DataFrame: genes (rows) × samples (columns).
    Raises RuntimeError if all candidates fail.
    """
    cfg = DATA_TYPE_CONFIG[data_type]
    hub = cfg["hub"]
    os.makedirs(cache_dir, exist_ok=True)

    last_err = None
    for template in cfg["xena_ids"]:
        dataset_id = template.format(code=cancer_code)
        leaf = dataset_id.rsplit("/", 1)[-1]
        # Per-cancer file → include cancer code in cache name; hub-wide file → don't
        per_cancer = "{code}" in template
        cache_file = os.path.join(
            cache_dir,
            f"{leaf}_{cancer_code}.tsv.gz" if per_cancer else f"{leaf}.tsv.gz",
        )

        if os.path.exists(cache_file):
            print(f"    Cache hit → {cache_file}")
        else:
            # Some Xena files are gzipped (.gz), some are plain TSV. Try both.
            downloaded = False
            for url in (f"{hub}/{dataset_id}.gz", f"{hub}/{dataset_id}"):
                print(f"    Trying {data_type}/{cancer_code}  ← {url}")
                try:
                    resp = requests.get(url, stream=True, timeout=600)
                    resp.raise_for_status()
                    with open(cache_file, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                    print(f"    Saved → {cache_file}")
                    downloaded = True
                    break
                except requests.HTTPError as e:
                    print(f"    [MISS] {url.rsplit('/', 1)[-1]}: {e.response.status_code}")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    last_err = e
            if not downloaded:
                continue

        print(f"    Reading {leaf}...")
        # Xena serves most files gzipped; some phenotype/survival files are plain TSV
        # but still served with .gz suffix. Try gzip first, fall back to plain.
        try:
            with gzip.open(cache_file, "rt") as fh:
                df = pd.read_csv(fh, sep="\t", index_col=0, low_memory=False)
        except (OSError, gzip.BadGzipFile):
            with open(cache_file, "rt") as fh:
                df = pd.read_csv(fh, sep="\t", index_col=0, low_memory=False)
        print(f"    Shape: {df.shape}")
        return df

    raise RuntimeError(f"All candidate dataset IDs failed for {data_type}/{cancer_code}: {last_err}")


# ─── Processing ───────────────────────────────────────────────────────────────

def normalize_sample_id(sid: str) -> str:
    """
    Normalize TCGA sample IDs for matching.
    Xena uses '-' separators and 15-char IDs like TCGA-A1-A0SB-01A.
    Our data uses IDs like TCGA-A1-A0SB-01 (no vial letter suffix).
    We truncate to the first 15 chars (4 fields) to align.
    """
    parts = sid.strip().split("-")
    if len(parts) >= 4:
        # Keep TCGA-XX-XXXX-NN (drop vial/portion/analyte suffixes)
        return "-".join(parts[:4])
    return sid.strip()


def process_directory(dir_name: str, cancer_codes: list, data_type: str, force: bool) -> str:
    """
    Download, combine, align, and save one data type for one dataset directory.
    Returns status string: 'saved', 'skipped', or 'failed'.
    """
    cfg       = DATA_TYPE_CONFIG[data_type]
    ref_dir   = os.path.join(DATA_DIR, dir_name)
    mut_path  = os.path.join(ref_dir, "mutation_data.csv")
    out_dir   = os.path.join(OUT_DIR, dir_name)
    out_path  = os.path.join(out_dir, cfg["out_file"])

    # Guard: reference directory and mutation file must exist
    if not os.path.isdir(ref_dir):
        print(f"  [SKIP] {dir_name}: reference directory not found ({ref_dir})")
        return "skipped"
    if not os.path.exists(mut_path):
        print(f"  [SKIP] {dir_name}: no mutation_data.csv in {ref_dir}")
        return "skipped"
    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] {dir_name}/{cfg['out_file']} already exists (use --force to overwrite)")
        return "skipped"

    os.makedirs(out_dir, exist_ok=True)

    # Load reference patient IDs and gene list from mutation data
    print(f"  Loading reference mutation_data.csv ...")
    mut_df    = pd.read_csv(mut_path, index_col=0)
    ref_ids   = mut_df.index.astype(str).tolist()   # ordered patient IDs
    ref_genes = mut_df.columns.tolist()             # ordered gene symbols
    print(f"  Reference: {len(ref_ids)} patients, {len(ref_genes)} genes")

    # Build a short-ID → original-ID lookup for matching
    short_to_ref = {normalize_sample_id(sid): sid for sid in ref_ids}

    samples_as_rows = cfg.get("samples_as_rows", False)
    align_genes     = cfg.get("align_genes", True)

    # Download and merge frames for all cancer codes
    frames = []
    for code in cancer_codes:
        try:
            raw = download_xena(code, data_type, CACHE_DIR)
            # Most matrices are genes × samples → transpose. Phenotype/survival
            # already have samples on rows, so skip the transpose.
            samples_df = raw.copy() if samples_as_rows else raw.T.copy()
            samples_df.index = samples_df.index.astype(str)
            frames.append(samples_df)
        except requests.HTTPError as e:
            print(f"    [WARN] HTTP error for {code}: {e}")
        except Exception as e:
            print(f"    [WARN] Failed {code}: {e}")

    if not frames:
        print(f"  [FAIL] {dir_name}: no {data_type} data downloaded")
        return "failed"

    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="first")]
    cols_label = "genes" if align_genes else "fields"
    print(f"  Combined {data_type}: {combined.shape} (samples × {cols_label})")

    # Build short-ID lookup for Xena samples
    xena_short = {normalize_sample_id(sid): sid for sid in combined.index}

    # Match patient IDs
    matched_ref  = []
    matched_xena = []
    for short_id, ref_id in short_to_ref.items():
        if short_id in xena_short:
            matched_ref.append(ref_id)
            matched_xena.append(xena_short[short_id])

    print(f"  Matched patients: {len(matched_ref)} / {len(ref_ids)}")
    if len(matched_ref) == 0:
        print(f"  [FAIL] {dir_name}: zero patient ID matches.")
        print(f"    Sample ref IDs  : {ref_ids[:3]}")
        print(f"    Sample xena IDs : {list(combined.index[:3])}")
        return "failed"

    # Align to matched samples
    matched = combined.loc[matched_xena].copy()
    matched.index = matched_ref   # restore original ref IDs

    if align_genes:
        # Filter to common genes
        common_genes = [g for g in ref_genes if g in matched.columns]
        print(f"  Common genes: {len(common_genes)} / {len(ref_genes)}")
        if len(common_genes) == 0:
            print(f"  [FAIL] {dir_name}: no gene overlap (check gene name format).")
            return "failed"

        # Reindex to exact shape of mutation_data.csv (fill missing with type-appropriate value)
        out = matched.reindex(index=ref_ids, columns=ref_genes, fill_value=cfg["fill"])
        out = out.fillna(cfg["fill"])
        if cfg["as_int"]:
            out = out.astype(int)
    else:
        # Phenotype/survival: keep all original columns; reindex rows to ref_ids only.
        # Missing patients become rows of NaN.
        out = matched.reindex(index=ref_ids)
        print(f"  Output: {out.shape} (samples × {len(out.columns)} fields)")

    out.to_csv(out_path)
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"  [SAVED] {out_path}  ({out.shape}, {sz_mb:.1f} MB)")
    return "saved"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download TCGA gene-level data from Xena")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Process only these dataset directories (default: all)")
    parser.add_argument("--types", nargs="+", default=list(DATA_TYPE_CONFIG.keys()),
                        choices=list(DATA_TYPE_CONFIG.keys()),
                        help=f"Data types to download (default: all = {list(DATA_TYPE_CONFIG.keys())})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and overwrite existing output files")
    args = parser.parse_args()

    targets = CANCER_MAP
    if args.datasets:
        targets = {k: v for k, v in CANCER_MAP.items() if k in args.datasets}
        missing = set(args.datasets) - set(CANCER_MAP)
        if missing:
            print(f"Warning: unknown dataset names: {missing}")
            print(f"Known datasets: {list(CANCER_MAP.keys())}")

    print("=" * 65)
    print("TCGA Gene-Level Data Downloader  (UCSC Xena)")
    print("=" * 65)
    print(f"Reference : {DATA_DIR}")
    print(f"Output    : {OUT_DIR}")
    print(f"Cache dir : {CACHE_DIR}")
    print(f"Datasets  : {list(targets.keys())}")
    print(f"Types     : {args.types}")
    print(f"Force     : {args.force}")
    print()

    results = {}   # (dir_name, data_type) -> status
    for dir_name, codes in targets.items():
        for data_type in args.types:
            print(f"\n{'─'*55}")
            print(f"  {dir_name} / {data_type}  →  {codes}")
            print(f"{'─'*55}")
            status = process_directory(dir_name, codes, data_type, args.force)
            results[(dir_name, data_type)] = status

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    for (dir_name, data_type), status in results.items():
        icon = {"saved": "✓", "skipped": "–", "failed": "✗"}.get(status, "?")
        print(f"  {icon}  {dir_name:<25} {data_type:<12} {status}")
    print()


if __name__ == "__main__":
    main()
