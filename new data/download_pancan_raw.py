#!/usr/bin/env python3
"""
Download raw TCGA Pan-Cancer (PANCAN) files from UCSC Xena's pancanatlas hub.

Source: https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)

Files are saved to data_tcga/pancan_raw/ as-is (gzipped where the hub serves
them gzipped, plain otherwise). No alignment, no patient/gene filtering — use
download_expression.py for the aligned-to-mutation_data.csv workflow.

Usage:
    cd /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga
    python download_pancan_raw.py            # download all 7 files
    python download_pancan_raw.py --force    # re-download even if cached
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote

import requests

HUB = "https://pancanatlas.xenahubs.net/download"
OUT_DIR = Path(__file__).resolve().parent / "pancan_raw"

# (label, dataset_id, suffix). suffix='.gz' means the hub serves it gzipped;
# suffix='' means plain text (still TSV inside).
FILES = [
    ("expression",       "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena",        ".gz"),
    ("cnv",              "broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.gene.xena", ".gz"),
    ("mutation",         "mc3.v0.2.8.PUBLIC.nonsilentGene.xena",                         ".gz"),
    ("curated_clinical", "Survival_SupplementalTable_S1_20171025_xena_sp",               ""),
    ("sample_phenotype", "TCGA_phenotype_denseDataOnlyDownload.tsv",                     ".gz"),
    ("molecular_subtype","TCGASubtype.20170308.tsv",                                     ".gz"),
    ("immune_subtype",   "Subtype_Immune_Model_Based.txt",                               ".gz"),
]


def download(label: str, dataset_id: str, suffix: str, force: bool) -> str:
    # S3 (the redirect target) rejects literal '+' — must be %2B-encoded.
    # quote() with default safe='/' leaves path separators alone.
    url = f"{HUB}/{quote(dataset_id)}{suffix}"
    out_name = f"{dataset_id}{suffix}".replace("/", "_")
    out_path = OUT_DIR / out_name

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [SKIP] {label:<18} {out_name}  ({size_mb:.1f} MB, cached)")
        return "skipped"

    print(f"  [GET ] {label:<18} {url}")
    try:
        resp = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        resp.raise_for_status()
        total = 0
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                total += len(chunk)
        print(f"  [SAVE] {label:<18} {out_path.name}  ({total / 1e6:.1f} MB)")
        return "saved"
    except requests.HTTPError as e:
        print(f"  [FAIL] {label:<18} HTTP {e.response.status_code}: {url}")
        if out_path.exists():
            out_path.unlink()
        return "failed"
    except Exception as e:
        print(f"  [FAIL] {label:<18} {e}")
        if out_path.exists():
            out_path.unlink()
        return "failed"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"PANCAN raw downloader → {OUT_DIR}")
    print(f"Hub: {HUB}\n")

    results = {}
    for label, ds, sfx in FILES:
        results[label] = download(label, ds, sfx, args.force)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, status in results.items():
        icon = {"saved": "OK", "skipped": "--", "failed": "XX"}[status]
        print(f"  [{icon}] {label:<18} {status}")

    if any(s == "failed" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
