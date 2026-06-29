#!/usr/bin/env python3
"""Build the PANCAN multimodal + subtype dataset for SMART.

Reads the raw UCSC Xena PANCAN files downloaded by ``download_pancan_raw.py``
(in ``pancan_raw/``) and emits a single, mutually-aligned multimodal dataset
under ``data/pancan/``:

    mm_expr.npy   (N, G) float32   log2 RSEM expression (EB++ adjusted)
    mm_cnv.npy    (N, G) float32   Gistic2 gene-level continuous copy number
    mm_mut.npy    (N, G) float32   MC3 non-silent gene-level binary mutation
    genes.txt     G gene symbols (shared across all three modalities, aligned)
    samples.txt   N sample barcodes (aligned to the array rows)
    labels.csv    sample, immune_subtype, molecular_subtype, cancer_type

Sample set N = samples present in *all three* omics matrices that also carry a
Thorsson immune-subtype call (the pan-cancer 6-class label). This single sample
and gene set is shared by every PANCAN experiment so that the expression-only
SMART run is a clean ablation baseline for the multimodal run.

``molecular_subtype`` is the curated TCGA ``Subtype_Selected`` call, kept only
for classes with >= MIN_MOL_SUPPORT members in N (the long tail of tiny
per-cancer subtypes is set to NA and excluded from that task).
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "pancan_raw"
OUT = HERE.parent / "data" / "pancan"

EXPR = RAW / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
CNV = RAW / "broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.gene.xena.gz"
MUT = RAW / "mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz"
IMMUNE = RAW / "Subtype_Immune_Model_Based.txt.gz"
MOLEC = RAW / "TCGASubtype.20170308.tsv.gz"
PHENO = RAW / "TCGA_phenotype_denseDataOnlyDownload.tsv.gz"

MIN_MOL_SUPPORT = 50   # drop molecular-subtype classes with fewer members than this


def _header_genes_samples(path: Path):
    """Cheaply read the gene (row index) and sample (header) lists of a
    genes x samples Xena matrix without loading the values."""
    with gzip.open(path, "rt") as f:
        samples = f.readline().rstrip("\n").split("\t")[1:]
        genes = [ln.split("\t", 1)[0] for ln in f]
    return genes, samples


def _read_matrix(path: Path, genes: list[str], samples: list[str]) -> np.ndarray:
    """Read a genes x samples matrix and return an (N_samples, G_genes) float32
    array aligned to the given `genes` and `samples` order."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df = df.reindex(index=genes, columns=samples)
    arr = df.T.to_numpy(dtype=np.float32)   # (samples, genes)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min_mol", type=int, default=MIN_MOL_SUPPORT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[build] reading headers ...", flush=True)
    eg, es = _header_genes_samples(EXPR)
    cg, cs = _header_genes_samples(CNV)
    mg, ms = _header_genes_samples(MUT)

    # ---- labels ----
    imm = pd.read_csv(IMMUNE, sep="\t").dropna(subset=["Subtype_Immune_Model_Based"])
    imm = imm.set_index("sample")["Subtype_Immune_Model_Based"].astype(str)
    mol = pd.read_csv(MOLEC, sep="\t").set_index("sampleID")["Subtype_Selected"].astype(str)
    phe = pd.read_csv(PHENO, sep="\t").set_index("sample")["_primary_disease"].astype(str)

    # ---- shared gene set (sorted for determinism) ----
    genes = sorted(set(eg) & set(cg) & set(mg))
    # ---- shared sample set: all three omics + an immune label ----
    samples = sorted((set(es) & set(cs) & set(ms)) & set(imm.index))
    print(f"[build] shared genes={len(genes)}  multimodal+immune samples={len(samples)}",
          flush=True)

    print("[build] reading expression ...", flush=True)
    X_expr = _read_matrix(EXPR, genes, samples)
    print("[build] reading cnv ...", flush=True)
    X_cnv = _read_matrix(CNV, genes, samples)
    print("[build] reading mutation ...", flush=True)
    X_mut = _read_matrix(MUT, genes, samples)

    # NaNs: expression/CNV missing -> 0; mutation missing -> 0 (no call = not mutated)
    X_expr = np.nan_to_num(X_expr, nan=0.0)
    X_cnv = np.nan_to_num(X_cnv, nan=0.0)
    X_mut = np.nan_to_num(X_mut, nan=0.0)

    # ---- label table aligned to samples ----
    immune = imm.reindex(samples).values
    molecular = mol.reindex(samples).astype("object").values
    cancer = phe.reindex(samples).astype("object").values
    # drop molecular tail: NA-suffixed or rare classes
    mol_clean = np.array([str(m) for m in molecular], dtype=object)
    mol_clean[pd.isna(molecular)] = "NA"
    is_na = np.array([m.endswith(".NA") or m == "NA" or m == "nan" for m in mol_clean])
    mol_clean[is_na] = np.nan
    vc = pd.Series(mol_clean).value_counts()
    keep = set(vc[vc >= args.min_mol].index)
    mol_final = np.array([m if (isinstance(m, str) and m in keep) else np.nan
                          for m in mol_clean], dtype=object)
    n_mol = int(pd.notna(mol_final).sum())
    n_mol_cls = len(set(m for m in mol_final if isinstance(m, str)))

    labels = pd.DataFrame({
        "sample": samples,
        "immune_subtype": immune,
        "molecular_subtype": mol_final,
        "cancer_type": cancer,
    })

    # ---- save ----
    np.save(args.out / "mm_expr.npy", X_expr)
    np.save(args.out / "mm_cnv.npy", X_cnv)
    np.save(args.out / "mm_mut.npy", X_mut)
    (args.out / "genes.txt").write_text("\n".join(genes) + "\n")
    (args.out / "samples.txt").write_text("\n".join(samples) + "\n")
    labels.to_csv(args.out / "labels.csv", index=False)

    print(f"[build] saved -> {args.out}", flush=True)
    print(f"  arrays: expr/cnv/mut = {X_expr.shape} float32", flush=True)
    print(f"  immune_subtype:    {len(set(immune))} classes, {len(samples)} samples", flush=True)
    print(f"  molecular_subtype: {n_mol_cls} classes (>= {args.min_mol} support), "
          f"{n_mol} labeled samples", flush=True)
    print("  immune value counts:\n", pd.Series(immune).value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
