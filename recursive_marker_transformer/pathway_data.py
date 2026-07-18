# ============================================================================
# bioMoR: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The bioMoR Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Loader for the Reactome / P-NET pathway-informed multi-omics cohorts.

Each ``data/<cohort>/`` (from the GitHub ``data-branch``) ships:

  * ``filtered_pathways.csv``  -- 1268 Reactome pathways (R-HSA-...) -> member genes
  * ``adjacency_matrix.csv``   -- 1268 x 1268 pathway->pathway hierarchy graph
  * ``mutation_data.csv``      -- patient x gene binary somatic mutation
  * ``cnv_data.csv``           -- patient x gene copy number in {-2..2} (absent for brca)
  * ``patient_labels.csv``     -- id,response  (binary or multiclass)

This module produces, for a cohort:

  * ``X``        -- (N, G) or (N, G, C) aligned omics (mut and/or cnv channels);
  * ``y``        -- (N,) int labels, ``classes`` the original label strings;
  * ``P``        -- (G, M) binary gene->pathway membership (Reactome), pathways
                    with < ``min_genes`` present members dropped;
  * ``centrality`` -- (M,) z-scored eigenvector centrality of the *kept* pathway
                    sub-hierarchy, the Reactome router prior (set_token_prior);
  * ``pathways`` -- the kept pathway ids; ``genes`` the gene order of X / P.

Gene symbols in the omics headers are Excel-mangled in places (``1-Mar`` ->
``MARCH1``, ``10-Sep`` -> ``SEPT10``); :func:`fix_symbol` reverses the common forms
before intersecting with the pathway gene lists. The membership and the omics
share one gene order so ``P`` lines up with the columns of ``X``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CHANNEL_SETS = {
    "mut":     ["mut"],
    "cnv":     ["cnv"],
    "mut_cnv": ["mut", "cnv"],
    "expr":    ["expr"],           # expression only (Xena-joined pancancer or 3-modal file)
    "mut_cnv_expr": ["mut", "cnv", "expr"],   # tri-modal concat (pan_meta_pri_3modal)
}
# each modality's per-gene omics file; 3-modal cohorts additionally ship expression.
_MODALITY_FILE = {"mut": "mutation_data.csv", "cnv": "cnv_data.csv",
                  "expr": "expression_data.csv"}


def _first_existing(root: Path, *names: str) -> Path:
    """Resolve a file that may be named differently across cohort variants
    (e.g. ``labels.csv`` vs ``patient_labels.csv``, ``pathways.csv`` vs
    ``filtered_pathways.csv``). Returns the first that exists, else the first name."""
    for n in names:
        if (root / n).exists():
            return root / n
    return root / names[0]

# Excel mangles a few gene symbols into dates: MARCH1 -> "1-Mar", SEPT10 ->
# "10-Sep", DEC1 -> "1-Dec". Reverse the common month forms.
_MONTH = {"Mar": "MARCH", "Sep": "SEPT", "Sept": "SEPT", "Dec": "DEC"}
_DATEY = re.compile(r"^(\d{1,2})-(Mar|Sep|Sept|Dec)$")


def fix_symbol(s: str) -> str:
    m = _DATEY.match(str(s))
    return f"{_MONTH[m.group(2)]}{int(m.group(1))}" if m else str(s)


@dataclass
class PathwayCohort:
    X: np.ndarray                 # (N, G) or (N, G, C) float32
    y: np.ndarray                 # (N,) int64
    classes: list                 # original label values, index = encoded id
    P: np.ndarray                 # (G, M) float32 binary membership
    centrality: np.ndarray        # (M,) float32 z-scored pathway-hierarchy prior
    adjacency: np.ndarray         # (M, M) float32 symmetric Reactome hierarchy graph
    genes: list                   # G gene symbols (order of X cols / P rows)
    pathways: list                # M kept pathway ids (order of P cols)
    patient_ids: list             # N patient ids (order of X rows)
    channels: list                # modality names making up C


def _read_omics(path: Path) -> tuple[list, list, np.ndarray]:
    """Read a patient x gene matrix -> (patient_ids, fixed_gene_names, values)."""
    df = pd.read_csv(path, index_col=0)
    genes = [fix_symbol(g) for g in df.columns]
    # Some per-gene matrices (notably the tri-modal expression file) ship sparse NaNs;
    # zero-fill them to match the Xena expression path (np.nan_to_num, nan=0.0) so the
    # model never sees NaN inputs (which propagate to NaN predictions -> roc_auc crash).
    X = np.nan_to_num(df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return df.index.astype(str).tolist(), genes, X


def _eigenvector_centrality(W: np.ndarray, iters: int = 200) -> np.ndarray:
    """Leading eigenvector (power iteration) of the affinity -> z-scored prior.
    Same routine as ``interaction._eigenvector_centrality`` but inlined (pure
    numpy) so this loader stays importable without torch."""
    n = W.shape[0]
    v = np.ones(n, dtype=np.float64) / np.sqrt(n)
    Wd = W.astype(np.float64)
    for _ in range(iters):
        v_new = Wd @ v
        nrm = np.linalg.norm(v_new)
        if nrm < 1e-12:
            break
        v = v_new / nrm
    c = np.abs(v)
    c = (c - c.mean()) / (c.std() + 1e-8)
    return c.astype(np.float32)


def _kept_adjacency(adj: pd.DataFrame, kept: list) -> np.ndarray:
    """Symmetric Reactome hierarchy graph restricted to the kept pathways (M, M)."""
    A = adj.reindex(index=kept, columns=kept).fillna(0.0).to_numpy(dtype=np.float32)
    return np.maximum(A, A.T)                    # treat the hierarchy as undirected


def _build_pathways(root: Path, genes: list, min_genes: int):
    """From ``filtered_pathways.csv`` + ``adjacency_matrix.csv`` and a candidate gene
    list, build the kept membership/graph. Returns
    ``(P, pathways, genes_kept, A_sub, centrality)``: membership ``P`` (G', M) over
    the genes that land in >=1 kept pathway, the kept pathway ids, that gene
    sub-list, the (M, M) symmetric hierarchy graph and its (M,) centrality prior.
    Pathways with < ``min_genes`` present members are dropped."""
    pw = pd.read_csv(_first_existing(root, "filtered_pathways.csv", "pathways.csv"))
    gidx = {g: i for i, g in enumerate(genes)}
    P_full = np.zeros((len(genes), len(pw)), dtype=np.float32)
    for j, gene_str in enumerate(pw["Genes"].fillna("")):
        for g in (x.strip() for x in str(gene_str).split(",")):
            i = gidx.get(g)
            if i is not None:
                P_full[i, j] = 1.0
    keep_pw_mask = P_full.sum(axis=0) >= min_genes
    P = P_full[:, keep_pw_mask]
    pathways = [pid for pid, k in zip(pw["Pathway_ID"], keep_pw_mask) if k]
    gene_in_pw = P.sum(axis=1) > 0
    genes_kept = [g for g, k in zip(genes, gene_in_pw) if k]
    P = P[gene_in_pw]
    adj = pd.read_csv(root / "adjacency_matrix.csv", index_col=0)
    A_sub = _kept_adjacency(adj, pathways)
    centrality = _eigenvector_centrality(A_sub)
    return P, pathways, genes_kept, A_sub, centrality


def load_cohort(cohort: str, channels: str = "mut_cnv", min_genes: int = 5,
                root: Optional[Path] = None) -> PathwayCohort:
    """Load one cohort aligned to a shared gene set across the requested omics."""
    root = (root or DATA_ROOT) / cohort
    names = CHANNEL_SETS[channels]

    # ---- omics: load each modality, intersect on the shared gene set ----
    mods, pids_ref, gene_sets = {}, None, []
    for n in names:
        p = root / _MODALITY_FILE[n]
        if not p.exists() or p.stat().st_size < 5:
            raise FileNotFoundError(f"{cohort}: modality {n!r} ({p.name}) is absent")
        pids, genes, vals = _read_omics(p)
        mods[n] = (pids, genes, vals)
        pids_ref = pids if pids_ref is None else pids_ref
        gene_sets.append(genes)

    # ---- labels ----
    # data.md: column 1 = patient ID, column 2 = ``response`` (the classification
    # target). Some cohorts (``pan_meta_pri``) carry extra descriptor columns
    # (``sample_type``, ``primary_disease``) AFTER ``response`` -- selecting the
    # last column there silently trains on the 32-class cancer type instead of the
    # binary metastatic-vs-primary target, so pick ``response`` explicitly.
    lab = pd.read_csv(_first_existing(root, "patient_labels.csv", "labels.csv"))
    idcol = lab.columns[0]
    ycol = "response" if "response" in lab.columns else lab.columns[1 if len(lab.columns) > 1 else -1]
    lab = lab.dropna(subset=[ycol])
    lab_map = dict(zip(lab[idcol].astype(str), lab[ycol]))

    # patients present in every modality AND labelled (preserve first-modality order)
    pid_sets = [set(m[0]) for m in mods.values()]
    keep_pid = set.intersection(*pid_sets) & set(lab_map)
    patient_ids = [p for p in pids_ref if p in keep_pid]

    # shared gene set (sorted for determinism)
    genes = sorted(set.intersection(*[set(g) for g in gene_sets]))

    # ---- pathways: membership + hierarchy over the shared gene set ----
    P, pathways, genes, A_sub, centrality = _build_pathways(root, genes, min_genes)

    # ---- assemble aligned X (N, G[, C]) over the kept genes ----
    chans = []
    for n in names:
        pids, gnames, vals = mods[n]
        prow = {p: r for r, p in enumerate(pids)}
        gcol = {g: c for c, g in enumerate(gnames)}
        r = np.array([prow[p] for p in patient_ids])
        c = np.array([gcol[g] for g in genes])
        chans.append(vals[np.ix_(r, c)])
    X = chans[0] if len(chans) == 1 else np.stack(chans, axis=-1)
    X = X.astype(np.float32)

    # ---- labels -> encoded ----
    y_raw = [lab_map[p] for p in patient_ids]
    classes = sorted(set(y_raw), key=lambda v: str(v))
    enc = {v: i for i, v in enumerate(classes)}
    y = np.array([enc[v] for v in y_raw], dtype=np.int64)

    return PathwayCohort(X=X, y=y, classes=classes, P=P, centrality=centrality,
                         adjacency=A_sub, genes=genes, pathways=pathways,
                         patient_ids=patient_ids, channels=names)


# Xena PANCAN expression (genes x samples) used for the pancancer_meta cohorts,
# which ship pathways + labels but no per-gene omics on the data-branch.
DEFAULT_EXPR = (Path(__file__).resolve().parents[1] / "new data" / "pancan_raw"
                / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz")


def load_pan_meta(label: str = "response", cohort: str = "pancancer_meta_pri",
                  min_genes: int = 5, expr_path: Optional[Path] = None,
                  root: Optional[Path] = None) -> PathwayCohort:
    """Load a pancancer-meta cohort: pathways/labels ship on the data-branch but
    NO per-gene omics, so the expression channel is joined from the UCSC Xena
    PANCAN matrix by TCGA barcode. ``label`` selects the head: ``response``
    (primary vs metastatic) or ``primary_disease`` (32-class cancer type)."""
    root = (root or DATA_ROOT) / cohort
    expr_path = expr_path or DEFAULT_EXPR

    lab = pd.read_csv(root / "patient_labels.csv")

    # The aligned expression matrix + pathway artifacts are label-independent and
    # expensive (full ~2GB Xena read), so cache them per (cohort, min_genes) and
    # reuse across both label heads and every ablation arm.
    cache = root / f"_expr_cache_mg{min_genes}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, genes, pathways = z["X"], list(z["genes"]), list(z["pathways"])
        P, A_sub, centrality = z["P"], z["adjacency"], z["centrality"]
        samples = list(z["samples"])
    else:
        all_ids = set(lab["id"].astype(str))
        # memory-frugal read: only the labelled sample columns, parsed as float32
        # (default float64 peaks several GB and OOMs on tight nodes).
        hdr = pd.read_csv(expr_path, sep="\t", index_col=0, nrows=0)
        idx_name = hdr.index.name or "sample"
        samp_cols = [c for c in hdr.columns if c in all_ids]
        expr = pd.read_csv(expr_path, sep="\t", index_col=0,
                           usecols=[idx_name] + samp_cols,
                           dtype={c: np.float32 for c in samp_cols})
        expr.index = expr.index.astype(str)
        expr = expr[~expr.index.duplicated(keep="first")]     # Xena has dup symbols
        samples = list(expr.columns)                          # labelled overlap
        genes_all = sorted(expr.index.tolist())
        P, pathways, genes, A_sub, centrality = _build_pathways(root, genes_all, min_genes)
        Xdf = expr.reindex(index=genes, columns=samples)
        X = np.nan_to_num(Xdf.T.to_numpy(dtype=np.float32), nan=0.0)   # (N, G)
        np.savez(cache, X=X, genes=np.array(genes), pathways=np.array(pathways),
                 P=P, adjacency=A_sub, centrality=centrality, samples=np.array(samples))

    # apply the requested label head, keeping only samples that carry it
    lab = lab.dropna(subset=[label])
    lab_map = dict(zip(lab["id"].astype(str), lab[label]))
    keep = [i for i, s in enumerate(samples) if s in lab_map]
    X = X[keep]
    samples = [samples[i] for i in keep]

    y_raw = [lab_map[s] for s in samples]
    classes = sorted(set(y_raw), key=lambda v: str(v))
    enc = {v: i for i, v in enumerate(classes)}
    y = np.array([enc[v] for v in y_raw], dtype=np.int64)

    return PathwayCohort(X=X, y=y, classes=classes, P=P, centrality=centrality,
                         adjacency=A_sub, genes=genes, pathways=pathways,
                         patient_ids=samples, channels=["expr"])


if __name__ == "__main__":   # quick sanity print (no torch needed)
    import sys
    for c in (sys.argv[1:] or ["prostate"]):
        ch = "mut" if c == "brca" else "mut_cnv"
        d = load_cohort(c, channels=ch)
        print(f"{c}: X={d.X.shape} y={d.y.shape} K={len(d.classes)} "
              f"G={len(d.genes)} M={len(d.pathways)} "
              f"prior[min,max]=[{d.centrality.min():.2f},{d.centrality.max():.2f}]")
