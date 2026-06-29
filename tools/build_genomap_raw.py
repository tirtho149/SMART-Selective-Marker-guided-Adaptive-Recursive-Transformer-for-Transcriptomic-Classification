#!/usr/bin/env python3
"""Tier-2: rebuild the genomap-paper raw scRNA-seq datasets into genomap features.

Downloads each dataset from its public accession, parses cells x genes + cell-type
labels, library-normalises + log1p, selects the top-``--hvg`` highly-variable genes,
runs genomap's own ``construct_genomap`` (Gromov-Wasserstein gene->grid placement,
Islam & Xing 2023) into an R x C genomap, flattens it to features, and writes the
capsule format SMART already consumes:

    data/singlecell/<name>/expression.csv.gz   cell_id, feat_0001 ... feat_RC
    data/singlecell/<name>/labels.csv          cell_id, label, class_name

Datasets (genomap-paper pancreas benchmark, all public on GEO/ArrayExpress):
    baron  GSE84133 (human, 4 donors)   muraro GSE85241   xin GSE81608
    wang   GSE83139                      segerstolpe E-MTAB-5061
(Broad SCP490/454/3 need a portal login and are not auto-downloadable -> skipped.)

Usage:
    python tools/build_genomap_raw.py --dataset baron --hvg 1936 --grid 44
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "genomap_raw"
OUT = ROOT / "data" / "singlecell"
sys.path.insert(0, str(ROOT / "genomap"))


# ----------------------------------------------------------------------------- parsers
def _baron() -> tuple[pd.DataFrame, pd.Series]:
    """GSE84133 human pancreas (4 donors). Each donor CSV: index, barcode,
    assigned_cluster (cell type), then gene columns."""
    tar = RAW / "GSE84133_RAW.tar"
    frames, labels = [], []
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if "human" not in m.name:
                continue
            with gzip.open(io.BytesIO(tf.extractfile(m).read()), "rt") as fh:
                df = pd.read_csv(fh, index_col=0)
            lab = df["assigned_cluster"].astype(str)
            expr = df.drop(columns=["barcode", "assigned_cluster"])
            expr.index = [f"{m.name.split('_')[1]}_{i}" for i in df.index]
            lab.index = expr.index
            frames.append(expr)
            labels.append(lab)
    X = pd.concat(frames).fillna(0.0)
    y = pd.concat(labels)
    return X, y


def _segerstolpe() -> tuple[pd.DataFrame, pd.Series]:
    """E-MTAB-5061: counts matrix (genes x 3514 cells) + sdrf metadata whose
    'Source Name' matches the matrix columns and 'inferred cell type' is the label."""
    sdrf = pd.read_csv(RAW / "segerstolpe_meta.dl", sep="\t")
    src = sdrf.columns[0]                                   # 'Source Name'
    ct = "Characteristics [inferred cell type]"
    lab_map = dict(zip(sdrf[src].astype(str), sdrf[ct].astype(str)))

    # Custom format: header has 3514 cell names; each data row is
    # gene + 3514 RPKM + 3514 counts + trailing tab (7030 fields). Take the COUNTS
    # block (second 3514 cols) and CPM+log1p it downstream like the other datasets.
    data = RAW / "segerstolpe_data.dl"
    with open(data) as fh:
        cells = fh.readline().rstrip("\n").split("\t")[1:]
    n = len(cells)
    df = pd.read_csv(data, sep="\t", header=None, skiprows=1)
    genes = df.iloc[:, 0].astype(str)
    counts = df.iloc[:, 1 + n: 1 + 2 * n].copy()           # the counts half
    counts.columns = cells
    counts.index = genes
    counts = counts[~counts.index.str.startswith("__")]    # drop htseq summary rows
    X = counts.T                                            # cells x genes
    drop = {"not applicable", "unclassified endocrine cell", "unclassified cell",
            "co-expression cell", "MHC class II cell", "nan"}
    y = pd.Series([lab_map.get(c, "nan") for c in X.index], index=X.index)
    mask = ~y.isin(drop)
    X, y = X[mask.values], y[mask.values]
    y = y.str.replace(" cell", "", regex=False)            # 'alpha cell' -> 'alpha'
    return X.fillna(0.0), y


PARSERS = {"baron": _baron, "segerstolpe": _segerstolpe}
ACCESSION = {"baron": "GSE84133", "segerstolpe": "E-MTAB-5061"}


# ----------------------------------------------------------------------------- pipeline
def _normalize_hvg(X: pd.DataFrame, n_hvg: int) -> pd.DataFrame:
    """CPM-style library normalisation + log1p, then top-n_hvg by variance."""
    M = X.to_numpy(dtype=np.float32)
    libs = M.sum(axis=1, keepdims=True)
    libs[libs == 0] = 1.0
    M = np.log1p(M / libs * 1e4)
    var = M.var(axis=0)
    keep = np.argsort(-var)[:min(n_hvg, M.shape[1])]
    keep.sort()
    return pd.DataFrame(M[:, keep], index=X.index, columns=X.columns[keep])


def build(name: str, n_hvg: int, grid: int, num_iter: int) -> None:
    if name not in PARSERS:
        raise SystemExit(f"no parser for {name!r}; available: {list(PARSERS)}")
    print(f"[{name}] parsing {ACCESSION.get(name,'?')} ...", flush=True)
    X, y = PARSERS[name]()
    print(f"[{name}] raw cells={X.shape[0]} genes={X.shape[1]} "
          f"classes={y.nunique()}", flush=True)

    Xh = _normalize_hvg(X, n_hvg)
    print(f"[{name}] HVG-selected -> {Xh.shape}; constructing {grid}x{grid} genomap ...",
          flush=True)
    from genomap.genomap import construct_genomap
    g = np.asarray(construct_genomap(Xh.to_numpy(dtype=np.float32), grid, grid,
                                     num_iter=num_iter))
    feats = g.reshape(g.shape[0], -1).astype(np.float32)        # (cells, grid*grid)
    print(f"[{name}] genomap features -> {feats.shape}", flush=True)

    classes = sorted(y.unique())
    enc = {c: i + 1 for i, c in enumerate(classes)}             # 1-based, capsule style
    cell_ids = [f"{name}_{i:06d}" for i in range(feats.shape[0])]

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cols = ["cell_id"] + [f"feat_{j+1:04d}" for j in range(feats.shape[1])]
    expr_df = pd.DataFrame(feats, columns=cols[1:])
    expr_df.insert(0, "cell_id", cell_ids)
    expr_df.to_csv(out / "expression.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"cell_id": cell_ids,
                  "label": [enc[v] for v in y.values],
                  "class_name": list(y.values)}).to_csv(out / "labels.csv", index=False)
    print(f"[{name}] wrote {out} (cells={feats.shape[0]} feats={feats.shape[1]} "
          f"classes={len(classes)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PARSERS))
    ap.add_argument("--hvg", type=int, default=1936)            # 44x44, capsule pancreas
    ap.add_argument("--grid", type=int, default=44)
    ap.add_argument("--num_iter", type=int, default=1000)
    args = ap.parse_args()
    build(args.dataset, args.hvg, args.grid, args.num_iter)


if __name__ == "__main__":
    main()
