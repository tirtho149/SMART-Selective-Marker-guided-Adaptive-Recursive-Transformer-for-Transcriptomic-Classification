"""bioMOR baseline backbone — the ONE apple-to-apple contract shared by every
baseline's `<method>_cv.py` runner, so all 10 baselines are comparable to bioMoR:

  * IDENTICAL folds: StratifiedKFold(n_splits=5, shuffle, random_state=42) with a
    within-train 10% stratified validation hold-out (72/8/20). Byte-identical to
    RecusrsiveQFormer/recursive_marker_transformer/cv.py::cv_folds. For the
    re-sourced single-cell datasets we instead load the saved folds.npz (same seed).
  * IDENTICAL metric: macro-F1 (+accuracy), summarised as mean/std(ddof=0) over folds.
  * IDENTICAL output schema: scores_<stamp>.csv with one row per fold + a mean/std row,
    columns [dataset, model, fold, macro_f1, accuracy, n_test]. So the paper table
    builder can drop any baseline in next to bioMoR.

Data lives in the bioMoR paper repo:
  MULTI-OMICS cohorts  : <BIOMOR>/data/<cohort>/{mutation_data.csv,cnv_data.csv,patient_labels.csv}
                         (patients in rows, genes in columns; label col = 2nd col of labels)
  3-modal              : <BIOMOR>/data/pan_meta_pri_3modal/{mutation,cnv,expression}_data.csv + labels.csv
  SINGLE-CELL          : <BIOMOR>/data/singlecell_resourced/<ds>/{adata.h5ad,folds.npz}

Set BIOMOR_ROOT to override the repo path (default: the sibling RecusrsiveQFormer checkout).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
N_FOLDS = 5
VAL_FRAC = 0.10

BIOMOR_ROOT = Path(os.environ.get(
    "BIOMOR_ROOT", "/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"))
DATA = BIOMOR_ROOT / "data"

MULTIOMICS_COHORTS = ["prostate", "blca", "stad", "brca", "pan_meta_pri"]
MULTIOMICS_3MODAL = ["pan_meta_pri_3modal"]
SC_DATASETS = ["Baron", "Lung", "Oesophagus", "Segerstolpe", "Spleen", "Tcell", "Xin"]


# --------------------------------------------------------------------------- #
# folds — byte-identical to bioMoR cv.cv_folds                                 #
# --------------------------------------------------------------------------- #
def cv_folds(y, n_folds=N_FOLDS, seed=SEED, val_frac=VAL_FRAC):
    """[(train_idx, val_idx, test_idx), ...] — same protocol as bioMoR."""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= n_folds:
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed).split(np.zeros(len(y)), y)
    else:
        outer = KFold(n_splits=n_folds, shuffle=True,
                      random_state=seed).split(np.zeros(len(y)))
    folds = []
    for tr_all, te in outer:
        _, cnt = np.unique(y[tr_all], return_counts=True)
        strat = y[tr_all] if cnt.min() >= 2 else None
        tr, va = train_test_split(tr_all, test_size=val_frac,
                                  random_state=seed, stratify=strat)
        folds.append((np.asarray(tr), np.asarray(va), np.asarray(te)))
    return folds


def _encode(labels):
    """Map raw label values -> contiguous 0..C-1 ints (sorted for determinism)."""
    labels = np.asarray(labels).ravel()
    uniq = np.unique(labels)
    remap = {v: i for i, v in enumerate(uniq)}
    return np.array([remap[v] for v in labels], dtype=np.int64), list(uniq)


# --------------------------------------------------------------------------- #
# multi-omics loader (P-NET cohorts)                                           #
# --------------------------------------------------------------------------- #
def _read_omics_csv(path):
    """patients x genes matrix. First column is the patient id (index)."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_omics(cohort, modalities=("mutation", "cnv"), root=None):
    """Return (X [N, sum F_m] float32, y int64 [N], meta dict).

    Intersects patients across every modality + labels, aligns genes per modality,
    concatenates modality blocks in the given order. CNV/expression are z-scored
    globally (per-fold z-scoring is the model's job if it wants it; global is the
    common baseline convention and keeps the shared split identical).
    meta = {patient_ids, feature_names, modality_dims, classes}.
    """
    root = Path(root) if root else DATA
    cdir = root / cohort
    lab_file = cdir / ("labels.csv" if (cdir / "labels.csv").exists()
                       else "patient_labels.csv")
    lab = pd.read_csv(lab_file, index_col=0)
    lab.index = lab.index.astype(str)
    label_col = lab.columns[0]                       # first non-index col is the label

    blocks, ids, names, dims = {}, None, [], {}
    for m in modalities:
        df = _read_omics_csv(cdir / f"{m}_data.csv")
        blocks[m] = df
        ids = df.index if ids is None else ids.intersection(df.index)
    ids = ids.intersection(lab.index)
    ids = sorted(ids)

    Xparts = []
    for m in modalities:
        sub = blocks[m].loc[ids]
        sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        arr = sub.to_numpy(np.float32)
        if m in ("cnv", "expression", "rna"):
            mu, sd = arr.mean(0, keepdims=True), arr.std(0, keepdims=True) + 1e-8
            arr = (arr - mu) / sd
        Xparts.append(arr)
        names += [f"{m}:{g}" for g in sub.columns]
        dims[m] = arr.shape[1]
    X = np.concatenate(Xparts, axis=1).astype(np.float32)
    y, classes = _encode(lab.loc[ids, label_col].to_numpy())
    meta = {"patient_ids": list(ids), "feature_names": names,
            "modality_dims": dims, "classes": classes}
    return X, y, meta


# --------------------------------------------------------------------------- #
# single-cell loader (re-sourced gene-annotated datasets)                      #
# --------------------------------------------------------------------------- #
def load_sc(ds, root=None):
    """Return (X [N,G] float32, y int64 [N], gene_symbols list)."""
    import anndata as ad
    root = Path(root) if root else DATA
    A = ad.read_h5ad(root / "singlecell_resourced" / ds / "adata.h5ad")
    X = A.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, np.float32)
    y = A.obs["label"].to_numpy().astype(np.int64)
    return X, y, [str(g) for g in A.var_names]


def load_sc_folds(ds, y=None, root=None):
    """Saved seed-42 5-fold split for a re-sourced SC dataset; falls back to
    computing cv_folds(y) if folds.npz is absent."""
    root = Path(root) if root else DATA
    p = root / "singlecell_resourced" / ds / "folds.npz"
    if p.exists():
        z = np.load(p)
        n = sum(1 for k in z.files if k.startswith("train_"))
        return [(z[f"train_{i}"], z[f"val_{i}"], z[f"test_{i}"]) for i in range(n)]
    if y is None:
        raise FileNotFoundError(f"{p} missing and no y given for fallback folds")
    return cv_folds(y)


# --------------------------------------------------------------------------- #
# metrics + score writer (common schema)                                       #
# --------------------------------------------------------------------------- #
def fold_metrics(y_true, y_pred):
    return (100.0 * f1_score(y_true, y_pred, average="macro"),
            100.0 * accuracy_score(y_true, y_pred))


def write_scores(work_dir, model, dataset, fold_f1, fold_acc, fold_ntest=None,
                 suffix="", stamp=None):
    """Write scores_<stamp>_<suffix>.csv (per-fold rows + a mean/std row) and
    return the path. Schema: dataset,model,fold,macro_f1,accuracy,n_test."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.today().strftime("%Y%m%d%H%M")
    fold_ntest = fold_ntest or [np.nan] * len(fold_f1)
    rows = [{"dataset": dataset, "model": model, "fold": i + 1,
             "macro_f1": float(f1), "accuracy": float(ac), "n_test": nt}
            for i, (f1, ac, nt) in enumerate(zip(fold_f1, fold_acc, fold_ntest))]
    f1a, aca = np.asarray(fold_f1, float), np.asarray(fold_acc, float)
    rows.append({"dataset": dataset, "model": model, "fold": "mean",
                 "macro_f1": float(f1a.mean()), "accuracy": float(aca.mean()),
                 "n_test": int(np.nansum(fold_ntest)) if fold_ntest else np.nan})
    rows.append({"dataset": dataset, "model": model, "fold": "std",
                 "macro_f1": float(f1a.std(ddof=0)), "accuracy": float(aca.std(ddof=0)),
                 "n_test": np.nan})
    out = work_dir / f"scores_{stamp}{('_' + suffix) if suffix else ''}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
