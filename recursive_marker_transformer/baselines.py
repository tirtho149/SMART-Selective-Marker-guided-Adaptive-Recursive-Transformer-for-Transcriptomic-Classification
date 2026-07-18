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

"""Classical / neural baselines for the genoNet BIO5 classification tasks.

Runs a battery of external baselines on the SAME data and SAME train/test split
as ``genonet_tasks.py`` (unified_bio5.csv, all 20530 genes), so the numbers are
directly comparable to bioMoR. One JSON per task is written to ``results_baselines/``
holding every baseline's accuracy / macro-F1 / weighted-F1.

Baselines (10) -- strong NONLINEAR / gradient-boosted tabular learners, the
competitive class for high-dimensional expression (linear models are excluded:
they are near-saturated on cohort classification and not the comparison of
interest here):
    Majority (DummyClassifier)        -- reference floor
    k-Nearest Neighbours
    RBF SVM (SVC)
    Random Forest
    Extra Trees
    Hist Gradient Boosting (sklearn)
    XGBoost
    LightGBM
    CatBoost
    MLP (one hidden layer)

Reproducible: fixed seed, train-split z-scoring, no network.

Usage:
    python -m recursive_marker_transformer.baselines               # all tasks
    python -m recursive_marker_transformer.baselines --tasks cancer_type
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .genonet_tasks import META_COLS, TASKS, _load_unified

warnings.filterwarnings("ignore")
SEED = 42


def _models(seed=SEED):
    """Return ordered (name, estimator) baselines."""
    return [
        ("Majority (floor)", DummyClassifier(strategy="most_frequent")),
        ("k-NN",             KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        ("RBF SVM",          SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=1000)),
        ("Random Forest",    RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                    n_jobs=-1, random_state=seed)),
        ("Extra Trees",      ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                                                  n_jobs=-1, random_state=seed)),
        ("Hist GBM",         HistGradientBoostingClassifier(max_iter=300, random_state=seed)),
        ("XGBoost",          XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                           subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                                           n_jobs=-1, random_state=seed, verbosity=0)),
        ("LightGBM",         LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.1,
                                            subsample=0.8, colsample_bytree=0.5, n_jobs=-1,
                                            random_state=seed, verbose=-1)),
        ("CatBoost",         CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                                random_seed=seed, verbose=0)),
        ("MLP (1x256)",      MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                                           early_stopping=True, random_state=seed)),
    ]


def run_task(task, X, labels, out_dir: Path, seed=SEED):
    y_raw = labels[task].values
    uniq = np.unique(y_raw)
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    G, K = X.shape[1], int(y.max() + 1)

    # identical split to genonet_tasks.py (test=0.2 stratified, seed 42)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)

    mu = X[tr].mean(0, keepdims=True)
    sd = X[tr].std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y[tr], y[te]

    print(f"\n########## {task}: N={len(y)} G={G} K={K} (train {len(tr)}, test {len(te)}) ##########",
          flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{task}.partial.json"
    rows = {}
    for name, model in _models(seed):
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            rows[name] = {
                "accuracy": float(accuracy_score(yte, yp)),
                "macro_f1": float(f1_score(yte, yp, average="macro")),
                "weighted_f1": float(f1_score(yte, yp, average="weighted")),
                "seconds": round(time.time() - t0, 1),
            }
            r = rows[name]
            print(f"  {name:20s} acc={r['accuracy']*100:5.1f} macroF1={r['macro_f1']*100:5.1f} "
                  f"wF1={r['weighted_f1']*100:5.1f}  ({r['seconds']}s)", flush=True)
        except Exception as e:
            rows[name] = {"error": str(e)}
            print(f"  {name:20s} ERROR {e}", flush=True)
        partial.write_text(json.dumps(
            {"task": task, "n_classes": int(K), "baselines": rows}, indent=1))

    res = {"task": task, "n_samples": int(len(y)), "n_genes": int(G),
           "n_classes": int(K), "n_train": int(len(tr)), "n_test": int(len(te)),
           "baselines": rows}
    with open(out_dir / f"{task}.json", "w") as f:
        json.dump(res, f, indent=1)
    partial.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/tcga/unified_bio5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_baselines"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    args = ap.parse_args()

    print(f"[baselines] loading {args.csv} ...", flush=True)
    X, labels, gene_cols = _load_unified(args.csv)
    print(f"[baselines] X={X.shape}  genes={len(gene_cols)}  tasks={args.tasks}", flush=True)
    for task in args.tasks:
        if (args.out / f"{task}.json").exists():
            print(f"[baselines] [skip] {task} (already done)", flush=True)
            continue
        run_task(task, X, labels, args.out)
    print("\n[baselines] done -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
