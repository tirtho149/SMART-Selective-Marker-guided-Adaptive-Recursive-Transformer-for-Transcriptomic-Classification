# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""External baselines for the PANCAN subtype / multimodal benchmark.

Same battery of 10 nonlinear tabular learners as ``baselines.py``, run on the
identical PANCAN sample set, gene set, and seeded stratified split as
``pancan_tasks.py`` so the numbers drop into the same tables next to SMART. For
a multimodal channel set the per-gene channels are concatenated into one feature
vector, which is the fair flat-feature baseline for the SMART fusion runs.

Usage:
    python -m recursive_marker_transformer.pancan_baselines \
        --task immune_subtype --channels expr expr_cnv_mut
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .baselines import _models, SEED
from .pancan_tasks import CHANNEL_SETS, _load_pancan, _stack_channels

warnings.filterwarnings("ignore")


def run_task(task, channel_set, mats, labels, out_dir: Path, seed=SEED):
    names = CHANNEL_SETS[channel_set]
    keep = labels[task].notna().values
    X = _stack_channels(mats, names)[keep]
    if X.ndim == 3:                       # (N,G,C) -> concat channels -> (N, G*C)
        X = X.reshape(X.shape[0], -1)
    y_raw = labels[task].values[keep]
    uniq = sorted(set(y_raw))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    G, K = X.shape[1], int(y.max() + 1)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    mu = X[tr].mean(0, keepdims=True)
    sd = X[tr].std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y[tr], y[te]

    print(f"\n########## {task} [{channel_set}] N={len(y)} F={G} K={K} "
          f"(train {len(tr)}, test {len(te)}) ##########", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{task}__{channel_set}.partial.json"
    rows = {}
    for name, model in _models(seed):
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            from sklearn.metrics import accuracy_score, f1_score
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
            {"task": task, "channel_set": channel_set, "n_classes": int(K),
             "baselines": rows}, indent=1))

    res = {"task": task, "channel_set": channel_set, "n_samples": int(len(y)),
           "n_features": int(G), "n_classes": int(K), "n_train": int(len(tr)),
           "n_test": int(len(te)), "baselines": rows}
    with open(out_dir / f"{task}__{channel_set}.json", "w") as f:
        json.dump(res, f, indent=1)
    partial.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pancan"))
    ap.add_argument("--out", type=Path, default=Path("results_pancan_baselines"))
    ap.add_argument("--task", type=str, default="immune_subtype",
                    choices=["immune_subtype", "molecular_subtype"])
    ap.add_argument("--channels", nargs="+", default=["expr"],
                    choices=list(CHANNEL_SETS.keys()))
    args = ap.parse_args()

    print(f"[pancan-baselines] loading {args.data} ...", flush=True)
    mats, labels, genes = _load_pancan(args.data)
    for cs in args.channels:
        if (args.out / f"{args.task}__{cs}.json").exists():
            print(f"[pancan-baselines] [skip] {args.task} [{cs}] (done)", flush=True)
            continue
        run_task(args.task, cs, mats, labels, args.out)
    print(f"\n[pancan-baselines] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
