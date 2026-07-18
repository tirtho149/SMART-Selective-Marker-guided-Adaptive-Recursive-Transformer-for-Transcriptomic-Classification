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

"""Hyperparameter tuning of bioMoR on the hard genoNet clinical tasks.

Grid-searches a small set of bioMoR hyperparameters (model width, marker count,
recursion depth, learning rate) on each clinical task and selects the configuration
by VALIDATION macro-F1 (never test), then the selected config's TEST macro-F1 is the
tuned number reported in the paper. This is a fair, standard tuning protocol: the
same stratified 80/15/15 train/val/test split as everything else, selection on val,
report on test.

All runs use the full gene vector (~20530 genes). Output: one JSON per
(task, config) under results_tune/, so the job is RESUMABLE.

    python -m recursive_marker_transformer.tune
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import RMTConfig
from .genonet_tasks import TASKS, _load_unified
from .sweeps import train_eval

# 16-point grid (width x markers x depth x lr) -- modest but covers the knobs.
GRID = []
for dm in (128, 256):
    for M in (256, 512):
        for K in (4, 6):
            for lr in (3e-4, 1e-3):
                GRID.append(dict(d_model=dm, d_ff=2 * dm, n_markers=M,
                                 recursion_depth=K, lr=lr))


def _tag(g):
    return f"d{g['d_model']}_m{g['n_markers']}_k{g['recursion_depth']}_lr{g['lr']:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/tcga/unified_bio5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_tune"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    print(f"[tune] loading {args.csv} ...", flush=True)
    X, labels, gene_cols = _load_unified(args.csv)
    print(f"[tune] X={X.shape} tasks={args.tasks} grid={len(GRID)} configs", flush=True)

    base = RMTConfig(
        heads=("cancer_type",), n_hvg=None, batch_size=32, d_model=128, d_ff=256,
        n_markers=256, marker_mode="router", recursion_mode="expert", recursion_depth=4,
        epochs=args.epochs, patience=args.patience, lr=3e-4, device=args.device,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        print(f"\n===== tuning {task} =====", flush=True)
        for g in GRID:
            f = args.out / f"{task}__{_tag(g)}.json"
            if f.exists():
                print(f"  [skip] {f.name}", flush=True)
                continue
            cfg = replace(base, **g)
            r = train_eval(cfg, X, labels, task, args.seed)
            r["config_tag"] = _tag(g)
            f.write_text(json.dumps(r, indent=1))
            print(f"  [done] {_tag(g):28s} val={r['val_macro_f1']*100:.1f} "
                  f"test={r['macro_f1']*100:.1f}", flush=True)

    # report best-by-validation per task
    print("\n==== tuned bioMoR (selected by val macro-F1) ====", flush=True)
    for task in args.tasks:
        runs = [json.loads(p.read_text()) for p in args.out.glob(f"{task}__*.json")]
        if not runs:
            continue
        best = max(runs, key=lambda r: r["val_macro_f1"])
        print(f"  {task:18s} cfg={best['config_tag']:28s} "
              f"val={best['val_macro_f1']*100:.1f} test_macroF1={best['macro_f1']*100:.1f} "
              f"test_acc={best['accuracy']*100:.1f}", flush=True)


if __name__ == "__main__":
    main()
