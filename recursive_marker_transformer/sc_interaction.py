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

"""Biology-informed-router ablation on the single-cell genomap datasets.

This is the headline experiment of the paper: does injecting the genomap
gene-gene interaction graph (co-expression-network centrality) as an annealed
prior on the recursion depth-router change cell-type recognition, and is it the
*real* co-expression structure that matters (coexpr) rather than any bias
(random)?  We sweep the prior over three modes per dataset across several seeds
and report accuracy / macro-F1 mean +/- std.

    none    -- original bioMoR router (no prior)
    coexpr  -- genomap correlation-graph eigenvector-centrality prior (proposed)
    random  -- degree-matched random-graph control (same sparsity, shuffled edges)

The genomap datasets are the *native* home of this prior (their features are the
genomap co-expression embedding itself), so unlike the bulk setting this is where
co-expression structure should be on-distribution. We report whatever the numbers
say. Resumable: one JSON per (dataset, mode, seed) under results_sc_interaction/.

    python -m recursive_marker_transformer.sc_interaction \
        --datasets tabula_muris pancreas --modes none coexpr random --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import RMTConfig
from .singlecell import HEAD, _load_dataset, _make_splits, _fit_eval
from .train import resolve_device


def run_one(name: str, data_root: Path, base: RMTConfig, mode: str, seed: int) -> dict:
    cfg = replace(base, seed=seed, gene_interaction=mode)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(cfg.device)

    X, y, split = _load_dataset(data_root / name)
    F, K = X.shape[1], int(y.max() + 1)
    tr, va, te = _make_splits(y, split, seed)
    cfg = replace(cfg, heads=(HEAD,), n_markers=min(cfg.n_markers, F))

    print(f"\n######### {name} mode={mode} seed={seed}: N={len(y)} F={F} K={K} "
          f"(train {len(tr)}, val {len(va)}, test {len(te)}) device={device} #########",
          flush=True)
    yt, yp, model = _fit_eval(X.astype(np.float32, copy=False), y, tr, va, te,
                              cfg, F, K, device)
    return {
        "dataset": name, "mode": mode, "seed": seed,
        "n_samples": int(len(y)), "n_features": int(F), "n_classes": int(K),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "transformer_params": int(model.transformer_param_count()),
        "total_params": int(model.total_param_count()),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "beta": cfg.router_prior_beta, "knn": cfg.interaction_knn,
        "anneal": cfg.router_prior_anneal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--out", type=Path, default=Path("results_sc_interaction"))
    ap.add_argument("--datasets", nargs="*", default=["tabula_muris", "pancreas"])
    ap.add_argument("--modes", nargs="*", default=["none", "coexpr", "random"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_markers", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--anneal", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--recursion_mode", type=str, default="expert",
                    choices=["fixed", "expert", "token"])
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = RMTConfig(
        heads=(HEAD,), n_hvg=None, batch_size=args.batch_size,
        d_model=args.d_model, d_ff=2 * args.d_model, n_markers=args.n_markers,
        marker_mode="router", recursion_mode=args.recursion_mode, recursion_depth=4,
        share_weights=True, epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, device=args.device,
        router_prior_beta=args.beta, interaction_knn=args.knn,
        router_prior_anneal=args.anneal,
    )

    for name in args.datasets:
        if not (args.data / name).exists():
            print(f"[skip] {name}: not found under {args.data}", flush=True)
            continue
        for mode in args.modes:
            for s in args.seeds:
                f = args.out / f"{name}__{mode}__seed{s}.json"
                if f.exists():
                    print(f"[skip] {f.name}", flush=True)
                    continue
                rec = run_one(name, args.data, base, mode, s)
                f.write_text(json.dumps(rec, indent=1, default=float))
                print(f"[done] {f.name}  acc={rec['accuracy']*100:.2f} "
                      f"macroF1={rec['macro_f1']*100:.2f}", flush=True)

    print("\n[sc-inter] done -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
