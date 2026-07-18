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

"""Multi-seed ablations + depth/marker sweeps on the HARD genoNet tasks.

The headline ablations in ``experiments.py`` run on the (near-saturated) 4-cohort
cancer-type task, where every variant scores 97-99% and differences are within
noise. This module re-runs the ablations -- plus a recursion-depth sweep and a
marker-count (M) sweep -- on the genuinely hard phenotype labels (pathologic
stage / T / N), across multiple seeds, so we can report mean +/- std and a real
depth/compute curve.

All runs use the full gene vector (all ~20530 genes) of unified_bio5.csv. Output:
one JSON per (experiment, task, variant, seed) under results_sweeps/<exp>/, so the
job is fully RESUMABLE -- an interrupted/requeued run skips finished cells.

    python -m recursive_marker_transformer.sweeps --exp ablate depth markers
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import RMTConfig
from .genonet_tasks import _DictLoader, _load_unified
from .losses import RMTLoss
from .model import RecursiveMarkerTransformer
from .train import _class_weights, _depth_stats, evaluate, resolve_device

HARD_TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N"]

# ablation variants -> config overrides (mirrors experiments.SUITE)
ABLATIONS = {
    "main":        dict(marker_mode="router",   share_weights=True,  recursion_mode="expert"),
    "mor_token":   dict(marker_mode="router",   recursion_mode="token"),
    "fixed_depth": dict(marker_mode="router",   recursion_mode="fixed"),
    "no_refine":   dict(marker_mode="router",   recursion_mode="expert", recursive_marker_refine=False),
    "depth1":      dict(marker_mode="router",   recursion_mode="expert", recursion_depth=1),
    "independent": dict(marker_mode="router",   recursion_mode="expert", share_weights=False),
    "sel_variance":dict(marker_mode="variance", recursion_mode="fixed",  compress_mode="drop"),
    "sel_random":  dict(marker_mode="random",   recursion_mode="fixed",  compress_mode="drop"),
}
DEPTHS = [1, 2, 4, 6, 8]
MARKERS = [32, 64, 128, 256, 512]


def _flops(active, cfg):
    """Token-count-aware effective vs fixed-depth FLOPs (attention O(a^2)+FFN O(a))."""
    def step(a):
        return a * a * cfg.d_model + a * cfg.d_model * cfg.d_ff
    M = cfg.n_markers
    nominal = cfg.recursion_depth * step(M)
    eff = float(sum(step(float(active[t])) for t in range(cfg.recursion_depth)))
    return int(round(eff)), int(round(nominal)), (eff / nominal if nominal else 1.0)


def train_eval(cfg: RMTConfig, X, labels, task, seed) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(cfg.device)
    dtypes = {task: "multiclass"}

    y_raw = labels[task].values
    remap = {v: i for i, v in enumerate(np.unique(y_raw))}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    G, K = X.shape[1], int(y.max() + 1)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.15, random_state=seed, stratify=y[tr])
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    cfg = replace(cfg, heads=(task,), n_hvg=None, n_markers=min(cfg.n_markers, G))
    dl_tr = _DictLoader(Xs, y, tr, cfg.batch_size, True, task)
    dl_va = _DictLoader(Xs, y, va, cfg.batch_size, False, task)
    dl_te = _DictLoader(Xs, y, te, cfg.batch_size, False, task)

    model = RecursiveMarkerTransformer(cfg, G, {task: K}, dtypes).to(device)
    model.set_gene_variance(torch.from_numpy(Xs[tr].var(0).astype(np.float32)))
    cw = _class_weights(torch.from_numpy(y[tr]), K).to(device)
    criterion = RMTLoss(cfg, dtypes, {task: cw})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        model.set_anneal(ep / max(cfg.epochs - 1, 1))
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = {h: v.to(device) for h, v in yb.items()}
            loss = criterion(model(xb), yb)["total"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        yt, yp = evaluate(model, dl_va, device, dtypes)[task]
        vf1 = f1_score(yt, yp, average="macro")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    yt, yp = evaluate(model, dl_te, device, dtypes)[task]
    res = {"task": task, "seed": int(seed), "n_classes": int(K),
           "accuracy": float(accuracy_score(yt, yp)),
           "macro_f1": float(f1_score(yt, yp, average="macro")),
           "weighted_f1": float(f1_score(yt, yp, average="weighted")),
           "val_macro_f1": float(best_f1),
           "transformer_params": int(model.transformer_param_count()),
           "recursion_depth": int(cfg.recursion_depth), "n_markers": int(cfg.n_markers),
           "d_model": int(cfg.d_model), "lr": float(cfg.lr)}
    try:
        _msd, _mi, active = _depth_stats(model, dl_te, device, cfg)
        eff, nom, sav = _flops(active, cfg)
        res.update(flops_eff=eff, flops_nominal=nom, compute_saving_ratio=sav,
                   mean_recursion_depth=float(_msd.mean()))
    except Exception as e:
        res["depth_stats_error"] = str(e)
    return res


def _run_cell(out: Path, name, cfg, X, labels, task, seed):
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{name}.json"
    if f.exists():
        print(f"  [skip] {f.name}", flush=True)
        return
    r = train_eval(cfg, X, labels, task, seed)
    f.write_text(json.dumps(r, indent=1))
    print(f"  [done] {f.name}  macroF1={r['macro_f1']*100:.1f} acc={r['accuracy']*100:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/tcga/unified_bio5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_sweeps"))
    ap.add_argument("--exp", nargs="*", default=["ablate", "depth", "markers"])
    ap.add_argument("--tasks", nargs="*", default=HARD_TASKS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sweep_seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    print(f"[sweeps] loading {args.csv} ...", flush=True)
    X, labels, gene_cols = _load_unified(args.csv)
    print(f"[sweeps] X={X.shape} exps={args.exp} tasks={args.tasks}", flush=True)

    base = RMTConfig(
        heads=("cancer_type",), n_hvg=None, batch_size=32, d_model=128, d_ff=256,
        n_markers=256, marker_mode="router", recursion_mode="expert", recursion_depth=4,
        epochs=args.epochs, patience=args.patience, lr=3e-4, device=args.device,
    )

    if "ablate" in args.exp:
        print("\n===== multi-seed ablations =====", flush=True)
        for task in args.tasks:
            for variant, kw in ABLATIONS.items():
                for s in args.seeds:
                    cfg = replace(base, **kw)
                    _run_cell(args.out / "ablate", f"{task}__{variant}__seed{s}",
                              cfg, X, labels, task, s)

    if "depth" in args.exp:
        print("\n===== recursion-depth sweep (main config) =====", flush=True)
        for task in args.tasks:
            for k in DEPTHS:
                for s in args.sweep_seeds:
                    cfg = replace(base, recursion_depth=k)
                    _run_cell(args.out / "depth", f"{task}__K{k}__seed{s}",
                              cfg, X, labels, task, s)

    if "markers" in args.exp:
        print("\n===== marker-count (M) sweep (main config) =====", flush=True)
        for task in args.tasks:
            for m in MARKERS:
                for s in args.sweep_seeds:
                    cfg = replace(base, n_markers=m)
                    _run_cell(args.out / "markers", f"{task}__M{m}__seed{s}",
                              cfg, X, labels, task, s)

    print("\n[sweeps] all cells complete -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
