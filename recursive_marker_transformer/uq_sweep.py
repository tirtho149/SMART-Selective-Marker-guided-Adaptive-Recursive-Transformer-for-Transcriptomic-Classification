# ============================================================================
# bioMoR -- UQ sweep: train each of the six configurations and record the log-prob /
# calibration uncertainty metrics (uq.py) on the test set, so the consolidated UQ
# table can ask whether the biological prior or adaptive depth improve uncertainty
# over a vanilla transformer. Multi-seed. One array task per configuration.
#   results_uq/<config>/s<seed>/<dataset>.json
#   python -m recursive_marker_transformer.uq_sweep --config bio --seeds 0 1 2
# ============================================================================
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import RMTConfig
from .singlecell import _DictLoader, _load_dataset, _make_splits, _fit_eval, HEAD
from .uq import predict_probs, predict_logits, uq_metrics, temperature_scaled_metrics
from .train import resolve_device

# 13-dataset paper: 9 genomap single-cell sets (materialised into data/singlecell/
# by tools/materialize_sc.py so this shares arrays with the learned Table-1 sweep).
SC = ["baron", "lung", "muraro", "oesophagus", "segerstolpe",
      "spleen", "tcell", "wang", "xin"]

# config -> overrides on the shared single-cell base config
CONFIGS = {
    "bio":         dict(gene_interaction="coexpr", recursion_mode="expert"),
    "none":        dict(gene_interaction="none", recursion_mode="expert"),
    "adaptive":    dict(gene_interaction="none", recursion_mode="expert"),
    "fixed":       dict(gene_interaction="none", recursion_mode="fixed"),
    "depth1":      dict(gene_interaction="none", recursion_mode="expert", recursion_depth=1),
    "independent": dict(gene_interaction="none", recursion_mode="expert", share_weights=False),
    # data-driven LEARNED gene-gene graph (Table-1 positive) -- does it help calibration?
    "learned":     dict(gene_interaction="none", recursion_mode="expert",
                        bio_learned_graph=True, bio_learned_rank=16),
}


def _base(seed, epochs):
    return RMTConfig(
        heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
        n_markers=128, marker_mode="router", recursion_mode="expert",
        recursion_depth=4, share_weights=True, seed=seed, epochs=epochs,
        patience=12, lr=1e-3, weight_decay=1e-5, device="cuda",
        gene_interaction="none", interaction_knn=16,
        router_prior_beta=1.0, router_prior_anneal=True)


COH = {"prostate": "mut_cnv", "blca": "mut_cnv", "stad": "mut_cnv", "brca": "mut_cnv"}
# cohort overrides (Reactome prior is the biological-prior analogue of co-expression)
COH_CONFIGS = {
    "bio":         dict(gene_interaction="reactome", recursion_mode="expert"),
    "none":        dict(gene_interaction="none", recursion_mode="expert"),
    "adaptive":    dict(gene_interaction="reactome", recursion_mode="expert"),
    "fixed":       dict(gene_interaction="reactome", recursion_mode="fixed"),
    "depth1":      dict(gene_interaction="reactome", recursion_mode="expert", recursion_depth=1),
    "independent": dict(gene_interaction="reactome", recursion_mode="expert", share_weights=False),
    # learned graph replaces the fixed Reactome prior (matches the Table-1 P-NET learned run)
    "learned":     dict(gene_interaction="none", recursion_mode="expert",
                        bio_learned_graph=True, bio_learned_rank=16),
}


def _run_cohorts(config, seeds, epochs, out, device):
    from .pathway_data import load_cohort, load_pan_meta
    from .pathway_tasks import PANMETA as _PANMETA, _fit_eval as _pw_fit
    from .pathway_warmstart import _splits as _pw_splits
    over = COH_CONFIGS[config]
    for task, chan in COH.items():
        bs = 128 if chan == "expr" else 32
        if task in _PANMETA:
            cohort_dir, label = _PANMETA[task]
            coh = load_pan_meta(label=label, cohort=cohort_dir, min_genes=5)
        else:
            coh = load_cohort(task, channels=chan, min_genes=5)
        X, y = coh.X, coh.y
        G, K = X.shape[1], int(y.max() + 1)
        C = 1 if X.ndim == 2 else X.shape[2]
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            tr, va, te = _pw_splits(y, seed)
            cfg = RMTConfig(heads=(task,), n_hvg=None, n_channels=C, batch_size=bs,
                            d_model=128, d_ff=256, n_markers=256, marker_mode="pathway",
                            recursion_depth=4, share_weights=True, seed=seed, epochs=epochs,
                            patience=8, lr=3e-4, weight_decay=1e-5, device="cuda",
                            gene_interaction="reactome",
                            pathway_pool=("sum" if task == "brca" else "mean"))
            cfg = replace(cfg, **over)
            dtypes = {task: "multiclass"}
            print(f"\n##### uq[coh] {config} {task} seed={seed} G={G} K={K} #####", flush=True)
            yt, yp, model, dl_te = _pw_fit(task, coh, X, y, tr, va, te, cfg, G, K, dtypes, device)
            # rebuild a val loader with the SAME z-scoring _pw_fit used, for temp scaling
            from .pathway_tasks import _zscore_train, _DictLoader as _PWLoader
            Xs = _zscore_train(X, tr)
            dl_va = _PWLoader(Xs, y, va, bs, False, task)
            yv, logv = predict_logits(model, dl_va, device, task)
            ytt, logt = predict_logits(model, dl_te, device, task)
            T, m, m_ts = temperature_scaled_metrics(logv, yv, logt, ytt)
            outd = out / config / f"s{seed}"
            outd.mkdir(parents=True, exist_ok=True)
            (outd / f"{task}.json").write_text(json.dumps(
                {"dataset": task, "config": config, "seed": seed,
                 "n_classes": K, "n_test": int(len(te)), **m, "temperature": T,
                 "nll_ts": m_ts["nll"], "ece_ts": m_ts["ece"], "brier_ts": m_ts["brier"],
                 "auroc_ts": m_ts["auroc"]}, indent=1))
            print(f"  [uq] {task} s{seed}: NLL={m['nll']:.3f} ECE={m['ece']:.3f} "
                  f"AUROC={m['auroc']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--datasets", nargs="*", default=SC)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--out", type=Path, default=Path("results_uq"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--cohorts", action="store_true", help="also run the P-NET cohorts")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    device = resolve_device(args.device)
    over = CONFIGS[args.config]

    for ds in args.datasets:
        if not (args.data / ds).exists():
            print(f"[uq] skip {ds} (missing)"); continue
        X, y, split = _load_dataset(args.data / ds)
        F, K = X.shape[1], int(y.max() + 1)
        Xf = X.astype(np.float32, copy=False)
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            tr, va, te = _make_splits(y, split, seed)
            cfg = replace(_base(seed, args.epochs), **over)
            print(f"\n##### uq {args.config} {ds} seed={seed} F={F} K={K} #####", flush=True)
            _, _, model = _fit_eval(Xf, y, tr, va, te, cfg, F, K, device)
            # rebuild the loaders exactly as _fit_eval z-scored them, to get logits
            mu = Xf[tr].mean(0, keepdims=True); sd = Xf[tr].std(0, keepdims=True) + 1e-6
            Xs = (Xf - mu) / sd
            dl_va = _DictLoader(Xs, y, va, cfg.batch_size, False)
            dl_te = _DictLoader(Xs, y, te, cfg.batch_size, False)
            yv, logv = predict_logits(model, dl_va, device, HEAD)
            yt, logt = predict_logits(model, dl_te, device, HEAD)
            # temperature scaling: fit T on val, apply to test (accuracy/F1 unchanged)
            T, m, m_ts = temperature_scaled_metrics(logv, yv, logt, yt)
            outd = args.out / args.config / f"s{seed}"
            outd.mkdir(parents=True, exist_ok=True)
            (outd / f"{ds}.json").write_text(json.dumps(
                {"dataset": ds, "config": args.config, "seed": seed,
                 "n_classes": K, "n_test": int(len(te)), **m, "temperature": T,
                 "nll_ts": m_ts["nll"], "ece_ts": m_ts["ece"], "brier_ts": m_ts["brier"],
                 "auroc_ts": m_ts["auroc"]}, indent=1))
            print(f"  [uq] {ds} s{seed}: NLL={m['nll']:.3f}->{m_ts['nll']:.3f} "
                  f"ECE={m['ece']:.3f}->{m_ts['ece']:.3f} T={T:.2f} AUROC={m['auroc']:.3f}", flush=True)

    if args.cohorts:
        _run_cohorts(args.config, args.seeds, max(40, args.epochs // 2), args.out, device)


if __name__ == "__main__":
    main()
