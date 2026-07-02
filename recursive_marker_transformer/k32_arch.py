# ============================================================================
# SMART -- one-off: Vanilla vs Recursive vs MoR at a FIXED depth on ONE dataset.
# Everything held fixed (dataset, d_model=96, n_markers=128, epochs, K) except the
# architecture knob, so the ONLY change between rows is independent-layers vs
# weight-shared vs adaptive-routed. Records accuracy, macro-F1, transformer/total
# params, and (for MoR) mean token depth + active fraction + recursion FLOPs.
#     python -m recursive_marker_transformer.k32_arch --dataset pancreas --K 32 --seeds 0 1
# -> results_k32_arch/<dataset>/<arch>_s<seed>.json
# ============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .config import RMTConfig
from .singlecell import HEAD, _load_dataset, _make_splits, _fit_eval
from .train import _depth_stats, resolve_device

ROOT = Path(__file__).resolve().parents[1]

# arch name -> the two knobs that define it (all else identical)
ARCHS = {
    "vanilla":   dict(share_weights=False, recursion_mode="fixed"),   # K independent blocks
    "recursive": dict(share_weights=True,  recursion_mode="fixed"),   # 1 shared block x K
    "mor":       dict(share_weights=True,  recursion_mode="expert"),  # adaptive expert-choice
}


def _flops_one_block(a, d, d_ff):
    """FLOPs of one shared-block application to `a` active tokens (paper Eq. phi)."""
    return 4.0 * a * a * d + 4.0 * a * d * d_ff


def run_arch(ds, arch, K, seed, epochs, data_dir, device):
    X, y, split = _load_dataset(data_dir / ds)
    F, C = X.shape[1], int(y.max() + 1)
    Xf = X.astype(np.float32, copy=False)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = _make_splits(y, split, seed)
    cfg = RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
                    n_markers=128, marker_mode="router", recursion_depth=K,
                    seed=seed, epochs=epochs, patience=12, lr=1e-3, weight_decay=1e-5,
                    device="cuda", **ARCHS[arch])
    yt, yp, model = _fit_eval(Xf, y, tr, va, te, cfg, F, C, device)
    test_f1 = 100 * f1_score(yt, yp, average="macro")
    test_acc = 100 * accuracy_score(yt, yp)

    M, d, d_ff = cfg.n_markers, cfg.d_model, cfg.d_ff
    out = {
        "dataset": ds, "arch": arch, "K": K, "seed": seed,
        "test_macro_f1": test_f1, "test_accuracy": test_acc,
        "transformer_params": model.transformer_param_count(),
        "total_params": model.total_param_count(),
        "n_markers": M, "d_model": d, "d_ff": d_ff,
        "flops_nominal": K * _flops_one_block(M, d, d_ff),   # all M tokens, K steps
    }

    if arch == "mor":
        # mean token depth + active fraction per step over the TEST set
        mu = Xf[tr].mean(0, keepdims=True); sd = Xf[tr].std(0, keepdims=True) + 1e-6
        Xs = ((Xf - mu) / sd).astype(np.float32)
        ds_te = TensorDataset(torch.from_numpy(Xs[te]), torch.from_numpy(y[te]))
        loader = [(xb, {HEAD: yb}) for xb, yb in DataLoader(ds_te, batch_size=128)]
        mean_depth, _idx, active = _depth_stats(model, loader, device, cfg)
        md = mean_depth.numpy()
        frac = [float(a) / len(md) for a in active.tolist()]
        out["mean_token_depth"] = float(md.mean())
        out["active_fraction_per_step"] = frac
        out["flops_effective"] = float(sum(_flops_one_block(f * M, d, d_ff) for f in frac))
        out["compute_saving"] = 1.0 - out["flops_effective"] / out["flops_nominal"]
    else:
        out["flops_effective"] = out["flops_nominal"]
        out["compute_saving"] = 0.0
    return out


SC = ["tabula_muris", "pancreas", "common_class", "prototype", "baron",
      "segerstolpe", "lung", "oesophagus", "spleen", "tcell"]
# P-NET / Reactome cohorts -> input channel(s). Same setup as depth_sweep._one_coh.
COH = {"prostate": "mut_cnv", "blca": "mut_cnv", "stad": "mut_cnv", "panmeta_subtype": "expr"}


def run_arch_coh(task, arch, K, seed, epochs, device):
    """One P-NET/Reactome cohort run for a given arch (Reactome pathway tokens)."""
    from .pathway_data import load_cohort, load_pan_meta
    from .pathway_tasks import PANMETA, _fit_eval as pw_fit
    from .pathway_warmstart import _splits as pw_splits
    from .router import default_capacity
    chan = COH[task]
    bs = 128 if chan == "expr" else 32
    if task in PANMETA:
        cohort_dir, label = PANMETA[task]
        coh = load_pan_meta(label=label, cohort=cohort_dir, min_genes=5)
    else:
        coh = load_cohort(task, channels=chan, min_genes=5)
    X, y = coh.X, coh.y
    G, C = X.shape[1], (1 if X.ndim == 2 else X.shape[2])
    Kc = int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = pw_splits(y, seed)
    cfg = RMTConfig(heads=(task,), n_hvg=None, n_channels=C, batch_size=bs, d_model=128,
                    d_ff=256, n_markers=256, marker_mode="pathway", recursion_depth=K,
                    seed=seed, epochs=epochs, patience=8, lr=3e-4, weight_decay=1e-5,
                    device="cuda", gene_interaction="reactome",
                    pathway_pool=("sum" if task == "brca" else "mean"), **ARCHS[arch])
    dtypes = {task: "multiclass"}
    yt, yp, model, _dl = pw_fit(task, coh, X, y, tr, va, te, cfg, G, Kc, dtypes, device)
    test_f1 = 100 * f1_score(yt, yp, average="macro")
    test_acc = 100 * accuracy_score(yt, yp)
    M, d, d_ff = cfg.n_markers, cfg.d_model, cfg.d_ff
    out = {"dataset": task, "arch": arch, "K": K, "seed": seed,
           "test_macro_f1": test_f1, "test_accuracy": test_acc,
           "transformer_params": model.transformer_param_count(),
           "total_params": model.total_param_count(),
           "n_markers": M, "d_model": d, "d_ff": d_ff,
           "flops_nominal": K * _flops_one_block(M, d, d_ff)}
    if arch == "mor":
        # expert-choice funnel capacity is deterministic -> effective FLOPs analytic
        frac = list(default_capacity(K))
        out["mean_token_depth"] = float(sum(frac))
        out["active_fraction_per_step"] = frac
        out["flops_effective"] = float(sum(_flops_one_block(f * M, d, d_ff) for f in frac))
        out["compute_saving"] = 1.0 - out["flops_effective"] / out["flops_nominal"]
    else:
        out["flops_effective"] = out["flops_nominal"]
        out["compute_saving"] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=SC + list(COH))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    for ds in args.datasets:
        is_coh = ds in COH
        if not is_coh and not (args.data / ds).exists():
            print(f"[k{args.K}] skip {ds} (missing)", flush=True); continue
        out = ROOT / f"results_k{args.K}_arch" / ds
        out.mkdir(parents=True, exist_ok=True)
        for arch in args.archs:
            for seed in args.seeds:
                p = out / f"{arch}_s{seed}.json"
                if p.exists() and not args.force:
                    print(f"[k{args.K}] skip {ds}/{arch} s{seed} (exists)", flush=True); continue
                print(f"\n##### k{args.K} {ds} arch={arch} K={args.K} seed={seed} #####", flush=True)
                r = (run_arch_coh(ds, arch, args.K, seed, args.epochs, device) if is_coh
                     else run_arch(ds, arch, args.K, seed, args.epochs, args.data, device))
                p.write_text(json.dumps(r, indent=1))
                print(f"  [k{args.K}] {ds}/{arch} s{seed}: F1={r['test_macro_f1']:.1f} "
                      f"acc={r['test_accuracy']:.1f} params={r['transformer_params']} "
                      f"save={r['compute_saving']*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
