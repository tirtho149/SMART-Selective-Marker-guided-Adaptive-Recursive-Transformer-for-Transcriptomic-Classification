# ============================================================================
# bioMoR -- MoR Table 14 / Figure 5 analogue: per-marker-token recursion depth.
# MoR Table 14 colours each subword token by how many recursions it received.
# bioMoR's expert-choice router assigns each MARKER TOKEN a survival depth; this
# script trains an expert-MoR model per genomap dataset and exports, over the test
# set, the mean recursion depth of each marker token and the fraction of tokens
# still active at each recursion step (Fig-5 token-count analysis).
#     python -m recursive_marker_transformer.depth_viz --datasets pancreas
# -> results_depth/<dataset>.json
# ============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import RMTConfig
from .singlecell import HEAD, _load_dataset, _make_splits, _fit_eval
from .train import _depth_stats, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--out", type=Path, default=Path("results_depth"))
    ap.add_argument("--datasets", nargs="*", default=["tabula_muris", "pancreas",
                                                      "common_class", "prototype"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        if not (args.data / ds).exists():
            print(f"[depth] skip {ds} (missing)"); continue
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        X, y, split = _load_dataset(args.data / ds)
        F, K = X.shape[1], int(y.max() + 1)
        tr, va, te = _make_splits(y, split, args.seed)
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        Xs = ((X - mu) / sd).astype(np.float32)
        cfg = RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=128, d_model=args.d_model,
                        d_ff=2 * args.d_model, n_markers=128, marker_mode="router",
                        recursion_mode="expert", recursion_depth=4, share_weights=True,
                        seed=args.seed, epochs=args.epochs, patience=12, lr=1e-3,
                        weight_decay=1e-5, device=args.device)
        print(f"\n########## depth {ds}: F={F} K={K} ##########", flush=True)
        _yt, _yp, model = _fit_eval(Xs, y, tr, va, te, cfg, F, K, device)

        # test loader for _depth_stats
        from torch.utils.data import DataLoader, TensorDataset
        ds_te = TensorDataset(torch.from_numpy(Xs[te]), torch.from_numpy(y[te]))
        loader = [(xb, {HEAD: yb}) for xb, yb in DataLoader(ds_te, batch_size=128)]
        mean_depth, marker_idx, active = _depth_stats(model, loader, device, cfg)
        md = mean_depth.numpy()
        res = {
            "dataset": ds, "recursion_depth": cfg.recursion_depth, "n_markers": int(len(md)),
            "mean_token_depth": float(md.mean()),
            "depth_histogram": np.histogram(md, bins=cfg.recursion_depth,
                                            range=(0, cfg.recursion_depth))[0].tolist(),
            "active_fraction_per_step": [float(a) / len(md) for a in active.tolist()],
            "top_token_depths": sorted(md.tolist(), reverse=True)[:10],
        }
        (args.out / f"{ds}.json").write_text(json.dumps(res, indent=1))
        print(f"  [depth] {ds}: mean_depth={md.mean():.2f}/{cfg.recursion_depth} "
              f"active/step={res['active_fraction_per_step']}", flush=True)


if __name__ == "__main__":
    main()
