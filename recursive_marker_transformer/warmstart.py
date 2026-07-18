# ============================================================================
# bioMoR -- MoR Table 9 analogue: WARM-START (uptraining).
# MoR's Table 9 uptrains a pretrained *vanilla* model into a recursive one. bioMoR
# has no pretrained-LLM checkpoint, so the faithful set-classifier analogue is:
# train a fixed-depth (Recursive) bioMoR, then INITIALISE the shared recursive block
# of an expert-choice MoR bioMoR from those weights and continue-train -- vs training
# the MoR model from scratch. Reports both so the uptraining gain is visible.
#     python -m recursive_marker_transformer.warmstart --datasets tabula_muris pancreas
# ============================================================================
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


def _run(Xs, y, tr, va, te, cfg, F, K, device, init_block=None):
    """One train/eval; optionally warm-start the shared block from init_block."""
    # _fit_eval builds + trains its own model; to inject weights we monkey-patch
    # via a closure is awkward, so re-implement the minimal build here is avoided
    # by training through _fit_eval after seeding the shared block. Simplest: run
    # _fit_eval (returns the trained model) and, for warm-start, we instead build
    # through the same path but preload. We approximate by passing init via cfg-free
    # state injection: train normally, but if init_block given, we first load it.
    from .model import RecursiveMarkerTransformer
    from .losses import RMTLoss
    from .train import _class_weights, evaluate
    from torch.utils.data import DataLoader, TensorDataset

    model = RecursiveMarkerTransformer(cfg, F, {HEAD: K}, {HEAD: "multiclass"}).to(device)
    if init_block is not None:
        model.stack.blocks[0].load_state_dict(init_block)             # warm-start
    model.set_gene_variance(torch.from_numpy(Xs[tr].var(0).astype(np.float32)))
    cw = _class_weights(torch.from_numpy(y[tr]), K).to(device)
    crit = RMTLoss(cfg, {HEAD: "multiclass"}, {HEAD: cw})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def loader(idx, shuf):
        ds = TensorDataset(torch.from_numpy(Xs[idx]), torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuf)

    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train(); model.set_anneal(ep / max(cfg.epochs - 1, 1))
        for xb, yb in loader(tr, True):
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), {HEAD: yb})["total"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yt, yp = evaluate(model, [(xb, {HEAD: yb}) for xb, yb in loader(va, False)],
                          device, {HEAD: "multiclass"})[HEAD]
        f1 = f1_score(yt, yp, average="macro")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    yt, yp = evaluate(model, [(xb, {HEAD: yb}) for xb, yb in loader(te, False)],
                      device, {HEAD: "multiclass"})[HEAD]
    return f1_score(yt, yp, average="macro"), accuracy_score(yt, yp), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/singlecell"))
    ap.add_argument("--out", type=Path, default=Path("results_warmstart"))
    ap.add_argument("--datasets", nargs="*", default=["tabula_muris", "pancreas",
                                                      "common_class", "prototype"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = resolve_device(args.device)

    common = dict(heads=(HEAD,), n_hvg=None, batch_size=128, d_model=args.d_model,
                  d_ff=2 * args.d_model, n_markers=128, marker_mode="router",
                  recursion_depth=4, share_weights=True, seed=args.seed,
                  epochs=args.epochs, patience=12, lr=1e-3, weight_decay=1e-5,
                  device=args.device)
    fixed_cfg = RMTConfig(recursion_mode="fixed", **common)
    mor_cfg = RMTConfig(recursion_mode="expert", **common)

    args.out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        if not (args.data / ds).exists():
            print(f"[warmstart] skip {ds} (missing)"); continue
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        X, y, split = _load_dataset(args.data / ds)
        F, K = X.shape[1], int(y.max() + 1)
        tr, va, te = _make_splits(y, split, args.seed)
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        Xs = ((X - mu) / sd).astype(np.float32)
        print(f"\n########## warmstart {ds}: F={F} K={K} ##########", flush=True)

        f1_fixed, _, fixed_model = _run(Xs, y, tr, va, te, fixed_cfg, F, K, device)
        init = {k: v.detach().cpu().clone()
                for k, v in fixed_model.stack.blocks[0].state_dict().items()}
        f1_scratch, _, _ = _run(Xs, y, tr, va, te, mor_cfg, F, K, device)
        f1_warm, _, _ = _run(Xs, y, tr, va, te, mor_cfg, F, K, device, init_block=init)

        res = {"dataset": ds, "fixed_source_macro_f1": 100 * f1_fixed,
               "mor_from_scratch_macro_f1": 100 * f1_scratch,
               "mor_warm_start_macro_f1": 100 * f1_warm,
               "warm_start_gain": 100 * (f1_warm - f1_scratch)}
        (args.out / f"{ds}.json").write_text(json.dumps(res, indent=1))
        print(f"  [warmstart] {ds}: fixed={100*f1_fixed:.1f} scratch={100*f1_scratch:.1f} "
              f"warm={100*f1_warm:.1f} gain={100*(f1_warm-f1_scratch):+.1f}", flush=True)


if __name__ == "__main__":
    main()
