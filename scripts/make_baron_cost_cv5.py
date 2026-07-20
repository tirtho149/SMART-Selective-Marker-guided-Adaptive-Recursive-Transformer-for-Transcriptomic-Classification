"""5-fold CV timed training on Baron for Vanilla / Recursive / MoR / bioMoR (Table-2 configs),
IN PARITY WITH TABLE 2 (unified 5-fold, seed 42, 20% test / 10%-of-train val). For each
architecture and fold it records the per-epoch validation macro-F1, training loss, and
cumulative A100 GPU-seconds; then aggregates across the 5 folds into per-epoch mean+/-SD
curves (plus mean+/-SD final test-F1). -> results/cv5/curves/baron_cost_cv5.json

Feeds the training-dynamics figures (make_baron_epoch_figs.py). bioMoR is the learned-graph
variant, matching the original curves. Run on an A100 (slurm) for consistent wall-clock.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from recursive_marker_transformer.singlecell import _load_dataset, _fit_eval, HEAD
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.cv import cv_folds, SEED, VAL_FRAC
from recursive_marker_transformer.bio_learned_genomap import _cfg as bio_cfg

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sc_cfg(mode, share):
    return RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
                     n_markers=128, marker_mode="router", recursion_mode=mode,
                     recursion_depth=4, share_weights=share, seed=SEED,
                     epochs=100, patience=15, lr=1e-3, weight_decay=1e-5)


def configs():
    return {
        "Vanilla":   sc_cfg("expert", False),   # K independent layers (no sharing)
        "Recursive": sc_cfg("fixed",  True),    # weight-shared, fixed depth
        "MoR":       sc_cfg("expert", True),    # weight-shared adaptive (MoR)
        "bioMoR":    bio_cfg("bio_both", 4, SEED, 100, n_markers=128),  # CANONICAL bioMoR (= Table 2)
    }


def aggregate(histories):
    """Per-epoch mean/SD across folds (folds that reached that epoch)."""
    maxep = max(len(h) for h in histories) if histories else 0
    agg = []
    for i in range(maxep):
        vf = [h[i]["val_f1"] for h in histories if len(h) > i]
        tl = [h[i]["train_loss"] for h in histories if len(h) > i]
        sc = [h[i]["sec"] for h in histories if len(h) > i]
        agg.append(dict(epoch=i + 1, n_folds=len(vf),
                        val_f1_mean=float(np.mean(vf)), val_f1_sd=float(np.std(vf)),
                        loss_mean=float(np.mean(tl)), loss_sd=float(np.std(tl)),
                        sec_mean=float(np.mean(sc))))   # mean cumulative A100 GPU-seconds
    return agg


def main():
    X, y, _ = _load_dataset(ROOT / "data" / "singlecell" / "baron")
    X = X.astype(np.float32); F = X.shape[1]; C = int(y.max() + 1)
    folds = list(cv_folds(y, n_folds=5, seed=SEED, val_frac=VAL_FRAC))
    out = {"_meta": {"dataset": "baron", "n_folds": 5, "seed": SEED, "device": DEV}}
    for name, cfg in configs().items():
        cfg.n_markers = min(cfg.n_markers, F)
        hists, tests = [], []
        model = None
        # seed ONCE per architecture before the fold loop -- matches run_dataset_cv /
        # run_cell_cv (which produced the Table-2 cells), so the numbers reproduce.
        torch.manual_seed(SEED); np.random.seed(SEED)
        for fi, (tr, va, te) in enumerate(folds):
            yt, yp, model = _fit_eval(X, y, tr, va, te, cfg, F, C, DEV)
            hists.append(getattr(model, "_history", []))
            tf = 100.0 * f1_score(yt, yp, average="macro"); tests.append(tf)
            print(f"[cv5-cost] {name:10s} fold {fi+1}/5 epochs={len(hists[-1]):3d} "
                  f"test_f1={tf:.1f}", flush=True)
        out[name] = dict(agg=aggregate(hists),
                         test_f1_mean=float(np.mean(tests)), test_f1_sd=float(np.std(tests)),
                         params=int(sum(p.numel() for p in model.parameters())))
        print(f"[cv5-cost] {name:10s} -> test-F1 {np.mean(tests):.1f}+/-{np.std(tests):.1f}",
              flush=True)
    d = ROOT / "results/cv5" / "curves"; d.mkdir(parents=True, exist_ok=True)
    (d / "baron_cost_cv5.json").write_text(json.dumps(out, indent=1))
    print(f"[cv5-cost] saved {d/'baron_cost_cv5.json'}", flush=True)


if __name__ == "__main__":
    main()
