"""Baron training curves for the paper: train loss vs epoch and val macro-F1 vs epoch.

Reruns ONE fold of the exact Table-2 bioMoR (learned graph, K=4) configuration on Baron
under the same 5-fold CV setup (seed 42, cv_folds fold 0), captures the per-epoch history
that the CV runner does not persist, and writes both a history JSON and figs/baron_curves.pdf.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recursive_marker_transformer.bio_learned_genomap import load_genomap, _cfg
from recursive_marker_transformer.singlecell import _fit_eval
from recursive_marker_transformer.cv import cv_folds, SEED, VAL_FRAC

ROOT = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    X, y = load_genomap("Baron")
    X = X.astype(np.float32)
    F, C = X.shape[1], int(y.max() + 1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = _cfg("learned", 4, SEED, 100, n_markers=128)
    cfg.n_markers = min(cfg.n_markers, F)

    tr, va, te = list(cv_folds(y, n_folds=5, seed=SEED, val_frac=VAL_FRAC))[0]
    print(f"[baron-curves] F={F} C={C} device={DEV} fold0 tr/va/te={len(tr)}/{len(va)}/{len(te)}", flush=True)
    yt, yp, model = _fit_eval(X, y, tr, va, te, cfg, F, C, DEV)
    hist = getattr(model, "_history", [])
    test_f1 = 100.0 * f1_score(yt, yp, average="macro")
    print(f"[baron-curves] epochs recorded={len(hist)} test macro-F1={test_f1:.2f}", flush=True)

    out_dir = ROOT / "results_cv5" / "curves"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baron_history.json").write_text(json.dumps(
        {"dataset": "Baron", "mode": "learned", "K": 4, "seed": SEED,
         "test_macro_f1": test_f1, "history": hist}, indent=1))

    ep = [h["epoch"] for h in hist]
    loss = [h["train_loss"] for h in hist]
    vf1 = [h["val_f1"] for h in hist]
    C_LOSS, C_F1 = "#0072B2", "#D55E00"
    fig, ax1 = plt.subplots(figsize=(5.0, 3.1))
    ax1.plot(ep, loss, color=C_LOSS, lw=1.8)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss", color=C_LOSS)
    ax1.tick_params(axis="y", labelcolor=C_LOSS)
    if ep:
        ax1.set_xticks(list(range(0, max(ep) + 1, 10)))
    ax1.grid(True, axis="x", ls=":", lw=0.5, alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ep, vf1, color=C_F1, lw=1.8, marker="o", markevery=10, ms=4)
    ax2.set_ylabel("validation macro-F1 (\%)".replace("\\", ""), color=C_F1)
    ax2.tick_params(axis="y", labelcolor=C_F1)
    ax1.set_title("Baron: bioMoR training curves (fold 0)", fontsize=10)
    fig.tight_layout()
    figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "baron_curves.pdf", bbox_inches="tight", dpi=600)
    print(f"[baron-curves] wrote {figs/'baron_curves.pdf'}", flush=True)


if __name__ == "__main__":
    main()
