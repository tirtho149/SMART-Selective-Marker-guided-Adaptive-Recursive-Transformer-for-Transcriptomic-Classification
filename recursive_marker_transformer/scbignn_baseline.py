# ============================================================================
# bioMoR -- external baseline: scBiGNN (faithful re-implementation).
#
# NEW FILE. Does not modify any existing module, script, or the paper. Reuses the
# repo's data loaders (load_genomap / load_cohort / load_pan_meta) and the shared
# 5-fold CV protocol (cv.cv_folds) verbatim, so the numbers land on byte-identical
# folds as the bioMoR Table 3 rows.
# ============================================================================
"""scBiGNN -- Bilevel Graph Representation Learning for Cell Type Classification.

Faithful re-implementation of:
    Yang, Fang, Zhang, Sun, Chawla, Xu, Wu.
    "scBiGNN: Bilevel Graph Representation Learning for Cell Type
    Classification." arXiv:2312.10310 (2023).

Architecture (two cooperating branches on the SAME PCA-reduced expression):
  * GNN branch  -- a cell-similarity KNN graph (k~=10-15) is built from the
    PCA-reduced features (cosine similarity, symmetrised, self-loops, GCN
    normalisation D^-1/2 (A+I) D^-1/2). A 2-layer GCN classifies each cell using
    its neighbourhood, i.e. it exploits cell-cell relational structure -- the
    mechanism scBiGNN adds over a plain classifier.
  * Feature (MLP) branch -- a 2-layer MLP classifies each cell from its own
    PCA feature vector alone (no graph).

Bilevel / EM-style coupling (mutual distillation): the two branches are trained
jointly. Each branch minimises cross-entropy against the labels PLUS a symmetric
KL term that pulls its softmax output toward the *detached* softmax of the other
branch. This is the practical, differentiable surrogate of scBiGNN's bilevel
EM optimisation where each branch's predictions act as soft targets ("teacher")
for the other. At test time the two branches are ENSEMBLED (mean of the softmax
probabilities) for the final prediction.

Full features are used (PCA to ~50-256 dims), NOT the 128-marker compression --
scBiGNN's contribution is the graph over cells, so we keep the input intact.

Protocol: identical shared folds via ``cv.cv_folds(y, n_folds=5, seed=SEED,
val_frac=VAL_FRAC)``; per fold: fit on train, early-stop on val macro-F1
(patience ~15, epochs ~100), score test macro-F1. Emits the standard bioMoR CV
JSON to ``results/cv5/scbignn/<cohort>.json``.

Usage:
    python -m recursive_marker_transformer.scbignn_baseline \
        --cohort Baron --epochs 100 --seed 42 --device cuda \
        --out results/cv5/scbignn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .cv import cv_folds, summarize, SEED, VAL_FRAC

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Cohort loading -- reuse the EXACT repo loaders so features/labels/folds match
# the bioMoR Table 3 rows.
#   * single-cell  (Baron, Muraro): bio_learned_genomap.load_genomap
#   * multi-omics  (pan_meta_pri = PM, pan_meta_pri_3modal = 3M):
#       pathway_data.load_cohort with the same channel sets used in Table 3
#       (PM -> mut_cnv, 3M -> mut_cnv_expr). Multi-channel (N,G,C) is flattened
#       to (N, G*C) -- the modality channels are concatenated, per the loaders.
# ---------------------------------------------------------------------------
# genomap single-cell suites accepted via load_genomap (Table 3 uses Lung);
# load_cohort_xy matches baselines11.py's loaders exactly so X/y/folds are identical.
SC_COHORTS = {"Baron", "Lung", "Muraro", "Oesophagus", "Segerstolpe", "Spleen", "Tcell", "Xin"}
MO_CHANNELS = {"pan_meta_pri": "mut_cnv", "pan_meta_pri_3modal": "mut_cnv_expr"}
COHORTS = sorted(SC_COHORTS) + sorted(MO_CHANNELS)


def load_cohort_xy(cohort: str):
    """Return (X float32 [N, F], y int64 [N]) for one of the 4 Table-3 cohorts."""
    if cohort in SC_COHORTS:
        from . import bio_learned_genomap as glm
        # The genomap suite was archived after Table 3; if the default root path
        # is gone, fall back to the archived copy so load_genomap resolves the
        # SAME .mat files (identical X, y) without editing that module.
        if not glm.GD.exists():
            alt = ROOT / "archive" / "misc" / "genomap_data"
            if alt.exists():
                glm.GD = alt
        X, y = glm.load_genomap(cohort)
        return X.astype(np.float32), y.astype(np.int64)
    if cohort in MO_CHANNELS:
        from .pathway_data import load_cohort
        coh = load_cohort(cohort, channels=MO_CHANNELS[cohort])
        X = coh.X
        if X.ndim == 3:                       # (N, G, C) -> concat channels (N, G*C)
            X = X.reshape(X.shape[0], -1)
        return X.astype(np.float32), coh.y.astype(np.int64)
    raise ValueError(f"unknown cohort {cohort!r}; choices = {COHORTS}")


# ---------------------------------------------------------------------------
# Graph construction: cosine KNN graph over PCA-reduced cells, GCN-normalised.
# ---------------------------------------------------------------------------
def build_knn_adjacency(Z: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """SPARSE GCN-normalised adjacency D^-1/2 (A + I) D^-1/2 from a cosine-KNN graph
    over rows of Z. A is symmetric (either-direction KNN) with self-loops. The KNN is
    computed in row-chunks so memory is O(N k + N*chunk), not O(N^2) -- this scales to
    tens of thousands of cells (a dense N x N graph OOMs on large single-cell suites)."""
    N = Z.shape[0]
    Zn = F.normalize(Z, dim=1)
    kk = min(k, N - 1)
    rows, cols = [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = Zn[s:e] @ Zn.t()                                     # (e-s, N)
        loc = torch.arange(e - s, device=Z.device)
        sim[loc, torch.arange(s, e, device=Z.device)] = -2.0      # exclude self
        idx = sim.topk(kk, dim=1).indices                          # (e-s, kk)
        rows.append((loc + s).unsqueeze(1).expand(-1, kk).reshape(-1))
        cols.append(idx.reshape(-1))
        del sim
    r = torch.cat(rows); c = torch.cat(cols)
    diag = torch.arange(N, device=Z.device)
    ri = torch.cat([r, c, diag])                                   # symmetrise + self-loops
    ci = torch.cat([c, r, diag])
    idx2 = torch.stack([ri, ci])
    A = torch.sparse_coo_tensor(idx2, torch.ones(idx2.shape[1], device=Z.device),
                                (N, N)).coalesce()
    vals = A.values().clamp(max=1.0)                               # binarise (dedup either-dir)
    deg = torch.zeros(N, device=Z.device).scatter_add_(0, A.indices()[0], vals).clamp(min=1e-8)
    dinv = deg.pow(-0.5)
    ii = A.indices()
    v = vals * dinv[ii[0]] * dinv[ii[1]]                           # D^-1/2 A D^-1/2
    return torch.sparse_coo_tensor(ii, v, (N, N)).coalesce()


# ---------------------------------------------------------------------------
# Branches.
# ---------------------------------------------------------------------------
class GCN(nn.Module):
    """2-layer GCN classifier over a fixed normalised adjacency."""
    def __init__(self, d_in: int, d_hid: int, n_cls: int, dropout: float = 0.3):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, n_cls)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj is a sparse (N,N) normalised adjacency; sparse.mm keeps memory O(nnz)
        h = torch.relu(self.w1(torch.sparse.mm(adj, x)))
        h = self.drop(h)
        return self.w2(torch.sparse.mm(adj, h))


class MLP(nn.Module):
    """2-layer MLP classifier on the cell's own features (feature branch)."""
    def __init__(self, d_in: int, d_hid: int, n_cls: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid, n_cls))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class scBiGNN(nn.Module):
    """Two-branch bilevel model: GCN branch + MLP branch, mutually distilled."""
    def __init__(self, d_in: int, n_cls: int, d_hid: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gcn = GCN(d_in, d_hid, n_cls, dropout)
        self.mlp = MLP(d_in, d_hid, n_cls, dropout)

    def forward(self, x, adj):
        return self.gcn(x, adj), self.mlp(x)


def _kl(student_logits, teacher_logits):
    """KL(teacher_soft || student_soft) with the teacher detached (distillation)."""
    p_t = F.softmax(teacher_logits.detach(), dim=1)
    logp_s = F.log_softmax(student_logits, dim=1)
    return F.kl_div(logp_s, p_t, reduction="batchmean")


# ---------------------------------------------------------------------------
# Fold-level fit / eval (transductive: the graph spans all cells in the fold;
# the loss is masked to the train cells, early-stop on val, score on test).
# ---------------------------------------------------------------------------
def fit_eval_fold(X, y, tr, va, te, n_cls, device,
                  pca_dim=128, k=15, d_hid=128, epochs=100, patience=15,
                  lr=1e-3, weight_decay=5e-4, distill_lambda=1.0, seed=SEED):
    """Train scBiGNN on one CV fold. PCA + scaler are fit on TRAIN ONLY (no test
    leakage); the KNN graph is built over all fold cells in that PCA space.
    Returns (y_test, y_pred_ensemble)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- feature reduction: fit on train rows only, apply to all ---
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)
    pdim = int(min(pca_dim, Xs.shape[1], len(tr) - 1))
    if pdim >= 2 and pdim < Xs.shape[1]:
        pca = PCA(n_components=pdim, random_state=seed).fit(Xs[tr])
        Z = pca.transform(Xs).astype(np.float32)
    else:
        Z = Xs
    Zt = torch.from_numpy(Z).to(device)
    yt = torch.from_numpy(y).long().to(device)

    adj = build_knn_adjacency(Zt, k)

    tr_t = torch.from_numpy(np.asarray(tr)).long().to(device)
    va_t = torch.from_numpy(np.asarray(va)).long().to(device)
    te_t = torch.from_numpy(np.asarray(te)).long().to(device)

    # class weights (balanced) computed on train labels
    cls, cnt = np.unique(y[tr], return_counts=True)
    w = np.ones(n_cls, dtype=np.float32)
    w[cls] = (len(tr) / (len(cls) * cnt)).astype(np.float32)
    cw = torch.from_numpy(w).to(device)

    model = scBiGNN(Z.shape[1], n_cls, d_hid=d_hid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss(weight=cw)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        g_logit, m_logit = model(Zt, adj)
        # each branch: CE against labels + mutual KL toward the OTHER (detached)
        loss = (ce(g_logit[tr_t], yt[tr_t]) + ce(m_logit[tr_t], yt[tr_t])
                + distill_lambda * _kl(g_logit[tr_t], m_logit[tr_t])
                + distill_lambda * _kl(m_logit[tr_t], g_logit[tr_t]))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            g_logit, m_logit = model(Zt, adj)
            prob = 0.5 * (F.softmax(g_logit, 1) + F.softmax(m_logit, 1))  # ensemble
            vp = prob[va_t].argmax(1).cpu().numpy()
            vf1 = f1_score(y[va], vp, average="macro")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        g_logit, m_logit = model(Zt, adj)
        prob = 0.5 * (F.softmax(g_logit, 1) + F.softmax(m_logit, 1))
        yp = prob[te_t].argmax(1).cpu().numpy()
    return y[te], yp


def run_cv(cohort, device, epochs=100, patience=15, seed=SEED, n_folds=5,
           pca_dim=128, k=15, d_hid=128, subset=None):
    """5-fold CV on the shared folds (cv.cv_folds). Returns the result dict."""
    X, y = load_cohort_xy(cohort)
    if subset is not None and subset < len(y):        # smoke-test speed only
        rng = np.random.RandomState(seed)
        # stratified-ish subset: keep it simple, random subset with all classes if possible
        idx = rng.permutation(len(y))[:subset]
        X, y = X[idx], y[idx]
        # re-encode labels to be contiguous
        uniq = np.unique(y)
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.array([remap[v] for v in y], dtype=np.int64)
    n_cls = int(y.max() + 1)
    F_ = X.shape[1]
    print(f"[scBiGNN] {cohort} N={len(y)} F={F_} C={n_cls} "
          f"pca={pca_dim} k={k} epochs={epochs} device={device}", flush=True)

    fold_f1, fold_acc, per_fold = [], [], []
    folds = cv_folds(y, n_folds=n_folds, seed=seed, val_frac=VAL_FRAC)
    for fi, (tr, va, te) in enumerate(folds):
        y_true, y_pred = fit_eval_fold(
            X, y, tr, va, te, n_cls, device,
            pca_dim=pca_dim, k=k, d_hid=d_hid, epochs=epochs,
            patience=patience, seed=seed)
        f1 = 100.0 * f1_score(y_true, y_pred, average="macro")
        acc = 100.0 * accuracy_score(y_true, y_pred)
        fold_f1.append(f1); fold_acc.append(acc)
        per_fold.append({"fold": fi, "macro_f1": f1, "accuracy": acc,
                         "n_test": int(len(te))})
        print(f"  fold {fi+1}/{n_folds}: macroF1={f1:.2f} acc={acc:.2f} (test {len(te)})",
              flush=True)

    return {
        "cohort": cohort, "model": "scBiGNN", "n_folds": n_folds, "seed": seed,
        "n_samples": int(len(y)), "n_features": int(F_), "n_classes": n_cls,
        "val_frac": VAL_FRAC, "pca_dim": int(min(pca_dim, F_)), "knn_k": k,
        "arxiv": "2312.10310",
        "cv_macro_f1": {"mean": summarize(fold_f1)["mean"], "sd": summarize(fold_f1)["std"]},
        "cv_accuracy": {"mean": summarize(fold_acc)["mean"], "sd": summarize(fold_acc)["std"]},
        "per_fold": per_fold,
    }


def main():
    ap = argparse.ArgumentParser(description="scBiGNN baseline (arXiv:2312.10310)")
    ap.add_argument("--cohort", required=True, choices=COHORTS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--knn_k", type=int, default=15)
    ap.add_argument("--d_hid", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "cv5" / "scbignn")
    ap.add_argument("--subset", type=int, default=None,
                    help="smoke-test only: random subsample of cells/samples")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from .train import resolve_device
    device = resolve_device(args.device)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.cohort}.json"
    if out_path.exists() and not args.force and args.subset is None:
        print(f"[scBiGNN] [skip] {out_path} exists", flush=True)
        return

    res = run_cv(args.cohort, device, epochs=args.epochs, patience=args.patience,
                 seed=args.seed, n_folds=args.n_folds, pca_dim=args.pca_dim,
                 k=args.knn_k, d_hid=args.d_hid, subset=args.subset)
    out_path.write_text(json.dumps(res, indent=1))
    print(f"[scBiGNN] {args.cohort} macroF1={res['cv_macro_f1']['mean']:.2f}"
          f"+/-{res['cv_macro_f1']['sd']:.2f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
