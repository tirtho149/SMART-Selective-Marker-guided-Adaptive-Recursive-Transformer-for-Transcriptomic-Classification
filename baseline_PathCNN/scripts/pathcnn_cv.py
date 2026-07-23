"""bioMOR CV runner for PathCNN (pathway-informed CNN).

Faithful torch re-implementation of the upstream Keras PathCNN pipeline
(pathway_cnn.py): per-pathway PCA of each omics block -> stacked pathway
"images" -> correlation-based pathway ordering -> 2D CNN
(Conv 32 -> Conv 64 -> MaxPool(4,2) -> Dropout -> Dense 64 -> Dense out).
Re-implemented in torch so it shares the /work .venv (no Keras/TF), but the
architecture + preprocessing match the upstream model exactly.

Driven by the shared backbone:
  * data   : bc.load_omics(cohort, modalities)
  * folds  : bc.cv_folds(y)  (bioMoR seed-42 CV5)
  * pathway grouping: data/<cohort>/filtered_pathways.csv if present, else a
    generic contiguous 50-gene grouping (documented in NOTES.md).
  * PCA is FIT PER FOLD on training data only (no leakage), applied to val/test.
  * output : bc.write_scores -> work_dirs/<cohort>/scores_*.csv

Usage:
    python scripts/pathcnn_cv.py --cohort prostate
    SMOKE=1 python scripts/pathcnn_cv.py --cohort prostate
"""
import argparse
import os
import random
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(BASE)
sys.path.insert(0, REPO)
import biomor_common as bc  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_COMPONENTS = 5   # PCA comps computed per pathway (upstream default)
N_PC = 2           # comps used per omics in the pathway image (upstream n_pc=2)


def setup_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_pathway_mapping(cohort, genes):
    gene_set = set(genes)
    pw_path = os.path.join(bc.DATA, cohort, "filtered_pathways.csv")
    mapping = {}
    if os.path.exists(pw_path):
        import pandas as pd
        df = pd.read_csv(pw_path)
        cl = {c.lower(): c for c in df.columns}
        gcol = next((cl[c] for c in ("genes", "gene", "gene_list", "members") if c in cl), None)
        ncol = next((cl[c] for c in ("pathway_id", "pathway_name", "pathway", "name") if c in cl),
                    df.columns[0])
        if gcol is not None:
            for _, row in df.iterrows():
                raw = str(row[gcol]); sep = "|" if "|" in raw else ","
                members = [g.strip() for g in raw.split(sep) if g.strip() in gene_set]
                if len(members) >= 5:
                    mapping[str(row[ncol])] = members
    if not mapping:
        block = 50
        for i in range(0, len(genes), block):
            members = list(genes[i:i + block])
            if len(members) >= 5:
                mapping[f"block_{i//block}"] = members
    return mapping


def pathway_pca_images(mut, cnv, mapping, gene_to_idx, pca_models=None, fit=False):
    """Return (N, P, N_PC*2) pathway image + fitted pca models (per pathway, per omics)."""
    pathways = list(mapping)
    P = len(pathways)
    N = mut.shape[0]
    img = np.zeros((N, P, N_PC * 2), dtype=np.float32)
    models = pca_models if pca_models is not None else {}
    for pj, pw in enumerate(pathways):
        idx = [gene_to_idx[g] for g in mapping[pw] if g in gene_to_idx]
        if not idx:
            continue
        for oi, mat in enumerate((mut, cnv)):
            sub = mat[:, idx]
            key = (pw, oi)
            n_comp = min(N_COMPONENTS, len(idx), mut.shape[0] - 1)
            if n_comp < 1:
                continue
            if fit:
                sc = StandardScaler()
                subs = np.nan_to_num(sc.fit_transform(sub))
                if np.std(subs) < 1e-10:
                    models[key] = None
                    continue
                pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
                comps = pca.fit_transform(subs)
                models[key] = (sc, pca)
            else:
                m = models.get(key)
                if m is None:
                    continue
                sc, pca = m
                subs = np.nan_to_num(sc.transform(sub))
                comps = pca.transform(subs)
            k = min(N_PC, comps.shape[1])
            img[:, pj, oi * N_PC:oi * N_PC + k] = comps[:, :k]
    return img, models


def order_by_correlation(img_train):
    N, P, F = img_train.shape
    flat = img_train.transpose(1, 0, 2).reshape(P, N * F)
    corr = np.corrcoef(flat)
    corr = np.nan_to_num(corr)
    order = [0]
    remaining = list(range(1, P))
    while remaining:
        last = order[-1]
        nxt = max(range(len(remaining)), key=lambda i: corr[last, remaining[i]])
        order.append(remaining.pop(nxt))
    return order


class PathCNN(nn.Module):
    """torch port of create_pathcnn_model (Conv 32 -> Conv 64 -> MaxPool(4,2))."""
    def __init__(self, img_rows, img_cols, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2)),
            nn.Dropout(0.25),
        )
        out_r = img_rows // 4
        out_c = img_cols // 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * out_r * out_c, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImgDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,P,F)
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]


def train_fold(Xtr, ytr, Xva, yva, Xte, yte, num_classes, device, epochs):
    model = PathCNN(Xtr.shape[1], Xtr.shape[2], num_classes).to(device)
    counts = np.bincount(ytr, minlength=num_classes)
    cw = torch.tensor([counts.sum() / (num_classes * c) if c > 0 else 0.0
                       for c in counts], dtype=torch.float32).to(device)
    opt = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    from sklearn.metrics import f1_score

    tr = DataLoader(ImgDataset(Xtr, ytr), batch_size=64, shuffle=True)
    va = DataLoader(ImgDataset(Xva, yva), batch_size=64)
    te = DataLoader(ImgDataset(Xte, yte), batch_size=64)

    best_f1, best_state, patience_left = -1.0, None, 25
    for epoch in range(1, epochs + 1):
        model.train()
        for x, t in tr:
            x, t = x.to(device), t.to(device)
            opt.zero_grad(); loss_fn(model(x), t).backward(); opt.step()
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, t in va:
                vp.append(model(x.to(device)).cpu().numpy()); vt.append(t.numpy())
        f1v = f1_score(np.concatenate(vt), np.concatenate(vp).argmax(1),
                       average="macro", zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 25
        else:
            patience_left -= 1
            if epoch >= min(50, epochs) and patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for x, t in te:
            tp.append(model(x.to(device)).cpu().numpy()); tt.append(t.numpy())
    return np.concatenate(tt), np.concatenate(tp).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--modalities", nargs="+", default=["mutation", "cnv"])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    epochs = 3 if SMOKE else args.epochs
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, meta = bc.load_omics(args.cohort, modalities=tuple(args.modalities))
    dims = [meta["modality_dims"][m] for m in args.modalities]
    assert len(set(dims)) == 1, f"PathCNN expects aligned gene sets, got {dims}"
    G = dims[0]
    genes = [n.split(":", 1)[1] for n in meta["feature_names"][:G]]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    mut = X[:, :G]                 # first block
    cnv = X[:, G:2 * G]            # second block
    num_classes = int(y.max() + 1)
    mapping = build_pathway_mapping(args.cohort, genes)
    print(f"device={device} cohort={args.cohort} genes={G} pathways={len(mapping)} "
          f"classes={num_classes} epochs={epochs}")

    folds = bc.cv_folds(y)
    if SMOKE:
        folds = folds[:1]

    f1s, accs, ns = [], [], []
    for k, (tr, va, te) in enumerate(folds):
        # PCA fit on train only, applied to all.
        img_tr, models = pathway_pca_images(mut[tr], cnv[tr], mapping, gene_to_idx, fit=True)
        order = order_by_correlation(img_tr)
        img_tr = img_tr[:, order, :]
        img_va, _ = pathway_pca_images(mut[va], cnv[va], mapping, gene_to_idx, models, fit=False)
        img_te, _ = pathway_pca_images(mut[te], cnv[te], mapping, gene_to_idx, models, fit=False)
        img_va = img_va[:, order, :]
        img_te = img_te[:, order, :]
        yt, yp = train_fold(img_tr, y[tr], img_va, y[va], img_te, y[te],
                            num_classes, device, epochs)
        f1, acc = bc.fold_metrics(yt, yp)
        f1s.append(f1); accs.append(acc); ns.append(len(te))
        print(f"  fold {k+1}: macro_f1={f1:.2f}  acc={acc:.2f}  n={len(te)}")

    wd = os.path.join(BASE, "work_dirs", args.cohort)
    out = bc.write_scores(wd, "PathCNN", args.cohort, f1s, accs, ns,
                          suffix="smoke" if SMOKE else "")
    print(f"macro_f1 mean={np.mean(f1s):.2f}  ->  {out}")


if __name__ == "__main__":
    main()
