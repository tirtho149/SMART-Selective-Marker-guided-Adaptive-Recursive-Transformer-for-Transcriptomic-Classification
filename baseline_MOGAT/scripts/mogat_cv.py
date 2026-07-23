#!/usr/bin/env python3
"""MOGAT runner on the bioMOR shared contract.

Reuses the UPSTREAM MOGAT GAT (lib.module2.Net) verbatim and the training design
of mogat_3mod_multiclass.py (one GAT embedding per modality similarity-graph,
concat embeddings + raw features -> MLP integration head), but:
  * loads cohorts + IDENTICAL seed-42 5-fold splits from biomor_common (bc),
  * supports 2 modalities (mutation+cnv) or 3 (+expression), binary or multiclass,
  * scores TEST fold with bc.fold_metrics and writes bc.write_scores into
    work_dirs/<cohort>/.

Env:
  COHORT      cohort name (default prostate)
  MODALITIES  comma list (default "mutation,cnv")
  SMOKE=1     1 fold, few epochs, few HP trials

Usage:
  COHORT=prostate SMOKE=1 python scripts/mogat_cv.py
"""
import os
import sys
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.data import Data
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)               # baseline_MOGAT
REPO = os.path.dirname(BASE)               # repo root
for p in (REPO, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import biomor_common as bc                 # noqa: E402
from lib import module2                    # noqa: E402  (upstream GAT)

SEED = 42
TOP_K = 5
GAT_LR = 1e-4
GAT_HID = 512
ADD_RAW_FEAT = True


def setup_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def split_views(X, meta, modalities):
    views, off = {}, 0
    for m in modalities:
        d = meta["modality_dims"][m]
        views[m] = X[:, off:off + d].astype(np.float32)
        off += d
    return views


# --- similarity graphs (transductive, over all patients) -------------------
def edges_from_similarity(sim, top_k=TOP_K):
    n = sim.shape[0]
    src, dst, w = [], [], []
    for i in range(n):
        s = sim[i].copy()
        s[i] = -np.inf
        for j in np.argsort(s)[-top_k:]:
            if s[j] > 0:
                src.append(i); dst.append(j); w.append(s[j])
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(w, dtype=torch.float32))


def jaccard_matrix(binary):
    b = (binary != 0)
    n = b.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        inter = (b[i] & b).sum(1)
        union = (b[i] | b).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.where(union > 0, inter / union, 0.0)
    return out


def build_graphs(views_norm, modalities):
    graphs = {}
    for m in modalities:
        X = views_norm[m]
        sim = jaccard_matrix(X) if m == "mutation" else np.nan_to_num(np.corrcoef(X), nan=0.0)
        graphs[m] = edges_from_similarity(sim)
    return graphs


# --- GAT embedding ---------------------------------------------------------
def train_gat_embedding(node_x, edge_index, edge_attr, y, tr_mask, va_mask,
                        out_size, device, max_epochs, min_epochs, patience):
    setup_seed()
    data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(device)
    data.train_mask = tr_mask
    data.valid_mask = va_mask
    model = module2.Net(in_size=node_x.shape[1], hid_size=GAT_HID, out_size=out_size).to(device)
    opt = Adam(model.parameters(), lr=GAT_LR)
    crit = nn.CrossEntropyLoss()
    best, wait, sel = np.inf, 0, None
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out, emb = model(data)
        loss = crit(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            out, emb = model(data)
            v = crit(out[data.valid_mask], data.y[data.valid_mask]).item()
        if v < best:
            best, wait, sel = v, 0, emb.detach().clone()
        else:
            wait += 1
        if epoch >= min_epochs and wait >= patience:
            break
    del model, opt, data
    clear_mem()
    return sel


# --- MLP integration head --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_size, hidden, num_classes, dropout=0.2):
        super().__init__()
        layers, prev = [], in_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(model, Xtr, ytr, Xva, yva, lr, device, max_epochs, patience):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    bs = min(16, Xtr.shape[0])
    nb = (Xtr.shape[0] + bs - 1) // bs
    best, wait, best_state = np.inf, 0, None
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(Xva), yva).item()
        if v < best:
            best, wait = v, 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def hp_search(Xtr, ytr, Xva, yva, num_classes, device, n_trials):
    setup_seed()
    hiddens = [[16], [32], [64], [128], [256], [512], [32, 32], [64, 32], [128, 64], [256, 128]]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    drops = [0.3, 0.5, 0.7]
    yva_np = yva.cpu().numpy()
    best_f1, best = -1.0, {"hidden": [128], "lr": 1e-3, "dropout": 0.5}
    for _ in range(n_trials):
        h = hiddens[np.random.randint(len(hiddens))]
        lr = lrs[np.random.randint(len(lrs))]
        dr = drops[np.random.randint(len(drops))]
        m = MLP(Xtr.shape[1], h, num_classes, dr).to(device)
        m = train_mlp(m, Xtr, ytr, Xva, yva, lr, device, max_epochs=100, patience=15)
        m.eval()
        with torch.no_grad():
            pred = m(Xva).argmax(1).cpu().numpy()
        f1 = f1_score(yva_np, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best = f1, {"hidden": h, "lr": lr, "dropout": dr}
        del m
        clear_mem()
    print(f"  best val macro-F1={best_f1:.3f} params={best}", flush=True)
    return best


def main():
    cohort = os.environ.get("COHORT", "prostate")
    modalities = tuple(os.environ.get("MODALITIES", "mutation,cnv").split(","))
    smoke = os.environ.get("SMOKE", "0") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed()

    gat_max = 4 if smoke else 200
    gat_min = 1 if smoke else 50
    gat_pat = 25
    hp_trials = 2 if smoke else 20
    mlp_max = 20 if smoke else 500
    mlp_pat = 50

    X, y, meta = bc.load_omics(cohort, modalities=modalities)
    num_classes = len(meta["classes"])
    views = split_views(X, meta, modalities)
    print(f"[MOGAT] cohort={cohort} N={len(y)} classes={num_classes} "
          f"mods={modalities} device={device}", flush=True)

    folds = bc.cv_folds(y)
    if smoke:
        folds = folds[:1]

    f1s, accs, ntests = [], [], []
    for k, (tr, va, te) in enumerate(folds, 1):
        print(f"\n===== fold {k}/{len(folds)} =====", flush=True)
        clear_mem()
        # Per-fold standardize cnv/expr on train rows; mutation kept binary.
        vn = {}
        for m in modalities:
            Xm = views[m].astype(np.float32)
            if m == "mutation":
                vn[m] = Xm
            else:
                mu = Xm[tr].mean(0, keepdims=True)
                sd = Xm[tr].std(0, keepdims=True) + 1e-8
                vn[m] = ((Xm - mu) / sd).astype(np.float32)
        graphs = build_graphs(vn, modalities)

        node_x = torch.tensor(np.concatenate([vn[m] for m in modalities], 1),
                              dtype=torch.float32).to(device)
        y_gpu = torch.tensor(y, dtype=torch.long).to(device)
        n = len(y)
        tr_mask = torch.zeros(n, dtype=torch.bool, device=device)
        va_mask = torch.zeros(n, dtype=torch.bool, device=device)
        tr_mask[torch.tensor(tr, dtype=torch.long)] = True
        va_mask[torch.tensor(va, dtype=torch.long)] = True

        embs = []
        for m in modalities:
            ei, ea = graphs[m]
            emb = train_gat_embedding(node_x, ei.to(device), ea.to(device), y_gpu,
                                      tr_mask, va_mask, num_classes, device,
                                      gat_max, gat_min, gat_pat)
            embs.append(emb)

        integrated = torch.cat(embs, dim=1)
        if ADD_RAW_FEAT:
            integrated = torch.cat([integrated, node_x], dim=1)

        tri = torch.tensor(tr, dtype=torch.long, device=device)
        vai = torch.tensor(va, dtype=torch.long, device=device)
        tei = torch.tensor(te, dtype=torch.long, device=device)
        Xtr, Xva, Xte = integrated[tri], integrated[vai], integrated[tei]
        ytr, yva = y_gpu[tri], y_gpu[vai]

        best = hp_search(Xtr, ytr, Xva, yva, num_classes, device, hp_trials)
        model = MLP(Xtr.shape[1], best["hidden"], num_classes, best["dropout"]).to(device)
        model = train_mlp(model, Xtr, ytr, Xva, yva, best["lr"], device, mlp_max, mlp_pat)
        model.eval()
        with torch.no_grad():
            pred = F.softmax(model(Xte), dim=1).argmax(1).cpu().numpy()

        f1, acc = bc.fold_metrics(y[te], pred)
        f1s.append(f1); accs.append(acc); ntests.append(len(te))
        print(f"  fold {k}: macro_f1={f1:.2f} acc={acc:.2f}", flush=True)
        del model, integrated, node_x
        clear_mem()

    out = bc.write_scores(
        os.path.join(BASE, "work_dirs", cohort),
        model="MOGAT", dataset=cohort,
        fold_f1=f1s, fold_acc=accs, fold_ntest=ntests,
        suffix="smoke" if smoke else "")
    print(f"\n[MOGAT] mean macro-F1={np.mean(f1s):.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
