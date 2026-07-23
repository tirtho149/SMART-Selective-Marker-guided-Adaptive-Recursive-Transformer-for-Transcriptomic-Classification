#!/usr/bin/env python3
"""MOGONET runner on the bioMOR shared contract.

Thin driver that reuses the UPSTREAM MOGONET model (models.py) and graph/adjacency
primitives (utils_adapted.py) verbatim, but:
  * loads cohorts + IDENTICAL seed-42 5-fold splits from biomor_common (bc),
  * supports 2- or 3-view (mut, cnv[, expression]) and multiclass,
  * selects the best-val-macro-F1 checkpoint and scores the TEST fold with
    bc.fold_metrics (macro-F1 + accuracy),
  * writes scores via bc.write_scores into work_dirs/<cohort>/.

Views (MOGONET convention here): view 1 = mutation (jaccard graph),
view 2 = cnv (cosine graph), view 3 = expression (cosine graph).

Env:
  COHORT   cohort name (default prostate)
  SMOKE=1  1 fold, few epochs
  MODALITIES  comma list (default "mutation,cnv")

Usage:
  COHORT=prostate SMOKE=1 python scripts/mogonet_cv.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)               # baseline_MOGONET
REPO = os.path.dirname(BASE)               # repo root (has biomor_common.py)
for p in (REPO, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import biomor_common as bc                                   # noqa: E402
from models import init_model_dict, init_optim               # noqa: E402
from utils_adapted import (                                  # noqa: E402
    one_hot_tensor, cal_sample_weight,
    gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# view id -> (modality name, graph metric)
METRIC = {"mutation": "jaccard", "cnv": "cosine", "expression": "cosine"}


def split_views(X, meta, modalities):
    """Slice the concatenated X into per-modality blocks (in load order)."""
    views, off = [], 0
    for m in modalities:
        d = meta["modality_dims"][m]
        views.append(X[:, off:off + d].astype(np.float32))
        off += d
    return views


def build_adj(data_full, tr_idx, sub_idx, adj_parameter, metric):
    """Adjacency for the train+sub subset (mirrors upstream gen_test_adj)."""
    param = cal_adj_mat_parameter(adj_parameter, data_full[tr_idx], metric)
    adj_tr = gen_adj_mat_tensor(data_full[tr_idx], param, metric).to(device)
    adj_sub = gen_test_adj_mat_tensor(
        data_full, list(tr_idx), list(sub_idx), param, metric).to(device)
    return adj_tr, adj_sub


def run_fold(views, y, tr, va, te, modalities, num_class,
             num_epoch_pretrain, num_epoch, lr_e_pretrain, lr_e, lr_c,
             adj_parameter, dim_he_list, patience, test_interval):
    num_view = len(views)
    metrics = [METRIC[m] for m in modalities]
    dim_hvcdn = pow(num_class, num_view)

    data_full = [torch.tensor(v, dtype=torch.float32, device=device) for v in views]

    y_tr = torch.tensor(y[tr], dtype=torch.long, device=device)
    onehot_tr = one_hot_tensor(y_tr, num_class).to(device)
    sw_tr = torch.tensor(cal_sample_weight(y[tr], num_class),
                         dtype=torch.float32, device=device)

    # Per-view adjacency for train, train+val, train+test.
    adj_tr, adj_val, adj_te = [], [], []
    for i in range(num_view):
        a_tr, a_val = build_adj(data_full[i], tr, va, adj_parameter, metrics[i])
        _, a_te = build_adj(data_full[i], tr, te, adj_parameter, metrics[i])
        adj_tr.append(a_tr)
        adj_val.append(a_val)
        adj_te.append(a_te)

    data_tr = [data_full[i][tr] for i in range(num_view)]
    # train+val / train+test stacked subsets, matching adjacency node order.
    data_val = [torch.cat([data_full[i][tr], data_full[i][va]], 0) for i in range(num_view)]
    data_te = [torch.cat([data_full[i][tr], data_full[i][te]], 0) for i in range(num_view)]
    ntr = len(tr)
    val_rel = list(range(ntr, ntr + len(va)))
    te_rel = list(range(ntr, ntr + len(te)))

    dim_list = [v.shape[1] for v in data_tr]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list,
                                 dim_hvcdn, gcn_dropout=0.2, device=device)

    def train_epoch(train_vcdn):
        crit = torch.nn.CrossEntropyLoss(reduction="none")
        for m in model_dict:
            model_dict[m].train()
        for i in range(num_view):
            opt = optim_dict[f"C{i+1}"]
            opt.zero_grad(set_to_none=True)
            emb = model_dict[f"E{i+1}"](data_tr[i], adj_tr[i])
            logits = model_dict[f"C{i+1}"](emb)
            loss = torch.mean(crit(logits, y_tr) * sw_tr)
            loss.backward()
            opt.step()
        if train_vcdn and num_view >= 2:
            optim_dict["C"].zero_grad(set_to_none=True)
            ci = [model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_tr[i], adj_tr[i]))
                  for i in range(num_view)]
            fused = model_dict["C"](ci)
            loss = torch.mean(crit(fused, y_tr) * sw_tr)
            loss.backward()
            optim_dict["C"].step()

    @torch.no_grad()
    def predict(data_list, adj_list, sub_rel):
        for m in model_dict:
            model_dict[m].eval()
        ci = [model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_list[i], adj_list[i]))
              for i in range(num_view)]
        c = model_dict["C"](ci) if num_view >= 2 else ci[0]
        return F.softmax(c[sub_rel, :], dim=1).float().cpu().numpy()

    # Pretrain encoders.
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for _ in range(num_epoch_pretrain):
        train_epoch(train_vcdn=False)

    # Full training w/ early stop on val macro-F1.
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    best_val, best_te_prob, wait = -1.0, None, 0
    for epoch in range(num_epoch + 1):
        train_epoch(train_vcdn=True)
        if epoch % test_interval == 0:
            vp = predict(data_val, adj_val, val_rel)
            vf1, _ = bc.fold_metrics(y[va], vp.argmax(1))
            tp = predict(data_te, adj_te, te_rel)
            if best_te_prob is None or vf1 > best_val:
                best_val, best_te_prob, wait = vf1, tp, 0
            else:
                wait += test_interval
            print(f"  epoch {epoch:4d} val_f1={vf1:.2f}"
                  + ("  <-best" if wait == 0 else ""), flush=True)
            if wait >= patience:
                break
    return best_te_prob.argmax(1), y[te]


def main():
    cohort = os.environ.get("COHORT", "prostate")
    smoke = os.environ.get("SMOKE", "0") == "1"
    modalities = tuple(os.environ.get("MODALITIES", "mutation,cnv").split(","))

    X, y, meta = bc.load_omics(cohort, modalities=modalities)
    num_class = len(meta["classes"])
    views = split_views(X, meta, modalities)
    print(f"[MOGONET] cohort={cohort} N={len(y)} classes={num_class} "
          f"views={modalities} dims={[v.shape[1] for v in views]} device={device}",
          flush=True)

    folds = bc.cv_folds(y)
    if smoke:
        folds = folds[:1]

    n_ep_pre = 5 if smoke else 200
    n_ep = 5 if smoke else 200
    test_interval = 5 if smoke else 50
    patience = 10 if smoke else 100
    dim_he_list = [400, 400, 200]

    f1s, accs, ntests = [], [], []
    for k, (tr, va, te) in enumerate(folds, 1):
        print(f"\n===== fold {k}/{len(folds)} =====", flush=True)
        y_pred, y_true = run_fold(
            views, y, tr, va, te, modalities, num_class,
            n_ep_pre, n_ep, 1e-4, 1e-4, 1e-4,
            adj_parameter=10, dim_he_list=dim_he_list,
            patience=patience, test_interval=test_interval)
        f1, acc = bc.fold_metrics(y_true, y_pred)
        f1s.append(f1); accs.append(acc); ntests.append(len(te))
        print(f"  fold {k}: macro_f1={f1:.2f} acc={acc:.2f}", flush=True)

    out = bc.write_scores(
        os.path.join(BASE, "work_dirs", cohort),
        model="MOGONET", dataset=cohort,
        fold_f1=f1s, fold_acc=accs, fold_ntest=ntests,
        suffix="smoke" if smoke else "")
    print(f"\n[MOGONET] mean macro-F1={np.mean(f1s):.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
