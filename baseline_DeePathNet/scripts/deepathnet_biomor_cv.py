"""bioMOR CV runner for DeePathNet (config-driven, DeePathNet-style).

Reuses the UNMODIFIED upstream `DeePathNet` transformer
(model_transformer_lrp.py::DeePathNet) but replaces the drug-response data prep
(training_prepare.prepare_data_cv, hardcoded /home/scai paths) with the shared
bioMOR backbone:
  * data   : bc.load_omics(cohort, modalities)  -> reshaped to (N, G, n_omics)
  * folds  : bc.cv_folds(y)  (bioMoR seed-42 CV5, byte-identical)
  * pathways: cohort's data/<cohort>/filtered_pathways.csv if present, else a
    generic contiguous gene grouping (documented in NOTES.md). DeePathNet needs a
    {pathway_name: [genes]} dict + a non-cancer-gene remainder.
  * metric : macro-F1 / accuracy per fold
  * output : bc.write_scores -> work_dirs/<cohort>/scores_*.csv

Config JSON (configs/biomor/<cohort>.json) provides task + model hyperparams.

Usage:
    python scripts/deepathnet_biomor_cv.py configs/biomor/prostate.json
    SMOKE=1 python scripts/deepathnet_biomor_cv.py configs/biomor/prostate.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))          # .../scripts
BASE = os.path.dirname(HERE)                               # baseline_DeePathNet
REPO = os.path.dirname(BASE)
sys.path.insert(0, HERE)          # upstream: model_transformer_lrp, utils
sys.path.insert(0, REPO)          # biomor_common
import biomor_common as bc  # noqa: E402
from model_transformer_lrp import DeePathNet  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"


class GeneOmicDataset(Dataset):
    """(N, G, n_omics) tensor + int label."""
    def __init__(self, X3, y):
        self.X = torch.from_numpy(X3).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]


def build_pathways(genes, cohort, min_genes=5, max_pathways=None):
    """Return (pathway_dict, non_cancer_genes).

    Prefers data/<cohort>/filtered_pathways.csv (columns include a name + a
    gene list). Falls back to a generic contiguous grouping over the gene axis
    so DeePathNet still gets a pathway-token structure (documented in NOTES.md).
    """
    gene_set = set(genes)
    pw_path = os.path.join(bc.DATA, cohort, "filtered_pathways.csv")
    pathway_dict = {}
    if os.path.exists(pw_path):
        import pandas as pd
        df = pd.read_csv(pw_path)
        # find a gene-list column and a name column heuristically
        name_col = next((c for c in df.columns
                         if c.lower() in ("name", "pathway_name", "pathway", "pathway_id")),
                        df.columns[0])
        gene_col = next((c for c in df.columns
                         if c.lower() in ("genes", "gene", "gene_list", "members")), None)
        if gene_col is not None:
            for _, row in df.iterrows():
                raw = str(row[gene_col])
                sep = "|" if "|" in raw else ","
                members = [g.strip() for g in raw.split(sep) if g.strip() in gene_set]
                if len(members) >= min_genes:
                    pathway_dict[str(row[name_col])] = members
    if not pathway_dict:
        # generic fallback: contiguous blocks of ~50 genes as "pathways"
        block = 50
        for i in range(0, len(genes), block):
            members = list(genes[i:i + block])
            if len(members) >= min_genes:
                pathway_dict[f"block_{i//block}"] = members
    if max_pathways:
        keys = list(pathway_dict)[:max_pathways]
        pathway_dict = {k: pathway_dict[k] for k in keys}
    covered = set(g for v in pathway_dict.values() for g in v)
    non_cancer = gene_set - covered
    return pathway_dict, non_cancer


def train_fold(X3tr, ytr, X3va, yva, X3te, yte, genes, cohort, cfg, device, num_classes):
    gene_to_id = {g: i for i, g in enumerate(genes)}
    id_to_gene = {i: g for g, i in gene_to_id.items()}
    pathway_dict, non_cancer = build_pathways(
        genes, cohort, min_genes=cfg.get("min_gene_num", 5),
        max_pathways=cfg.get("max_pathways"))

    # DeePathNet's non_cancer_layer is nn.Linear(len(non_cancer)*n_omics, dim);
    # an empty remainder (all genes covered by pathway tokens) would give an
    # in_features=0 Linear. In that case run cancer-only (pathway tokens only).
    only_cancer = cfg.get("cancer_only", False) or (len(non_cancer) == 0)

    n_omics = X3tr.shape[2]
    model = DeePathNet(
        n_omics, num_classes, gene_to_id, id_to_gene, pathway_dict, non_cancer,
        embed_dim=cfg["dim"], depth=cfg["depth"], num_heads=cfg["heads"],
        mlp_ratio=cfg["mlp_ratio"], out_mlp_ratio=cfg["out_mlp_ratio"],
        qkv_bias=True, drop_rate=cfg.get("dropout", 0.0),
        pathway_drop_rate=cfg.get("pathway_dropout", 0.0),
        only_cancer_genes=only_cancer, tissues=None,
    ).to(device)

    counts = np.bincount(ytr, minlength=num_classes)
    cw = torch.tensor([counts.sum() / (num_classes * c) if c > 0 else 0.0
                       for c in counts], dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    bs = cfg["batch_size"]
    pin = device.type == "cuda"
    tr = DataLoader(GeneOmicDataset(X3tr, ytr), batch_size=bs, shuffle=True,
                    drop_last=cfg.get("drop_last", False), pin_memory=pin)
    va = DataLoader(GeneOmicDataset(X3va, yva), batch_size=bs, pin_memory=pin)
    te = DataLoader(GeneOmicDataset(X3te, yte), batch_size=bs, pin_memory=pin)

    from sklearn.metrics import f1_score
    best_f1, best_state, patience_left = -1.0, None, cfg.get("patience", 25)
    for epoch in range(1, cfg["num_of_epochs"] + 1):
        model.train()
        for x, t in tr:
            x, t = x.to(device, non_blocking=pin), t.to(device, non_blocking=pin)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            opt.step()
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, t in va:
                vp.append(model(x.to(device)).cpu().numpy())
                vt.append(t.numpy())
        f1v = f1_score(np.concatenate(vt), np.concatenate(vp).argmax(1),
                       average="macro", zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.get("patience", 25)
        else:
            patience_left -= 1
            if epoch >= cfg.get("min_epochs", 50) and patience_left <= 0:
                print(f"    early stop @ {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for x, t in te:
            tp.append(model(x.to(device)).cpu().numpy())
            tt.append(t.numpy())
    return np.concatenate(tt), np.concatenate(tp).argmax(1)


def main():
    cfg = json.load(open(sys.argv[1]))
    cohort = cfg["cohort"]
    modalities = tuple(cfg.get("modalities", ["mutation", "cnv"]))
    if SMOKE:
        cfg["num_of_epochs"] = min(cfg["num_of_epochs"], 3)
        cfg["min_epochs"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42)); np.random.seed(cfg.get("seed", 42))
    print(f"device={device}  cohort={cohort}  modalities={modalities}  "
          f"epochs={cfg['num_of_epochs']}")

    X, y, meta = bc.load_omics(cohort, modalities=modalities)
    n_omics = len(modalities)
    # bioMoR loader concatenates equal-length modality blocks over the SAME gene
    # set (per-modality dims identical), so reshape to (N, G, n_omics).
    dims = [meta["modality_dims"][m] for m in modalities]
    assert len(set(dims)) == 1, f"DeePathNet expects aligned gene sets, got {dims}"
    G = dims[0]
    # feature_names are 'mod:gene'; take gene order from the first modality block.
    genes = [n.split(":", 1)[1] for n in meta["feature_names"][:G]]
    X3 = np.stack([X[:, i * G:(i + 1) * G] for i in range(n_omics)], axis=-1).astype(np.float32)
    num_classes = int(y.max() + 1)
    print(f"X3={X3.shape}  genes={G}  classes={num_classes}  labels={np.bincount(y)}")

    folds = bc.cv_folds(y)
    if SMOKE:
        folds = folds[:1]

    f1s, accs, ns = [], [], []
    for k, (tr, va, te) in enumerate(folds):
        # z-score per fold on train stats (mutation block is 0/1, harmless).
        mu = X3[tr].mean((0, 1), keepdims=True)
        sd = X3[tr].std((0, 1), keepdims=True) + 1e-8
        Xn = (X3 - mu) / sd
        yt, yp = train_fold(Xn[tr], y[tr], Xn[va], y[va], Xn[te], y[te],
                            genes, cohort, cfg, device, num_classes)
        f1, acc = bc.fold_metrics(yt, yp)
        f1s.append(f1); accs.append(acc); ns.append(len(te))
        print(f"  fold {k+1}: macro_f1={f1:.2f}  acc={acc:.2f}  n={len(te)}")

    wd = os.path.join(BASE, "work_dirs", cohort)
    out = bc.write_scores(wd, "DeePathNet", cohort, f1s, accs, ns,
                          suffix="smoke" if SMOKE else "")
    print(f"macro_f1 mean={np.mean(f1s):.2f}  ->  {out}")


if __name__ == "__main__":
    main()
