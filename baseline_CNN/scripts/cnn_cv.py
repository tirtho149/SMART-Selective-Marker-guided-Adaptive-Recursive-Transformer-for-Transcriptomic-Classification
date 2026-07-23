"""Thin bioMOR CV runner for the CNN baselines (Early- and Late-Integration).

Wraps the UNMODIFIED upstream CNN architectures (eiCNN.py::EarlyIntegrationCNN,
liCNN.py::LateIntegrationCNN) but drives them with the shared backbone:
  * data      : bc.load_omics(cohort, modalities)
  * folds     : bc.cv_folds(y)   (bioMoR seed-42 CV5, byte-identical)
  * metric    : macro-F1 / accuracy per fold
  * output    : bc.write_scores -> baseline_CNN/work_dirs/<cohort>/scores_*.csv

Supports binary + multiclass, 2-modal (mutation+cnv) and 3-modal
(mutation+cnv+expression). Late integration keeps one Conv1d branch per modality
(generalised from the upstream 2-branch liCNN to N modalities).

Usage:
    python scripts/cnn_cv.py --model ei --cohort prostate
    python scripts/cnn_cv.py --model li --cohort pan_meta_pri_3modal --modalities mutation cnv expression
    SMOKE=1 python scripts/cnn_cv.py --model ei --cohort prostate   # 1 fold / few epochs
"""
import argparse
import os
import random
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)          # for biomor_common
sys.path.insert(0, os.path.dirname(HERE))  # baseline_CNN root (upstream modules)
import biomor_common as bc  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# Upstream architectures (verbatim from eiCNN.py / liCNN.py)                    #
# --------------------------------------------------------------------------- #
class EarlyIntegrationCNN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        conv1_out = in_dim - 1000 + 1
        pool1_out = conv1_out // 100
        conv2_out = pool1_out - 50 + 1
        pool2_out = conv2_out // 10
        linear_input = pool2_out * 16
        self.FC = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1000), nn.ReLU(), nn.MaxPool1d(100),
            nn.Conv1d(32, 16, kernel_size=50), nn.ReLU(), nn.MaxPool1d(10),
            nn.Flatten(),
            nn.Linear(int(linear_input), 50), nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x):
        return self.FC(x.unsqueeze(1))   # logits (CrossEntropyLoss handles softmax)


class LateIntegrationCNN(nn.Module):
    """Generalised liCNN: one Conv1d branch per modality block, then merge MLP."""
    def __init__(self, modality_dims, num_classes):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.branches = nn.ModuleList()
        total = 0
        for d in self.modality_dims:
            conv_out = (d - 300 + 1) // 100
            self.branches.append(nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=300), nn.ReLU(),
                nn.MaxPool1d(100), nn.Flatten()))
            total += conv_out * 32
        self.FC_merge = nn.Sequential(
            nn.Linear(total, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        outs, off = [], 0
        for d, br in zip(self.modality_dims, self.branches):
            outs.append(br(x[:, off:off + d].unsqueeze(1)))
            off += d
        return self.FC_merge(torch.cat(outs, dim=1))


class OmicsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]


class EarlyStopping:
    def __init__(self, patience=25, delta=0.001, stop=50):
        self.patience, self.delta, self.stop = patience, delta, stop
        self.counter, self.best, self.early_stop = 0, None, False

    def __call__(self, score, epoch):
        if epoch <= self.stop:
            self.best = score
            self.counter = 0
            return
        if self.best is None:
            self.best = score
        elif score < self.best - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best = score
            self.counter = 0


def macro_f1(y_true, prob):
    from sklearn.metrics import f1_score
    return f1_score(y_true, prob.argmax(1), average="macro", zero_division=0)


def run_fold(model_kind, Xtr, ytr, Xva, yva, Xte, yte, dims, num_classes,
             device, epochs, batch_size=16):
    if model_kind == "ei":
        model = EarlyIntegrationCNN(in_dim=Xtr.shape[1], num_classes=num_classes)
    else:
        model = LateIntegrationCNN(modality_dims=dims, num_classes=num_classes)
    model = model.to(device)

    counts = np.bincount(ytr, minlength=num_classes)
    cw = torch.tensor([counts.sum() / (num_classes * c) if c > 0 else 0.0
                       for c in counts], dtype=torch.float32).to(device)
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    stopper = EarlyStopping()

    pin = device.type == "cuda"
    tr = DataLoader(OmicsDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, pin_memory=pin)
    va = DataLoader(OmicsDataset(Xva, yva), batch_size=batch_size, pin_memory=pin)
    te = DataLoader(OmicsDataset(Xte, yte), batch_size=batch_size, pin_memory=pin)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, t in tr:
            x, t = x.to(device, non_blocking=pin), t.to(device, non_blocking=pin)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            opt.step()
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, t in va:
                vp.append(model(x.to(device)).cpu().numpy())
                vt.append(t.numpy())
        f1v = macro_f1(np.concatenate(vt), np.concatenate(vp))
        stopper(f1v, epoch)
        if stopper.early_stop:
            print(f"    early stop @ {epoch}")
            break

    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for x, t in te:
            tp.append(model(x.to(device)).cpu().numpy())
            tt.append(t.numpy())
    return np.concatenate(tt), np.concatenate(tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ei", "li"], required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--modalities", nargs="+", default=["mutation", "cnv"])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    epochs = 3 if SMOKE else args.epochs
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={args.model}  cohort={args.cohort}  "
          f"modalities={args.modalities}  epochs={epochs}")

    X, y, meta = bc.load_omics(args.cohort, modalities=tuple(args.modalities))
    dims = [meta["modality_dims"][m] for m in args.modalities]
    num_classes = int(y.max() + 1)
    print(f"X={X.shape}  classes={num_classes}  dims={dims}  labels={np.bincount(y)}")

    folds = bc.cv_folds(y)
    if SMOKE:
        folds = folds[:1]

    f1s, accs, ns = [], [], []
    for k, (tr, va, te) in enumerate(folds):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]).astype(np.float32)
        Xva = sc.transform(X[va]).astype(np.float32)
        Xte = sc.transform(X[te]).astype(np.float32)
        yt, yp = run_fold(args.model, Xtr, y[tr], Xva, y[va], Xte, y[te],
                          dims, num_classes, device, epochs)
        f1, acc = bc.fold_metrics(yt, yp.argmax(1))
        f1s.append(f1); accs.append(acc); ns.append(len(te))
        print(f"  fold {k+1}: macro_f1={f1:.2f}  acc={acc:.2f}  n={len(te)}")

    model_name = f"CNN_{'ei' if args.model=='ei' else 'li'}"
    wd = os.path.join(os.path.dirname(HERE), "work_dirs", args.cohort)
    out = bc.write_scores(wd, model_name, args.cohort, f1s, accs, ns,
                          suffix="smoke" if SMOKE else "")
    print(f"macro_f1 mean={np.mean(f1s):.2f}  ->  {out}")


if __name__ == "__main__":
    main()
