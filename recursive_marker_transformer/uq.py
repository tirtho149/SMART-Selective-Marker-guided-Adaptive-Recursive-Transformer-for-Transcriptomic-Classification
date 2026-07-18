# ============================================================================
# bioMoR -- log-probability-driven uncertainty quantification (UQ).
#
# From the model's test-set softmax probabilities we compute the standard
# calibration / log-prob UQ metrics: negative log-likelihood (NLL), expected
# calibration error (ECE, 15 equal-width confidence bins), Brier score, mean
# confidence, and the AUROC with which max-probability separates correct from
# incorrect predictions (confidence as a failure detector). All are label-free of
# the architecture's claims, so they fairly test whether the biological prior or
# adaptive depth buy any *uncertainty* quality over a vanilla transformer.
#
#   python -m recursive_marker_transformer.uq_sweep      # produces results_uq/
# ============================================================================
from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def predict_probs(model, loader, device, head):
    """Run a multiclass head and return (y_true (N,), probs (N,C))."""
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logit = model(xb.to(device))["logits"][head]
        ps.append(torch.softmax(logit, dim=-1).cpu().numpy())
        ys.append(yb[head].numpy() if hasattr(yb[head], "numpy") else np.asarray(yb[head]))
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def predict_logits(model, loader, device, head):
    """Run a multiclass head and return (y_true (N,), logits (N,C)) -- raw pre-softmax."""
    model.eval()
    ys, ls = [], []
    for xb, yb in loader:
        logit = model(xb.to(device))["logits"][head]
        ls.append(logit.cpu().numpy())
        ys.append(yb[head].numpy() if hasattr(yb[head], "numpy") else np.asarray(yb[head]))
    return np.concatenate(ys), np.concatenate(ls)


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y_true, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al. 2017): fit a single scalar T>0 that minimises
    NLL of softmax(logits / T) on a held-out (validation) set. Dividing all logits by
    one positive scalar is monotonic, so the arg-max -- and thus accuracy/macro-F1 --
    is UNCHANGED; only the confidence is recalibrated. Returns the fitted T."""
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
    if z.ndim != 2 or z.shape[0] == 0:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)                 # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    ce = torch.nn.CrossEntropyLoss()

    def _closure():
        opt.zero_grad()
        loss = ce(z / log_t.exp(), y)
        loss.backward()
        return loss
    try:
        opt.step(_closure)
        T = float(log_t.exp().item())
        return T if math.isfinite(T) and T > 0 else 1.0
    except Exception:
        return 1.0


def temperature_scaled_metrics(val_logits, val_y, test_logits, test_y):
    """Fit T on the validation logits, apply to the test logits, and return
    (T, raw_metrics, temperature_scaled_metrics). accuracy/F1 are unchanged by T."""
    T = fit_temperature(val_logits, val_y)
    zt = np.asarray(test_logits, dtype=np.float64)
    raw = uq_metrics(test_y, _softmax(zt))
    ts = uq_metrics(test_y, _softmax(zt / T))
    return T, raw, ts


def uq_metrics(y_true, probs, n_bins: int = 15):
    """Log-prob / calibration UQ metrics from softmax probabilities.
    Returns dict with nll, ece, brier, conf, auroc (all floats). Lower is better
    for nll/ece/brier; higher is better for conf-AUROC."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    N, C = p.shape
    idx = np.arange(N)

    nll = float(-np.log(p[idx, y_true]).mean())
    onehot = np.zeros_like(p)
    onehot[idx, y_true] = 1.0
    brier = float(((p - onehot) ** 2).sum(1).mean())

    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y_true).astype(np.float64)
    # ECE: |accuracy - confidence| weighted by bin mass
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * (m.mean())
    # AUROC of confidence as a correctness detector (skip if one class only)
    auroc = float("nan")
    if 0 < correct.sum() < N:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(correct, conf))
        except Exception:
            pass
    return {"nll": nll, "ece": float(ece), "brier": brier,
            "conf": float(conf.mean()), "auroc": auroc}
