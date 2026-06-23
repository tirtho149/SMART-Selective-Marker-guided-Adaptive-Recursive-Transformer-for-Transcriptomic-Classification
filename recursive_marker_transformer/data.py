# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Data layer: a thin wrapper over ``genomic_dataloader.build_loaders``.

Adds two things the bundled loader does not provide and that the model needs:

1. An optional high-variance-gene (HVG) pre-filter, fit on the **training**
   split only. The selection mirrors ``genomap``'s ``select_n_features`` (top-n
   by variance); we reimplement the three-line core here to avoid importing the
   whole ``genomap`` package (its ``util_Sig`` pulls in the optimal-transport
   stack at module import).
2. A contiguous label remap per multiclass head, fit on the training split, so
   any cohort subset stays in ``[0, n_classes)`` for ``CrossEntropyLoss`` (the
   raw ``cancer_type`` codes are global and non-contiguous on a subset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from genomic_dataloader import build_loaders

from .config import RMTConfig


@dataclass
class DataBundle:
    loaders: Dict[str, "torch.utils.data.DataLoader"]
    meta: object                       # genomic_dataloader.LoaderMeta
    gene_names: List[str]              # after HVG filter
    n_genes: int
    hvg_index: Optional[np.ndarray]    # column indices kept, or None
    head_n_classes: Dict[str, int]     # after contiguous remap
    head_dtypes: Dict[str, str]
    class_weights: Dict[str, np.ndarray]
    label_maps: Dict[str, Dict[int, int]]   # head -> {raw_value: contiguous_idx}
    gene_variance: np.ndarray          # raw train per-gene variance, kept genes


def _collect_train_stats(loader, heads):
    """One pass over the raw train loader: gather label columns and the per-gene
    variance of the (z-scored) features. Accumulators are gene-length vectors, so
    memory stays tiny (no full-X materialisation)."""
    ys: Dict[str, list] = {}
    n = 0
    s = sq = None
    for xb, yb in loader:
        xd = xb.double()
        s = xd.sum(0) if s is None else s + xd.sum(0)
        sq = (xd ** 2).sum(0) if sq is None else sq + (xd ** 2).sum(0)
        n += xb.shape[0]
        for h in heads:
            ys.setdefault(h, []).append(yb[h].numpy())
    labels = {h: np.concatenate(v) for h, v in ys.items()}
    scaled_var = (sq / n - (s / n) ** 2).clamp(min=0).numpy()
    return labels, scaled_var


def build_data(cfg: RMTConfig) -> DataBundle:
    cohorts = list(cfg.cohorts) if cfg.cohorts is not None else None
    loaders, meta = build_loaders(
        heads=list(cfg.heads),
        cohorts=cohorts,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    Y_train, scaled_var = _collect_train_stats(loaders["train"], cfg.heads)

    # ---- HVG filter (fit on train) -------------------------------------
    # Loaders return z-scored features, so recover the true raw train variance
    # exactly: var(raw) = var(scaled) * std_floored^2 (holds for every gene,
    # including the near-constant ones the scaler floors to std=1.0).
    train_std = np.asarray(meta.scaler.std, dtype=np.float64)
    raw_var = scaled_var * (train_std ** 2)
    hvg_index = None
    gene_names = list(meta.gene_names)
    if cfg.n_hvg is not None and cfg.n_hvg < meta.n_genes:
        hvg_index = np.sort(np.argsort(raw_var)[::-1][: cfg.n_hvg])
        gene_names = [gene_names[i] for i in hvg_index]

    gene_variance = raw_var if hvg_index is None else raw_var[hvg_index]

    # ---- contiguous label maps (fit on train) --------------------------
    label_maps: Dict[str, Dict[int, int]] = {}
    head_n_classes = dict(meta.head_n_classes)
    for h in cfg.heads:
        if meta.head_dtypes[h] == "multiclass":
            uniq = sorted(int(v) for v in np.unique(Y_train[h]))
            label_maps[h] = {v: i for i, v in enumerate(uniq)}
            head_n_classes[h] = len(uniq)

    hvg_t = None if hvg_index is None else torch.as_tensor(hvg_index.copy(), dtype=torch.long)

    # Wrap each loader so batches come out HVG-sliced + label-remapped.
    wrapped = {
        split: _MappedLoader(dl, hvg_t, label_maps, meta.head_dtypes)
        for split, dl in loaders.items()
    }

    return DataBundle(
        loaders=wrapped,
        meta=meta,
        gene_names=gene_names,
        n_genes=len(gene_names),
        hvg_index=hvg_index,
        head_n_classes=head_n_classes,
        head_dtypes=dict(meta.head_dtypes),
        class_weights=dict(meta.class_weights),
        label_maps=label_maps,
        gene_variance=gene_variance.astype(np.float32),
    )


class _MappedLoader:
    """Iterable that HVG-slices x and remaps multiclass labels per batch."""

    def __init__(self, loader, hvg_index, label_maps, head_dtypes):
        self._loader = loader
        self._hvg = hvg_index
        self._maps = label_maps
        self._dtypes = head_dtypes
        # Precompute lookup tensors for fast vectorised remap. Unseen raw values
        # (a class absent from train but present in val/test) map to 0 and are
        # clamped in __iter__ so they never index out of bounds.
        self._lut = {}
        for h, m in label_maps.items():
            size = max(m) + 1
            lut = torch.zeros(size, dtype=torch.long)
            for raw, idx in m.items():
                lut[raw] = idx
            self._lut[h] = lut

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        for xb, yb in self._loader:
            if self._hvg is not None:
                xb = xb.index_select(1, self._hvg)
            out = {}
            for h, v in yb.items():
                lut = self._lut.get(h)
                if lut is not None:
                    idx = v.long().clamp(min=0, max=lut.numel() - 1)
                    out[h] = lut[idx]
                else:
                    out[h] = v
            yield xb, out
