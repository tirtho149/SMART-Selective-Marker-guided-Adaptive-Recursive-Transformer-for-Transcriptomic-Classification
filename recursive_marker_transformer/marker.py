# ============================================================================
# bioMoR: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The bioMoR Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Stages 2-3: learnable marker selection + recursive compression.

The novelty of RMT lives here. A ``MarkerHead`` scores every gene; the top-K
genes (global, batch-independent, so markers are interpretable gene identities)
become *marker tokens*. Every non-marker gene is folded into its nearest marker
(cosine similarity in embedding space), compressing N gene tokens into M
marker-anchored cluster tokens -- turning attention cost from O(N^2) to O(M^2).

The hard top-K is non-differentiable in *which* genes are chosen, so selected
markers are multiplied by a soft gate ``sigmoid(score)``; gradients flow to the
head and push useful markers' scores up (straight-through style). During
recursion the ``RefineHead`` re-scores the current marker tokens, so markers
that stop being informative get down-weighted -- the closed feedback loop.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MarkerHead(nn.Module):
    """Per-gene importance score from the (batch-independent) gene identity."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, gene_identity: torch.Tensor) -> torch.Tensor:
        return self.net(gene_identity).squeeze(-1)          # (N,)


class RefineHead(nn.Module):
    """Per-token gate from the *current* contextual marker embedding (B, M, d)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens).squeeze(-1)                 # (B, M)


class ConcreteSelector(nn.Module):
    """Concrete / Gumbel-softmax differentiable feature selection
    (Balin, Abid & Zou 2019; Jang, Gu & Poole 2017).

    Holds ``M`` selectors, each a learnable distribution over *all* ``N`` genes.
    During training each selector draws a temperature-annealed Gumbel-softmax
    sample (near-uniform when hot, near one-hot when cold), so gradients reach
    every gene and the model learns *which* genes to keep -- unlike hard top-k,
    which can only re-rank a frozen set. At eval we take the hard argmax gene.
    """

    def __init__(self, n_genes: int, n_markers: int,
                 temp_start: float = 1.0, temp_end: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.n_markers = min(n_markers, n_genes)
        # Peaked init: each selector starts ~one-hot on a distinct random gene
        # (a spike large enough to dominate the softmax over N genes), so training
        # begins at random-selection quality and *improves* -- avoiding the
        # near-uniform "mush" cold start that needs hundreds of epochs to escape.
        logits = 0.01 * torch.randn(self.n_markers, n_genes)
        spike = torch.randperm(n_genes)[: self.n_markers]
        logits[torch.arange(self.n_markers), spike] = 10.0
        self.logits = nn.Parameter(logits)
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.register_buffer("temp", torch.tensor(float(temp_start)))

    def set_progress(self, p: float):
        p = min(1.0, max(0.0, float(p)))
        self.temp.fill_(self.temp_start * (self.temp_end / self.temp_start) ** p)

    def weights(self, gene_identity: torch.Tensor = None) -> torch.Tensor:
        """(M, N) selection weights: Gumbel-softmax in train, hard one-hot in eval."""
        if self.training:
            u = torch.rand_like(self.logits).clamp_(1e-9, 1.0)
            g = -torch.log(-torch.log(u))
            return torch.softmax((self.logits + g) / self.temp.clamp_min(1e-4), dim=1)
        idx = self.logits.argmax(dim=1)
        return torch.zeros_like(self.logits).scatter_(1, idx.unsqueeze(1), 1.0)

    def selected_indices(self, gene_identity: torch.Tensor = None) -> torch.Tensor:
        return self.logits.argmax(dim=1)


class SlotRouter(nn.Module):
    """Cross-attention 'slot' router -- the best-practice router for marker
    selection (Set Transformer induced points / Perceiver / Slot Attention).

    ``M`` learnable marker queries cross-attend over all ``N`` gene embeddings
    (keys), with a temperature-annealed softmax *over genes* so gradient reaches
    every gene every step (unlike hard top-k routing, which cannot explore). Each
    query has its own parameters, so slots specialise on distinct genes. At eval
    each slot collapses to its arg-max gene for a discrete, interpretable marker.
    """

    def __init__(self, n_genes: int, n_markers: int, d_model: int,
                 temp_start: float = 1.0, temp_end: float = 0.3):
        super().__init__()
        self.n_genes = n_genes
        self.n_markers = min(n_markers, n_genes)
        self.d_model = d_model
        self.queries = nn.Parameter(0.02 * torch.randn(self.n_markers, d_model))
        self.key = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.register_buffer("temp", torch.tensor(float(temp_start)))

    def set_progress(self, p: float):
        p = min(1.0, max(0.0, float(p)))
        self.temp.fill_(self.temp_start * (self.temp_end / self.temp_start) ** p)

    def _logits(self, gene_identity: torch.Tensor) -> torch.Tensor:
        k = self.key(gene_identity)                          # (N, d)
        return (self.queries @ k.t()) * self.scale           # (M, N)

    def weights(self, gene_identity: torch.Tensor) -> torch.Tensor:
        logits = self._logits(gene_identity)
        if self.training:
            return torch.softmax(logits / self.temp.clamp_min(1e-4), dim=1)
        idx = logits.argmax(dim=1)
        return torch.zeros_like(logits).scatter_(1, idx.unsqueeze(1), 1.0)

    def selected_indices(self, gene_identity: torch.Tensor) -> torch.Tensor:
        return self._logits(gene_identity).argmax(dim=1)


class PathwayPooler(nn.Module):
    """Fixed Reactome gene->pathway membership pooling -> ``M`` *pathway tokens*.

    A drop-in selector (same ``weights`` / ``selected_indices`` interface as
    ``SlotRouter``), so ``model.py``'s selector branch pools both the gene-identity
    matrix and the per-channel patient values by pathway membership with no other
    change. This is P-NET's (Elmarakeby et al. 2021) gene->pathway sparse layer,
    recast as interpretable tokens that then flow into the recursive MoR stack:
    each token *is* a curated Reactome pathway, unlike the learned/abstract marker
    slots.

    ``membership`` is the binary ``(N_genes, M_pathways)`` matrix. The pooling is
    ``pool="mean"`` (row-normalised, a token is the *mean* of its members --
    scale-free, right for dense assays like CNV) or ``pool="sum"`` (raw
    membership, a token is the *sum/burden* of its members -- right for SPARSE
    binary mutation, where mean-pooling washes the per-pathway signal to a near
    constant and the model collapses). A learnable per-pathway gate
    ``sigmoid(g_m)`` lets the model down-weight uninformative pathways -- the
    'selective' property of bioMoR, now at pathway granularity. The membership is
    fixed (biology, not learned); only the gate trains.
    """

    def __init__(self, membership: torch.Tensor, pool: str = "mean"):
        super().__init__()
        P = membership.float()                                  # (N, M)
        self.n_genes, self.n_markers = P.shape
        self.pool = pool
        if pool == "sum":
            W = P.t().contiguous()                             # (M, N) raw -> burden
        elif pool == "mean":
            col = P.sum(dim=0).clamp_min(1.0)                  # members per pathway
            W = (P / col).t().contiguous()                    # (M, N) rows ~sum to 1
        else:
            raise ValueError(f"Unknown pathway pool: {pool!r}")
        self.register_buffer("Wn", W, persistent=True)
        self.gate = nn.Parameter(torch.zeros(self.n_markers))  # sigmoid(0)=0.5

    def weights(self, gene_identity: torch.Tensor = None) -> torch.Tensor:
        """(M, N) pooling weights = per-pathway gate x normalised membership."""
        return torch.sigmoid(self.gate).unsqueeze(1) * self.Wn

    def selected_indices(self, gene_identity: torch.Tensor = None) -> torch.Tensor:
        """Each token's identity is its pathway index (tokens ARE pathways)."""
        return torch.arange(self.n_markers, device=self.Wn.device)


class MarkerModule(nn.Module):
    def __init__(self, d_model: int, n_genes: int, n_markers: int, mode: str = "learnable"):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.n_markers = min(n_markers, n_genes)
        self.mode = mode
        self.head = MarkerHead(d_model) if mode == "learnable" else None
        self.refine = RefineHead(d_model)
        # Fixed random marker panel (drawn once with the seeded global RNG) so
        # the "random" baseline keeps a stable marker set across batches/epochs.
        rand_idx, _ = torch.sort(torch.randperm(n_genes)[: self.n_markers])
        self.register_buffer("random_markers", rand_idx, persistent=True)

    # ---- selection -----------------------------------------------------
    def select(
        self,
        gene_identity: torch.Tensor,          # (N, d)
        variance: Optional[torch.Tensor],     # (N,) input variance, for "variance" mode
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (marker_idx (M,), scores (N,), init_gate (M,))."""
        n = gene_identity.shape[0]
        m = min(self.n_markers, n)
        if self.mode == "learnable":
            scores = self.head(gene_identity)                       # (N,)
            marker_idx = torch.topk(scores, m).indices
            gate = torch.sigmoid(scores[marker_idx])                # (M,) differentiable
        elif self.mode == "variance":
            scores = variance if variance is not None else gene_identity.var(dim=1)
            marker_idx = torch.topk(scores, m).indices
            gate = torch.ones(m, device=gene_identity.device)
        elif self.mode == "random":
            scores = torch.zeros(n, device=gene_identity.device)
            marker_idx = self.random_markers.to(gene_identity.device)
            gate = torch.ones(marker_idx.shape[0], device=gene_identity.device)
        else:
            raise ValueError(f"Unknown marker_mode: {self.mode}")
        marker_idx, _ = torch.sort(marker_idx)
        return marker_idx, scores, gate

    # ---- compression ---------------------------------------------------
    def aggregate(
        self,
        tokens: torch.Tensor,         # (B, N, d) per-cell gene tokens
        gene_identity: torch.Tensor,  # (N, d)
        marker_idx: torch.Tensor,     # (M,)
        gate: torch.Tensor,           # (M,) or (B, M)
    ) -> torch.Tensor:
        """Fold non-marker genes into their nearest marker -> (B, M, d)."""
        B, N, d = tokens.shape
        M = marker_idx.shape[0]
        device = tokens.device

        # Nearest-marker assignment by cosine similarity (batch-independent).
        ident = nn.functional.normalize(gene_identity, dim=1)        # (N, d)
        marker_ident = ident[marker_idx]                             # (M, d)
        sims = ident @ marker_ident.t()                              # (N, M)
        assign = sims.argmax(dim=1)                                  # (N,) -> slot
        # Force markers to map to their own slot.
        assign[marker_idx] = torch.arange(M, device=device)

        # Gated marker tokens.
        if gate.dim() == 1:
            gate = gate.unsqueeze(0).expand(B, -1)                   # (B, M)
        marker_tokens = tokens[:, marker_idx, :] * gate.unsqueeze(-1)

        # Mean of assigned non-marker tokens per slot.
        is_marker = torch.zeros(N, dtype=torch.bool, device=device)
        is_marker[marker_idx] = True
        nm = ~is_marker
        neigh_sum = torch.zeros(B, M, d, device=device)
        counts = torch.zeros(M, device=device)
        if nm.any():
            nm_assign = assign[nm]                                   # (n_non,)
            neigh_sum.index_add_(1, nm_assign, tokens[:, nm, :])
            counts.index_add_(0, nm_assign, torch.ones(nm.sum(), device=device))
        neigh_mean = neigh_sum / counts.clamp(min=1).view(1, M, 1)

        return marker_tokens + neigh_mean                            # (B, M, d)

    # ---- recursive refinement -----------------------------------------
    def refine_gate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Re-score current marker tokens -> per-cell gate (B, M)."""
        return torch.sigmoid(self.refine(tokens))
