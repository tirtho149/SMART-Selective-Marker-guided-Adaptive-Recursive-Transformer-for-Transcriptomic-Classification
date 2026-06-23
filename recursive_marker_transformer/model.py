"""The Recursive Marker Transformer.

Pipeline (proposal Stages 1-5):

    gene expression
      -> GeneEmbedding              (B, N, d)
      -> MarkerModule.select        top-K marker genes (global, interpretable)
      -> MarkerModule.aggregate     compress to (B, M, d) marker-anchored tokens
      -> RecursiveStack (K x shared) with recursive marker refinement
      -> mean-pool markers -> per-head linear classifier
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import RMTConfig
from .embedding import GeneEmbedding
from .marker import ConcreteSelector, MarkerModule, SlotRouter
from .recursion import RecursiveStack


class RecursiveMarkerTransformer(nn.Module):
    def __init__(self, cfg: RMTConfig, n_genes: int, head_n_classes: Dict[str, int],
                 head_dtypes: Dict[str, str]):
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes
        self.head_dtypes = head_dtypes

        self.embed = GeneEmbedding(n_genes, cfg.d_model, cfg.dropout)
        self.marker = MarkerModule(cfg.d_model, n_genes, cfg.n_markers, cfg.marker_mode)
        # Soft selectors produce M marker tokens directly with all-gene gradient.
        if cfg.marker_mode == "concrete":
            self.selector = ConcreteSelector(n_genes, cfg.n_markers)
        elif cfg.marker_mode == "router":
            self.selector = SlotRouter(n_genes, cfg.n_markers, cfg.d_model)
            if getattr(cfg, "peak_init", True):
                self._peak_init_router()
        else:
            self.selector = None
        self.stack = RecursiveStack(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            depth=cfg.recursion_depth, share_weights=cfg.share_weights,
            adaptive_depth=cfg.adaptive_depth, marker_ffn=cfg.marker_ffn,
            recursion_mode=cfg.recursion_mode, router_capacity=cfg.router_capacity,
            router_alpha=cfg.router_alpha, router_temp=cfg.router_temp,
            router_type=cfg.router_type,
        )
        # One classifier per head; binary heads use a single logit.
        self.classifiers = nn.ModuleDict({
            h: nn.Linear(cfg.d_model, 1 if head_dtypes[h] == "binary" else head_n_classes[h])
            for h in cfg.heads
        })
        # Auxiliary head for the marker-sufficiency loss (primary head only).
        self._primary = cfg.heads[0]
        prim_out = 1 if head_dtypes[self._primary] == "binary" else head_n_classes[self._primary]
        self.aux_marker_head = nn.Linear(cfg.d_model, prim_out)

        self.register_buffer("gene_variance", torch.zeros(n_genes), persistent=False)

    def set_gene_variance(self, variance: torch.Tensor) -> None:
        self.gene_variance.copy_(variance.to(self.gene_variance))

    @torch.no_grad()
    def _peak_init_router(self):
        """Point each router query at a distinct gene's key so attention starts
        peaked (random-selection quality) rather than uniform mush -- the same
        fix that lets Concrete learn in a small epoch budget."""
        ident = self.embed.gene_identity()                      # (N, d)
        k = self.selector.key(ident)                            # (N, d)
        kn = nn.functional.normalize(k, dim=1)
        genes = torch.randperm(self.n_genes)[: self.selector.n_markers]
        # Strong peak so each slot starts ~one-hot on a distinct gene; the exact
        # constant is uncritical as long as the softmax over N genes is peaked.
        self.selector.queries.copy_(kn[genes] * 60.0)

    def set_anneal(self, progress: float) -> None:
        """Advance the selector temperature schedule (explore early, exploit late).
        No-op for fixed selection modes, and disabled when anneal_markers is off
        (temperature then stays at its hot start value)."""
        if not getattr(self.cfg, "anneal_markers", True):
            return
        if self.selector is not None and hasattr(self.selector, "set_progress"):
            self.selector.set_progress(progress)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        gene_identity = self.embed.gene_identity()              # (N, d)

        if self.selector is not None:
            # Soft selection (Concrete or cross-attention router): soft (train) /
            # hard (eval) selection over ALL genes -> M marker tokens directly, so
            # gradients reach every gene (the property hard top-k routing lacks).
            w = self.selector.weights(gene_identity)           # (M, N)
            sel_ident = w @ gene_identity                      # (M, d)
            sel_value = x @ w.t()                              # (B, M) selected expression
            cluster = sel_ident.unsqueeze(0) + self.embed.value_proj(sel_value.unsqueeze(-1))
            marker_idx = self.selector.selected_indices(gene_identity)
            scores = w.max(dim=0).values                       # (N,) per-gene max selection weight
            marker_ident = nn.functional.normalize(sel_ident, dim=1)
        else:
            tokens = self.embed(x)                             # (B, N, d)
            marker_idx, scores, init_gate = self.marker.select(gene_identity, self.gene_variance)
            if self.cfg.compress_mode == "drop":
                # Strict selection: use ONLY the selected marker genes (no folding).
                gate_b = init_gate.unsqueeze(0).expand(x.shape[0], -1)   # (B, M)
                cluster = tokens[:, marker_idx, :] * gate_b.unsqueeze(-1)
            else:
                cluster = self.marker.aggregate(tokens, gene_identity, marker_idx, init_gate)
            marker_ident = nn.functional.normalize(gene_identity[marker_idx], dim=1)

        # Marker-sufficiency aux loss: probe whether the (pre-recursion) marker
        # tokens alone are task-sufficient, which trains selection toward
        # discriminative genes.
        aux_logits = self.aux_marker_head(cluster.mean(dim=1))

        refine_fn = self.marker.refine_gate if self.cfg.recursive_marker_refine else None
        h, route_info = self.stack(cluster, refine_fn)          # (B, M, d), routing info
        pooled = h.mean(dim=1)                                  # (B, d)

        logits = {head: clf(pooled) for head, clf in self.classifiers.items()}

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "scores": scores,
            "marker_idx": marker_idx,
            "marker_ident": marker_ident,
            # MoR routing: per-marker-token recursion depth (importance signal)
            # and the raw router losses (z-loss, load-balancing) for the objective.
            "recursion_depth_per_token": route_info["depth_per_token"],
            "router_z_loss": route_info["z_loss"],
            "router_balance_loss": route_info["balance_loss"],
        }

    def transformer_param_count(self) -> int:
        """Parameters in the recursive transformer stack only (the quantity the
        weight-sharing claim is about)."""
        return sum(p.numel() for p in self.stack.parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
