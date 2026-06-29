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
from .marker import ConcreteSelector, MarkerModule, PathwayPooler, SlotRouter
from .recursion import RecursiveStack


class RecursiveMarkerTransformer(nn.Module):
    def __init__(self, cfg: RMTConfig, n_genes: int, head_n_classes: Dict[str, int],
                 head_dtypes: Dict[str, str],
                 pathway: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes
        self.head_dtypes = head_dtypes

        self.embed = GeneEmbedding(n_genes, cfg.d_model, cfg.dropout,
                                   n_channels=getattr(cfg, "n_channels", 1))
        self.marker = MarkerModule(cfg.d_model, n_genes, cfg.n_markers, cfg.marker_mode)
        # Soft selectors produce M marker tokens directly with all-gene gradient.
        if cfg.marker_mode == "concrete":
            self.selector = ConcreteSelector(n_genes, cfg.n_markers)
        elif cfg.marker_mode == "router":
            self.selector = SlotRouter(n_genes, cfg.n_markers, cfg.d_model)
            if getattr(cfg, "peak_init", True):
                self._peak_init_router()
        elif cfg.marker_mode == "pathway":
            # Reactome gene->pathway membership pooling (M = #pathways, fixed by
            # biology). M tokens = curated pathways, each pooling its member genes.
            if pathway is None:
                raise ValueError("marker_mode='pathway' requires the (N_genes, "
                                 "M_pathways) membership matrix")
            self.selector = PathwayPooler(pathway, pool=getattr(cfg, "pathway_pool", "mean"))
        else:
            self.selector = None
        self.stack = RecursiveStack(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            depth=cfg.recursion_depth, share_weights=cfg.share_weights,
            adaptive_depth=cfg.adaptive_depth, marker_ffn=cfg.marker_ffn,
            recursion_mode=cfg.recursion_mode, router_capacity=cfg.router_capacity,
            router_alpha=cfg.router_alpha, router_temp=cfg.router_temp,
            router_type=cfg.router_type,
            share_strategy=getattr(cfg, "share_strategy", "cycle"),
            n_unique_blocks=getattr(cfg, "n_unique_blocks", None),
            step_cache=getattr(cfg, "step_cache", False),
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
        # Biology-informed router: per-gene network-centrality prior (genomap
        # gene-gene interaction graph). Zero unless set; beta_t is the annealed
        # additive-bias strength, updated by set_anneal.
        self.register_buffer("gene_centrality", torch.zeros(n_genes), persistent=False)
        self._use_prior = getattr(cfg, "gene_interaction", None) not in (None, "none")
        self._prior_beta = float(getattr(cfg, "router_prior_beta", 0.0))
        # Per-token prior for pathway tokens: Reactome pathway-hierarchy centrality
        # (M,), one entry per pathway token. Used in place of the per-gene
        # gene_centrality when the tokens ARE pathways (marker_mode='pathway').
        M_tok = self.selector.n_markers if cfg.marker_mode == "pathway" else 1
        self.register_buffer("token_prior", torch.zeros(M_tok), persistent=False)
        self._use_token_prior = False
        # Pathway-hierarchy attention bias (Reactome pathway->pathway graph): an
        # additive (M,M) bias on the self-attention logits so a pathway token
        # attends to its hierarchy neighbours. Zero unless installed.
        self.register_buffer("pathway_attn", torch.zeros(M_tok, M_tok),
                             persistent=False)
        self._use_attn_bias = False
        self._attn_lambda = float(getattr(cfg, "pathway_attn_lambda", 0.0))

    def set_gene_variance(self, variance: torch.Tensor) -> None:
        self.gene_variance.copy_(variance.to(self.gene_variance))

    def set_gene_interaction(self, centrality: torch.Tensor) -> None:
        """Install the genomap gene-gene-interaction centrality prior (N,)."""
        self.gene_centrality.copy_(centrality.to(self.gene_centrality))
        self._use_prior = True

    def set_token_prior(self, prior: torch.Tensor) -> None:
        """Install a per-pathway-token prior (M,) -- e.g. Reactome pathway-graph
        eigenvector centrality. Added (annealed by beta_t) to the depth-router
        logits exactly like the gene prior, but indexed by token, not gene."""
        self.token_prior.copy_(prior.to(self.token_prior))
        self._use_token_prior = True

    def set_pathway_adjacency(self, adjacency: torch.Tensor) -> None:
        """Install the Reactome pathway->pathway hierarchy graph (M, M) as an
        additive attention bias: lambda on hierarchy-adjacent pathway pairs, 0
        elsewhere (self-attention on the diagonal is left unbiased). Pathway
        tokens then attend preferentially along the curated Reactome graph."""
        A = adjacency.to(self.pathway_attn).float()
        A = torch.maximum(A, A.t())
        A.fill_diagonal_(0.0)
        self.pathway_attn.copy_(self._attn_lambda * (A > 0).float())
        self._use_attn_bias = True

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
        """Advance the explore->exploit schedules. (1) the selector temperature
        (gated by anneal_markers); (2) the biological-prior strength beta_t, which
        decays linearly to 0 over training when router_prior_anneal is on (a
        warm-start prior that hands off to the data-driven router), else stays at
        beta_0."""
        progress = min(1.0, max(0.0, float(progress)))
        if getattr(self.cfg, "anneal_markers", True) and \
                self.selector is not None and hasattr(self.selector, "set_progress"):
            self.selector.set_progress(progress)
        if getattr(self.cfg, "router_prior_anneal", True):
            self._prior_beta = float(getattr(self.cfg, "router_prior_beta", 0.0)) * (1.0 - progress)
        else:
            self._prior_beta = float(getattr(self.cfg, "router_prior_beta", 0.0))

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        gene_identity = self.embed.gene_identity()              # (N, d)

        if self.selector is not None:
            # Soft selection (Concrete or cross-attention router): soft (train) /
            # hard (eval) selection over ALL genes -> M marker tokens directly, so
            # gradients reach every gene (the property hard top-k routing lacks).
            w = self.selector.weights(gene_identity)           # (M, N)
            sel_ident = w @ gene_identity                      # (M, d)
            if x.dim() == 3:
                # Multimodal: pool each gene-aligned channel by the same marker
                # weights, then fuse the C channels at the shared value projection.
                sel_value = torch.einsum("bnc,mn->bmc", x, w)  # (B, M, C)
                cluster = sel_ident.unsqueeze(0) + self.embed.value_proj(sel_value)
            else:
                sel_value = x @ w.t()                          # (B, M) selected expression
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

        # Biology-informed router: gather the centrality prior for the selected
        # markers and pass it (with the annealed strength beta_t) to the depth
        # router. For pathway tokens the prior is per-pathway (token_prior);
        # otherwise it is the per-gene centrality gathered at the selected markers
        # (marker_idx is the per-slot arg-max gene). Either way prior is (M,).
        if self._use_token_prior:
            prior = self.token_prior
            prior_weight = self._prior_beta
        elif self._use_prior:
            prior = self.gene_centrality[marker_idx]
            prior_weight = self._prior_beta
        else:
            prior, prior_weight = None, 0.0

        refine_fn = self.marker.refine_gate if self.cfg.recursive_marker_refine else None
        attn_bias = self.pathway_attn if self._use_attn_bias else None
        h, route_info = self.stack(cluster, refine_fn, prior=prior,
                                   prior_weight=prior_weight,
                                   attn_bias=attn_bias)         # (B, M, d), routing info
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
