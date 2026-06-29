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

"""Stage 1: gene embedding.

Each gene becomes a token whose embedding combines a learned gene-identity
vector (indexed by gene position) with a projection of the per-gene value
channels -- the standard scGPT / Geneformer gene+value scheme, generalised to
``n_channels`` aligned assays. With ``n_channels == 1`` the channel is the scalar
expression value (the original SMART input); with ``n_channels > 1`` the channels
are the gene-aligned multimodal measurements (expression, copy-number, mutation),
fused by a single shared value projection so every modality of a gene lands on
the same token. Cost is O(N), so it runs over all genes before any compression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeneEmbedding(nn.Module):
    def __init__(self, n_genes: int, d_model: int, dropout: float = 0.1,
                 n_channels: int = 1):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.n_channels = n_channels
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.value_proj = nn.Linear(n_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("gene_ids", torch.arange(n_genes), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N) scalar or (B, N, C) multichannel -> (B, N, d_model) tokens."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)                              # (B, N, 1)
        gene = self.gene_emb(self.gene_ids)                  # (N, d)
        value = self.value_proj(x)                           # (B, N, d)
        tokens = gene.unsqueeze(0) + value                   # broadcast over batch
        return self.drop(self.norm(tokens))

    def gene_identity(self) -> torch.Tensor:
        """The batch-independent gene-identity matrix (N, d) for marker scoring
        and nearest-marker assignment."""
        return self.gene_emb(self.gene_ids)
