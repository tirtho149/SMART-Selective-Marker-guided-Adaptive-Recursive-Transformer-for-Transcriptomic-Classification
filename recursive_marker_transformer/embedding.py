"""Stage 1: gene embedding.

Each gene becomes a token whose embedding combines a learned gene-identity
vector (indexed by gene position) with a projection of the scalar expression
value -- the standard scGPT / Geneformer gene+value scheme. Cost is O(N), so it
runs over all genes before any compression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeneEmbedding(nn.Module):
    def __init__(self, n_genes: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("gene_ids", torch.arange(n_genes), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N) scalar expression -> (B, N, d_model) tokens."""
        gene = self.gene_emb(self.gene_ids)                  # (N, d)
        value = self.value_proj(x.unsqueeze(-1))             # (B, N, d)
        tokens = gene.unsqueeze(0) + value                   # broadcast over batch
        return self.drop(self.norm(tokens))

    def gene_identity(self) -> torch.Tensor:
        """The batch-independent gene-identity matrix (N, d) for marker scoring
        and nearest-marker assignment."""
        return self.gene_emb(self.gene_ids)
