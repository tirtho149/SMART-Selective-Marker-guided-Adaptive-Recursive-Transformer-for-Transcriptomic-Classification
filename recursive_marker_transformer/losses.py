"""Composite objective:  L = task + lambda*marker + gamma*diversity + beta*compression.

- task        : weighted CE (multiclass) / BCE (binary), summed over heads.
- marker      : CE of an auxiliary classifier fed only the selected marker tokens
                -> pushes the marker head to pick task-sufficient genes.
- diversity   : off-diagonal energy of the marker-identity Gram matrix
                -> prevents all markers collapsing to the same direction.
- compression : mean importance (sigmoid of scores) -> sparse marker distribution.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RMTConfig


class RMTLoss(nn.Module):
    def __init__(self, cfg: RMTConfig, head_dtypes: Dict[str, str],
                 class_weights: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.cfg = cfg
        self.head_dtypes = head_dtypes
        self.class_weights = class_weights or {}
        self._primary = cfg.heads[0]

    def _task_term(self, head: str, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.head_dtypes[head] == "binary":
            return F.binary_cross_entropy_with_logits(logit.squeeze(-1), target.float())
        w = self.class_weights.get(head)
        return F.cross_entropy(logit, target.long(), weight=w)

    def forward(self, out: Dict[str, object], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = out["logits"]
        task = sum(self._task_term(h, logits[h], targets[h]) for h in logits)

        # marker-sufficiency aux loss on the primary head
        marker = self._task_term(self._primary, out["aux_logits"], targets[self._primary])

        ident = out["marker_ident"]                          # (M, d), L2-normalised
        gram = ident @ ident.t()
        m = gram.shape[0]
        off = gram - torch.eye(m, device=gram.device)
        diversity = (off ** 2).sum() / (m * (m - 1) + 1e-8)

        if self.cfg.marker_mode == "learnable":
            compression = torch.sigmoid(out["scores"]).mean()
        else:
            compression = torch.zeros((), device=ident.device)

        # MoR router regularisation: z-loss (both regimes) + token-choice
        # load-balancing. Zero when recursion_mode == "fixed".
        z = out.get("router_z_loss", torch.zeros((), device=ident.device))
        bal = out.get("router_balance_loss", torch.zeros((), device=ident.device))
        router = self.cfg.router_z_coeff * z + self.cfg.router_balance_coeff * bal

        total = (task
                 + self.cfg.lambda_marker * marker
                 + self.cfg.gamma_diversity * diversity
                 + self.cfg.beta_compression * compression
                 + router)
        return {
            "total": total, "task": task, "marker": marker,
            "diversity": diversity, "compression": compression, "router": router,
        }
