"""Recursive Marker Transformer (RMT).

A parameter-efficient transformer for gene-expression classification where
parameter efficiency is an architectural property: a learnable marker head
selects discriminative genes, non-markers are compressed into marker-anchored
cluster tokens, and a single transformer block is applied K times with
recursive marker refinement.
"""

from .config import RMTConfig
from .model import RecursiveMarkerTransformer

__all__ = ["RMTConfig", "RecursiveMarkerTransformer"]
