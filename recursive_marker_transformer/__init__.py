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
