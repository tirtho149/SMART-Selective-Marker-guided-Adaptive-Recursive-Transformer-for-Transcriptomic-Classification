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

"""Ablation 9: do the learned markers overlap real biology?

Tests whether the top markers written by ``train.py`` (``markers_top.csv``) are
enriched for Reactome pathways more than a random gene panel of the same size,
via the hypergeometric test. Reuses ``TherapAgent``'s Reactome GMT ingestion
(``TherapAgent/path/reactome.py::load_gmt``).

    python -m recursive_marker_transformer.bio_enrichment --markers markers_top.csv

Requires internet on first run (downloads ReactomePathways.gmt.zip, then cached).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "TherapAgent"))

GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def _load_markers(path: str):
    genes = []
    with open(path) as f:
        for row in csv.DictReader(f):
            genes.append(row["gene"].upper())
    return genes


def _n_enriched(marker_set, gene_sets, universe, alpha=0.05):
    """Count pathways significantly enriched for ``marker_set`` (Bonferroni)."""
    N = len(universe)
    n = len(marker_set & universe)
    hits = 0
    tested = 0
    for genes in gene_sets.values():
        K = len(genes & universe)
        if K < 5:
            continue
        k = len(marker_set & genes & universe)
        if k == 0:
            continue
        tested += 1
        p = hypergeom.sf(k - 1, N, K, n)
        if p < alpha:
            hits += 1
    bonf = alpha / max(tested, 1)
    # recount with Bonferroni threshold
    hits = 0
    for genes in gene_sets.values():
        K = len(genes & universe)
        if K < 5:
            continue
        k = len(marker_set & genes & universe)
        if k == 0:
            continue
        if hypergeom.sf(k - 1, N, K, n) < bonf:
            hits += 1
    return hits, tested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markers", default="markers_top.csv")
    ap.add_argument("--top", type=int, default=200, help="use top-N markers")
    ap.add_argument("--trials", type=int, default=20, help="random-panel baseline draws")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from path.reactome import load_gmt  # TherapAgent

    cache = _REPO / "TherapAgent" / "binn" / "cache"
    os.makedirs(cache, exist_ok=True)
    print("[bio] loading Reactome gene sets ...")
    gene_sets = {k: {g.upper() for g in v} for k, v in load_gmt(cache, GMT_URL).items()}

    universe = set().union(*gene_sets.values())
    markers = _load_markers(args.markers)[: args.top]
    marker_set = set(markers) & universe
    print(f"[bio] universe={len(universe)} genes | markers in universe={len(marker_set)}/{len(markers)}")

    obs_hits, tested = _n_enriched(marker_set, gene_sets, universe)
    print(f"[bio] learned markers: {obs_hits} enriched pathways (of {tested} tested, Bonferroni 0.05)")

    rng = np.random.default_rng(args.seed)
    uni_list = sorted(universe)
    rand_hits = []
    for _ in range(args.trials):
        panel = set(rng.choice(uni_list, size=len(marker_set), replace=False))
        h, _ = _n_enriched(panel, gene_sets, universe)
        rand_hits.append(h)
    rand_hits = np.array(rand_hits)
    pval = float((rand_hits >= obs_hits).mean())
    print(f"[bio] random panels (n={args.trials}): mean={rand_hits.mean():.1f} "
          f"std={rand_hits.std():.1f} max={rand_hits.max()}")
    print(f"[bio] empirical p(random >= learned) = {pval:.3f} "
          f"-> {'enriched vs random' if pval < 0.05 else 'NOT distinguishable from random'}")


if __name__ == "__main__":
    main()
