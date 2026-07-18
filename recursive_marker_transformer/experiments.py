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

"""Run the full experiment suite for the GenomicRecursiveFormer paper.

Every result is written as JSON to ``--outdir`` so the paper-generation step
(``make_paper.py``) can pull real numbers straight from disk. Re-running this
script regenerates the numbers; re-running ``make_paper.py`` regenerates the
paper. Scale is controlled entirely by CLI flags / environment so the same code
produces a quick smoke paper or a full-scale one.

    python -m recursive_marker_transformer.experiments --epochs 12 --n_hvg 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import torch

from .config import RMTConfig
from .model import RecursiveMarkerTransformer
from .train import run


def param_efficiency_table(base: RMTConfig, depths=(1, 2, 4, 6, 8), n_genes=2000):
    """Transformer-stack parameter counts, shared vs independent, across depth.
    These are exact and require no training (the architectural claim)."""
    hc, hd = {"cancer_type": 4}, {"cancer_type": "multiclass"}
    rows = []
    for k in depths:
        entry = {"depth": k}
        for share in (True, False):
            cfg = replace(base, recursion_depth=k, share_weights=share)
            m = RecursiveMarkerTransformer(cfg, n_genes, hc, hd)
            key = "shared" if share else "independent"
            entry[f"{key}_params"] = m.transformer_param_count()
        entry["ratio"] = entry["independent_params"] / max(entry["shared_params"], 1)
        rows.append(entry)
    return rows


# Ablation variants -> config overrides. Each is one training run.
# The marker-selection study uses compress_mode="drop": only the selected genes
# are seen, so selection quality (learned vs random vs variance) is isolated.
SUITE = {
    # headline model: cross-attention marker router + shared recursion +
    # Mixture-of-Recursions expert-choice routing (per-token adaptive depth).
    # Also the "learned (router)" row of the selection study.
    "main":           dict(marker_mode="router",   share_weights=True, recursion_mode="expert"),
    # routing study (reproduces the MoR paper's comparison + uniform baseline)
    "mor_token":      dict(marker_mode="router", recursion_mode="token"),
    "fixed_depth":    dict(marker_mode="router", recursion_mode="fixed"),
    # marker-selection study: does *learning* the markers help, and *how*?
    # Run at FIXED depth so selection quality is isolated from the (lossy) MoR
    # router; the clean "router (learned)" row of this study is `fixed_depth`.
    "sel_concrete":   dict(marker_mode="concrete", recursion_mode="fixed"),
    "sel_variance":   dict(marker_mode="variance", recursion_mode="fixed", compress_mode="drop"),
    "sel_random":     dict(marker_mode="random",   recursion_mode="fixed", compress_mode="drop"),
    # architecture ablations (on the router model)
    "no_refine":      dict(marker_mode="router", recursive_marker_refine=False),
    "depth1":         dict(marker_mode="router", recursion_depth=1),
    "independent":    dict(marker_mode="router", share_weights=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("RMT_EPOCHS", 12)))
    ap.add_argument("--n_hvg", type=int, default=int(os.environ.get("RMT_NHVG", 2000)))
    ap.add_argument("--d_model", type=int, default=int(os.environ.get("RMT_DMODEL", 96)))
    ap.add_argument("--n_markers", type=int, default=int(os.environ.get("RMT_NMARKERS", 200)))
    ap.add_argument("--depth", type=int, default=int(os.environ.get("RMT_DEPTH", 4)))
    ap.add_argument("--heads", type=str, default=os.environ.get("RMT_HEADS", "cancer_type"))
    ap.add_argument("--only", type=str, default="", help="comma list to run a subset of the suite")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = RMTConfig(
        heads=tuple(args.heads.split(",")),
        n_hvg=args.n_hvg, d_model=args.d_model, d_ff=2 * args.d_model,
        n_markers=args.n_markers, recursion_depth=args.depth, epochs=args.epochs,
        patience=max(args.epochs, 6), recursion_mode="expert",
    )

    # Architectural parameter-efficiency table (instant, no training).
    pe = param_efficiency_table(base, n_genes=args.n_hvg)
    with open(os.path.join(args.outdir, "param_efficiency.json"), "w") as f:
        json.dump(pe, f, indent=2)
    print(f"[exp] wrote param_efficiency.json ({len(pe)} depths)")

    todo = SUITE if not args.only else {k: SUITE[k] for k in args.only.split(",")}
    manifest = {"base": base.as_dict(), "runs": []}
    for name, kw in todo.items():
        print(f"\n########## experiment: {name} ##########")
        t0 = time.time()
        cfg = replace(base, **kw)
        # Per-run marker file so the headline model's gene panel is preserved
        # (a shared markers_top.csv would be overwritten by the last run).
        res = run(cfg, markers_path=os.path.join(args.outdir, f"markers_{name}.csv"))
        res["name"] = name
        res["wall_seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(args.outdir, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        manifest["runs"].append(name)
        print(f"[exp] wrote {name}.json ({res['wall_seconds']}s)")

    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[exp] suite complete -> {args.outdir}/  (runs: {manifest['runs']})")


if __name__ == "__main__":
    main()
