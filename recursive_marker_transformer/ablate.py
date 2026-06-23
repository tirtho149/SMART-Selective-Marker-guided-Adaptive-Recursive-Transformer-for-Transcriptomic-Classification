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

"""Ablation presets (proposal Ablations 1-9).

    python -m recursive_marker_transformer.ablate --which 3
    python -m recursive_marker_transformer.ablate --which 4 --base epochs=10 n_hvg=2000

Each ablation yields one or more RMTConfig variants run back-to-back; results
are printed as a compact table. Ablation 9 (biological enrichment) is handled
separately in ``bio_enrichment.py`` since it needs Reactome gene sets, not a
training run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Dict, List, Tuple

from .config import RMTConfig
from .train import run, _parse_overrides


def variants(which: int, base: RMTConfig) -> List[Tuple[str, RMTConfig]]:
    if which == 1:   # recursion depth: with vs without
        return [("no-recursion (K=1)", replace(base, recursion_depth=1)),
                ("recursive (K=4)", replace(base, recursion_depth=4))]
    if which == 2:   # marker learning vs random genes
        return [("learnable-markers", replace(base, marker_mode="learnable")),
                ("random-markers", replace(base, marker_mode="random"))]
    if which == 3:   # parameter sharing vs independent layers
        return [("shared (RMT)", replace(base, share_weights=True)),
                ("independent layers", replace(base, share_weights=False))]
    if which == 4:   # number of markers
        return [(f"M={m}", replace(base, n_markers=m)) for m in (100, 500, 1000, 2000)]
    if which == 5:   # recursion depth sweep (+ adaptive)
        out = [(f"K={k}", replace(base, recursion_depth=k)) for k in (1, 2, 4, 6, 8)]
        out.append(("K=4 adaptive", replace(base, recursion_depth=4, adaptive_depth=True)))
        return out
    if which == 6:   # marker selection method
        return [("variance", replace(base, marker_mode="variance")),
                ("learnable", replace(base, marker_mode="learnable"))]
    if which == 7:   # shared FFN vs dedicated marker FFN
        return [("shared-ffn", replace(base, marker_ffn=False)),
                ("marker-ffn", replace(base, marker_ffn=True))]
    if which == 8:   # compression ratio (via M relative to n_hvg)
        n = base.n_hvg or 20000
        return [(f"~{n//m}x (M={m})", replace(base, n_markers=m))
                for m in (n // 5, n // 10, n // 20, n // 50) if m >= 16]
    raise SystemExit(f"Ablation {which} has no training preset (see bio_enrichment.py for 9).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, required=True)
    ap.add_argument("base", nargs="*", help="key=value overrides applied to every variant")
    args = ap.parse_args()

    base = RMTConfig.from_overrides(**_parse_overrides(args.base))
    rows: Dict[str, dict] = {}
    for name, cfg in variants(args.which, base):
        print(f"\n########## Ablation {args.which}: {name} ##########")
        res = run(cfg)
        prim = cfg.heads[0]
        rows[name] = {
            "macro_f1": res["heads"][prim]["macro_f1"],
            "accuracy": res["heads"][prim]["accuracy"],
            "tf_params": res["transformer_params"],
            "flops": res["approx_flops_per_sample"],
        }

    print(f"\n===== Ablation {args.which} summary (primary head: {base.heads[0]}) =====")
    print(f"{'variant':<24}{'macroF1':>9}{'acc':>8}{'tf_params':>14}{'flops':>16}")
    for name, r in rows.items():
        print(f"{name:<24}{r['macro_f1']:>9.4f}{r['accuracy']:>8.4f}"
              f"{r['tf_params']:>14,}{r['flops']:>16,}")


if __name__ == "__main__":
    main()
