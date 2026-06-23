"""Reviewer-requested extras on the main 4-cohort cancer-type task:

  multiseed : the headline SMART config over 5 seeds -> mean +/- std on the
              primary task (reviewer asked for variance on the main result).
  init_anneal : isolates the two router design choices the paper argues for --
              peaked initialisation and temperature annealing -- as a 2x2 grid
              (peak_init in {on,off} x anneal in {on,off}) over 3 seeds.

Uses the same TCGA 4-cohort loader and config as the headline run (train.run).
Resumable: one JSON per cell under results_extra/.

    python -m recursive_marker_transformer.extra_experiments --exp multiseed init_anneal
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import RMTConfig
from .train import run

# headline config (matches the full GPU sweep that produced results/main.json)
BASE = dict(heads=("cancer_type",), n_hvg=4000, d_model=128, d_ff=256, n_markers=256,
            marker_mode="router", recursion_mode="expert", recursion_depth=4,
            epochs=25, patience=25)


def _cell(out: Path, name, cfg):
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{name}.json"
    if f.exists():
        print(f"  [skip] {f.name}", flush=True)
        return
    r = run(cfg, markers_path=str(out / f"markers_{name}.csv"))
    h = r["heads"]["cancer_type"]
    rec = {"macro_f1": h["macro_f1"], "accuracy": h["accuracy"],
           "transformer_params": r.get("transformer_params"),
           "total_params": r.get("total_params"),
           "seed": cfg.seed, "peak_init": cfg.peak_init,
           "anneal_markers": cfg.anneal_markers}
    f.write_text(json.dumps(rec, indent=1))
    print(f"  [done] {name}  macroF1={h['macro_f1']*100:.2f} acc={h['accuracy']*100:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_extra"))
    ap.add_argument("--exp", nargs="*", default=["multiseed", "init_anneal"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ia_seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    if "multiseed" in args.exp:
        print("\n===== multi-seed main (cancer_type) =====", flush=True)
        for s in args.seeds:
            cfg = RMTConfig(**BASE, seed=s)
            _cell(args.out / "multiseed", f"seed{s}", cfg)

    if "init_anneal" in args.exp:
        print("\n===== init/anneal ablation (cancer_type) =====", flush=True)
        for pk in (True, False):
            for an in (True, False):
                for s in args.ia_seeds:
                    cfg = RMTConfig(**BASE, seed=s, peak_init=pk, anneal_markers=an)
                    tag = f"peak{int(pk)}_anneal{int(an)}_seed{s}"
                    _cell(args.out / "init_anneal", tag, cfg)

    print("\n[extra] done -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
