"""PM (pan_meta_pri) routing-interpretability experiment, IN PARITY WITH TABLE 2.

For recursion depths K=1..4 and BOTH expert-choice and token-choice routing, train the
canonical bioMoR (bio_both, marker_mode=pathway) under the unified 5-fold CV (seed 42, 20%
test / 10%-of-train val) that produced Table 2, and record each Reactome pathway token's
MEAN RECURSION DEPTH over the held-out folds. A pathway routed to greater depth is KEPT (the
router keeps spending compute on it); a pathway that exits early is DROPPED. Writes
results/cv5/pm_routing/{pm_routing.json, pm_routing_report.md}.

Usage:  python scripts/pm_routing_experiment.py            # full: 2 routings x 4 depths x 5 folds
        python scripts/pm_routing_experiment.py --smoke    # expert K=2, 1 fold, 3 epochs
"""
import argparse, json, sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.pathway_data import load_cohort
from recursive_marker_transformer.pathway_tasks import _fit_eval
from recursive_marker_transformer.train import _depth_stats
from recursive_marker_transformer.cv import cv_folds, SEED, VAL_FRAC

TASK = "pan_meta_pri"; CH = "mut_cnv"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = {TASK: "multiclass"}


def base_cfg(routing, K, epochs, patience):
    # canonical bioMoR multi-omics (bio_both): learned pathway graph at BOTH sites, sum pool.
    return RMTConfig(heads=(TASK,), n_hvg=None, batch_size=32, d_model=128, d_ff=256,
                     n_markers=256, marker_mode="pathway", recursion_mode=routing,
                     recursion_depth=K, share_weights=True, seed=SEED, epochs=epochs,
                     patience=patience, lr=3e-4, weight_decay=1e-2, device=DEV,
                     gene_interaction="none", pathway_pool="sum",
                     pathway_learned_graph=True, pathway_learned_fuse=True,
                     bio_graph_router=True)


def run_config(routing, K, coh, X, y, G, C, nclass, folds, epochs, patience):
    torch.manual_seed(SEED); np.random.seed(SEED)   # parity: seed once per config (as run_cv)
    cfg = replace(base_cfg(routing, K, epochs, patience), n_channels=C)
    M = len(coh.pathways)
    depth_sum = np.zeros(M); active_sum = np.zeros(K); f1s = []
    for fi, (tr, va, te) in enumerate(folds):
        yt, yp, model, dl_te, _ = _fit_eval(TASK, coh, X, y, tr, va, te, cfg, G, nclass, DTYPES, DEV)
        f1s.append(100.0 * f1_score(yt, yp, average="macro"))
        msd, _midx, act = _depth_stats(model, dl_te, DEV, cfg)      # (M,), _, (K,)
        depth_sum += np.asarray(msd, dtype=float)
        active_sum += np.asarray(act, dtype=float)
        print(f"[pm-routing] {routing:6s} K={K} fold {fi+1}/{len(folds)} "
              f"macroF1={f1s[-1]:.1f}", flush=True)
    n = len(folds)
    return {"routing": routing, "K": K,
            "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
            "mean_depth": (depth_sum / n).tolist(),
            "active_per_step": (active_sum / n).tolist()}


def write_md(out, path, topn=25):
    paths = out["pathways"]; L = []
    L += ["# PM routing interpretability: which pathways are kept vs dropped", ""]
    L += [f"**Cohort:** `{out['task']}` ({out['channels']}) — {out['n_samples']} samples, "
          f"{out['n_pathways']} Reactome pathway tokens, {out['n_classes']} classes.  "]
    L += [f"**Protocol:** unified 5-fold CV, seed {out['seed']} (20% test / 10%-of-train val) "
          f"— identical folds to Table 2. Model: canonical bioMoR (`bio_both`, learned pathway "
          f"graph at both sites, sum-pooled).  "]
    L += ["**Keep/drop signal:** each pathway token's *mean recursion depth* over the held-out "
          "test folds. Under **expert-choice** a capacity funnel keeps a shrinking top-$k$ each "
          "step, so depth $\\in[0,K]$ counts how many steps the pathway survived; under "
          "**token-choice** each pathway self-gates one depth $\\in[1,K]$. Higher depth = the "
          "router **keeps** allocating compute to that pathway; the minimum = it is **dropped** "
          "(exits early).", ""]
    # summary
    L += ["## Summary", "",
          "| Routing | K | Macro-F1 | mean depth | active tokens per step | kept-to-K |",
          "|---|---|---|---|---|---|"]
    for r in out["results"]:
        md = np.array(r["mean_depth"]); K = r["K"]
        act = ", ".join(f"{a:.0f}" for a in r["active_per_step"])
        kept = int((md >= K - 0.5).sum())
        L.append(f"| {r['routing']} | {K} | {r['f1_mean']:.1f}$\\pm${r['f1_sd']:.1f} | "
                 f"{md.mean():.2f} | {act} | {kept}/{len(md)} |")
    L.append("")
    # per-config kept/dropped lists
    for r in out["results"]:
        md = np.array(r["mean_depth"]); K = r["K"]; order = np.argsort(-md)
        L += [f"## {r['routing']}-choice, K={K}  (Macro-F1 {r['f1_mean']:.1f}$\\pm${r['f1_sd']:.1f})", ""]
        if K == 1:
            L += ["_K=1 has a single pass, so there is no keep/drop decision "
                  "(every pathway is kept to depth 1)._", ""]
            continue
        L += [f"**KEPT — top {topn} deepest-routed pathways:**"]
        L += [f"{j+1}. `{paths[i]}` — depth {md[i]:.2f}" for j, i in enumerate(order[:topn])]
        L += ["", f"**DROPPED — bottom {topn} earliest-exit pathways:**"]
        L += [f"{j+1}. `{paths[i]}` — depth {md[i]:.2f}" for j, i in enumerate(order[::-1][:topn])]
        L += [""]
    path.write_text("\n".join(L))


def main(smoke=False):
    coh = load_cohort(TASK, channels=CH, min_genes=5)
    X = coh.X.astype(np.float32); y = coh.y
    G = X.shape[1]; nclass = int(y.max() + 1); C = 1 if X.ndim == 2 else X.shape[2]
    print(f"[pm-routing] {TASK} X={X.shape} pathways={len(coh.pathways)} classes={nclass} "
          f"channels={C} device={DEV}", flush=True)
    grid = [("expert", 2)] if smoke else [(r, K) for r in ("expert", "token") for K in (1, 2, 3, 4)]
    epochs, patience = (3, 2) if smoke else (100, 15)
    results = []
    for routing, K in grid:
        folds = list(cv_folds(y, n_folds=5, seed=SEED, val_frac=VAL_FRAC))
        if smoke:
            folds = folds[:1]
        results.append(run_config(routing, K, coh, X, y, G, C, nclass, folds, epochs, patience))
    out = {"task": TASK, "channels": CH, "n_pathways": len(coh.pathways), "n_classes": nclass,
           "n_samples": int(len(y)), "seed": SEED, "folds": 1 if smoke else 5,
           "pathways": list(coh.pathways), "results": results}
    d = ROOT / "results" / "cv5" / "pm_routing"; d.mkdir(parents=True, exist_ok=True)
    (d / "pm_routing.json").write_text(json.dumps(out, indent=1))
    write_md(out, d / "pm_routing_report.md")
    print(f"[pm-routing] wrote {d/'pm_routing.json'} and {d/'pm_routing_report.md'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
