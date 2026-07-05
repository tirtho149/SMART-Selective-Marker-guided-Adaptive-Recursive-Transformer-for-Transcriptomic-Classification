#!/usr/bin/env python3
"""scGPT foundation-model baseline on the CURRENT P-NET multi-omics cohorts
(prostate, blca, stad): frozen pan-cancer cell embedding + LogisticRegression
probe. Directly comparable to SMART's pathway_tasks numbers.

Comparability
-------------
Data + split taken VERBATIM from recursive_marker_transformer.pathway_tasks:
  * data:  load_cohort(<cohort>, channels="mut_cnv")
  * split: the exact two train_test_split calls (same random_state=seed, same
           conditional stratify) copied from pathway_tasks.run (lines ~188-194).
  * metric: sklearn macro-F1 on the held-out TEST split, x100. Seeds 0,1,2.
The LR probe is fit on train+val (tr+va), evaluated on the identical te split.

mut_cnv -> pseudo-expression reduction
--------------------------------------
coh.X for mut_cnv is (N, G, 2): ch0 = binary somatic mutation {0,1}, ch1 = copy
number {-2..2}. scGPT rank-bins a non-negative per-gene value, so we reduce to
  pseudo_gene = |cnv| + mut_indicator      (>= 0)  -- per-gene alteration burden.
Documented in the output JSON "note".

flash-attn absent -> use_fast_transformer=False. Requires an A100/H200-class GPU
(CC7.0/V100 previously broke scGPT).

Run inside the scgpt env:
  /work/mech-ai-scratch/tirtho/.venv_scgpt/bin/python run_scgpt_pnet.py \
      --cohorts prostate blca stad --seeds 0 1 2
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
# local repo scgpt FIRST (torchtext-free shim), then the project root for the loader
sys.path.insert(0, str(REPO / "lit_pipeline/baseline_repos/scGPT"))
sys.path.insert(0, str(REPO))
OUT = REPO / "results_fm_pnet"
CKPT = REPO / "lit_pipeline/baseline_repos/scGPT/checkpoints/pan-cancer"

REDUCTION_NOTE = (
    "P-NET mut_cnv channels reduced to a non-negative per-gene pseudo-expression "
    "'alteration burden' = |cnv| + mut_indicator (cnv in {-2..2}, mut in {0,1}); "
    "fed to scGPT's rank-binning tokenizer. Frozen pan-cancer embedding + LR probe. "
    "Bulk mutation/CNV on a single-cell RNA foundation model is OOD (reported)."
)


def pseudo_expression(coh):
    X = coh.X
    ci = {c: i for i, c in enumerate(coh.channels)}
    if X.ndim == 2:
        return np.abs(X).astype(np.float32)
    mut = X[..., ci["mut"]] if "mut" in ci else np.zeros(X.shape[:2], np.float32)
    cnv = X[..., ci["cnv"]] if "cnv" in ci else np.zeros(X.shape[:2], np.float32)
    return (np.abs(cnv) + (mut != 0).astype(np.float32)).astype(np.float32)


def make_split(coh, seed):
    """VERBATIM copy of pathway_tasks.run split (lines ~188-194)."""
    from sklearn.model_selection import train_test_split
    y = coh.y
    idx = np.arange(len(y))
    _, cnt = np.unique(y, return_counts=True)
    strat = y if cnt.min() >= 2 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    _, cnt = np.unique(y[tr], return_counts=True)
    strat = y[tr] if cnt.min() >= 2 else None
    tr, va = train_test_split(tr, test_size=0.15, random_state=seed, stratify=strat)
    return tr, va, te


def run(cohort, seed, smoke):
    import anndata as ad, pandas as pd, torch
    from scgpt.tasks import embed_data
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from recursive_marker_transformer.pathway_data import load_cohort

    coh = load_cohort(cohort, channels="mut_cnv")
    Xg = pseudo_expression(coh)                      # (N,G) >=0
    genes = list(coh.genes)
    # dedup symbols keeping first (scGPT matches gene-symbol vocab)
    seen, cols, syms = set(), [], []
    for j, s in enumerate(genes):
        if s and s != "?" and s not in seen:
            seen.add(s); cols.append(j); syms.append(s)
    X = Xg[:, cols].astype(np.float32)
    y = coh.y
    # A few P-NET patients carry NO mapped mutation/CNV alteration -> an all-zero row,
    # which scGPT's rank-binning tokenizer cannot process (empty positive set -> the
    # row.max() ValueError seen on prostate/stad). Give such patients a tiny uniform
    # baseline: an honest "no detected alteration" cell, no label leakage, split intact.
    zero_rows = X.sum(axis=1) == 0
    if zero_rows.any():
        X[zero_rows] = 1e-3
        print(f"[{cohort} s{seed}] {int(zero_rows.sum())} all-zero patient rows -> uniform baseline", flush=True)

    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame({"label": y.astype(str)}))
    a.var["gene_name"] = syms
    a.var_names = syms
    print(f"[{cohort} s{seed}] N={len(y)} G={len(genes)} deduped_genes={len(syms)} "
          f"K={len(coh.classes)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = embed_data(a, str(CKPT), gene_col="gene_name", device=dev,
                   use_fast_transformer=False, return_new_adata=False)
    emb = np.asarray(a.obsm["X_scGPT"])
    # scGPT's embed_data drops genes absent from its vocab; coverage = matched/total
    try:
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        vocab = GeneVocab.from_file(CKPT / "vocab.json")
        matched = [s for s in syms if s in vocab]
        cov = len(matched) / max(1, len(genes))
    except Exception:
        cov = float("nan")

    tr, va, te = make_split(coh, seed)
    fit_idx = np.concatenate([tr, va])
    if smoke:
        fit_idx = fit_idx[:min(64, len(fit_idx))]
    clf = LogisticRegression(max_iter=2000).fit(emb[fit_idx], y[fit_idx])
    yp = clf.predict(emb[te])
    return {
        "cohort": cohort, "method": "scGPT (pan-cancer)", "seed": int(seed),
        "test_macro_f1": float(f1_score(y[te], yp, average="macro")) * 100.0,
        "test_accuracy": float(accuracy_score(y[te], yp)) * 100.0,
        "n_samples": int(len(y)), "n_classes": int(len(coh.classes)),
        "n_genes": int(len(genes)), "gene_coverage": float(cov),
        "note": REDUCTION_NOTE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["prostate", "blca", "stad"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    for cohort in args.cohorts:
        (OUT / cohort).mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            try:
                res = run(cohort, seed, args.smoke)
                op = OUT / cohort / f"scGPT_s{seed}.json"
                op.write_text(json.dumps(res, indent=1))
                print(f"[{cohort} s{seed}] macroF1={res['test_macro_f1']:.2f} "
                      f"acc={res['test_accuracy']:.2f} cov={res['gene_coverage']:.3f} "
                      f"-> {op}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[{cohort} s{seed}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
