#!/usr/bin/env python3
"""Geneformer foundation-model baseline on the CURRENT P-NET multi-omics cohorts
(prostate, blca, stad), directly comparable to SMART's pathway_tasks numbers.

Comparability
-------------
Data + split are taken VERBATIM from recursive_marker_transformer.pathway_tasks:
  * data:  load_cohort(<cohort>, channels="mut_cnv")
  * split: the exact two train_test_split calls (same random_state=seed, same
           conditional stratify) copied from pathway_tasks.run (lines ~188-194).
  * metric: sklearn macro-F1 on the held-out TEST split, x100.
Seeds 0,1,2; report mean+/-std over seeds.

mut_cnv -> pseudo-expression reduction (P-NET channels are mutation/CNV, NOT RNA)
-------------------------------------------------------------------------------
coh.X for mut_cnv is (N, G, 2): channel 0 = binary somatic mutation {0,1},
channel 1 = copy-number in {-2..2}. Geneformer's rank-value tokenizer expects a
non-negative per-gene magnitude (pseudo-count), so we reduce the two channels to
  pseudo_gene = |cnv| + mut_indicator      (>= 0)
i.e. a per-gene *alteration burden*: amplifications and deletions both contribute
their CNV dosage magnitude, +1 if the gene is somatically mutated; unaltered
genes -> 0 and are naturally ranked last / dropped. This is documented in the
output JSON "note" field.

Method: fine-tune BertForSequenceClassification (same pattern as run_geneformer.py).

Run inside the geneformer env:
  /work/mech-ai-scratch/tirtho/.venv_geneformer/bin/python run_geneformer_pnet.py \
      --cohorts prostate blca stad --seeds 0 1 2
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
GF = REPO / "lit_pipeline/baseline_repos/Geneformer"
OUT = REPO / "results_fm_pnet"
sys.path.insert(0, str(REPO))

REDUCTION_NOTE = (
    "P-NET mut_cnv channels reduced to a non-negative per-gene pseudo-expression "
    "'alteration burden' = |cnv| + mut_indicator (cnv in {-2..2}, mut in {0,1}); "
    "fed to Geneformer's rank-value tokenizer as pseudo-counts. Bulk mutation/CNV "
    "on a single-cell RNA foundation model is out-of-distribution (reported)."
)


def pseudo_expression(coh):
    """(N,G,2) mut+cnv -> (N,G) non-negative alteration burden."""
    X = coh.X
    ci = {c: i for i, c in enumerate(coh.channels)}
    if X.ndim == 2:  # single channel fallback
        return np.abs(X).astype(np.float32)
    mut = X[..., ci["mut"]] if "mut" in ci else np.zeros(X.shape[:2], np.float32)
    cnv = X[..., ci["cnv"]] if "cnv" in ci else np.zeros(X.shape[:2], np.float32)
    return (np.abs(cnv) + (mut != 0).astype(np.float32)).astype(np.float32)


def gene_mapping(genes):
    """HUGO symbols -> Ensembl ids present in Geneformer's token vocab."""
    name2id = pickle.load(open(GF / "geneformer/gene_name_id_dict_gc104M.pkl", "rb"))
    tokd = pickle.load(open(GF / "geneformer/token_dictionary_gc104M.pkl", "rb"))
    cols, ens = [], []
    for j, sym in enumerate(genes):
        g = name2id.get(sym)
        if g and g in tokd:
            cols.append(j); ens.append(g)
    return cols, ens


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


class _Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, feats):
        import torch
        L = max(len(f["input_ids"]) for f in feats)
        ids, att, lab = [], [], []
        for f in feats:
            x = list(f["input_ids"])[:4096]
            pad = L - len(x)
            ids.append(x + [self.pad_id] * pad)
            att.append([1] * len(x) + [0] * pad)
            lab.append(int(f["labels"]))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
                "labels": torch.tensor(lab)}


def run(cohort, seed, model_dir, work, epochs, smoke):
    sys.path.insert(0, str(GF))
    import anndata as ad, pandas as pd, datasets, torch
    from geneformer import TranscriptomeTokenizer
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (BertForSequenceClassification, Trainer, TrainingArguments)
    from recursive_marker_transformer.pathway_data import load_cohort

    coh = load_cohort(cohort, channels="mut_cnv")
    Xg = pseudo_expression(coh)                      # (N,G) >=0
    tr, va, te = make_split(coh, seed)
    cols, ens = gene_mapping(coh.genes)
    cov = len(cols) / max(1, len(coh.genes))
    print(f"[{cohort} s{seed}] N={len(coh.y)} G={len(coh.genes)} "
          f"usable_genes={len(cols)} ({cov*100:.1f}%) K={len(coh.classes)} "
          f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    if not cols:
        raise RuntimeError(f"{cohort}: 0/{len(coh.genes)} genes map to Geneformer vocab")

    counts = np.clip(np.rint(Xg[:, cols]), 0, None).astype(np.float32)
    split = np.array(["test"] * len(coh.y), dtype=object)
    split[tr] = "train"; split[va] = "train"     # probe/train on tr+va, eval on te
    obs = pd.DataFrame({"label": coh.y.astype(str), "split": split,
                        "n_counts": counts.sum(1)})
    var = pd.DataFrame(index=np.arange(len(ens))); var["ensembl_id"] = ens
    a = ad.AnnData(X=counts, obs=obs, var=var)
    keep = np.asarray(a.X.sum(1)).ravel() > 0
    a = a[keep].copy()

    wd = Path(work); wd.mkdir(parents=True, exist_ok=True)
    h5dir = wd / "h5"; h5dir.mkdir(exist_ok=True)
    tag = f"{cohort}_s{seed}"
    a.write_h5ad(h5dir / f"{tag}.h5ad")

    tok = TranscriptomeTokenizer({"label": "label", "split": "split"}, nproc=4)
    tok.tokenize_data(str(h5dir), str(wd), tag, file_format="h5ad")
    ds = datasets.load_from_disk(str(wd / f"{tag}.dataset"))

    classes = sorted(set(ds["label"]))
    c2i = {c: i for i, c in enumerate(classes)}
    ds = ds.map(lambda e: {"labels": c2i[e["label"]]})
    train_ds = ds.filter(lambda e: e["split"] == "train")
    test_ds = ds.filter(lambda e: e["split"] == "test")
    if smoke:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        test_ds = test_ds.select(range(min(16, len(test_ds))))

    pad_id = pickle.load(open(GF / "geneformer/token_dictionary_gc104M.pkl", "rb")).get("<pad>", 0)
    model = BertForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(classes))
    args = TrainingArguments(
        output_dir=str(wd / "trainer"), per_device_train_batch_size=8,
        per_device_eval_batch_size=16, num_train_epochs=1 if smoke else epochs,
        learning_rate=5e-5, warmup_ratio=0.1, weight_decay=0.01, logging_steps=25,
        fp16=torch.cuda.is_available(), report_to=[], save_strategy="no", seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=_Collator(pad_id))
    trainer.train()

    pred = trainer.predict(test_ds)
    yp = pred.predictions.argmax(-1)
    gt = pred.label_ids
    return {
        "cohort": cohort, "method": "Geneformer (V2-104M_CLcancer)", "seed": int(seed),
        "test_macro_f1": float(f1_score(gt, yp, average="macro")) * 100.0,
        "test_accuracy": float(accuracy_score(gt, yp)) * 100.0,
        "n_samples": int(len(coh.y)), "n_classes": int(len(classes)),
        "n_genes": int(len(coh.genes)), "gene_coverage": float(cov),
        "note": REDUCTION_NOTE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["prostate", "blca", "stad"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--model", default=str(GF / "Geneformer-V2-104M_CLcancer"))
    ap.add_argument("--work", default=str(OUT / "_geneformer_work"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    for cohort in args.cohorts:
        (OUT / cohort).mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            try:
                res = run(cohort, seed, args.model,
                          Path(args.work) / f"{cohort}_s{seed}", args.epochs, args.smoke)
                op = OUT / cohort / f"Geneformer_s{seed}.json"
                op.write_text(json.dumps(res, indent=1))
                print(f"[{cohort} s{seed}] macroF1={res['test_macro_f1']:.2f} "
                      f"acc={res['test_accuracy']:.2f} -> {op}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[{cohort} s{seed}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
