<div align="center">

# bioMoR — Biology-guided Adaptive Recursive Transformer for Transcriptomic Classification

**Koushik Howlader¹ · Tirtho Roy¹ · Md Tauhidul Islam² · Wei Le¹**

¹ Iowa State University   ² Stanford University

*Reference implementation and full reproduction harness for the paper in `paper/` (`main.tex` + `supplementary.tex`). Manuscript under review — not yet accepted.*

</div>

---

## What this is

**bioMoR** is a biology-guided adaptive recursive transformer for transcriptomic
classification. It compresses the input into a small set of interpretable marker (single-cell)
or Reactome pathway (multi-omics) tokens, applies **one weight-shared block recursively**, and
combines data-driven and biology-prior scores to allocate deeper computation to informative
tokens. A *learned* low-rank interaction graph is injected at **both** the embedding and the
router (zero-init graph-conv) — the paper's main result. Everything is evaluated under a unified
**5-fold cross-validation** protocol (seed 42) across 8 single-cell suites and Reactome
multi-omics cohorts.

This repository reproduces **every table and figure in the paper** from the committed results in
`results/`, and documents the full **DATA → TRAINING → RESULTS → PAPER** pipeline to regenerate
those results from scratch on a GPU cluster.

## The pipeline at a glance

```
 (1) DATA            (2) TRAINING (GPU/slurm)      (3) RESULTS→PAPER (CPU, secs)   (4) PAPER
 data/  ──────────►  slurm/*.sbatch  ───────────►  results/cv5,repro,depth  ────►  scripts/refresh_cv5.sh
 (provided,           python -m recursive_          (committed JSON)                 → cv5_*.tex + figs/
  gitignored)          marker_transformer.<mod>                                      → pdflatex main/supp
```

Steps 3–4 need **no GPU** and run in seconds; step 2 is the only GPU stage. If you only want the
paper from the committed results, jump to **Step 3**.

## Setup

```bash
# Python 3.11 + torch 2.10.0+cu128 (CUDA build first on a GPU box, then the rest).
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt

export PY=/work/mech-ai-scratch/tirtho/.venv/bin/python   # or ./.venv/bin/python
```

Run every command **from the repository root**. GPU jobs go through **slurm** (never the login
node); the sbatch scripts self-activate the venv.

## Repository layout

```
.
├── recursive_marker_transformer/   # model + training package (source of truth)
│   ├── singlecell.py  pathway_tasks.py  bio_learned_genomap.py   # training entry points
│   ├── bio_redesign_curated.py  baselines11.py  scbignn_baseline.py
│   └── cv.py           # THE shared 5-fold split (seed 42) every experiment imports
├── genomic_dataloader/ genomap/ bio_networks/   # import-time training dependencies
├── data/               # datasets (provided, gitignored — see Step 1)
├── results/            # committed results the paper is built from
│   ├── cv5/            #   5-fold-CV JSON — feeds almost every table/figure (+ scbignn/, pm_routing/)
│   ├── repro/          #   PATH-protocol JSON — supplementary pos-F1 table
│   └── depth/          #   per-pathway depth panels
├── scripts/            # table/figure builders + refresh_cv5.sh
├── slurm/              # SLURM jobs that (re)produce results/ from scratch
├── paper/              # main.tex, supplementary.tex, cv5_*.tex fragments, figs/, refs.bib
└── archive/            # superseded code/results (gitignored; not needed to reproduce)
```

---

## Step 1 — Data (provided, gitignored)

`data/` is not script-generated; it ships alongside the code (large, gitignored). Each cohort
directory must contain the exact files its loader reads:

- **Single-cell** — two forms of the same 8 suites (Baron, Lung, Muraro, Oesophagus, Segerstolpe,
  Spleen, T-cell, Xin):
  - `singlecell.py` reads `data/singlecell/<name>/` with `expression.csv.gz` (index `cell_id`) +
    `labels.csv` (`cell_id,label`) — derived from genomap capsule 6967747 via the archived
    converter `archive/misc/tools/convert_capsule_to_csv.py`.
  - `bio_learned_genomap.py` and `baselines11.py` read the genomap `.mat/.npy` form from
    `genomap_data/` (auto-falls-back to `archive/misc/genomap_data/`, where the suites now live —
    verified). The two forms carry the same cells/labels.
- **Multi-omics / Reactome** (`recursive_marker_transformer/pathway_data.py`): each
  `data/{prostate,blca,stad,pan_meta_pri}/` with `filtered_pathways.csv`, `adjacency_matrix.csv`
  (the provided Reactome pathway graph), `mutation_data.csv`, `cnv_data.csv`, `patient_labels.csv`;
  `data/pan_meta_pri_3modal/` additionally has `expression_data.csv`.

There is no `make-data` step — verify the files above exist before training.

---

## Step 2 — Training (GPU, via slurm)

Retraining regenerates the JSON in `results/`. Each SLURM array writes into the subtree its
table/figure reads; all jobs use `--cv_folds 5 --seed 42 --epochs 100 --patience 15` (the shared
folds from `cv.py`). Submit from the repo root; `*_nova.sbatch` twins run the same command on the
`nova` account. Entry points are `python -m recursive_marker_transformer.<module>`.

| Result subtree (`results/…`) | SLURM job | Core command |
|---|---|---|
| `cv5/sc/<v>` (SC ladder) | `slurm/run_cv5_sc.sbatch` (64) | `$PY -m …singlecell --data data/singlecell --datasets $DS --d_model 96 --n_markers 128 --recursion_mode {expert,fixed,token} [--recursion_depth K] [--no_share_weights] --out results/cv5/sc/$V` |
| `cv5/mo/<v>` (MO ladder) | `slurm/run_cv5_mo.sbatch` (40) + `run_cv5_panmeta_fix.sbatch` | `$PY -m …pathway_tasks --task $T --channels $CH --marker_mode pathway --n_markers 256 --d_model 128 --batch_size 32 --gene_interaction reactome --out results/cv5/mo/$V` |
| `cv5/biomor_canonical{,_mo}` (headline bioMoR) | `run_canonical_biomor_{sc,mo}.sbatch` | SC: `$PY -m …bio_learned_genomap --dataset $D --modes bio_both --K 4 --n_markers 128 --out results/cv5/biomor_canonical` ; MO: `…pathway_tasks … --pathway_learned_graph --pathway_learned_fuse --pathway_attn_bias --recursion_mode token --out …_mo` |
| `cv5/biomor_ladder{,_mo}` | `run_biomorboth_ladder_{sc,mo,mo_pancan}.sbatch`, `run_cv5_tokenk.sbatch` | bioMoR at each `(recursion_mode,K)` rung |
| `cv5/inject_mo/<cond>` (injection fig) | `run_injection_{mo,pancan,3m}.sbatch` | `…pathway_tasks … --out results/cv5/inject_mo/$NAME`; cond flags none / `--bio_graph_router` (router) / `--pathway_learned_fuse` (embed) / both |
| `cv5/baselines` (Table 3) | `run_cv5_baselines.sbatch`, `run_baselines_newmo.sbatch` | `$PY -m …baselines11 --datasets $D --cv_folds 5 --out results/cv5/baselines` |
| `cv5/scbignn` (Table 3, scBiGNN) | `slurm/run_scbignn.sbatch` (Lung, PM) | `$PY -m …scbignn_baseline --cohort {Lung,pan_meta_pri} --pca_dim 128 --knn_k 15 --out results/cv5/scbignn` |
| `cv5/scaling_*` (supp. scaling) | `run_cv5_scale_{sc_gen,sc_biomor,mo_gen,mo_biomor}.sbatch` | same modules swept over `--d_model {96..352}` |
| `cv5/biorouter_ablation` (fig) | `run_biorouter_{ablation,prostate}.sbatch` | `bio_learned_genomap` / `bio_redesign_curated --modes $MODE` |
| `cv5/pm_routing` (interp. fig+tables) | `run_pm_routing.sbatch` | `$PY scripts/pm_routing_experiment.py` |
| `depth/` (supp. depth fig) | `run_prostate_panels.sbatch` | `$PY scripts/prostate_depth_panels.py` |
| `repro/ladder` (supp. pos-F1) | `run_ladder_posf1.sbatch`, `run_repro_all.sbatch` | `$PY scripts/reproduce_path.py --path_protocol …` |

Submit an array, e.g. `sbatch slurm/run_cv5_sc.sbatch`. Partial results render as `run…`
placeholders, so the paper always compiles while jobs are in flight.

---

## Step 3 — Results → Paper (CPU, seconds)

One command regenerates **all** table fragments + figures and recompiles both PDFs:

```bash
bash scripts/refresh_cv5.sh
```

Or run each piece individually. **Every table/figure maps to exactly one command:**

| Paper artifact | Regenerate with | Reads from `results/` |
|---|---|---|
| `cv5_main_table.tex` — Table 2 (efficiency ladder) | `$PY scripts/build_cv5_tex.py` | `cv5/{sc,mo,biomor_canonical,biomor_ladder,inject_mo,biomor_ladder_mo}` |
| `cv5_baselines_table.tex` — **Table 3** (classical + scBiGNN) | `$PY scripts/build_cv5_tex.py` | `cv5/{baselines,scbignn,biomor_canonical,inject_mo}` |
| `cv5_scaling_table.tex` — supp. scaling | `$PY scripts/build_cv5_tex.py` | `cv5/scaling_*` |
| `cv5_posf1_table.tex` — supp. positive-class F1 | `$PY scripts/build_posf1_table.py` | `repro/ladder` |
| `pm_depth_tables.tex` — supp. per-pathway depth | `$PY scripts/build_pm_depth_tables.py` | `cv5/pm_routing` |
| `figs/biorouter_bars.pdf` — Fig. injection ablation | `$PY scripts/make_biorouter_bars.py` | `cv5/biorouter_ablation` |
| `figs/baron_loss.pdf`, `baron_val_f1.pdf` — Fig. training dynamics | `$PY scripts/make_baron_epoch_figs.py` | `cv5/curves` |
| `figs/pm_depth_expert_vs_token.pdf` — Fig. recursion-depth keep/drop | `$PY scripts/make_pm_depth_figure.py` | `cv5/pm_routing` |
| `figs/fig2_depth.pdf` — supp. routing-depth panels | `$PY scripts/make_fig2_depth.py` | `depth` |
| `figs/pareto_efficiency.pdf` — Fig. accuracy–compute Pareto | `$PY scripts/pareto_prototype.py` | `cv5/` |
| `figs/overview.pdf` — Fig. 1 schematic | *static asset (hand-made)* | — |

`build_cv5_tex.py` writes three fragments at once (`cv5_{main,scaling,baselines}_table.tex`);
`refresh_cv5.sh` calls all nine generators, then compiles.

## Step 4 — Build the paper

```bash
cd paper
for doc in main supplementary; do
  pdflatex -interaction=nonstopmode $doc.tex && bibtex $doc \
    && pdflatex -interaction=nonstopmode $doc.tex \
    && pdflatex -interaction=nonstopmode $doc.tex
done
```

(`scripts/refresh_cv5.sh` already runs this full `pdflatex×2 → bibtex → pdflatex×2` sequence after
regenerating the fragments, so references resolve.)

## Reproducibility

Every table fragment regenerates **byte-identical** from the committed `results/`, all figure
generators run clean, and the headline numbers (avg macro-F1 67.1→73.5, 75% fewer parameters,
up to 58% fewer FLOPs, and the injection ordering None<Router<Embedding<Both) are consistent with
the generated tables. The **scBiGNN** baseline (Table 3) is a faithful re-implementation
(arXiv:2312.10310) trained on the same 5-fold folds via `slurm/run_scbignn.sbatch`.

## Citation

> **Status:** manuscript **under review** — not yet accepted or published.

```bibtex
@unpublished{howlader2026biomor,
  title  = {Biology-guided Adaptive Recursive Transformer for Transcriptomic Classification},
  author = {Howlader, Koushik and Roy, Tirtho and Islam, Md Tauhidul and Le, Wei},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

See `LICENSE`. Copyright © 2026 the authors. All rights reserved.
