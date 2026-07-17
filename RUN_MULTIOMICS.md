# Running the Multi-Omics Experiments (Table 2 & Table 3)

This runbook reproduces the **multi-omics** results for the two main tables of the
SMART / bioMoR paper, **each table separately**. Single-cell datasets are covered by the
genomap runner; here we document only the multi-omics (Reactome / P-NET) cohorts.

All multi-omics runs use the **provided** Reactome pathway graph
(`data/<cohort>/adjacency_matrix.csv`) — never a co-expression graph computed on the sparse
mutation/CNV data — and mutation-burden pooling (`--pathway_pool sum`) for the sparse
mutation channel.

## 0. Environment & protocol

```bash
source /work/mech-ai-scratch/tirtho/.venv/bin/activate     # torch 2.10 cu128
export PYTHONPATH=$PWD
```

Shared protocol (identical across both tables): `marker_mode=pathway`, `recursion_mode=expert`,
`K=4`, `n_markers=256`, `d_model=128`, `batch_size=32`, `lr=3e-4`, `epochs=100`, `patience=15`,
**5-fold CV, seed 42**. Runner: `python -m recursive_marker_transformer.pathway_tasks`.

Cohorts & channels:

| Tag | `--task` | `--channels` | Note |
|---|---|---|---|
| Prostate | `prostate` | `mut_cnv` | P-NET primary-vs-metastatic (clean) |
| BLCA | `blca` | `mut_cnv` | |
| STAD | `stad` | `mut_cnv` | |
| BRCA | `brca` | `mut` | mutation-only |
| PM | `pan_meta_pri` | `mut_cnv` | pan-cancer (melanoma-confounded, appendix) |
| PC | `panmeta_response` | `expr` | pan-cancer (melanoma-confounded, appendix) |
| 3M | `pan_meta_pri_3modal` | `mut_cnv_expr` | tri-modal (melanoma-confounded, appendix) |

> **Confound note:** PM/PC/3M metastatic class = 100% melanoma (SKCM), so a linear probe
> hits ~99%. Report them as a confounded/appendix sanity task, not headline.

---

## Table 2 — efficiency ladder (multi-omics rows)

Table 2's bioMoR column = the **both-sites** model (`bio_both`), read from
`results_cv5/inject_mo/both/` (K=4 headline) and `results_cv5/biomor_ladder_mo/<cfg>/`
(K2/K3/token ladder rows). The Vanilla / Recursive / MoR baseline rows are the
no-biology pathway ladder in `results_cv5/mo/`.

**bioMoR ladder (K=4 headline + K2/K3/token):**
```bash
sbatch slurm/run_injection_mo.sbatch          # cond 'both' -> inject_mo/both (K=4 headline)
sbatch slurm/run_biomorboth_ladder_mo.sbatch  # bio_both at K2/K3 + token variants
```

**Or a single cohort/config by hand** (headline K=4 both):
```bash
python -m recursive_marker_transformer.pathway_tasks \
  --task prostate --channels mut_cnv --marker_mode pathway --recursion_mode expert \
  --recursion_depth 4 --gene_interaction none \
  --pathway_learned_graph --pathway_learned_fuse --bio_graph_router \
  --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 \
  --batch_size 32 --lr 3e-4 --cv_folds 5 --device cuda --out results_cv5/inject_mo/both
```

**Baseline MoR rows** (no biology) live in `results_cv5/mo/<arm>/` (arms: `independent`,
`fixed[_k2/_k3]`, `shared`/`expert_k2`/`expert_k3`, `token[_k2/_k3]`); regenerate with the
same command but `--gene_interaction none` and **no** `--pathway_learned_graph/--bio_graph_router`.

Render: `python build_cv5_tex.py` → `paper/cv5_main_table.tex`.

---

## Table 3 — injection-site ablation (multi-omics rows)

Where should the biological interaction enter? Four conditions per cohort, identical
protocol, differing **only** in the injection site (all write to
`results_cv5/inject_mo/<cond>/`):

| Condition | Flags | Meaning |
|---|---|---|
| `none` | `--gene_interaction none` | no biology |
| `router` | `--gene_interaction none --pathway_learned_graph --no_pathway_prop --bio_graph_router` | graph feeds ONLY the depth router |
| `embed` | `--gene_interaction none --pathway_learned_graph --pathway_learned_fuse` | graph smooths the token embedding |
| `both` | `--gene_interaction none --pathway_learned_graph --pathway_learned_fuse --bio_graph_router` | both (= bioMoR) |

**Run the P-NET cohorts + pan-cancer:**
```bash
sbatch slurm/run_injection_mo.sbatch       # prostate/blca/stad/brca x {none,router,embed,both}
sbatch slurm/run_injection_pancan.sbatch   # PM (pan_meta_pri) + PC (panmeta_response)
sbatch slurm/run_injection_3m.sbatch       # 3M (pan_meta_pri_3modal), tri-modal, high-mem
```

**Single cell by hand** (e.g. STAD, `both`):
```bash
python -m recursive_marker_transformer.pathway_tasks \
  --task stad --channels mut_cnv --marker_mode pathway --recursion_mode expert \
  --recursion_depth 4 --gene_interaction none \
  --pathway_learned_graph --pathway_learned_fuse --bio_graph_router \
  --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 \
  --batch_size 32 --lr 3e-4 --cv_folds 5 --device cuda --out results_cv5/inject_mo/both
```

Render: `python build_injection_table.py` → `paper/cv5_injection_table.tex`.

---

## Redesigned bio-router (what the flags mean)

The router-site biology is a **zero-init graph-conv residual** on the depth-router logits
(`recursive_marker_transformer/router.py`), fed the neighbourhood mean **and the
high-frequency contrast** `[A H, H − A H]` through a nonlinear MLP whose output layer is
zero-initialised (starts as a no-op → cannot collapse; learns to help). A per-step scalar
gate `σ(g)` keeps "both" at embedding-only until the router earns its weight. Formal proof:
`BIO_LEARNED_GRAPH_PROOF.md` §8 (Theorem 2 monotone safety, Prop 4 why the old static
centrality prior collapsed, Cor C5 both-sites dominance).

## Live table refresh

`sbatch slurm/run_refresh_loop.sbatch` regenerates `paper/cv5_{main,injection}_table.tex`
every 90 s (pending cells show `run…`); the Overleaf sync pushes them.
