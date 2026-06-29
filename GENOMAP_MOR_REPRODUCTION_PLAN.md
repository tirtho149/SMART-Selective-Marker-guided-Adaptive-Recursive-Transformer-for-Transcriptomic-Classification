# Plan: reproduce the MoR-paper tables with SMART, on the full genomap dataset suite

> Goal (user): use **all genomap-paper datasets** (Islam & Xing, *Nat Commun* 2023,
> s41467-023-36383-6), following that paper's experimental *pattern*, and reproduce
> **every table of the MoR paper** (Bae et al. 2025, arXiv:2507.10524) — but with SMART
> as the model. **No TCGA** anywhere (already dropped from the rendered paper). Justify
> the three SMART claims — **adaptive recursion loop, token reduction, parameter
> reduction** — with the corresponding MoR-style experiments.

## A. Data — the genomap suite (the "pattern")
The genomap paper's task is **cell-type classification on genomap-imaged scRNA-seq**.
Its datasets (accessions): Baron `GSE84133`, Muraro `GSE85241`, Segerstolpe `E-MTAB-5061`,
Xin `GSE81608`, Wang `GSE83139`, Tabula Muris `GSE109774`, T-cell landscape `SCP490`,
proto-vertebrate `SCP454`, retinal bipolar `SCP3`, ischaemic `PRJEB31843`.

**Local now** (genomap Code Ocean capsule 6967747, already converted, `data/singlecell/`):
| dataset | cells | genomap feats | classes | split |
|---|---|---|---|---|
| tabula_muris | 54,865 | 1089 (33×33) | 55 | yes |
| pancreas | 14,767 | 1936 (44×44) | 15 | yes |
| common_class | 27,499 | 1089 | 19 | none |
| prototype | 90,579 | 752 | 10 | none |

- **Tier 1 (now):** run on all **4** local capsule datasets = "maximum genomap" with zero
  new data work. Paper currently uses only TM+pancreas → add common_class + prototype.
- **Tier 2 (optional, more data):** rebuild the individual raw datasets (Baron/Muraro/
  Segerstolpe/Xin/Wang pancreas; SCP490/454/3) into genomap features via genomap's own
  `construct_genomap` (the `genomap/` package is vendored here), then run the same suite.
  This matches the genomap paper one-dataset-at-a-time pattern most literally.
- The new **pathway/P-NET** cohorts (prostate/blca/stad/brca/panmeta) ride the same
  experiment grid so the three claims are shown on bulk multi-omics too.

## B. MoR table-by-table reproduction map
**ALL 14 tables are reproduced** (user: "exactly all the tables"). None dropped — the
two autoregressive-LLM-only tables get a documented set-classifier analogue:
- **T9 (uptraining)** → **warm-start**: initialise the shared recursive block from a
  trained *fixed-depth / vanilla* SMART, then continue-train into the recursive model;
  report the gain vs from-scratch. (= "convert a non-recursive model into a recursive one".)
- **T12 (KV-cache sharing)** → **step-cache**: compute the attention key/value projections
  at recursion step 1 and **reuse vs recompute** them across the K steps; report the
  accuracy/compute trade-off. (The one-shot-encoder analogue of MoR's recursion-wise KV cache.)
Both are labelled "adapted (non-autoregressive analogue)" in the caption — honest, not faked.

Legend: **[R]** reproduce directly · **[A]** adapt to the set-classifier setting (incl. T9/T12).

| MoR table | What it is | SMART/genomap reproduction | claim |
|---|---|---|---|
| **T3** MoR vs Recursive vs Vanilla @ fixed FLOPs/tokens | headline | **[R]** SMART-MoR(expert) vs Recursive(fixed-depth, shared) vs Vanilla(K independent blocks / K=1) on each genomap dataset; report acc, params, FLOPs, mean depth — *the* main table | all 3 |
| **T4** expert- & token-choice router ablation (sampling, aux-loss, balance, router linear/MLP, z-loss) | the screenshot | **[R]** router ablation: expert vs token × {router_type linear/MLP, router_z_coeff, balance_coeff, capacity, aux marker-loss on/off} on TM+pancreas | adaptive loop |
| **T1/T5/T8 + Fig6** parameter-sharing schemes (Cycle, Middle-Cycle, Sequence, Middle-Sequence) | param sharing | **[A]** implement the 4 sharing schemes for the recursive block (new code, see §D); table across 2 model sizes × datasets | param reduction |
| **T6** model-size variants (N-emb, d_model, N_head, d_head, d_inter, vocab, L_ctx) | config | **[R]** SMART size variants (d_model, n_heads, d_ff, K, M, #params transformer/total) used in the scaling study | config |
| **T7** isoFLOP across 3 compute budgets (NLL) | scaling | **[A]** accuracy across 3 model-size/compute budgets for MoR/Recursive/Vanilla (= Fig 3 analog table) | adaptive+param |
| **T10/T11** expert/token router under different routing configs | deep routing | **[R]** capacity-funnel schedules, temperature, α (router_alpha) sweeps | adaptive loop |
| **T13** relaxing parameter + KV sharing | mixed | **[A]** parameter-sharing relaxation (shared→partially-tied→independent) × step-cache on/off | param reduction |
| **T14** per-token recursion-depth visualization | interpretability | **[R]** per-marker / per-pathway recursion depth heatmap (we already log `depth_per_token`) → "which genes/pathways recurse deepest" | adaptive loop |
| **T2** routing + cache summary | summary | **[A]** routing-strategy summary + step-cache column (the set-encoder analogue of recursion-wise KV cache) | — |
| **T9** uptraining → **warm-start** | adapted | **[A]** init shared block from a trained fixed-depth SMART, continue-train; gain vs from-scratch | param/adaptive |
| **T12** KV-cache sharing → **step-cache** | adapted | **[A]** reuse step-1 attention K/V across the K recursions vs recompute; acc/compute trade-off | adaptive loop |

Figures: **Fig 3** (val-loss vs compute → acc vs compute, 3 archs) **[R/A]**; **Fig 5**
(compute-optimal + learned per-recursion token counts + tie rate → we have active-per-step
& depth) **[A]**; **Fig 4** (throughput–quality Pareto → FLOPs/latency vs acc) **[A]**;
**Fig 7/8** (hidden/key/value L2-norm & cosine across layers → hidden-state norm/cosine
across the K recursions) **[A]**; **Fig 1/2** architecture (reuse `assets/`).

## C. Experiment suite → the three claims
All on every genomap dataset (Tier 1: 4 datasets) + pathway cohorts, genomap-paper
protocol (lr 1e-3, wd 1e-5, batch 128, 150 ep, patience 15, d_model 96, M=128), 3 seeds
for mean±std. Reuses `run_sc_arch.sbatch` variants + `singlecell.py`/`token_reduction.py`.

1. **Adaptive loop** — `shared`(expert) vs `token` vs `fixed` vs `depth1`(K=1) → T3, T4, T10/11, T14.
2. **Token reduction** — M-sweep (`token_reduction.py`, M∈{32,64,128,256,…}) + selection
   baselines (`marker_random`,`marker_var`) + compute-saving from the routing funnel → T4(sel), Fig 5.
3. **Parameter reduction** — `shared` vs `independent` + the 4 sharing schemes (§D) +
   #params/×reduction (ratio≈K) → T1/5/6/8/13, Fig 3.

## D. New code needed (small, additive)
- **Sharing schemes** (`recursion.py`): Cycle / Middle-Cycle / Sequence / Middle-Sequence
  block-assignment over K steps (currently only fully-shared vs fully-independent). New
  `cfg.share_strategy`. Needed for T1/T5/T8.
- **Depth-visualization** export (have `depth_per_token`; add a per-feature aggregation +
  plot in `make_paper.py`) for T14.
- **Scaling driver**: a sweep over {d_model, K} × {MoR, Recursive, Vanilla} for T3/T6/T7/Fig3
  (extend `experiments.py`/`sweeps.py`).
- **make_paper**: most builders exist (`arch_table`, `param_table`, `selection_table`,
  `main_sc_table`, `ladder_table`, `dataset_overview_table`); add T3/T4/T6-styled emitters
  and the genomap-suite loaders for all 4 datasets.

## E. Compute & sequencing
1. Extend `run_sc_arch.sbatch` datasets `tabula_muris pancreas` → **+ common_class prototype**
   (reuse existing TM+pancreas JSONs; only the 2 new datasets run). 7 variants × 3 seeds × 2 new ds.
2. `token_reduction` M-sweep on all 4 datasets.
3. Scaling sweep (sizes × 3 archs) for T3/T6/T7/Fig3.
4. Sharing-scheme job once §D lands.
5. Pathway-cohort arch ablation (mirror, on prostate/blca/stad/brca/panmeta).
6. Regenerate paper: `python -m recursive_marker_transformer.make_paper --outdir paper`.

**No TCGA**: `make_paper.py` is already the TCGA-free single-cell generator; the TCGA
generator stays archived at `make_paper_tcga_backup.py`. Confirm 0 TCGA refs after rebuild.
