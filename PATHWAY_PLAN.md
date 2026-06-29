# Pathway-Structured SMART (Reactome / P-NET data) — Plan

> New benchmark + component for SMART. Status legend: [x] done · [~] in progress · [ ] todo.
> Origin: the GitHub `data-branch` ships P-NET-style (Elmarakeby et al., *Nature* 2021)
> pathway-informed multi-omics for 5 cohorts. User instruction = "incorporate the
> pathway-based data". This is the *curated* biological prior that the data-driven
> co-expression router ([[BIO_ROUTER_PLAN]]) left as an honest open question
> (`coexpr ≈ random ≈ none`).

## 0. The data (now local under `data/<cohort>/`)
Pulled from `origin/data-branch` via
`git checkout origin/data-branch -- data/{prostate,brca,blca,stad,pan_meta_pri}`.

| Cohort | Patients | Label | Mut genes | CNV genes | Pathways |
|---|---|---|---|---|---|
| prostate | 1011 | binary 678/333 (primary vs metastatic — P-NET task) | 8434 | 8434 | 1268 |
| blca | 404 | binary 131/273 | 23384 | 23384 | 1268 |
| stad | 414 | binary 186/228 | 23384 | 23384 | 1268 |
| brca | 518 | 5-class (PAM50-like) | 40543 | none | 1268 |
| pan_meta_pri | 8893 | 32-class cancer type | none | none | 1268 |

Per cohort:
- `filtered_pathways.csv` — 1268 Reactome pathways `R-HSA-…` → member-gene lists (build `P ∈ {0,1}^{G×1268}`).
- `adjacency_matrix.csv` — 1268×1268 Reactome pathway→pathway hierarchy graph `A`.
- `mutation_data.csv` — patient×gene binary mutation.
- `cnv_data.csv` — patient×gene copy-number in {-2..2}.
- `patient_labels.csv` — `id,response`.

**Gotchas:**
- Excel-corrupted gene symbols in mut/cnv headers (`1-Mar`=MARCH1, `10-Sep`=SEPT10, …) —
  must be de-corrupted before intersecting with pathway gene lists.
- `brca` has no CNV; `pan_meta_pri` & `brca` have no omics here → join to the Xena
  expression matrix from `new data/build_pancan.py` (keyed by sample id) for those two.
- mut/cnv share the same gene order per cohort (verify), so stacking is direct.

## 1. Idea (one line)
Make biology *structural*, not just a soft prior: pool genes into **Reactome pathway
tokens** (interpretable, fixed gene→pathway sparsity = P-NET's first layer) and let the
recursive MoR stack operate over pathways, optionally biased by the Reactome hierarchy.

## 2. Implementation map
- [x] `recursive_marker_transformer/pathway_data.py` — load+align mut/cnv, de-corrupt
  gene symbols, build membership `P` (drop <min_genes pathways + orphan genes), pathway
  sub-hierarchy eigenvector centrality, seeded split-ready arrays. Smoke-validated on
  prostate/blca/stad/brca (torch-free; numpy/pandas only).
- [x] `recursive_marker_transformer/pathway_tasks.py` — task runner (parallels
  `pancan_tasks.py:run/main`); tasks = prostate/blca/stad/brca; channels mut/cnv/both;
  result tag `task__channels__mode__prior`.
- [x] **Pathway tokens** — `marker_mode="pathway"` (`config.py`); `PathwayPooler` in
  `marker.py`; wired in `model.py` selector branch. M tokens = `P`-pooled per-channel
  features + learnable per-pathway gate (keeps "selective"). Forward+backward verified;
  gate receives gradient.
- [x] **Reactome router prior** — `gene_interaction="reactome"` (`config.py`); per-token
  `token_prior` buffer + `set_token_prior` in `model.py`; forward prefers it over the
  per-gene `gene_centrality`; reuses the existing `set_anneal` β_t schedule + router
  plumbing untouched. `_use_token_prior=True` verified, β=1.0 anneals.
- [ ] (opt) **Hierarchy attention bias** — `A` as an additive attention mask in
  `recursion.py` so pathway tokens attend along the Reactome hierarchy.
- [x] `run_pathway.sbatch` — GPU runner (mirror `run_pancan.sbatch`); full token-type ×
  prior × modality grid. NOT yet submitted.
- [ ] `make_paper.py` — pathway-token method subsection + P-NET benchmark table +
  curated-vs-coexpr ablation.
- [x] **brca collapse FIXED** — mutation is ~0.4% sparse, so mean-pooling washes the
  per-pathway signal to a constant → collapse (acc 6.7 / F1 4.2). Added `pathway_pool`
  ("mean"|"sum"); **sum = pathway mutation burden** restores signal (smoke F1 6.3→26.3).
- [x] **pancancer_meta_pri** wired — ships pathways+labels but NO omics, so expression
  is joined from the Xena PANCAN matrix by TCGA barcode (8586/8893 overlap, 10013 genes
  in pathways). `load_pan_meta` (cached .npz, float32+usecols frugal read). Two heads:
  `panmeta_response` (primary vs metastatic, 8225/361) and `panmeta_subtype` (32-class).
- [ ] (still) expression join could also rescue brca CNV-less / pan_meta_pri sibling — n/a now.

## VALIDATED DIMENSIONS (smoke, min_genes=5)
| cohort | N | G(in-pw) | M(pathways) | C | K |
|---|---|---|---|---|---|
| prostate | 1011 | 5050 | 1223 | 2 | 2 |
| blca | 404 | 10333 | 1258 | 2 | 2 |
| stad | 414 | 10333 | 1258 | 2 | 2 |
| brca | 518 | 10684 | 1260 | 1 (mut) | 5 |

## 3. Ablations (the paper story)
1. **Token type:** learned marker (current) vs **pathway** vs random-grouping control.
2. **Router prior:** none vs coexpr (data, the negative result) vs **reactome** (curated) vs random.
   Claim: `reactome > coexpr ≈ none` ⇒ curated structure, not "any graph", is what helps.
3. **Modality:** mut-only vs cnv-only vs mut+cnv (existing `n_channels` path).
4. **Baseline:** P-NET on prostate (their headline) for apples-to-apples.

## 4. Run commands (target)
```bash
python -m recursive_marker_transformer.pathway_tasks \
    --task prostate --channels mut_cnv --marker_mode pathway \
    --gene_interaction reactome --device cuda
# or GPU: sbatch run_pathway.sbatch
```
