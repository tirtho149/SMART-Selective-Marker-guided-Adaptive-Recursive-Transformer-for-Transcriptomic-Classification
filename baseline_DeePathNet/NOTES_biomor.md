# DeePathNet on bioMOR — wiring notes

## What changed vs upstream
- Upstream `scripts/deepathnet_cv.py` + `utils/training_prepare.py` are hardcoded
  to CCLE/GDSC drug-response data under `/home/scai/...` and do MSE regression /
  drug-response multilabel. Those paths do not exist here.
- New thin runner: `scripts/deepathnet_biomor_cv.py` (config-driven, same style).
  It reuses the **unmodified** `DeePathNet` transformer from
  `scripts/model_transformer_lrp.py`, but swaps the data layer for the shared
  `biomor_common` backbone:
  - data via `bc.load_omics(cohort, modalities)`, reshaped to `(N, G, n_omics)`;
  - folds via `bc.cv_folds(y)` (byte-identical bioMoR seed-42 CV5);
  - CrossEntropy classification head (binary + multiclass) instead of MSE;
  - macro-F1 early stopping on the inner val split;
  - output via `bc.write_scores` -> `work_dirs/<cohort>/scores_*.csv`.
- Configs live in `configs/biomor/<cohort>.json`.

## Pathway grouping choice
DeePathNet needs a `{pathway_name: [gene,...]}` dict plus a non-cancer-gene
remainder to form pathway tokens. Resolution order per cohort:
1. `data/<cohort>/filtered_pathways.csv` if present (auto-detects a name column
   and a gene-list column; `|` or `,` separated), keeping pathways with >=5 of
   the cohort's genes.
2. **Fallback (generic):** contiguous 50-gene blocks over the gene axis as
   pseudo-pathways. This preserves DeePathNet's token structure when no curated
   pathway file matches the cohort's gene set. `max_pathways` (default 300) caps
   the token count for memory.

## Env
Shared `/work/mech-ai-scratch/tirtho/.venv` (torch 2.10). The DeePathNet model is
plain `nn.Module` + einops; no torch-1.10 pin needed.

## Metric
macro-F1 per fold, mean/std over 5 folds (schema: dataset,model,fold,macro_f1,accuracy,n_test).
