# MOGAT on the bioMOR shared contract

**Runner:** `scripts/mogat_cv.py` — reuses the upstream MOGAT GAT
(`lib/module2.Net`, GATConv) and the training design of
`mogat_3mod_multiclass.py`: one GAT embedding per modality patient-similarity
graph (mutation→jaccard, cnv/expr→correlation), concat embeddings + raw
features → MLP integration head (random HP search on validation macro-F1).
Loads cohorts via `biomor_common.load_omics`, uses IDENTICAL seed-42 CV5 folds
(`biomor_common.cv_folds`), writes `work_dirs/<cohort>/scores_<stamp>.csv` via
`biomor_common.write_scores`.

- Supports 2-modality (mut+cnv) and 3-modality (+expression); binary + multiclass.
- Metric: macro-F1 per fold + mean/std.

**Env:** isolated `/work/mech-ai-scratch/tirtho/.venvs_baselines/mogat` — a light
venv holding only the pure-python `torch_geometric==2.6.1` (+ light deps).
The heavy stack (torch 2.10+cu128, numpy, sklearn, pandas) is reused from the
shared venv via `PYTHONPATH` (installing torch_geometric with `--no-deps` avoids
shadowing numpy). PyG 2.6 needs no compiled torch-scatter/torch-sparse.

**Cohorts:** prostate, blca, stad, brca (5-class), pan_meta_pri,
pan_meta_pri_3modal.

**Run:** `sbatch scripts/build_and_smoke.sbatch` then
`sbatch scripts/full_array.sbatch` (array 0-5).
