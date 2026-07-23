# MOGONET on the bioMOR shared contract

**Runner:** `scripts/mogonet_cv.py` — thin driver over the upstream MOGONET
model (`models.py`: GCN encoders + VCDN fusion) and graph primitives
(`utils_adapted.py`). Loads a cohort via `biomor_common.load_omics`, builds
IDENTICAL seed-42 CV5 folds via `biomor_common.cv_folds`, trains per fold
(pretrain encoders → train with VCDN, early-stop on validation macro-F1),
predicts the test fold, and writes the common-schema
`work_dirs/<cohort>/scores_<stamp>.csv` via `biomor_common.write_scores`.

- Views: 1=mutation (jaccard graph), 2=cnv (cosine), 3=expression (cosine).
- Supports 2- or 3-modality and binary or multiclass (BRCA 5-class works).
- Metric: macro-F1 per fold + mean/std.

**Env:** shared venv `/work/mech-ai-scratch/tirtho/.venv` (torch 2.10+cu128,
pure-torch — no extra installs).

**Cohorts:** prostate, blca, stad, brca (5-class), pan_meta_pri,
pan_meta_pri_3modal (3-modal mut+cnv+expression).

**Run:** `sbatch scripts/full_array.sbatch` (array 0-5). Smoke:
`sbatch scripts/smoke.sbatch`.
