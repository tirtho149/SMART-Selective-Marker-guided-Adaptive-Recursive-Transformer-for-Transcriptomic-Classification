# P-NET on the bioMOR shared contract

**Approach:** keep the upstream P-NET Keras model intact (`build_pnet2`:
Diagonal gene layer + SparseTF pathway-masked layer + deep-supervision heads)
and drive it through the existing `train/run_me.py` → CrossvalidationPipeline,
but wire it to the bioMOR contract:

- **`pipeline/crossvalidation_pipeline.py`** — patched to load the FULL cohort
  (`Data.get_data()`, reader's patient-sorted order) and split with
  `biomor_common.cv_folds` (IDENTICAL seed-42 CV5), instead of its own
  `StratifiedKFold(random_state=123)`. Per-fold CNV z-scoring (train stats only)
  is applied on the gene-grouped `[g_mut, g_cnv]` matrix; the reader's own
  z-score is disabled (`zscore_cnv=False`). After CV it emits the common-schema
  `work_dirs/<cohort>/scores_<stamp>.csv` via `biomor_common.write_scores`.
  `PNET_SMOKE=1` runs 1 fold / 3 epochs.
- **param files** (`train/params/P1000/pnet/crossvalidation_average_reg_10_tanh_{prostate,blca,stad,pan_meta_pri}.py`)
  — pointed at real cohort dirs (`BIOMOR_DATA_ROOT`), `selected_genes_filename=None`
  (use all common genes), `zscore_cnv=False`, and `n_hidden_layers=1`
  (flat `filtered_pathways.csv` → single sparse pathway layer; loss_weights=[2,7],
  n_outputs=2, monitor=val_o2_f1).
- **`data/pathways/reactome.py`** — honors `PNET_REACTOME_DIR` so concurrent
  array tasks build per-cohort Reactome files without clobbering.

**Scope:** P-NET as ported is **2-modality (mut+cnv) and binary only** (sigmoid /
binary_crossentropy). So it runs on prostate, blca, stad, pan_meta_pri.
It does NOT support brca (5-class) or the 3-modal cohort — those are covered by
MOGONET/MOGAT. Reactome pathway/adjacency files are built per cohort from each
cohort's `filtered_pathways.csv`.

**Env:** isolated `/work/mech-ai-scratch/tirtho/.venvs_baselines/pnet`
(TensorFlow 2.19 + CUDA, built inside the SLURM job).

**Run:** `sbatch scripts/build_and_smoke.sbatch` (builds venv + Reactome +
1-fold smoke), then `sbatch scripts/full_array.sbatch` (array 0-3).
