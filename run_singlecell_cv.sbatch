#!/usr/bin/env bash
#SBATCH --job-name=smart-sc-cv
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-cv-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-cv-%A_%a.out
# =============================================================================
# Single-cell datasets, same protocol as the TCGA cohort CV: stratified 5-fold
# CV, macro-F1 mean+/-std, for all THREE model types. One array task per variant
# (0=shared, 1=independent, 2=mor). Datasets: tabula_muris, common_class,
# prototype, pancreas (genomap capsule 6967747).
# Output: results_singlecell_cv/<variant>/<dataset>.json
#   sbatch run_singlecell_cv.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VARIANTS=(shared independent mor)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
case "$V" in
  shared)      MODEFLAGS="--recursion_mode expert" ;;
  independent) MODEFLAGS="--recursion_mode expert --no_share_weights" ;;
  mor)         MODEFLAGS="--recursion_mode token" ;;
esac

echo "[sc-cv] variant=$V host=$(hostname) date=$(date)"
nvidia-smi -L

$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results_singlecell_cv/$V" \
    --datasets tabula_muris common_class prototype pancreas \
    --epochs 10 --d_model 96 --n_markers 128 --batch_size 512 \
    --lr 3e-4 --patience 5 --device cuda --cv_folds 5 $MODEFLAGS
echo "[sc-cv] variant=$V DONE rc=$? date=$(date)"
