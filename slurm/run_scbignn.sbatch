#!/usr/bin/env bash
#SBATCH --job-name=cv5-scbignn
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scbignn-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scbignn-%A_%a.out
# scBiGNN baseline (arXiv:2312.10310), 5-fold CV on the SHARED bioMoR folds
# (cv.cv_folds, seed 42, epochs=100 patience=15). One array task per Table-3 cohort.
# Table 3 rows = Lung (single-cell) and PM=pan_meta_pri (multi-omics, mut+CNV).
#   0 Lung           (single-cell)
#   1 pan_meta_pri   (PM, mut_cnv; multi-omics)
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
# Self-activate the venv: submitted jobs do not inherit the interactive shell env.
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

COHORTS=(Lung pan_meta_pri)
COH="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
OUT="$PROJ/results/cv5/scbignn"

echo "[cv5-scbignn] $(date) cohort=$COH host=$(hostname)"; nvidia-smi -L || true
python -m recursive_marker_transformer.scbignn_baseline \
    --cohort "$COH" --epochs 100 --patience 15 --seed 42 \
    --pca_dim 128 --knn_k 15 --device cuda --out "$OUT"
echo "[cv5-scbignn] DONE rc=$? $(date) cohort=$COH"
