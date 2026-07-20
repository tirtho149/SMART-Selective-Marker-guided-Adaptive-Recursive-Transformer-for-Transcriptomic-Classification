#!/usr/bin/env bash
#SBATCH --job-name=scbignn-smoke
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scbignn-smoke-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scbignn-smoke-%A_%a.out
# GPU SMOKE TEST for the scBiGNN baseline (NOT run on the login node).
# 1 fold, 3 epochs, small subset -> just proves the code runs end-to-end on GPU
# for one single-cell and one multi-omics cohort and writes valid JSON.
#   0 Lung           (single-cell)
#   1 pan_meta_pri   (PM, mut_cnv; multi-omics)
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

COHORTS=(Lung pan_meta_pri)
COH="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
OUT="$PROJ/results/cv5/scbignn_smoke"

echo "[scbignn-smoke] $(date) cohort=$COH host=$(hostname)"; nvidia-smi -L || true
python -m recursive_marker_transformer.scbignn_baseline \
    --cohort "$COH" --epochs 3 --patience 3 --n_folds 2 --subset 500 \
    --pca_dim 64 --knn_k 10 --seed 42 --device cuda --force --out "$OUT"
echo "[scbignn-smoke] DONE rc=$? $(date) cohort=$COH"
