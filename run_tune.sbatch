#!/usr/bin/env bash
#SBATCH --job-name=smart-tune
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tune-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tune-%j.out
# Hyperparameter tuning of SMART on the 5 hard clinical genoNet tasks: 16-point
# grid (width x markers x depth x lr), selection by validation macro-F1, all 20530
# genes. Resumable per (task,config) -> results_tune/
#   sbatch run_tune.sbatch
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[tune] host=$(hostname) date=$(date)"; nvidia-smi -L
$PY -m recursive_marker_transformer.tune \
  --csv data/tcga/unified_bio5.csv --out results_tune --device cuda
echo "[tune] DONE rc=$? date=$(date)"
