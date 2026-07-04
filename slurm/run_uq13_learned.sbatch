#!/usr/bin/env bash
#SBATCH --job-name=smart-uq13-learned
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq13-learned-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq13-learned-%j.out
# =============================================================================
# Table 4 addendum: calibration (NLL/ECE/AUROC) for the LEARNED data-driven
# gene-gene graph -- the Table-1 accuracy winner. Does it also improve calibration?
# 9 genomap single-cell sets + 4 P-NET cohorts, 3 seeds.
# Output: results_uq13/learned/s<seed>/<ds>.json
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[uq13-learned] host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.uq_sweep --config learned --seeds 0 1 2 \
    --epochs 80 --cohorts --out results_uq13 --device cuda
echo "[uq13-learned] DONE rc=$? date=$(date)"
