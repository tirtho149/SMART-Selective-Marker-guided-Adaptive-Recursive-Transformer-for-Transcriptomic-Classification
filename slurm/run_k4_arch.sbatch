#!/usr/bin/env bash
#SBATCH --job-name=smart-k4arch
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k4arch-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k4arch-%j.out
# Vanilla vs Recursive vs MoR at fixed depth K=4 (accuracy peak) on pancreas.
# Everything fixed (d_model=96, n_markers=128, epochs=60, K=4) except the arch knob.
# 3 seeds so the accuracy column is trustworthy.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
echo "[k4arch] start $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.k32_arch \
    --dataset pancreas --K 4 --seeds 0 1 2 \
    --archs vanilla recursive mor --epochs 60 --device cuda
echo "[k4arch] DONE rc=$? $(date)"
