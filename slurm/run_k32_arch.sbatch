#!/usr/bin/env bash
#SBATCH --job-name=smart-k32arch
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k32arch-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k32arch-%j.out
# Vanilla vs Recursive vs MoR at fixed depth K=32 on pancreas.
# Everything fixed (d_model=96, n_markers=128, epochs=60, K=32) except the arch knob.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
echo "[k32arch] start $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.k32_arch \
    --dataset pancreas --K 32 --seeds 0 1 \
    --archs vanilla recursive mor --epochs 60 --device cuda
echo "[k32arch] DONE rc=$? $(date)"
