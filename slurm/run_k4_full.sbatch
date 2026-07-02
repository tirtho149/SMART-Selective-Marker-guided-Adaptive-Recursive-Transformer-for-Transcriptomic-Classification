#!/usr/bin/env bash
#SBATCH --job-name=smart-k4full
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k4full-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k4full-%j.out
# Full-suite Vanilla vs Recursive vs MoR at fixed depth K=4 on all 10 genomap
# single-cell datasets. Everything fixed except the arch knob; 3 seeds each.
# Resumable: existing results_k4_arch/<ds>/<arch>_s<seed>.json are skipped.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
echo "[k4full] start $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.k32_arch \
    --K 4 --seeds 0 1 2 --archs vanilla recursive mor --epochs 60 --device cuda
echo "[k4full] DONE rc=$? $(date)"
