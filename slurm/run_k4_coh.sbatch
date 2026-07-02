#!/usr/bin/env bash
#SBATCH --job-name=smart-k4coh
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k4coh-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/k4coh-%j.out
# P-NET / Reactome cohorts: Vanilla vs Recursive vs MoR at K=4 (Reactome pathway
# tokens, d=128, M=256). 3 seeds each; resumable (skips existing json).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
echo "[k4coh] start $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.k32_arch \
    --datasets prostate blca stad panmeta_subtype \
    --K 4 --seeds 0 1 2 --archs vanilla recursive mor --epochs 60 --device cuda
echo "[k4coh] DONE rc=$? $(date)"
