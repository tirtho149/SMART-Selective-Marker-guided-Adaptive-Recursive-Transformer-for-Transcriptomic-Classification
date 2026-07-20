#!/usr/bin/env bash
#SBATCH --job-name=flops-all
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/flops-all-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/flops-all-%j.out
# REAL forward-pass FLOPs (torch FlopCounterMode) for the full Table-2 ladder on ALL 14
# datasets (8 single-cell + 6 multi-omics) on one A100 -> results/cv5/curves/all_flops.json.
# Big mem for the pan-cancer / 3M cohort loads (avoids login-node OOM).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[flops-all] $(date) host=$(hostname)"; nvidia-smi -L || true
"$PY" scripts/measure_flops_all.py
echo "[flops-all] DONE rc=$? $(date)"
