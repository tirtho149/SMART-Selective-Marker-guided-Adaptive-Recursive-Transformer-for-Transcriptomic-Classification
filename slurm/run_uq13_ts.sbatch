#!/usr/bin/env bash
#SBATCH --job-name=smart-uq13-ts
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq13ts-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq13ts-%A_%a.out
# Table 4 with TEMPERATURE SCALING: re-run all UQ configs (incl. learned), saving raw
# AND temperature-scaled (val-fit T) NLL/ECE/AUROC per (config,seed,dataset). Accuracy
# is unchanged by T. Overwrites results_uq13/<config>/s<seed>/<ds>.json (richer json).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
CFG=(bio none adaptive fixed depth1 independent learned)
C=${CFG[$SLURM_ARRAY_TASK_ID]}
echo "[uq13ts] config=$C host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.uq_sweep --config $C --seeds 0 1 2 \
    --epochs 80 --cohorts --out results_uq13 --device cuda
echo "[uq13ts] config=$C DONE rc=$? date=$(date)"
