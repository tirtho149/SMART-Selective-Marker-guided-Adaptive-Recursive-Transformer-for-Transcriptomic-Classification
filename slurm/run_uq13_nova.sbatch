#!/usr/bin/env bash
#SBATCH --job-name=smart-uq13
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq13-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq13-%A_%a.out
# =============================================================================
# 13-dataset paper -- Table 4 (calibration/UQ): NLL / ECE per config. One array task
# per configuration, 3 seeds, over the 9 genomap single-cell sets + 4 P-NET cohorts
# (brca; set in uq_sweep.SC / COH). Output: results_uq13/<config>/s<seed>/<ds>.json
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
CFG=(bio none adaptive fixed depth1 independent)
C=${CFG[$SLURM_ARRAY_TASK_ID]}
echo "[uq13] config=$C host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.uq_sweep --config $C --seeds 0 1 2 \
    --epochs 80 --cohorts --out results_uq13 --device cuda
echo "[uq13] config=$C DONE rc=$? date=$(date)"
