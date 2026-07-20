#!/usr/bin/env bash
#SBATCH --job-name=muraro-flops
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/muraro-flops-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/muraro-flops-%j.out
# REAL forward-pass FLOPs (torch FlopCounterMode) for every Figure-5 architecture x depth
# point on Muraro, on a single A100. -> results/cv5/curves/muraro_flops.json (device-
# independent: reproduces the CPU counts exactly; A100 run is the official record).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[muraro-flops] $(date) host=$(hostname)"; nvidia-smi -L || true
"$PY" scripts/measure_flops_muraro.py
echo "[muraro-flops] DONE rc=$? $(date)"
