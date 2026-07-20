#!/usr/bin/env bash
#SBATCH --job-name=pm-routing
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --constraint="a100|h200|l40s|rtxpro6000|v100"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pm-routing-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pm-routing-%j.out
# PM routing-interpretability experiment (K=1..4, expert & token, 5-fold parity with Table 2):
# records each pathway's mean recursion depth (kept vs dropped) -> results/cv5/pm_routing/.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[pm-routing] $(date) host=$(hostname)"; nvidia-smi -L || true
"$PY" scripts/pm_routing_experiment.py
echo "[pm-routing] DONE rc=$? $(date)"
