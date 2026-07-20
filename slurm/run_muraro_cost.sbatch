#!/usr/bin/env bash
#SBATCH --job-name=muraro-cost
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/muraro-cost-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/muraro-cost-%j.out
# Timed A100 training on Muraro for Vanilla / Recursive / MoR / bioMoR (Table-2 configs):
# records per-epoch {val_f1, cumulative GPU-seconds}, final test-F1 and param count, for
# the real-hardware compute-cost-vs-accuracy figure. -> results/cv5/curves/muraro_cost.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[muraro-cost] $(date) host=$(hostname)"; nvidia-smi -L || true
"$PY" scripts/make_baron_cost.py --dataset muraro
echo "[muraro-cost] DONE rc=$? $(date)"
