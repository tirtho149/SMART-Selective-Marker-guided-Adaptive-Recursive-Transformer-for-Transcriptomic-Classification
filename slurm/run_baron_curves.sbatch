#!/usr/bin/env bash
#SBATCH --job-name=baron-curves
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-curves-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-curves-%j.out
# One Baron bioMoR run (Table-2 learned/K4 config, fold 0) to capture per-epoch
# train-loss and val-macro-F1 curves and render paper/figs/baron_curves.pdf.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[baron-curves] $(date) host=$(hostname)"; nvidia-smi -L || true
"$PY" make_baron_curves.py
echo "[baron-curves] DONE rc=$? $(date)"
