#!/usr/bin/env bash
# Build the PANCAN multimodal + subtype dataset (data/pancan/) on a compute node
# with enough memory to read the wide Xena matrices. Idempotent.
#   sbatch run_build_pancan.sbatch
#SBATCH --job-name=pancan-build
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-build-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-build-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
echo "[pancan-build] host=$(hostname) date=$(date)"
$PY "new data/build_pancan.py"
RC=$?
echo "[pancan-build] DONE rc=$RC date=$(date)"
ls -lh data/pancan/
