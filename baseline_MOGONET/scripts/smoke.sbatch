#!/usr/bin/env bash
#SBATCH --job-name=mogonet-smoke
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGONET/work_dirs/smoke_%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGONET/work_dirs/smoke_%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGONET
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
COHORT=prostate SMOKE=1 python scripts/mogonet_cv.py
echo "[mogonet-smoke] rc=$?"
