#!/usr/bin/env bash
#SBATCH --job-name=bc-selftest
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/_selftest_%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/_selftest_%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
python _selftest_common.py
echo "[selftest] rc=$?"
