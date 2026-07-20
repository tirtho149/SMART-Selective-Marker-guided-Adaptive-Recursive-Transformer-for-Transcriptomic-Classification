#!/usr/bin/env bash
#SBATCH --job-name=trace-json
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:08:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/trace-json-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/trace-json-%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
/work/mech-ai-scratch/tirtho/.venv/bin/python scripts/trace_table_json.py
echo "[trace] DONE rc=$?"
