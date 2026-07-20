#!/usr/bin/env bash
#SBATCH --job-name=fig2-bars
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fig2-bars-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fig2-bars-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
echo "[fig2-bars] host=$(hostname) $(date)"
/work/mech-ai-scratch/tirtho/.venv/bin/python scripts/make_fig2_era_bars.py
echo "[fig2-bars] DONE rc=$? $(date)"
