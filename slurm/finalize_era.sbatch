#!/usr/bin/env bash
#SBATCH --job-name=finalize-era
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/finalize-era-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/finalize-era-%j.out
# Runs AFTER the four ERA arrays (afterany). Regenerates Table 2 (build_cv5_tex.py -> the bioMoR
# rows now all read the ERA ladder) and Figure 2 (make_biology_combo_bars.py -> keeps only
# datasets with strict mean(ERA)>E>R>A) and recompiles the paper. Writes a completion summary.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MARK="$PROJ/RUN_COMPLETE_ERA.txt"
{
  echo "=== finalize-era $(date) ==="
  echo "--- Table 2 (bioMoR = ERA everywhere) ---"
  python scripts/build_cv5_tex.py
  echo "--- Figure 2 (all 14 datasets; keep strict ERA>E>R>A) ---"
  python scripts/make_biology_combo_bars.py
  echo "--- full refresh (tables + figs + paper compile) ---"
  bash scripts/refresh_cv5.sh 2>&1 | tail -25
  echo "=== DONE $(date) ==="
} | tee "$MARK"
