#!/usr/bin/env bash
#SBATCH --job-name=val-bundle
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:08:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/val-bundle-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/val-bundle-%j.out
set -uo pipefail
BUN=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/paper/tabledata
REAL=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/paper
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$BUN"; mkdir -p paper
for s in build_cv5_tex build_injection_table build_posf1_table build_pm_depth_tables; do
  $PY scripts/$s.py >/dev/null 2>&1 && echo "  ran $s" || echo "  FAILED $s"
done
echo "=== diff bundle-generated .tex vs the real paper fragments ==="
for t in cv5_main_table cv5_baselines_table cv5_scaling_table cv5_injection_table cv5_posf1_table pm_depth_tables; do
  if [ -f "$BUN/paper/$t.tex" ] && [ -f "$REAL/$t.tex" ]; then
    if diff -q "$BUN/paper/$t.tex" "$REAL/$t.tex" >/dev/null; then echo "  IDENTICAL  $t.tex"; else echo "  DIFFERS    $t.tex"; fi
  else echo "  MISSING    $t.tex (bundle=$( [ -f "$BUN/paper/$t.tex" ] && echo y||echo n) real=$( [ -f "$REAL/$t.tex" ] && echo y||echo n))"; fi
done
echo "[val] DONE"
