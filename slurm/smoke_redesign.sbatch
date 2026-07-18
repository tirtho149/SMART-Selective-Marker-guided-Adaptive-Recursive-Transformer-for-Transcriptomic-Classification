#!/usr/bin/env bash
#SBATCH --job-name=smoke-redesign
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-redesign-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-redesign-%j.out
# Fast CPU smoke of the redesigned graph-router: 3 modes on a small dataset, 2 epochs,
# 2 folds -- just confirms the code runs end-to-end (no correctness claim).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset Xin \
    --modes learned_bio route_graph bio_both --cv_folds 2 --K 4 --epochs 2 --patience 2 \
    --n_markers 64 --device cpu --out /tmp/smoke_redesign && echo "SMOKE_OK rc=$?"
