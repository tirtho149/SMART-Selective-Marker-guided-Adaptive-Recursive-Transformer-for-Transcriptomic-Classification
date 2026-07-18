#!/usr/bin/env bash
#SBATCH --job-name=smoke-3m
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-3m-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-3m-%j.out
# Confirm the tri-modal 3M (pan_meta_pri_3modal, mut+CNV+expr) loads and runs one fold.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"
$PY -m recursive_marker_transformer.pathway_tasks \
    --task pan_meta_pri_3modal --channels mut_cnv_expr --marker_mode pathway \
    --gene_interaction none --pathway_learned_graph --pathway_learned_fuse --bio_graph_router \
    --pathway_pool sum --recursion_mode expert --recursion_depth 4 \
    --n_markers 128 --d_model 64 --epochs 2 --patience 2 --batch_size 32 \
    --cv_folds 2 --device cpu --out /tmp/smoke_3m && echo "SMOKE_3M_OK rc=$?"
