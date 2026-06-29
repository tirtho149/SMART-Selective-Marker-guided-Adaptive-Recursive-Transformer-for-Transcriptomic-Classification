#!/usr/bin/env bash
# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
# ============================================================================

#SBATCH --job-name=smart-pancan-base
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-base-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-base-%j.out
# =============================================================================
# 10 nonlinear tabular baselines on the SAME PANCAN split as SMART:
#   immune_subtype    : expr  and  expr+cnv+mut (concatenated flat features)
#   molecular_subtype : expr
#   sbatch run_pancan_baselines.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[pancan-base] host=$(hostname) date=$(date)"

$PY -m recursive_marker_transformer.pancan_baselines \
  --task immune_subtype --channels expr expr_cnv_mut --out results_pancan_baselines

$PY -m recursive_marker_transformer.pancan_baselines \
  --task molecular_subtype --channels expr --out results_pancan_baselines

RC=$?
echo "[pancan-base] DONE rc=$RC date=$(date)"
