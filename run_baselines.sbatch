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
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

#SBATCH --job-name=smart-baselines
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baselines-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baselines-%j.out
# 14 external baselines on the 6 BIO5 genoNet tasks (all 20530 genes), identical
# split to genonet_tasks.py -> results_baselines/. sklearn (CPU); a GPU slot is
# requested only to schedule on scavenger.
#   sbatch run_baselines.sbatch
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[baselines] host=$(hostname) date=$(date)"
$PY -m recursive_marker_transformer.baselines --csv data/tcga/unified_bio5.csv --out results_baselines
echo "[baselines] DONE rc=$? date=$(date)"
