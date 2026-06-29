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

#SBATCH --job-name=smart-pancan
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-%j.out
# =============================================================================
# SMART on the TCGA PANCAN benchmark (data/pancan/, 7806 samples x 18000 shared
# genes; expression + CNV + mutation gene-aligned channels):
#   (1) subtype classification on EXPRESSION
#         immune_subtype   (Thorsson pan-cancer, 6 classes)
#         molecular_subtype (curated Subtype_Selected, long tail trimmed)
#   (2) multimodal fusion on immune_subtype
#         expr -> expr+cnv -> expr+cnv+mut   (clean within-model ablation)
#   sbatch run_pancan.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[pancan] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[pancan] torch',torch.__version__,'cuda',torch.cuda.is_available())"

# (1)+(2) immune subtype: expression + the two fused channel sets
$PY -m recursive_marker_transformer.pancan_tasks \
  --task immune_subtype --channels expr expr_cnv expr_cnv_mut \
  --out results_pancan --epochs 40 --d_model 128 --n_markers 256 \
  --batch_size 64 --lr 3e-4 --patience 8 --device cuda

# (1) molecular subtype on expression (+ full fusion for completeness)
$PY -m recursive_marker_transformer.pancan_tasks \
  --task molecular_subtype --channels expr expr_cnv_mut \
  --out results_pancan --epochs 40 --d_model 128 --n_markers 256 \
  --batch_size 64 --lr 3e-4 --patience 8 --device cuda

RC=$?
echo "[pancan] DONE rc=$RC date=$(date)"
