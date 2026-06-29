#!/usr/bin/env bash
# ============================================================================
# SMART -- Reactome pathway-hierarchy ATTENTION-BIAS ablation (with vs without).
# Pathway tokens attend along the pathway->pathway graph; the matched control is
# the same token-choice router with the bias off. Runs alongside the main
# expert-grid job (different result tags, no collision).
#   sbatch run_pathway_attnbias.sbatch
# ============================================================================
#SBATCH --job-name=smart-pwattn
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
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwattn-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwattn-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[pwattn] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[pwattn] torch',torch.__version__,'cuda',torch.cuda.is_available())"

OUT=results_pathway
COMMON="--out $OUT --epochs 40 --d_model 128 --n_markers 256 --batch_size 32 \
        --lr 3e-4 --patience 8 --device cuda"

for TASK in prostate blca stad; do
  # control: token-choice router, NO bias
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task $TASK --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --recursion_mode token $COMMON
  # treatment: Reactome pathway-hierarchy attention bias ON
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task $TASK --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --pathway_attn_bias $COMMON
done

RC=$?
echo "[pwattn] DONE rc=$RC date=$(date)"
