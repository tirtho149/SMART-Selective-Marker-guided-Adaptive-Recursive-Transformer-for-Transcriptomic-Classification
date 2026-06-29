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

#SBATCH --job-name=smart-pathway
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pathway-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pathway-%j.out
# =============================================================================
# SMART on the Reactome / P-NET pathway-informed multi-omics cohorts
# (data/<cohort>/: mutation + CNV per gene, Reactome gene->pathway membership +
# pathway->pathway hierarchy). M tokens = Reactome pathways.
#
# Ablation grid (per cohort):
#   token type:   pathway  vs  router (learned marker tokens)
#   router prior: reactome (curated hierarchy) vs none
#   modality:     mut | cnv | mut_cnv   (prostate sweep)
#
#   sbatch run_pathway.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[pathway] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[pathway] torch',torch.__version__,'cuda',torch.cuda.is_available())"

OUT=results_pathway
COMMON="--out $OUT --epochs 40 --d_model 128 --n_markers 256 --batch_size 32 \
        --lr 3e-4 --patience 8 --device cuda"

# ---- binary cohorts: full token-type x prior ablation on mut+cnv ----
for TASK in prostate blca stad; do
  for MODE in pathway router; do
    for GI in reactome none; do
      # 'reactome' prior only applies to pathway tokens; skip router+reactome
      [ "$MODE" = "router" ] && [ "$GI" = "reactome" ] && continue
      $PY -m recursive_marker_transformer.pathway_tasks \
        --task $TASK --channels mut_cnv --marker_mode $MODE --gene_interaction $GI $COMMON
    done
  done
done

# ---- brca (mutation only, 5-class subtype) ----
# mutation is ~0.4% sparse, so pathway tokens pool by SUM/burden (mean washes the
# signal to a constant -> collapse); the router baseline selects genes (pool n/a).
for GI in reactome none; do
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task brca --channels mut --marker_mode pathway --gene_interaction $GI \
    --pathway_pool sum $COMMON
done
$PY -m recursive_marker_transformer.pathway_tasks \
  --task brca --channels mut --marker_mode router --gene_interaction none $COMMON

# ---- pancancer_meta_pri (expression joined from Xena; no omics on the branch) --
#   panmeta_response = primary vs metastatic (binary); panmeta_subtype = 32-class
for TASK in panmeta_response panmeta_subtype; do
  for GI in reactome none; do
    $PY -m recursive_marker_transformer.pathway_tasks \
      --task $TASK --channels expr --marker_mode pathway --gene_interaction $GI \
      $COMMON --batch_size 128
  done
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task $TASK --channels expr --marker_mode router --gene_interaction none \
    $COMMON --batch_size 128
done

# ---- modality ablation on prostate (pathway + reactome) ----
$PY -m recursive_marker_transformer.pathway_tasks \
  --task prostate --channels mut cnv --marker_mode pathway --gene_interaction reactome $COMMON

# ---- hierarchy attention-bias ablation (token-choice; with vs without) --------
# Pathway tokens attend along the Reactome pathway->pathway graph. The matched
# control is the SAME token-choice router with the bias OFF, isolating the graph.
for TASK in prostate blca stad; do
  # without bias (token-choice control)
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task $TASK --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --recursion_mode token $COMMON
  # with Reactome hierarchy attention bias
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task $TASK --channels mut_cnv --marker_mode pathway --gene_interaction reactome \
    --pathway_attn_bias $COMMON
done

RC=$?
echo "[pathway] DONE rc=$RC date=$(date)"
