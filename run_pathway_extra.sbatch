#!/usr/bin/env bash
# ============================================================================
# SMART -- brca collapse fix (burden/sum pooling) + pancancer_meta_pri (expression
# joined from the Xena PANCAN matrix; the branch ships pathways/labels but no omics).
#   panmeta_response = primary vs metastatic (binary, imbalanced 8225/361)
#   panmeta_subtype  = 32-class cancer type
#   sbatch run_pathway_extra.sbatch
# ============================================================================
#SBATCH --job-name=smart-pwextra
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwextra-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwextra-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[pwextra] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[pwextra] torch',torch.__version__,'cuda',torch.cuda.is_available())"

OUT=results_pathway
COMMON="--out $OUT --epochs 40 --d_model 128 --n_markers 256 --lr 3e-4 \
        --patience 8 --device cuda"

# ---- brca fix: pathway tokens with burden(sum) pooling for sparse mutation ----
for GI in reactome none; do
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task brca --channels mut --marker_mode pathway --gene_interaction $GI \
    --pathway_pool sum --batch_size 32 $COMMON
done

# ---- pancancer_meta_pri: two heads x {pathway+/-prior, router baseline} --------
for TASK in panmeta_response panmeta_subtype; do
  for GI in reactome none; do
    $PY -m recursive_marker_transformer.pathway_tasks \
      --task $TASK --channels expr --marker_mode pathway --gene_interaction $GI \
      --batch_size 128 $COMMON
  done
  $PY -m recursive_marker_transformer.pathway_tasks \
    --task $TASK --channels expr --marker_mode router --gene_interaction none \
    --batch_size 128 $COMMON
done

RC=$?
echo "[pwextra] DONE rc=$RC date=$(date)"
