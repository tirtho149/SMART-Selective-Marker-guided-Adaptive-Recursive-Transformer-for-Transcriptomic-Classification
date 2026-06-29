#!/usr/bin/env bash
# ============================================================================
# Close the GitHub-data gap: run the remaining pathway cohorts (brca, panmeta_response)
# through the SAME architecture ablation as prostate/blca/stad/panmeta_subtype, so all
# 6 data-branch cohorts appear in the MoR tables.
#   brca: mutation-only 5-class -> pathway_pool=sum (sparse-mutation fix)
#   panmeta_response: expression, binary primary-vs-metastatic
#   out: results_pathway_arch/<variant>/<tag>.json
#   sbatch run_pathway_arch_extra.sbatch          # array 0-4 (recursion/sharing variants)
# ============================================================================
#SBATCH --job-name=smart-pwarchx
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarchx-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarchx-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(shared independent token fixed depth1)
declare -a FL
FL[0]="--recursion_mode expert"
FL[1]="--recursion_mode expert --no_share_weights"
FL[2]="--recursion_mode token"
FL[3]="--recursion_mode fixed"
FL[4]="--recursion_mode expert --recursion_depth 1"
V=${NAMES[$SLURM_ARRAY_TASK_ID]}; F="${FL[$SLURM_ARRAY_TASK_ID]}"
OUT="results_pathway_arch/$V"
C="--marker_mode pathway --gene_interaction reactome --out $OUT \
   --epochs 40 --d_model 128 --n_markers 256 --lr 3e-4 --patience 8 --device cuda --seed 0"
echo "[pwarchx] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

# brca: mutation-only 5-class, burden(sum) pooling
$PY -m recursive_marker_transformer.pathway_tasks --task brca --channels mut --pathway_pool sum --batch_size 32 $C $F
# panmeta_response: expression, binary
$PY -m recursive_marker_transformer.pathway_tasks --task panmeta_response --channels expr --batch_size 128 $C $F
echo "[pwarchx] variant=$V DONE rc=$? date=$(date)"
