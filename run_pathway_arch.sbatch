#!/usr/bin/env bash
# ============================================================================
# Run the GitHub pathway/P-NET cohorts through the 14-table architecture ablation
# (adaptive loop + parameter reduction), so the MoR tables include the new data.
# Pathway tokens + Reactome prior; one array task per recursion/sharing variant.
#   0 shared(expert)  1 independent  2 token  3 fixed  4 depth1(K=1)
#   tasks: prostate/blca/stad (mut_cnv) + panmeta_subtype (expr, 32-class)
#   out: results_pathway_arch/<variant>/<tag>.json
#   sbatch run_pathway_arch.sbatch          # array 0-4
# ============================================================================
#SBATCH --job-name=smart-pwarch
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarch-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarch-%A_%a.out
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
echo "[pwarch] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

for T in prostate blca stad; do
  $PY -m recursive_marker_transformer.pathway_tasks --task $T --channels mut_cnv $C $F
done
$PY -m recursive_marker_transformer.pathway_tasks --task panmeta_subtype --channels expr --batch_size 128 $C $F
echo "[pwarch] variant=$V DONE rc=$? date=$(date)"
