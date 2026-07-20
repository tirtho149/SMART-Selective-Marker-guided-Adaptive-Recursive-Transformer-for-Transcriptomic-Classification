#!/usr/bin/env bash
#SBATCH --job-name=stad-attn-ablation
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/stad-attn-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/stad-attn-%A_%a.out
# EXPLORATORY: STAD (stomach) under the canonical multi-omics bioMoR config, WITH vs
# WITHOUT the pathway attention bias (--pathway_attn_bias), on the shared 5-fold folds
# (cv.cv_folds, seed 42). Same everything else, so the delta isolates the attention bias.
#   0 with_attn : canonical config (includes --pathway_attn_bias)
#   1 no_attn   : identical config MINUS --pathway_attn_bias
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# canonical multi-omics bioMoR config (matches slurm/run_canonical_biomor_mo.sbatch)
BASE=(--task stad --channels mut_cnv --marker_mode pathway --gene_interaction reactome
      --pathway_learned_graph --pathway_learned_fuse --pathway_pool sum --recursion_mode token
      --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 --cv_folds 5
      --seed 42 --device cuda)

case "$SLURM_ARRAY_TASK_ID" in
  0) NAME=with_attn; EXTRA=(--pathway_attn_bias) ;;
  1) NAME=no_attn;   EXTRA=() ;;
esac
OUT="$PROJ/results/cv5/stad_attn_ablation/$NAME"

echo "[stad-attn] $(date) arm=$NAME host=$(hostname)"; nvidia-smi -L || true
python -m recursive_marker_transformer.pathway_tasks "${BASE[@]}" "${EXTRA[@]}" --out "$OUT"
echo "[stad-attn] DONE rc=$? arm=$NAME $(date)"
