#!/usr/bin/env bash
#SBATCH --job-name=mo7-smoke
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --array=2,6
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo7-smoke-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo7-smoke-%A_%a.out
# Multi-omics (PM=pan_meta_pri mut_cnv, 3M=pan_meta_pri_3modal mut_cnv_expr) 7-COMBINATION
# biology ablation, SAME protocol as the injection figure (5-fold CV seed 42, epochs 100 pat 15,
# marker_mode pathway, K=4, d=128, M=256). Three binary mechanisms, all in token mode (attention
# bias requires recursion_mode=token): E=embedding (learned graph + fuse), R=router (graph-conv),
# A=attention bias (Reactome pathway attention). 7 non-empty combos x 2 datasets = 14 tasks.
#   index = dataset_idx*7 + combo_idx ;  datasets=(PM 3M), combos=(E R A ER EA RA ERA)
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

TASKS=(pan_meta_pri:mut_cnv pan_meta_pri_3modal:mut_cnv_expr)
COMBOS=(E R A ER EA RA ERA)
di=$(( SLURM_ARRAY_TASK_ID / 7 )); ci=$(( SLURM_ARRAY_TASK_ID % 7 ))
IFS=: read -r TASK CH <<< "${TASKS[$di]}"
M="${COMBOS[$ci]}"

BASE=(--task "$TASK" --channels "$CH" --marker_mode pathway --gene_interaction none
      --recursion_mode token --recursion_depth 4 --pathway_pool sum --n_markers 256 --d_model 128
      --epochs 3 --patience 2 --batch_size 32 --lr 3e-4 --cv_folds 2 --seed 42 --device cuda)

# per-combo mechanism flags. E present -> learned graph + input smoothing (fuse); E absent but
# R/A present -> learned graph but no input smoothing (--no_pathway_prop). R -> bio_graph_router;
# A -> pathway_attn_bias.
case "$M" in
  E)   FLAGS=(--pathway_learned_graph --pathway_learned_fuse) ;;
  R)   FLAGS=(--pathway_learned_graph --no_pathway_prop --bio_graph_router) ;;
  A)   FLAGS=(--pathway_learned_graph --no_pathway_prop --pathway_attn_bias) ;;
  ER)  FLAGS=(--pathway_learned_graph --pathway_learned_fuse --bio_graph_router) ;;
  EA)  FLAGS=(--pathway_learned_graph --pathway_learned_fuse --pathway_attn_bias) ;;
  RA)  FLAGS=(--pathway_learned_graph --no_pathway_prop --bio_graph_router --pathway_attn_bias) ;;
  ERA) FLAGS=(--pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_attn_bias) ;;
esac
OUT="$PROJ/results/cv5/mo_7combo_smoke/${TASK}/${M}"

echo "[mo7-smoke] $(date) task=$TASK ch=$CH combo=$M host=$(hostname)"; nvidia-smi -L || true
python -m recursive_marker_transformer.pathway_tasks "${BASE[@]}" "${FLAGS[@]}" --out "$OUT"
echo "[mo7-smoke] DONE rc=$? task=$TASK combo=$M $(date)"
