#!/usr/bin/env bash
#SBATCH --job-name=fig2-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-29%15
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fig2-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fig2-mo-%A_%a.out
# Figure 2 (None/E/R/A/ERA) for ALL 6 multi-omics cohorts, token mode, SAME 5-fold CV protocol
# (pathway, m256, d128, batch32, lr3e-4, e100, p15, sum). 5 conds x 6 cohorts = 30 tasks.
# E=learned graph + fuse (embedding smoothing); R=graph-conv router; A=Reactome attention bias.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COHS=(prostate blca stad pan_meta_pri panmeta_response pan_meta_pri_3modal)
CHS=(mut_cnv mut_cnv mut_cnv mut_cnv expr mut_cnv_expr)
CONDS=(N E R A ERA)
i=$SLURM_ARRAY_TASK_ID
ni=$(( i % 6 )); M=${CONDS[$(( i / 6 ))]}
coh=${COHS[$ni]}; CH=${CHS[$ni]}
BASE=(--task "$coh" --channels "$CH" --marker_mode pathway --gene_interaction none
      --recursion_mode token --recursion_depth 4 --pathway_pool sum --n_markers 256 --d_model 128
      --epochs 100 --patience 15 --batch_size 32 --lr 3e-4 --cv_folds 5 --seed 42 --device cuda)
case "$M" in
  N)   FLAGS=() ;;
  E)   FLAGS=(--pathway_learned_graph --pathway_learned_fuse) ;;
  R)   FLAGS=(--pathway_learned_graph --no_pathway_prop --bio_graph_router) ;;
  A)   FLAGS=(--pathway_learned_graph --no_pathway_prop --pathway_attn_bias) ;;
  ERA) FLAGS=(--pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_attn_bias) ;;
esac
echo "[fig2-mo] coh=$coh cond=$M host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks "${BASE[@]}" "${FLAGS[@]}" \
    --out "results/cv5/fig2_mo/${coh}/${M}"
echo "[fig2-mo] DONE coh=$coh cond=$M rc=$? $(date)"
