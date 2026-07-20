#!/usr/bin/env bash
#SBATCH --job-name=none7-token
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/none7-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/none7-%A_%a.out
# No-biology (None) baseline in TOKEN mode for all 4 Figure-2 datasets, SAME protocol as the
# 7-combo runs (5-fold CV seed 42, epochs 100 pat 15). Completes the None/E/R/A/ERA figure.
#   0 Baron (SC)  1 Muraro (SC)  2 PM=pan_meta_pri (MO)  3 3M=pan_meta_pri_3modal (MO)
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

case "$SLURM_ARRAY_TASK_ID" in
  0) python -m recursive_marker_transformer.bio_learned_genomap --dataset Baron --modes N \
        --cv_folds 5 --K 4 --epochs 100 --patience 15 --n_markers 128 --seed 42 --device cuda \
        --out "$PROJ/results/cv5/baron_7combo" ;;
  1) python -m recursive_marker_transformer.bio_learned_genomap --dataset Muraro --modes N \
        --cv_folds 5 --K 4 --epochs 100 --patience 15 --n_markers 128 --seed 42 --device cuda \
        --out "$PROJ/results/cv5/muraro_7combo" ;;
  2) python -m recursive_marker_transformer.pathway_tasks --task pan_meta_pri --channels mut_cnv \
        --marker_mode pathway --gene_interaction none --recursion_mode token --recursion_depth 4 \
        --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 \
        --lr 3e-4 --cv_folds 5 --seed 42 --device cuda --out "$PROJ/results/cv5/mo_7combo/pan_meta_pri/N" ;;
  3) python -m recursive_marker_transformer.pathway_tasks --task pan_meta_pri_3modal --channels mut_cnv_expr \
        --marker_mode pathway --gene_interaction none --recursion_mode token --recursion_depth 4 \
        --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 \
        --lr 3e-4 --cv_folds 5 --seed 42 --device cuda --out "$PROJ/results/cv5/mo_7combo/pan_meta_pri_3modal/N" ;;
esac
echo "[none7] DONE rc=$? task=$SLURM_ARRAY_TASK_ID $(date)"
