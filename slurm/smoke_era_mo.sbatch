#!/usr/bin/env bash
#SBATCH --job-name=smoke-era-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-era-mo-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-era-mo-%j.out
# CPU smoke: multi-omics ERA (pathway_attn_bias) in EXPERT-choice routing must now stay expert
# (no forced token) and exercise the per-step (k,k) pathway-attention sub-mask gather.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m recursive_marker_transformer.pathway_tasks --task prostate --channels mut_cnv \
    --marker_mode pathway --recursion_mode expert --recursion_depth 4 --gene_interaction none \
    --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_attn_bias \
    --pathway_pool sum --n_markers 128 --d_model 64 --epochs 2 --patience 5 --batch_size 32 \
    --lr 3e-4 --cv_folds 2 --device cpu --out "$PROJ/results/cv5/_smoke_era_mo/expert"
echo "[smoke-era-mo] rc=$? DONE $(date)"
