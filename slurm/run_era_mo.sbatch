#!/usr/bin/env bash
#SBATCH --job-name=era-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/era-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/era-mo-%A_%a.out
# ERA on the remaining multi-omics datasets for Table 2 (prostate/blca/stad Reactome + PC).
# PM & 3M already running in mo_7combo. Same protocol; token mode. -> results/cv5/era/<task>/ERA
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
TASKS=(prostate:mut_cnv blca:mut_cnv stad:mut_cnv panmeta_response:expr)
IFS=: read -r TASK CH <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
python -m recursive_marker_transformer.pathway_tasks --task "$TASK" --channels "$CH" \
    --marker_mode pathway --gene_interaction none --recursion_mode token --recursion_depth 4 \
    --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 \
    --lr 3e-4 --cv_folds 5 --seed 42 --device cuda \
    --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_attn_bias \
    --out "$PROJ/results/cv5/era/${TASK}/ERA"
echo "[era-mo] DONE rc=$? task=$TASK $(date)"
