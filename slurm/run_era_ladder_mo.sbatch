#!/usr/bin/env bash
#SBATCH --job-name=era-ladder-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-35%18
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/era-ladder-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/era-ladder-mo-%A_%a.out
# NEW canonical bioMoR = ERA (embedding + fuse + graph-router + Reactome ATTENTION bias) across
# the FULL ladder for Table 2, multi-omics. 6 (K,routing) configs x 6 cohorts = 36 tasks.
# Identical protocol to the old bio_both MO ladder (pathway, m256, d128, batch32, lr3e-4, e100,
# p15, sum), CV5 seed42 -- the ONLY change is A=--pathway_attn_bias. Every bioMoR row = ERA.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COHS=(prostate blca stad pan_meta_pri panmeta_response pan_meta_pri_3modal)
CHS=(mut_cnv mut_cnv mut_cnv mut_cnv expr mut_cnv_expr)
CFG_K=(2 3 4 2 3 4); CFG_R=(expert expert expert token token token)
CFG_N=(expert_k2 expert_k3 expert_k4 token_k2 token_k3 token_k4)
i=$SLURM_ARRAY_TASK_ID
ni=$(( i % 6 )); ci=$(( i / 6 ))
coh=${COHS[$ni]}; CH=${CHS[$ni]}
K=${CFG_K[$ci]}; R=${CFG_R[$ci]}; N=${CFG_N[$ci]}
echo "[era-mo] coh=$coh ch=$CH cfg=$N host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --recursion_mode "$R" \
    --recursion_depth "$K" --gene_interaction none \
    --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_attn_bias \
    --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 \
    --batch_size 32 --lr 3e-4 --cv_folds 5 --device cuda \
    --out "results/cv5/biomor_era_ladder_mo/$N"
echo "[era-mo] DONE coh=$coh cfg=$N rc=$? $(date)"
