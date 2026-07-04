#!/usr/bin/env bash
#SBATCH --job-name=smart-pwarch13
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarch13-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarch13-%A_%a.out
# =============================================================================
# 13-dataset paper -- architecture/routing ablation on the 4 P-NET cohorts
# (prostate, blca, stad, brca -- brca REPLACES panmeta_subtype). Feeds Table 5
# ladder + Table 1 accuracy for the P-NET side. One array task per config, 3 seeds,
# all 4 cohorts. Output: results_pw13/<variant>/s<seed>/<tag>.json
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(shared independent token fixed depth1 none)
declare -a FLAGS
FLAGS[0]="--gene_interaction reactome --recursion_mode expert"                       # shared expert MoR (headline)
FLAGS[1]="--gene_interaction reactome --recursion_mode expert --no_share_weights"    # independent (vanilla)
FLAGS[2]="--gene_interaction reactome --recursion_mode token"                        # token-choice
FLAGS[3]="--gene_interaction reactome --recursion_mode fixed"                        # fixed depth
FLAGS[4]="--gene_interaction reactome --recursion_mode expert --recursion_depth 1"   # K=1
FLAGS[5]="--gene_interaction none --recursion_mode expert"                           # general router (no prior)

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[pwarch13] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

COHORTS=(prostate blca stad brca)
for SEED in 0 1 2; do
  for TASK in "${COHORTS[@]}"; do
    $PY -m recursive_marker_transformer.pathway_tasks \
        --task $TASK --channels mut_cnv --marker_mode pathway \
        --n_markers 256 --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
        --batch_size 32 --seed $SEED --device cuda \
        --out results_pw13/$V/s$SEED $F
  done
done
echo "[pwarch13] variant=$V DONE rc=$? date=$(date)"
