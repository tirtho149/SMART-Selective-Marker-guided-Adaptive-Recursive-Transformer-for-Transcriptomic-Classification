#!/usr/bin/env bash
#SBATCH --job-name=smart-token13
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/token13-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/token13-%A_%a.out
# =============================================================================
# 13-dataset paper -- Table 3 (token reduction): macro-F1 as marker budget M shrinks.
# One array task per M, 3 seeds, over the 9 genomap single-cell sets + 4 P-NET cohorts
# (brca). Output: results_token13/M<M>/s<seed>/<ds>.json (SC) and
# results_pwtoken13/M<M>/s<seed>/<tag>.json (cohorts).
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
M=(16 32 64 128 256); MM=${M[$SLURM_ARRAY_TASK_ID]}
echo "[token13] M=$MM host=$(hostname) date=$(date)"; nvidia-smi -L || true

COHORTS=(prostate blca stad brca)
for SEED in 0 1 2; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_token13/M$MM/s$SEED" \
      --datasets baron lung muraro oesophagus segerstolpe spleen tcell wang xin \
      --epochs 80 --d_model 96 --n_markers $MM --batch_size 128 --lr 0.001 \
      --weight_decay 0.00001 --patience 12 --device cuda --recursion_mode expert --seed $SEED
  for TASK in "${COHORTS[@]}"; do
    $PY -m recursive_marker_transformer.pathway_tasks \
        --task $TASK --channels mut_cnv --marker_mode router --gene_interaction none \
        --n_markers $MM --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
        --batch_size 32 --seed $SEED --device cuda \
        --out "results_pwtoken13/M$MM/s$SEED"
  done
done
echo "[token13] M=$MM DONE rc=$? date=$(date)"
