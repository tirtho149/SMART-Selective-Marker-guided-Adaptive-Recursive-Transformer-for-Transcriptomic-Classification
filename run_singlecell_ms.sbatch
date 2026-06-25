#!/usr/bin/env bash
#SBATCH --job-name=smart-sc-ms
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-ms-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-ms-%A_%a.out
# Multi-seed single-cell under the genomap protocol: one array task per SEED (0-4),
# each runs all 3 variants x {tabula_muris, pancreas}. Test sets are the paper's
# fixed shipped splits; seeds vary init + val carve -> accuracy mean+/-std.
# Output: results_singlecell_ms/<variant>/seed<S>/<dataset>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SEED=$SLURM_ARRAY_TASK_ID
echo "[sc-ms] seed=$SEED $(date)"; nvidia-smi -L
COMMON="--data data/singlecell --datasets tabula_muris pancreas --epochs 150 \
        --d_model 96 --n_markers 128 --batch_size 128 --lr 0.001 \
        --weight_decay 0.00001 --patience 15 --device cuda --seed $SEED"
for VAR in shared independent mor; do
  case "$VAR" in
    shared) MF="--recursion_mode expert" ;;
    independent) MF="--recursion_mode expert --no_share_weights" ;;
    mor) MF="--recursion_mode token" ;;
  esac
  $PY -m recursive_marker_transformer.singlecell $COMMON $MF \
      --out "results_singlecell_ms/$VAR/seed$SEED"
  echo "[sc-ms] seed=$SEED $VAR rc=$? $(date)"
done
echo "[sc-ms] seed=$SEED DONE $(date)"
