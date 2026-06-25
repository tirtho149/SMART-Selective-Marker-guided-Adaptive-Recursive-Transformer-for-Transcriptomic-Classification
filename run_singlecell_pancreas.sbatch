#!/usr/bin/env bash
#SBATCH --job-name=smart-sc-panc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-panc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-panc-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
VARIANTS=(shared independent mor); V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
case "$V" in
  shared) MF="--recursion_mode expert" ;;
  independent) MF="--recursion_mode expert --no_share_weights" ;;
  mor) MF="--recursion_mode token" ;;
esac
echo "[sc-panc] variant=$V $(date)"; nvidia-smi -L
$PY -m recursive_marker_transformer.singlecell --data data/singlecell \
    --out "results_singlecell_paper/$V" --datasets pancreas \
    --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda $MF
echo "[sc-panc] $V DONE rc=$? $(date)"
