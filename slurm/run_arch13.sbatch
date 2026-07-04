#!/usr/bin/env bash
#SBATCH --job-name=smart-arch13
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/arch13-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/arch13-%A_%a.out
# =============================================================================
# 13-dataset paper -- architecture/routing ablation on the 9 genomap single-cell
# datasets. Feeds Table 5 (Vanilla/Recursive/MoR ladder) + Table 1 accuracy cross-
# check. One array task per variant, 3 seeds. Trains on the SAME materialised arrays
# as the learned Table-1 sweep. Output: results_arch13/<variant>/s<seed>/<ds>.json
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(shared independent token fixed depth1 marker_random marker_var)
declare -a FLAGS
FLAGS[0]="--recursion_mode expert"
FLAGS[1]="--recursion_mode expert --no_share_weights"
FLAGS[2]="--recursion_mode token"
FLAGS[3]="--recursion_mode fixed"
FLAGS[4]="--recursion_mode expert --recursion_depth 1"
FLAGS[5]="--recursion_mode expert --marker_mode random"
FLAGS[6]="--recursion_mode expert --marker_mode variance"

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[arch13] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

for SEED in 0 1 2; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_arch13/$V/s$SEED" \
      --datasets baron lung muraro oesophagus segerstolpe spleen tcell wang xin \
      --epochs 100 --d_model 96 --n_markers 128 --batch_size 128 \
      --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda \
      --seed $SEED $F
done
echo "[arch13] variant=$V DONE rc=$? date=$(date)"
