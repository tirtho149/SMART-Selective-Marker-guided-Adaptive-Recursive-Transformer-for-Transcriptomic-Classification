#!/usr/bin/env bash
# ============================================================================
# Push the Tier-2 genomap datasets (baron, segerstolpe) through the FULL 14-table
# experiment suite so they appear in every MoR table alongside the 4 capsule
# datasets. One array task per experiment family; writes into the same result dirs.
#   0 arch  1 sharing  2 stepcache  3 routing  4 scaling  5 warmstart  6 depth
#   sbatch run_tier2_suite.sbatch          # array 0-6
# ============================================================================
#SBATCH --job-name=smart-tier2suite
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tier2suite-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tier2suite-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DS="baron segerstolpe"
SC() { $PY -m recursive_marker_transformer.singlecell --data data/singlecell \
        --datasets $DS --d_model 96 --n_markers 128 --batch_size 128 --lr 0.001 \
        --weight_decay 0.00001 --patience 12 --device cuda --seed 0 "$@"; }
echo "[tier2suite] task=$SLURM_ARRAY_TASK_ID host=$(hostname) date=$(date)"; nvidia-smi -L || true

case $SLURM_ARRAY_TASK_ID in
 0) # arch ablation (seed 0) -> results_singlecell_arch/<v>/s0
    SC --out results_singlecell_arch/shared/s0       --epochs 150 --recursion_mode expert
    SC --out results_singlecell_arch/independent/s0  --epochs 150 --recursion_mode expert --no_share_weights
    SC --out results_singlecell_arch/token/s0        --epochs 150 --recursion_mode token
    SC --out results_singlecell_arch/fixed/s0        --epochs 150 --recursion_mode fixed
    SC --out results_singlecell_arch/depth1/s0       --epochs 150 --recursion_mode expert --recursion_depth 1
    SC --out results_singlecell_arch/marker_random/s0 --epochs 150 --recursion_mode expert --marker_mode random
    SC --out results_singlecell_arch/marker_var/s0   --epochs 150 --recursion_mode expert --marker_mode variance ;;
 1) # sharing schemes (K=6) -> results_sharing/<tag>
    SC --out results_sharing/shared_nu1          --epochs 80 --recursion_mode fixed --recursion_depth 6 --share_strategy cycle --n_unique_blocks 1
    SC --out results_sharing/independent_nu6     --epochs 80 --recursion_mode fixed --recursion_depth 6 --share_strategy cycle --n_unique_blocks 6
    SC --out results_sharing/cycle_nu3           --epochs 80 --recursion_mode fixed --recursion_depth 6 --share_strategy cycle --n_unique_blocks 3
    SC --out results_sharing/sequence_nu3        --epochs 80 --recursion_mode fixed --recursion_depth 6 --share_strategy sequence --n_unique_blocks 3
    SC --out results_sharing/middle_cycle_nu3    --epochs 80 --recursion_mode fixed --recursion_depth 6 --share_strategy middle_cycle --n_unique_blocks 3
    SC --out results_sharing/middle_sequence_nu3 --epochs 80 --recursion_mode fixed --recursion_depth 6 --share_strategy middle_sequence --n_unique_blocks 3 ;;
 2) # step-cache -> results_stepcache/<off|on>
    SC --out results_stepcache/off --epochs 80 --recursion_mode fixed
    SC --out results_stepcache/on  --epochs 80 --recursion_mode fixed --step_cache ;;
 3) # routing configs -> results_routing/<tag>
    SC --out results_routing/exp_linear --epochs 80 --recursion_mode expert --router_type linear
    SC --out results_routing/exp_mlp    --epochs 80 --recursion_mode expert --router_type mlp
    SC --out results_routing/exp_temp2  --epochs 80 --recursion_mode expert --router_type linear --router_temp 2.0
    SC --out results_routing/tok_linear --epochs 80 --recursion_mode token  --router_type linear
    SC --out results_routing/tok_mlp    --epochs 80 --recursion_mode token  --router_type mlp
    SC --out results_routing/tok_bal01  --epochs 80 --recursion_mode token  --router_type linear --router_balance_coeff 0.01 ;;
 4) # scaling -> results_scaling/<arch>_d<D>
    for D in 48 96 192 384; do
      SC --out results_scaling/vanilla_d$D   --epochs 80 --recursion_mode fixed  --no_share_weights --d_model $D --recursion_depth 4
      SC --out results_scaling/recursive_d$D --epochs 80 --recursion_mode fixed                     --d_model $D --recursion_depth 4
      SC --out results_scaling/mor_d$D       --epochs 80 --recursion_mode expert                    --d_model $D --recursion_depth 4
    done ;;
 5) # warm-start -> results_warmstart
    $PY -m recursive_marker_transformer.warmstart --data data/singlecell --out results_warmstart \
        --datasets $DS --epochs 80 --d_model 96 --device cuda --seed 0 ;;
 6) # depth viz -> results_depth
    $PY -m recursive_marker_transformer.depth_viz --data data/singlecell --out results_depth \
        --datasets $DS --epochs 60 --d_model 96 --device cuda ;;
esac
echo "[tier2suite] task=$SLURM_ARRAY_TASK_ID DONE rc=$? date=$(date)"
