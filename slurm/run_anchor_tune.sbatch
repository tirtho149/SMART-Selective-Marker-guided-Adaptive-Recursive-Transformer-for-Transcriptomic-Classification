#!/usr/bin/env bash
#SBATCH --job-name=smart-anchtune
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/anchtune-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/anchtune-%A_%a.out
# Round 1: make the bio-informed learned router WIN via a PERSISTENT anchor (floor>0)
# + strength/rank grid. Config selected by validation macro-F1, evaluated on test.
#   base   : learned (random init)             results_anchor_tune/base
#   cfgA   : lam0=0.5 floor=0.3 rank=16         results_anchor_tune/cfgA
#   cfgB   : lam0=1.0 floor=0.3 rank=32         results_anchor_tune/cfgB
#   cfgC   : lam0=2.0 floor=0.5 rank=32         results_anchor_tune/cfgC
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Xin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[anchtune] $DS $(date)"; nvidia-smi -L || true
run() { $PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
        --seeds 0 1 2 --K 4 --epochs 50 --device cuda "$@"; }
run --modes learned        --out results_anchor_tune/base
run --modes learned_anchor --out results_anchor_tune/cfgA --anchor_lambda 0.5 --anchor_floor 0.3 --anchor_rank 16
run --modes learned_anchor --out results_anchor_tune/cfgB --anchor_lambda 1.0 --anchor_floor 0.3 --anchor_rank 32
run --modes learned_anchor --out results_anchor_tune/cfgC --anchor_lambda 2.0 --anchor_floor 0.5 --anchor_rank 32
echo "[anchtune] DONE rc=$? $(date)"
