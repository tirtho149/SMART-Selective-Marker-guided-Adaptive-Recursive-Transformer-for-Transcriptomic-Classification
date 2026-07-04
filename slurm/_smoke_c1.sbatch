#!/usr/bin/env bash
#SBATCH --job-name=smart-c1-smoke
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/c1-smoke-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/c1-smoke-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1
echo "[c1-smoke] $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset Spleen \
    --modes none learned smooth_coexpr smooth_random route_coexpr route_random \
    --seeds 0 1 --epochs 40 --device cuda --out /tmp/c1_smoke --force
echo "[c1-smoke] DONE rc=$? $(date)"
