#!/usr/bin/env bash
#SBATCH --job-name=smart-fuse-smoke
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fuse-smoke-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fuse-smoke-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1
echo "[fuse-smoke] $(date)"; nvidia-smi -L || true
echo "==== Spleen (good coexpr) ===="
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset Spleen \
    --modes learned learned_fused learned_fused_rand --seeds 0 1 --epochs 40 --device cuda \
    --out /tmp/fuse_smoke --force
echo "==== blca (Reactome) ===="
$PY -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort blca --task response \
    --modes learned learned_fused --seeds 0 1 --epochs 40 --cap_genes 3000 --device cuda \
    --out /tmp/fuse_smoke_pnet --force
echo "[fuse-smoke] DONE rc=$? $(date)"
