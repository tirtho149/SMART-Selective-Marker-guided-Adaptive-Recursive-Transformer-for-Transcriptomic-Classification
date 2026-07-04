#!/usr/bin/env bash
#SBATCH --job-name=smart-lb-pnet
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/lb-pnet-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/lb-pnet-%A_%a.out
# Table-1 5th column: LEARNED (bio-init) on the P-NET cohorts. Learned low-rank gene
# graph warm-started from the curated Reactome gene-gene operator, then refined
# end-to-end. One array task per cohort; learned_bio x 10 seeds.
# Output merges into results_bio_curated/pnet/<coh>__response/learned_bio_s<seed>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad brca)
COHORT="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
echo "[lb-pnet] start $(date) cohort=$COHORT"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort "$COHORT" --task response \
    --modes learned_bio --seeds 0 1 2 3 4 5 6 7 8 9 \
    --K 4 --epochs 60 --cap_genes 3000 --device cuda
echo "[lb-pnet] DONE rc=$? $(date)"
