#!/usr/bin/env bash
#SBATCH --job-name=smart-lb-genomap
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/lb-genomap-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/lb-genomap-%A_%a.out
# Table-1 5th column: LEARNED (bio-init) on the genomap single-cell suite. Learned
# low-rank gene graph warm-started from the co-expression graph (top-r eigenvectors),
# then refined end-to-end. One array task per dataset; learned_bio x 10 seeds.
# Output merges into results_learned_genomap/<Ds>/learned_bio_s<seed>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[lb-genomap] start $(date) dataset=$DS"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes learned_bio --seeds 0 1 2 3 4 5 6 7 8 9 \
    --K 4 --epochs 60 --device cuda
echo "[lb-genomap] DONE rc=$? $(date)"
