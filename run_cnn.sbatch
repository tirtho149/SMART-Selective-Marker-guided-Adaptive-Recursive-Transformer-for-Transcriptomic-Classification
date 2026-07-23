#!/usr/bin/env bash
#SBATCH --job-name=biomor-cnn
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/cnn_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/cnn_%A_%a.out
set -uo pipefail
ROOT=/work/mech-ai-scratch/tirtho/bioMOR-baselines
cd "$ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1

# task layout: (model, cohort, modalities)
MODELS=(ei ei ei ei ei ei li li li li li li)
COHORTS=(prostate blca stad brca pan_meta_pri pan_meta_pri_3modal \
         prostate blca stad brca pan_meta_pri pan_meta_pri_3modal)
MODS=("mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv expression" \
      "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv expression")

i=$SLURM_ARRAY_TASK_ID
echo "task $i: model=${MODELS[$i]} cohort=${COHORTS[$i]} mods=${MODS[$i]}"
python baseline_CNN/scripts/cnn_cv.py --model "${MODELS[$i]}" --cohort "${COHORTS[$i]}" --modalities ${MODS[$i]}
echo "rc=$?  DONE task $i"
