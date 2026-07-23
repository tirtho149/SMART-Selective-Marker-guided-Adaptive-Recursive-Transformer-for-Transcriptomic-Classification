#!/usr/bin/env bash
#SBATCH --job-name=biomor-dpn
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/dpn_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/dpn_%A_%a.out
set -uo pipefail
ROOT=/work/mech-ai-scratch/tirtho/bioMOR-baselines
cd "$ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1

CFGS=(prostate blca stad brca pan_meta_pri pan_meta_pri_3modal)
c=${CFGS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: cohort=$c"
python baseline_DeePathNet/scripts/deepathnet_biomor_cv.py \
    baseline_DeePathNet/configs/biomor/${c}.json
echo "rc=$?  DONE $c"
