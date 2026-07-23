#!/usr/bin/env bash
#SBATCH --job-name=biomor-pathcnn
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/pathcnn_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/pathcnn_%A_%a.out
set -uo pipefail
ROOT=/work/mech-ai-scratch/tirtho/bioMOR-baselines
cd "$ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1

# PathCNN builds pathway images from mutation+cnv (2-modal cohorts only).
COHORTS=(prostate blca stad brca pan_meta_pri)
c=${COHORTS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: cohort=$c"
python baseline_PathCNN/scripts/pathcnn_cv.py --cohort "$c"
echo "rc=$?  DONE $c"
