#!/usr/bin/env bash
#SBATCH --job-name=biomor-pathformer
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/pathformer_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/pathformer_%A_%a.out
set -uo pipefail
ROOT=/work/mech-ai-scratch/tirtho/bioMOR-baselines
cd "$ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1

COHORTS=(prostate blca stad brca pan_meta_pri pan_meta_pri_3modal)
MODS=("mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv" "mutation cnv expression")
i=$SLURM_ARRAY_TASK_ID
c=${COHORTS[$i]}
echo "task $i: cohort=$c mods=${MODS[$i]}"
python baseline_Pathformer/scripts/pathformer_cv.py --cohort "$c" --modalities ${MODS[$i]}
echo "rc=$?  DONE $c"
