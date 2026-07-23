#!/usr/bin/env bash
#SBATCH --job-name=mogat-full
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGAT/work_dirs/full_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGAT/work_dirs/full_%A_%a.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGAT
export PYTHONUNBUFFERED=1

SHARED=/work/mech-ai-scratch/tirtho/.venv
VENV=/work/mech-ai-scratch/tirtho/.venvs_baselines/mogat
source "$VENV/bin/activate"
export PYTHONPATH="$SHARED/lib/python3.11/site-packages:${PYTHONPATH:-}"

COHORTS=(prostate blca stad brca pan_meta_pri pan_meta_pri_3modal)
MODS=(mutation,cnv mutation,cnv mutation,cnv mutation,cnv mutation,cnv mutation,cnv,expression)
C=${COHORTS[$SLURM_ARRAY_TASK_ID]}
M=${MODS[$SLURM_ARRAY_TASK_ID]}
echo "[mogat-full] cohort=$C mods=$M"
COHORT=$C MODALITIES=$M python scripts/mogat_cv.py
echo "[mogat-full] cohort=$C rc=$?"
